//! Type-2 equivalence sequentialization for edge orders using orbit (bits-back) coding.
//!
//! We treat an ordered edge list as the "ordered object" (length m).
//! The unordered object is the *set* of edges (i.e., the graph).
//!
//! At each step k, we:
//!   (1) build an orbit partition of the remaining-edge graph G_k
//!   (2) decode an orbit id o_k ~ q(o_k | G_k) (ANS pop; bits-back)
//!   (3) deterministically choose a representative edge inside that orbit
//!   (4) remove that edge -> G_{k-1}
//!   (5) encode the removed edge with p( edge | G_{k-1} )  (ANS push)
//
// This matches the control flow of AutoregressivePrefixShuffleCodec:
// orbit is decoded first (pop), then a slice is encoded (push).  :contentReference[oaicite:2]{index=2}

use crate::autoregressive::prefix_orbit::FixPrefixOrbitCodec;
use crate::autoregressive::{
    InnerSliceCodecs, PrefixFn, PrefixingChain,
};
use crate::codec::{Codec, LogUniform, Message, Uniform};
use crate::graph::{ColorRefinement, EdgeIndex, PlainGraph, Undirected};
use crate::permutable::{Len, Permutable, Unordered, Hashing};
use itertools::Itertools;
use std::marker::PhantomData;
use std::fmt::Debug;
// use std::collections::BTreeMap;
use crate::autoregressive::UnfusedAutoregressiveShuffleCodec;


/// An ordered edge list (the thing we permute).
/// `len_active` is the current prefix length in the prefix chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeList {
    pub n: usize,
    pub edges: Vec<EdgeIndex>, // length m_total
}

impl Permutable for EdgeList {
    fn len(&self) -> usize {
        self.edges.len()
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.edges.swap(i, j);
    }
}

/// Prefix = remaining edges graph G_k (as a set) + an ordering container that we can swap/pop from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgePrefix {
    /// Remaining graph G_k as a PlainGraph.
    pub g: PlainGraph<Undirected>,
    /// Ordered container of remaining edges (we remove by swapping target to the back).
    pub list: EdgeList,
    /// Active prefix length (first `len_active` entries in list.edges are "still present").
    pub len_active: usize,
}

impl Permutable for EdgePrefix {
    fn len(&self) -> usize {
        self.len_active
    }

    fn swap(&mut self, i: usize, j: usize) {
        // Only swap within active region.
        assert!(i < self.len_active && j < self.len_active);
        self.list.swap(i, j);
    }
}

/// Slice = one removed edge.
pub type EdgeSlice = EdgeIndex;

fn canon_edge((u,v): EdgeIndex) -> EdgeIndex { if u < v {(u,v)} else {(v,u)} }


/// Prefixing chain that removes edges from the remaining graph.
/// pop_slice removes the *last active* edge (after swaps by orbit selection).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgePrefixingChain {
    phantom: PhantomData<()>,
}

impl EdgePrefixingChain {
    pub fn new() -> Self {
        Self { phantom: PhantomData }
    }
}

impl PrefixingChain for EdgePrefixingChain {
    type Prefix = EdgePrefix;
    type Full = EdgeList;
    type Slice = EdgeSlice;

    fn pop_slice(&self, prefix: &mut Self::Prefix) -> Self::Slice {
        // Remove the last active edge from the ordered list.
        let idx = prefix.len_active - 1;
        let e = canon_edge(prefix.list.edges[idx]);
        prefix.len_active -= 1;

        // Remove from remaining graph G_k -> G_{k-1}.
        // (Graph is undirected PlainGraph; remove both directed entries handled internally.) :contentReference[oaicite:3]{index=3}
        prefix.g.remove_plain_edge(e);

        e
    }

    fn push_slice(&self, prefix: &mut Self::Prefix, slice: &Self::Slice) {
        let e = canon_edge(*slice);
        let idx = prefix.len_active;
        prefix.list.edges[idx] = e;
        prefix.len_active += 1;
        prefix.g.insert_plain_edge(e);
    }

    fn prefix(&self, full: Self::Full) -> Self::Prefix {
        // Build the remaining graph G_m from all edges.
        let mut g = PlainGraph::plain_empty(full.n);
        for &e in &full.edges {
            g.insert_plain_edge(canon_edge(e));
        }
        let len_active = full.edges.len();
        EdgePrefix { g, list: full, len_active }
    }

    fn full(&self, prefix: Self::Prefix) -> Self::Full {
        prefix.list
    }
}

/// --- Orbit IDs for edges (your "sig") --------------------------------------
///
/// We approximate edge orbits using:
///   - node colors from ColorRefinement on the remaining graph
///   - degrees
///   - (min/max) normalization for undirected edges
///
/// This is exactly your sig(e) idea; the only subtlety is:
/// edges in the same bucket should share the SAME computed id at that step,
/// but the id can change after removals because colors/degrees change. :contentReference[oaicite:4]{index=4}
fn edge_sig(node_color: &[usize], deg: &[usize], (u, v): EdgeIndex) -> (usize, usize, usize, usize) {
    let (cu, cv) = (node_color[u], node_color[v]);
    let (du, dv) = (deg[u], deg[v]);
    let c0 = cu.min(cv);
    let c1 = cu.max(cv);
    let d0 = du.min(dv);
    let d1 = du.max(dv);
    (c0, c1, d0, d1)
}


/// A simple stable hash into a single OrdSymbol.
/// We keep it as u64 so it can be ranked to orbit ids by FixPrefixOrbitCodec. :contentReference[oaicite:5]{index=5}
fn hash_sig(sig: (usize, usize, usize, usize)) -> u64 {
    let (a, b, c, d) = sig;

    let mut x: u64 = (a as u64).wrapping_mul(0x9E3779B97F4A7C15);
    x ^= (b as u64)
        .wrapping_add(0xD1B54A32D192ED03)
        .rotate_left(17);
    x ^= (c as u64)
        .wrapping_mul(0x94D049BB133111EB)
        .rotate_left(31);
    x ^= (d as u64)
        .wrapping_mul(0xBF58476D1CE4E5B9)
        .rotate_left(47);

    x
}



/// Orbit-codecs factory: given a prefix (remaining graph), build a distribution over edge-orbit ids.
/// This implements q(o_k | G_k) ∝ |orbit| by using FixPrefixOrbitCodec masses = orbit sizes. :contentReference[oaicite:6]{index=6}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeOrbitCodecs {
    pub cr: ColorRefinement,
}

impl PrefixFn<EdgePrefixingChain> for EdgeOrbitCodecs {
    type Output = FixPrefixOrbitCodec;

    fn apply(&self, x: &EdgePrefix) -> Self::Output {
        // Remaining graph G_k:
        let g = &x.g;

        // Node "colors": use ColorRefinement hashes. :contentReference[oaicite:7]{index=7}
        // ColorRefinement::apply returns Vec<u64> (node hashes) for graphs in this repo. :contentReference[oaicite:8]{index=8}
        let colors: Vec<usize> = <ColorRefinement as Hashing<PlainGraph<Undirected>>>::apply(&self.cr, g);


        
        // Degrees:
        let deg = g.degrees().collect_vec();

        // Compute edge ids for the *ordered container positions* [0..m_total),
        // but only first len_active are relevant at this step.
        let mut ids: Vec<u64> = Vec::with_capacity(x.len_active);
        for &e in x.list.edges[..x.len_active].iter() {
            let e = canon_edge(e);
            let sig = edge_sig(&colors, &deg, e);
            ids.push(hash_sig(sig));
        }

        FixPrefixOrbitCodec::new(ids, x.len_active)
    }

    fn update_after_pop_slice(&self, image: &mut Self::Output, x: &EdgePrefix, _slice: &EdgeSlice) {
        // Simplicity: recompute from scratch (correct, but not the fastest).
        // You can incrementally update later.
        *image = self.apply(x);
    }

    fn update_after_push_slice(&self, image: &mut Self::Output, x: &EdgePrefix, _slice: &EdgeSlice) {
        *image = self.apply(x);
    }

    fn swap(&self, image: &mut Self::Output, i: usize, j: usize) {
        // Keep the orbit codec consistent with the prefix swaps.
        image.swap(i, j);
    }
}

// impl OrbitCodecs<EdgePrefixingChain> for EdgeOrbitCodecs {}

/// --- p(edge | G_{k-1}) -----------------------------------------------------
///
/// “True orbit-coded p” in your terms = p models the removed edge under the
/// *post-removal* remaining graph G_{k-1}.
///
/// This trait lets you plug in any model you want.
/// The default implementation below is a simple uniform over a candidate set,
/// but the interface supports a fully-featured orbit-coded p.

// #[derive(Clone, Copy, Debug)]
// pub struct EndpointsCodec {
//     pub n: usize,
// }

// impl Codec for EndpointsCodec {
//     type Symbol = (usize, usize);

//     fn push(&self, m: &mut Message, x: &Self::Symbol) {
//         let (u, v) = *x;
//         debug_assert!(u < self.n && v < self.n && u != v);

//         // encode u in [0, n)
//         EndpointsCodec{ n }.push(m, &u);

//         // encode v in [0, n-1) after "skipping" u
//         let v_packed = if v < u { v } else { v - 1 };
//         EndpointsCodec{self.n - 1}.push(m, &v_packed);
//     }

//     fn pop(&self, m: &mut Message) -> Self::Symbol {
//         let v_packed = EndpointsCodec{self.n - 1}.pop(m);
//         let u = EndpointsCodec{ n }.pop(m);

//         let v = if v_packed < u { v_packed } else { v_packed + 1 };
//         (u, v)
//     }

//     fn bits(&self, x: &Self::Symbol) -> Option<f64> {
//         let (u, v) = *x;
//         if u >= self.n || v >= self.n || u == v { return None; }
//         Some(EndpointsCodec{ n }.uni_bits() + EndpointsCodec{self.n - 1}.uni_bits())
//     }
// }


// pub trait EdgeRemovalModel: Clone + Debug + Eq + PartialEq + Send + Sync + 'static {
//     /// Build a codec to encode the removed edge given the *current* prefix (which is already G_{k-1}).
//     fn codec_for_removed_edge(&self, prefix_after_removal: &EdgePrefix) -> Uniform;
// }


// /// Minimal baseline: encode the removed edge as a uniform index into the active edge list.
// /// (You will replace this with your option (B) model.)
// #[derive(Clone, Debug, Eq, PartialEq)]
// pub struct UniformEdgeModel;

// impl EdgeRemovalModel for UniformEdgeModel {
//     fn codec_for_removed_edge(&self, prefix_after_removal: &EdgePrefix) -> Uniform {
//         // We encode an index into [0..m_total], but only the removed edge itself is known at that moment.
//         // Simpler: encode endpoints directly with Uniform(n) twice (wastes bits).
//         let n = prefix_after_removal.list.n;
//         #[derive(Clone, Debug, Eq, PartialEq)]
//         struct EndpointsCodec {
//             n: usize,
//         }
//         impl Codec for EndpointsCodec {
//             type Symbol = EdgeSlice;
//             fn push(&self, m: &mut Message, x: &Self::Symbol) {
//                 let (u, v) = *x;
//                 Uniform::new(self.n).push(m, &u);
//                 Uniform::new(self.n).push(m, &v);
//             }
//             fn pop(&self, m: &mut Message) -> Self::Symbol {
//                 let u = Uniform::new(self.n).pop(m);
//                 let v = Uniform::new(self.n).pop(m);
//                 (u, v)
//             }
//             fn bits(&self, _x: &Self::Symbol) -> Option<f64> {
//                 Some(2.0 * Uniform::new(self.n).uni_bits())
//             }
//         }
//         Uniform::new(n)

//     }
// }



/// Orbit-coded p model (baseline):
/// - Candidate set A(G) = all non-edges among 0..n-1.
/// - Bucket candidates by (approx) orbit key.
/// - p(bucket) ∝ bucket size; uniform within bucket.

/// p = uniform over candidate edges, factorized as:
///   choose orbit rank with prob ∝ orbit size (FixPrefixOrbitCodec categorical),
///   then choose uniformly within that orbit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UniformCandidateP {
    pub cr: ColorRefinement,
}

impl UniformCandidateP {
    pub fn new(wl_rounds: usize) -> Self {
        Self { cr: ColorRefinement::new(wl_rounds, false) }
    }
}

/// Slice codec built at each step from current G_{k-1}.
#[derive(Clone, Debug)]
pub struct UniformCandidatePEdgeCodec {
    candidates: Vec<EdgeIndex>,     // missing edges, sorted
    orbit: FixPrefixOrbitCodec,     // built over candidates, using post-add orbit ids
    members_by_orbit: Vec<Vec<usize>>, 
}

impl UniformCandidatePEdgeCodec {
    pub fn build(prefix_after_removal: &EdgePrefix, cr: &ColorRefinement) -> Self {
        let g = &prefix_after_removal.g;
        let n = prefix_after_removal.list.n;

        // 1) enumerate candidates A(G_{k-1}) = all missing edges (fixed-n)
        let mut candidates = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                let e = canon_edge((u, v));
                if g.has_edge(&e) { continue; }
                candidates.push(e);
            }
        }
        candidates.sort();

        // 2) compute OrbitId per candidate using Option-1 symbol definition:
        //    orbit id of e in (G_{k-1} ∪ {e})
        //
        //    Important: OrbitId type must satisfy OrdSymbol + Default (your snippet).
        //    We'll use u64 (it usually implements those in this repo).
        let mut orbit_ids_u64: Vec<u64> = Vec::with_capacity(candidates.len());
        for &e in &candidates {
            let e = canon_edge(e);
            let mut g_plus = g.clone();
            g_plus.insert_plain_edge(e);

            // colors/degrees in post-add graph
            let colors: Vec<usize> = <ColorRefinement as Hashing<PlainGraph<Undirected>>>::apply(cr, &g_plus);

            let deg = g_plus.degrees().collect_vec();

            // reuse your existing edge_sig/hash_sig helpers (computed on g_plus)
            let sig = edge_sig(&colors, &deg, e);
            orbit_ids_u64.push(hash_sig(sig));
        }

        // 3) Build FixPrefixOrbitCodec over candidates. Masses = orbit sizes.
        let orbit = FixPrefixOrbitCodec::new(orbit_ids_u64, candidates.len());
        // build deterministic orbit members
        let num_orbits = 1 + *orbit.ids.iter().max().unwrap_or(&0);
        let mut members_by_orbit = vec![Vec::<usize>::new(); num_orbits];
        for i in 0..candidates.len() {
            members_by_orbit[orbit.ids[i]].push(i);
        }
        for mem in &mut members_by_orbit {
            mem.sort_by_key(|&idx| candidates[idx]); // stable deterministic order
        }

        Self { candidates, orbit, members_by_orbit }
    }

    fn locate_candidate(&self, e: EdgeIndex) -> usize {
        let e = canon_edge(e);
        self.candidates
            .binary_search(&e)
            .expect("edge not in A(G_{k-1}) (encoder/decoder mismatch)")
    }

    // fn position_in_orbit(members: &[usize], element_idx: usize) -> usize {
    //     // members is a Vec<usize> of element indices belonging to the orbit rank.
    //     // VecOrbits::new typically stores indices in sorted/insertion order; to be safe:
    //     members.iter().position(|&j| j == element_idx).unwrap()
    // }
}

impl Codec for UniformCandidatePEdgeCodec {
    type Symbol = EdgeIndex;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        // element index in candidate list
        let element_idx = self.locate_candidate(*x);

        // // orbit rank for this element (already computed by FixPrefixOrbitCodec::ranks)
        let orbit_rank = self.orbit.ids[element_idx];

        // --- encode orbit_rank with mass ∝ orbit size ---
        // FenwickTree<usize> is used as categorical in FixPrefixOrbitCodec::new.
        // In this repo it is used as a codec over indices (push/pop). If your FenwickTree
        // methods are named differently, swap these two calls accordingly.
        self.orbit.push(m, &orbit_rank);
        

        let members = &self.members_by_orbit[orbit_rank];
        let within = members.iter().position(|&j| j == element_idx).unwrap();
        Uniform::new(members.len()).push(m, &within);

    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        // decode orbit_rank first
        let orbit_rank = self.orbit.pop(m);
        let members = &self.members_by_orbit[orbit_rank];

        let within = Uniform::new(members.len()).pop(m);
        let element_idx = members[within];
        self.candidates[element_idx]

    }

    fn bits(&self, _x: &Self::Symbol) -> Option<f64> { None }
}

pub trait EdgeRemovalModel: Clone + Debug + Eq + PartialEq + Send + Sync + 'static {
    type SliceCodec: Codec<Symbol = EdgeSlice> + Clone + Send + Sync + 'static;
    fn codec_for_removed_edge(&self, prefix_after_removal: &EdgePrefix) -> Self::SliceCodec;
}

impl EdgeRemovalModel for UniformCandidateP {
    type SliceCodec = UniformCandidatePEdgeCodec;

    fn codec_for_removed_edge(&self, prefix_after_removal: &EdgePrefix) -> Self::SliceCodec {
        UniformCandidatePEdgeCodec::build(prefix_after_removal, &self.cr)
    }
}



/// Slice codecs wrapper required by the autoregressive engine:
/// it must produce a slice codec given the current prefix, and support the fused push/pop.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeSliceCodecs<M: EdgeRemovalModel> {
    pub chain: EdgePrefixingChain,
    pub model: M,
    pub m: usize,
}

impl<M: EdgeRemovalModel> Len for EdgeSliceCodecs<M> {
    fn len(&self) -> usize {
        self.m
    }
}

impl<M: EdgeRemovalModel> InnerSliceCodecs<EdgePrefixingChain> for EdgeSliceCodecs<M> {
    fn prefixing_chain(&self) -> EdgePrefixingChain {
        self.chain.clone()
    }

    fn empty_prefix(&self) -> impl Codec<Symbol = EdgePrefix> {
    #[derive(Clone, Debug, Eq, PartialEq)]
    struct EmptyPrefixNm;

    impl Codec for EmptyPrefixNm {
        type Symbol = EdgePrefix;

        fn push(&self, m: &mut Message, x: &Self::Symbol) {
            // At the end of encoding, x should be the empty remaining graph G_0:
            debug_assert_eq!(x.len_active, 0);

            let n = x.list.n;
            let m_edges = x.list.edges.len();

            // IMPORTANT (ANS stack order):
            // Whatever you push last is popped first.
            // We want pop() to read n first, then m.
            // So we push m first, then n.
            LogUniform::max().push(m, &m_edges);
            LogUniform::max().push(m, &n);
        }

        fn pop(&self, m: &mut Message) -> Self::Symbol {
            // Inverse of push(): pop n first (it was pushed last), then m.
            let n = LogUniform::max().pop(m);
            let m_edges = LogUniform::max().pop(m);

            // Build the empty prefix state expected by the shuffle decoder:
            // - empty graph on n nodes
            // - edge list of length m (preallocated, values irrelevant while len_active=0)
            // - len_active=0
            let g = PlainGraph::<Undirected>::plain_empty(n);

            // Fill with a dummy edge index; it's never read while len_active==0.
            // (push_slice() will overwrite positions 0..m-1 during decoding.)
            let dummy = (0usize, 0usize);
            let list = EdgeList {
                n,
                edges: vec![dummy; m_edges],
            };

            EdgePrefix { g, list, len_active: 0 }
        }

        fn bits(&self, x: &Self::Symbol) -> Option<f64> {
            let n = x.list.n;
            let m_edges = x.list.edges.len();
            // Must match push() order/contents; sum is commutative anyway.
            Some(LogUniform::max().bits(&n)? + LogUniform::max().bits(&m_edges)?)
        }
    }

    EmptyPrefixNm
}

}

impl<M: EdgeRemovalModel> PrefixFn<EdgePrefixingChain> for EdgeSliceCodecs<M> {
    type Output = M::SliceCodec;

    fn apply(&self, x: &EdgePrefix) -> Self::Output {
        self.model.codec_for_removed_edge(x)
    }

    fn update_after_pop_slice(&self, image: &mut Self::Output, x: &EdgePrefix, _slice: &EdgeSlice) {
        *image = self.apply(x);
    }

    fn update_after_push_slice(&self, image: &mut Self::Output, x: &EdgePrefix, _slice: &EdgeSlice) {
        *image = self.apply(x);
    }

    fn swap(&self, _image: &mut Self::Output, _i: usize, _j: usize) {}
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Type2EdgeOrbitGraphCodec<M: EdgeRemovalModel> {
    pub convs: usize,
    pub model: M,
}

impl<M: EdgeRemovalModel> Codec for Type2EdgeOrbitGraphCodec<M> {
    type Symbol = Unordered<PlainGraph<Undirected>>;

    fn push(&self, msg: &mut Message, Unordered(g): &Self::Symbol) {
        let n = g.len();
        let edges = g.edge_indices();
        let inner = type2_edge_orbit_codec(n, edges.clone(), self.convs, self.model.clone());
        inner.push(msg, &Unordered(EdgeList { n, edges }));
    }

    fn pop(&self, msg: &mut Message) -> Self::Symbol {
        // n,m will be decoded by empty_prefix() inside inner.pop(),
        // so we can use a dummy inner with (n=0, edges=[]).
        let inner = type2_edge_orbit_codec(0, vec![], self.convs, self.model.clone());
        let Unordered(edge_list) = inner.pop(msg);

        let mut g = PlainGraph::<Undirected>::plain_empty(edge_list.n);
        for &e in &edge_list.edges {
            g.insert_plain_edge(canon_edge(e));
        }
        Unordered(g)
    }

    fn bits(&self, _x: &Self::Symbol) -> Option<f64> { None }
}


/// Public constructor:
/// returns a full autoregressive shuffle codec for unordered edge sets (graphs).
pub fn type2_edge_orbit_codec<M: EdgeRemovalModel>(
    n: usize,
    edges: Vec<EdgeIndex>,
    convs: usize,
    model: M,
) -> UnfusedAutoregressiveShuffleCodec<EdgePrefixingChain, EdgeSliceCodecs<M>, EdgeOrbitCodecs> {
    // Build "full" ordered object:
    let full = EdgeList { n, edges };
    let m = full.len();

    let chain = EdgePrefixingChain::new();
    let slices = EdgeSliceCodecs { chain: chain.clone(), model, m };
    let orbit_codecs = EdgeOrbitCodecs { cr: ColorRefinement::new(convs, false) };

    UnfusedAutoregressiveShuffleCodec::new(slices, orbit_codecs)
}

/// Helper to wrap as Unordered<EdgeList> if you want to use the Codec trait directly.
pub fn unordered_edge_list(n: usize, edges: Vec<EdgeIndex>) -> Unordered<EdgeList> {
    Unordered(EdgeList { n, edges })
}
