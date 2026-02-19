//! Type-2 equivalence for edge orders using orbit (bits-back) coding.
//!
//! Implements autoregressive shuffle coding where edges are removed one at a time,
//! with orbit-based bits-back coding to exploit graph symmetries.
//!
//! At each step k:
//!   1. Build orbit partition of edges in G_k using color refinement
//!   2. Decode orbit id o_k ~ q(o_k | G_k) via ANS pop (bits-back)
//!   3. Select representative edge from that orbit
//!   4. Remove edge to get G_{k-1}
//!   5. Encode removed edge with p(edge | G_{k-1}) via ANS push

use crate::autoregressive::prefix_orbit::FixPrefixOrbitCodec;
use crate::autoregressive::{InnerSliceCodecs, PrefixFn, PrefixingChain, UnfusedAutoregressiveShuffleCodec};
use crate::codec::{Codec, LogUniform, Message, Uniform};
use crate::graph::{ColorRefinement, EdgeIndex, PlainGraph, Undirected};
use crate::permutable::{Hashing, Len, Permutable, Unordered};
use itertools::Itertools;
use std::fmt::Debug;
use std::marker::PhantomData;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

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

// ============================================================================
// PREFIXING CHAIN
// ============================================================================

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

// ============================================================================
// ORBIT COMPUTATION (q codec helpers)
// ============================================================================

/// Compute edge signature for orbit partitioning.
///
/// We approximate edge orbits using:
///   - node colors from ColorRefinement on the remaining graph
///   - degrees
///   - (min/max) normalization for undirected edges
///
/// Edges in the same bucket share the same computed id at that step,
/// but the id can change after removals because colors/degrees change.
fn edge_sig(node_color: &[usize], deg: &[usize], (u, v): EdgeIndex) -> (usize, usize, usize, usize) {
    let (cu, cv) = (node_color[u], node_color[v]);
    let (du, dv) = (deg[u], deg[v]);
    let c0 = cu.min(cv);
    let c1 = cu.max(cv);
    let d0 = du.min(dv);
    let d1 = du.max(dv);
    (c0, c1, d0, d1)
}


/// Hash the edge signature into a single u64 for orbit partitioning.
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
/// This implements q(o_k | G_k) ∝ |orbit| by using FixPrefixOrbitCodec masses = orbit sizes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeOrbitCodecs {
    pub cr: ColorRefinement,
}

impl PrefixFn<EdgePrefixingChain> for EdgeOrbitCodecs {
    type Output = FixPrefixOrbitCodec;

    fn apply(&self, x: &EdgePrefix) -> Self::Output {
        // Remaining graph G_k:
        let g = &x.g;

        // Node colors from color refinement.
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

// ============================================================================
// P MODEL (edge removal model)
// ============================================================================

/// Uniform distribution over candidate edges (missing edges in G_{k-1}).
///
/// The p codec encodes edges uniformly among all candidate (missing) edges.
/// Orbit-based bits-back coding is handled by q (EdgeOrbitCodecs), not here.
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
/// Encodes edges uniformly among all candidates (missing edges).
#[derive(Clone, Debug)]
pub struct UniformCandidatePEdgeCodec {
    candidates: Vec<EdgeIndex>,     // missing edges, sorted
}

impl UniformCandidatePEdgeCodec {
    pub fn build(prefix_after_removal: &EdgePrefix, _cr: &ColorRefinement) -> Self {
        let g = &prefix_after_removal.g;
        let n = prefix_after_removal.list.n;

        // Enumerate candidates A(G_{k-1}) = all missing edges (fixed-n)
        let mut candidates = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                let e = canon_edge((u, v));
                if g.has_edge(&e) { continue; }
                candidates.push(e);
            }
        }
        candidates.sort();

        Self { candidates }
    }

    fn locate_candidate(&self, e: EdgeIndex) -> usize {
        let e = canon_edge(e);
        self.candidates
            .binary_search(&e)
            .expect("edge not in A(G_{k-1}) (encoder/decoder mismatch)")
    }
}

impl Codec for UniformCandidatePEdgeCodec {
    type Symbol = EdgeIndex;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        // Encode edge uniformly among all candidates.
        // The orbit structure for bits-back is handled by q (EdgeOrbitCodecs), not p.
        let element_idx = self.locate_candidate(*x);
        Uniform::new(self.candidates.len()).push(m, &element_idx);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let element_idx = Uniform::new(self.candidates.len()).pop(m);
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

// ============================================================================
// INTEGRATION (slice codecs, graph codec)
// ============================================================================

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
        // Peek n and m from message to create codec with correct dimensions.
        // (ANS stack order: n was pushed last, so pop it first; then m.)
        let n = LogUniform::max().pop(msg);
        let m = LogUniform::max().pop(msg);

        // Push them back so inner.pop() can decode them via empty_prefix().
        LogUniform::max().push(msg, &m);
        LogUniform::max().push(msg, &n);

        // Create dummy edges vector of correct length.
        let dummy_edges = vec![(0usize, 0usize); m];
        let inner = type2_edge_orbit_codec(n, dummy_edges, self.convs, self.model.clone());
        let Unordered(edge_list) = inner.pop(msg);

        let mut g = PlainGraph::<Undirected>::plain_empty(edge_list.n);
        for &e in &edge_list.edges {
            g.insert_plain_edge(canon_edge(e));
        }
        Unordered(g)
    }

    fn bits(&self, _x: &Self::Symbol) -> Option<f64> { None }
}

// ============================================================================
// PUBLIC API
// ============================================================================

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

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{Codec, Message};

    #[test]
    fn test_type2_edge_orbit_roundtrip() {
        // Small triangle graph: 3 nodes, 3 edges
        let mut g = PlainGraph::<Undirected>::plain_empty(3);
        g.insert_plain_edge((0, 1));
        g.insert_plain_edge((1, 2));
        g.insert_plain_edge((0, 2));

        let codec = Type2EdgeOrbitGraphCodec {
            convs: 2,
            model: UniformCandidateP::new(2),
        };

        let original = Unordered(g);
        let mut msg = Message::random(42);
        codec.push(&mut msg, &original);
        let decoded = codec.pop(&mut msg);

        // Verify graphs have the same edges
        let mut orig_edges = original.0.edge_indices();
        let mut dec_edges = decoded.0.edge_indices();
        orig_edges.sort();
        dec_edges.sort();
        assert_eq!(orig_edges, dec_edges);
    }

    #[test]
    fn test_type2_edge_orbit_4cycle() {
        // 4-cycle: 0-1-2-3-0 (4 nodes, 4 edges)
        let mut g = PlainGraph::<Undirected>::plain_empty(4);
        g.insert_plain_edge((0, 1));
        g.insert_plain_edge((1, 2));
        g.insert_plain_edge((2, 3));
        g.insert_plain_edge((3, 0));

        let codec = Type2EdgeOrbitGraphCodec {
            convs: 2,
            model: UniformCandidateP::new(2),
        };

        let original = Unordered(g);
        let mut msg = Message::random(123);
        codec.push(&mut msg, &original);
        let decoded = codec.pop(&mut msg);

        let mut orig_edges = original.0.edge_indices();
        let mut dec_edges = decoded.0.edge_indices();
        orig_edges.sort();
        dec_edges.sort();
        assert_eq!(orig_edges, dec_edges);
    }

    #[test]
    fn test_type2_edge_orbit_path() {
        // Path: 0-1-2-3-4 (5 nodes, 4 edges)
        let mut g = PlainGraph::<Undirected>::plain_empty(5);
        g.insert_plain_edge((0, 1));
        g.insert_plain_edge((1, 2));
        g.insert_plain_edge((2, 3));
        g.insert_plain_edge((3, 4));

        let codec = Type2EdgeOrbitGraphCodec {
            convs: 2,
            model: UniformCandidateP::new(2),
        };

        let original = Unordered(g);
        let mut msg = Message::random(123);
        codec.push(&mut msg, &original);
        let decoded = codec.pop(&mut msg);

        let mut orig_edges = original.0.edge_indices();
        let mut dec_edges = decoded.0.edge_indices();
        orig_edges.sort();
        dec_edges.sort();
        assert_eq!(orig_edges, dec_edges);
    }

}
