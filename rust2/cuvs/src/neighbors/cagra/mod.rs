/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CAGRA: GPU-accelerated graph-based approximate nearest neighbor search.
//!
//! CAGRA builds a k-NN graph on the GPU and uses it for fast approximate
//! nearest neighbor queries.  It offers state-of-the-art throughput for both
//! small and large batch sizes.

mod index;
mod params;

pub use crate::neighbors::filters::SearchFilter;
pub use index::Index;
pub use params::{
    AceParams, CompressionParams, ExtendParams, IndexParams, IvfPqGraphBuildParams, SearchParams,
};

use crate::error::LibraryError;
use crate::ffi;

// ---------------------------------------------------------------------------
// Rust-native enums with From conversions to/from FFI
// ---------------------------------------------------------------------------

/// Algorithm for building the internal k-NN graph.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphBuildAlgo {
    /// Automatically select the best algorithm.
    Auto,
    /// Build using IVF-PQ.
    IvfPq,
    /// Build using NN-Descent.
    NnDescent,
    /// Build using iterative CAGRA search.
    IterativeCagraSearch,
    /// Build using ACE (Augmented Core Extraction) for large datasets.
    Ace,
}

impl From<GraphBuildAlgo> for ffi::cuvsCagraGraphBuildAlgo {
    fn from(v: GraphBuildAlgo) -> Self {
        match v {
            GraphBuildAlgo::Auto => Self::AUTO_SELECT,
            GraphBuildAlgo::IvfPq => Self::IVF_PQ,
            GraphBuildAlgo::NnDescent => Self::NN_DESCENT,
            GraphBuildAlgo::IterativeCagraSearch => Self::ITERATIVE_CAGRA_SEARCH,
            GraphBuildAlgo::Ace => Self::ACE,
        }
    }
}

impl From<ffi::cuvsCagraGraphBuildAlgo> for GraphBuildAlgo {
    fn from(v: ffi::cuvsCagraGraphBuildAlgo) -> Self {
        match v {
            ffi::cuvsCagraGraphBuildAlgo::AUTO_SELECT => Self::Auto,
            ffi::cuvsCagraGraphBuildAlgo::IVF_PQ => Self::IvfPq,
            ffi::cuvsCagraGraphBuildAlgo::NN_DESCENT => Self::NnDescent,
            ffi::cuvsCagraGraphBuildAlgo::ITERATIVE_CAGRA_SEARCH => Self::IterativeCagraSearch,
            ffi::cuvsCagraGraphBuildAlgo::ACE => Self::Ace,
        }
    }
}

/// Search kernel implementation.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchAlgo {
    /// Single CTA -- best for large batch sizes.
    SingleCta,
    /// Multi CTA -- best for small batch sizes.
    MultiCta,
    /// Multi kernel -- best for small batch sizes.
    MultiKernel,
    /// Automatically select the best kernel.
    Auto,
}

impl From<SearchAlgo> for ffi::cuvsCagraSearchAlgo {
    fn from(v: SearchAlgo) -> Self {
        match v {
            SearchAlgo::SingleCta => Self::SINGLE_CTA,
            SearchAlgo::MultiCta => Self::MULTI_CTA,
            SearchAlgo::MultiKernel => Self::MULTI_KERNEL,
            SearchAlgo::Auto => Self::AUTO,
        }
    }
}

impl From<ffi::cuvsCagraSearchAlgo> for SearchAlgo {
    fn from(v: ffi::cuvsCagraSearchAlgo) -> Self {
        match v {
            ffi::cuvsCagraSearchAlgo::SINGLE_CTA => Self::SingleCta,
            ffi::cuvsCagraSearchAlgo::MULTI_CTA => Self::MultiCta,
            ffi::cuvsCagraSearchAlgo::MULTI_KERNEL => Self::MultiKernel,
            ffi::cuvsCagraSearchAlgo::AUTO => Self::Auto,
        }
    }
}

/// Hash-table mode used during search.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum HashMode {
    /// Standard hash table.
    Hash,
    /// Small hash table optimised for low memory.
    Small,
    /// Automatically select the best mode.
    Auto,
}

impl From<HashMode> for ffi::cuvsCagraHashMode {
    fn from(v: HashMode) -> Self {
        match v {
            HashMode::Hash => Self::HASH,
            HashMode::Small => Self::SMALL,
            HashMode::Auto => Self::AUTO_HASH,
        }
    }
}

impl From<ffi::cuvsCagraHashMode> for HashMode {
    fn from(v: ffi::cuvsCagraHashMode) -> Self {
        match v {
            ffi::cuvsCagraHashMode::HASH => Self::Hash,
            ffi::cuvsCagraHashMode::SMALL => Self::Small,
            ffi::cuvsCagraHashMode::AUTO_HASH => Self::Auto,
        }
    }
}

/// Strategy for selecting CAGRA graph parameters from HNSW-like inputs.
///
/// Used with [`IndexParams::from_hnsw_params`].
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum HnswHeuristicType {
    /// Produce a graph with search performance similar to an HNSW graph.
    SimilarSearchPerformance,
    /// Produce a graph with the same binary size as the equivalent HNSW graph.
    SameGraphFootprint,
}

impl From<HnswHeuristicType> for ffi::cuvsCagraHnswHeuristicType {
    fn from(v: HnswHeuristicType) -> Self {
        match v {
            HnswHeuristicType::SimilarSearchPerformance => {
                Self::CUVS_CAGRA_HEURISTIC_SIMILAR_SEARCH_PERFORMANCE
            }
            HnswHeuristicType::SameGraphFootprint => {
                Self::CUVS_CAGRA_HEURISTIC_SAME_GRAPH_FOOTPRINT
            }
        }
    }
}

impl From<ffi::cuvsCagraHnswHeuristicType> for HnswHeuristicType {
    fn from(v: ffi::cuvsCagraHnswHeuristicType) -> Self {
        match v {
            ffi::cuvsCagraHnswHeuristicType::CUVS_CAGRA_HEURISTIC_SIMILAR_SEARCH_PERFORMANCE => {
                Self::SimilarSearchPerformance
            }
            ffi::cuvsCagraHnswHeuristicType::CUVS_CAGRA_HEURISTIC_SAME_GRAPH_FOOTPRINT => {
                Self::SameGraphFootprint
            }
        }
    }
}

/// Error type for CAGRA operations.
#[derive(Debug, thiserror::Error)]
pub enum CagraError {
    /// The C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// A file path contained an interior NUL byte.
    #[error("path contains interior NUL byte")]
    InvalidPath(#[from] std::ffi::NulError),
    /// A parameter value failed validation.
    #[error("invalid parameter: {0}")]
    Validation(String),
}
