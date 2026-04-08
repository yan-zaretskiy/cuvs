/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! IVF-PQ: inverted-file index with product-quantized vectors.
//!
//! Partitions the dataset into `n_lists` Voronoi cells via k-means and
//! compresses each vector with product quantization. This gives a compact
//! in-memory footprint while retaining high recall for large-scale search.

mod index;
mod params;

pub use index::{Index, PrecomputedIndex};
pub use params::{IndexParams, SearchParams};

use crate::error::LibraryError;
use crate::ffi;

// ---------------------------------------------------------------------------
// Rust-native enums with From conversions to/from FFI
// ---------------------------------------------------------------------------

/// Strategy for creating PQ codebooks.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodebookGen {
    /// One codebook per PQ subspace.
    PerSubspace,
    /// One codebook per IVF cluster.
    PerCluster,
}

impl From<CodebookGen> for ffi::cuvsIvfPqCodebookGen {
    fn from(v: CodebookGen) -> Self {
        match v {
            CodebookGen::PerSubspace => Self::CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE,
            CodebookGen::PerCluster => Self::CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER,
        }
    }
}

impl From<ffi::cuvsIvfPqCodebookGen> for CodebookGen {
    fn from(v: ffi::cuvsIvfPqCodebookGen) -> Self {
        match v {
            ffi::cuvsIvfPqCodebookGen::CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE => {
                Self::PerSubspace
            }
            ffi::cuvsIvfPqCodebookGen::CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER => Self::PerCluster,
        }
    }
}

/// Memory layout of IVF-PQ list data.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum ListLayout {
    /// Codes stored contiguously (one vector after another).
    Flat,
    /// Codes interleaved for optimised search performance.
    Interleaved,
}

impl From<ListLayout> for ffi::cuvsIvfPqListLayout {
    fn from(v: ListLayout) -> Self {
        match v {
            ListLayout::Flat => Self::CUVS_IVF_PQ_LIST_LAYOUT_FLAT,
            ListLayout::Interleaved => Self::CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED,
        }
    }
}

impl From<ffi::cuvsIvfPqListLayout> for ListLayout {
    fn from(v: ffi::cuvsIvfPqListLayout) -> Self {
        match v {
            ffi::cuvsIvfPqListLayout::CUVS_IVF_PQ_LIST_LAYOUT_FLAT => Self::Flat,
            ffi::cuvsIvfPqListLayout::CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED => Self::Interleaved,
        }
    }
}

/// Lookup-table dtype used during IVF-PQ search.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum LutDType {
    /// 32-bit floating-point lookup tables.
    F32,
    /// 16-bit floating-point lookup tables.
    F16,
    /// 8-bit unsigned lookup tables.
    U8,
}

impl From<LutDType> for ffi::cudaDataType_t {
    fn from(v: LutDType) -> Self {
        match v {
            LutDType::F32 => ffi::cudaDataType_t_CUDA_R_32F,
            LutDType::F16 => ffi::cudaDataType_t_CUDA_R_16F,
            LutDType::U8 => ffi::cudaDataType_t_CUDA_R_8U,
        }
    }
}

/// Accumulator dtype used for internal IVF-PQ distance computation.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum InternalDistanceDType {
    /// 32-bit floating-point accumulators.
    F32,
    /// 16-bit floating-point accumulators.
    F16,
}

impl From<InternalDistanceDType> for ffi::cudaDataType_t {
    fn from(v: InternalDistanceDType) -> Self {
        match v {
            InternalDistanceDType::F32 => ffi::cudaDataType_t_CUDA_R_32F,
            InternalDistanceDType::F16 => ffi::cudaDataType_t_CUDA_R_16F,
        }
    }
}

/// GEMM dtype used to identify coarse clusters to probe.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoarseSearchDType {
    /// 32-bit floating-point coarse search.
    F32,
    /// 16-bit floating-point coarse search.
    F16,
    /// 8-bit signed coarse search.
    I8,
}

impl From<CoarseSearchDType> for ffi::cudaDataType_t {
    fn from(v: CoarseSearchDType) -> Self {
        match v {
            CoarseSearchDType::F32 => ffi::cudaDataType_t_CUDA_R_32F,
            CoarseSearchDType::F16 => ffi::cudaDataType_t_CUDA_R_16F,
            CoarseSearchDType::I8 => ffi::cudaDataType_t_CUDA_R_8I,
        }
    }
}

/// Error type for IVF-PQ operations.
#[derive(Debug, thiserror::Error)]
pub enum IvfPqError {
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
