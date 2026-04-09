/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! IVF-Flat: inverted-file index with uncompressed (flat) vectors.
//!
//! Partitions the dataset into `n_lists` Voronoi cells via k-means and
//! stores the raw vectors in each cell. At search time only the `n_probes`
//! closest cells are scanned, giving sub-linear search cost.

mod index;
mod params;

pub use crate::neighbors::filters::SearchFilter;
pub use index::Index;
pub use params::{IndexParams, SearchParams};

use crate::dlpack::DLPackError;
use crate::error::LibraryError;

/// Error type for IVF-Flat operations.
#[derive(Debug, thiserror::Error)]
pub enum IvfFlatError {
    /// The C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
    /// A file path contained an interior NUL byte.
    #[error("path contains interior NUL byte")]
    InvalidPath(#[from] std::ffi::NulError),
    /// A parameter value failed validation.
    #[error("invalid parameter: {0}")]
    Validation(String),
}
