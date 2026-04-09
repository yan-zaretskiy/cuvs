/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Vamana: DiskANN-compatible graph construction and serialization.
//!
//! The current cuVS C API for Vamana exposes index build and serialization,
//! but does not yet expose search or deserialize entry points.

mod index;
mod params;

pub use index::Index;
pub use params::IndexParams;

use crate::error::LibraryError;

/// Error type for Vamana operations.
#[derive(Debug, thiserror::Error)]
pub enum VamanaError {
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
