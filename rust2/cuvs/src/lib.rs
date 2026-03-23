/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuVS: Safe Rust bindings for GPU-accelerated vector search.

use std::marker::PhantomData;

use cuvs_sys as ffi;

pub mod error;
pub mod logging;
pub mod resources;
pub mod version;

/// Marker that prevents `Send` and `Sync` on any type containing it.
/// Used on all GPU-bound handles that are tied to a CUDA context and thread.
pub(crate) type NotSend = PhantomData<*const ()>;
