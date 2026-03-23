/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Raw FFI bindings to libcuvs_c.
//!
//! This crate provides auto-generated bindings to the cuVS C API via bindgen.
//! For a safe, idiomatic Rust API, use the `cuvs` crate instead.

// Suppress warnings from bindgen-generated code
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused_attributes)]

// Use pre-generated bindings by default.
// When the `generate-bindings` feature is enabled, use freshly generated ones.
#[cfg(all(feature = "generate-bindings", docsrs))]
compile_error!("generate-bindings is incompatible with docs.rs builds");

#[cfg(feature = "generate-bindings")]
include!(concat!(env!("OUT_DIR"), "/cuvs_bindings.rs"));

#[cfg(not(feature = "generate-bindings"))]
include!("bindings.rs");
