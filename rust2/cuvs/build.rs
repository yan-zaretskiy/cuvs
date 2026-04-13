/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::env;

fn add_runtime_search_path(var_name: &str) {
    if let Ok(lib_path) = env::var(var_name) {
        // Add the discovered shared-library directory to the runtime search
        // path of final targets built in this package (tests, examples,
        // benches, binaries). Cargo carries link-time search paths from
        // dependencies, but that does not guarantee the loader can find those
        // DSOs at runtime unless the path is embedded here or supplied through
        // loader environment variables.
        println!("cargo:rustc-link-arg=-Wl,-rpath={lib_path}");
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_CUVS_C_LIB");
    add_runtime_search_path("DEP_CUVS_C_LIB");

    if env::var_os("CARGO_FEATURE_TORCH").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=DEP_TCH_LIBTORCH_LIB");
    add_runtime_search_path("DEP_TCH_LIBTORCH_LIB");

    // Link libtorch_cuda so the CUDA dispatch backend registers its kernels.
    // Without this, tch operations on Device::Cuda fail at runtime with
    // "operator not available for CUDA backend".
    //
    // We use `rustc-link-lib` (not `rustc-link-arg`) because the former
    // propagates transitively to downstream crates that depend on `cuvs`,
    // whereas `rustc-link-arg` only applies to the declaring crate's own
    // targets.
    //
    // The `+verbatim` modifier passes `--no-as-needed` for this specific lib
    // so the linker doesn't strip it even though no Rust code references its
    // symbols directly — the CUDA kernels register via a C++ global
    // constructor.
    println!("cargo:rustc-link-lib=dylib:+verbatim=libtorch_cuda.so");
    println!("cargo:rustc-link-lib=dylib:+verbatim=libtorch.so");
}
