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

    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");
}
