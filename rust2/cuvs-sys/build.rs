/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;

use cmake_package::find_package;

fn main() {
    // Discover installed cuVS via CMake package config.
    let package = find_package("cuvs")
        .components(["c_api".to_owned()])
        .find()
        .expect("Could not find the cuVS cmake package.");

    let target = package
        .target("cuvs::c_api")
        .expect("cuvs::c_api target not found");
    target.link();

    // Generate bindings with bindgen. OUT_DIR is guaranteed by Cargo to be set when running a build script
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let mut builder = bindgen::Builder::default()
        .header("cuvs_c_wrapper.h")
        .must_use_type("cuvsError_t")
        .allowlist_function("cuvs.*")
        .allowlist_type("(cuvs|DL).*")
        .rustified_enum("(cuvs|DL).*");

    // Add include directories from the CMake target.
    for inc in &target.include_directories {
        builder = builder.clang_arg(format!("-I{}", inc));
    }

    builder
        .generate()
        .expect("bindgen failed to generate cuvs bindings")
        .write_to_file(out_dir.join("cuvs_bindings.rs"))
        .expect("failed to write cuvs_bindings.rs");
}
