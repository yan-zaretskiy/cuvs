/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use cmake_package::{find_package, CMakePackage};

const CUVS_COMPONENT: &str = "c_api";

/// Try to find a pip-installed cuVS by locating `site-packages/libcuvs`.
/// Checks both the active venv (via VIRTUAL_ENV) and the system python.
fn pip_cuvs_prefix() -> Option<PathBuf> {
    // If a venv is active, check its site-packages first.
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        if let Some(prefix) = find_libcuvs_in_prefix(Path::new(&venv)) {
            return Some(prefix);
        }
    }

    // Fall back to asking python3 for its site-packages.
    let output = Command::new("python3")
        .arg("-c")
        .arg("import sysconfig; print(sysconfig.get_path('purelib'))")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let site_packages = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let prefix = site_packages.join("libcuvs");
    prefix.is_dir().then_some(prefix)
}

/// Look for `lib/python*/site-packages/libcuvs` under a given prefix.
fn find_libcuvs_in_prefix(prefix: &Path) -> Option<PathBuf> {
    let lib_dir = prefix.join("lib");
    for entry in std::fs::read_dir(&lib_dir).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        if name.as_encoded_bytes().starts_with(b"python") {
            let candidate = entry.path().join("site-packages").join("libcuvs");
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Discover the cuVS CMake package.
///
/// 1. Try the default CMake search paths (works for conda, system installs).
/// 2. If that fails, check if cuVS was pip-installed and retry with that prefix.
///
fn discover_cuvs() -> CMakePackage {
    // Attempt 1: default CMake discovery (conda, system, or user-set CMAKE_PREFIX_PATH).
    if let Ok(package) = find_package("cuvs")
        .components([CUVS_COMPONENT.to_owned()])
        .find()
    {
        return package;
    }

    // Attempt 2: pip-installed cuVS (libcuvs lives inside site-packages/libcuvs).
    // The pip package uses lib64/ which CMake doesn't search via prefix paths,
    // so we set cuvs_DIR directly to the cmake config directory.
    if let Some(prefix) = pip_cuvs_prefix() {
        let cmake_dir = prefix.join("lib64/cmake/cuvs");
        if cmake_dir.is_dir() {
            std::env::set_var("cuvs_DIR", &cmake_dir);
            if let Ok(package) = find_package("cuvs")
                .components([CUVS_COMPONENT.to_owned()])
                .find()
            {
                return package;
            }
        }
    }

    eprintln!(
        "error: Could not find the cuVS CMake package.\n\
         Install cuVS via one of:\n\
         - conda: conda install -c rapidsai libcuvs\n\
         - pip:   pip install cuvs-cu13\n\
         Or set CMAKE_PREFIX_PATH to point to your cuVS installation."
    );
    std::process::exit(1);
}

fn main() {
    println!("cargo:rerun-if-changed=cuvs_c_wrapper.h");
    println!("cargo:rerun-if-changed=vendor/dlpack/dlpack.h");
    println!("cargo:rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=cuvs_DIR");

    let package = discover_cuvs();

    let target = package
        .target("cuvs::c_api")
        .expect("cuvs::c_api target not found");
    target.link();

    // Generate FFI bindings.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set by Cargo"));

    let include_args = target
        .include_directories
        .iter()
        .map(|inc| format!("-I{}", inc));

    bindgen::Builder::default()
        .header("cuvs_c_wrapper.h")
        .must_use_type("cuvsError_t")
        .allowlist_function("cuvs.*")
        .allowlist_type("(cuvs|DL).*")
        .rustified_enum("(cuvs|DL).*")
        .clang_args(include_args)
        .clang_arg("-Ivendor") // vendored dlpack header
        .generate()
        .expect("bindgen failed to generate cuvs bindings")
        .write_to_file(out_dir.join("cuvs_bindings.rs"))
        .expect("failed to write cuvs_bindings.rs");
}
