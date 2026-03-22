/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use cmake_package::find_package;

const CUVS_COMPONENT: &str = "c_api";
#[cfg(feature = "vendored")]
const DEFAULT_CPP_SOURCE: &str = "../../cpp";

// ---------------------------------------------------------------------------
// CMake package discovery
// ---------------------------------------------------------------------------

/// Run CMake `find_package(cuvs)` and extract the include directory.
/// Calls `CMakeTarget::link()` to emit the full set of cargo link directives,
/// preserving all link libraries, directories, and options from the CMake target.
fn try_find_cuvs_package() -> Option<String> {
    let package = find_package("cuvs")
        .components([CUVS_COMPONENT.to_owned()])
        .find()
        .ok()?;
    let target = package.target("cuvs::c_api")?;
    let include_dir = target.include_directories.first()?.clone();
    target.link();
    Some(include_dir)
}

/// Try to discover a pip-installed cuVS.
/// The pip package places files under `site-packages/libcuvs/lib64/`, which
/// CMake doesn't search via prefix paths, so we set `cuvs_DIR` directly.
fn try_discover_pip() -> Option<String> {
    let prefix = pip_cuvs_prefix()?;
    let cmake_dir = prefix.join("lib64/cmake/cuvs");
    if !cmake_dir.is_dir() {
        return None;
    }
    // SAFETY: build scripts are single-threaded, so mutating the process
    // environment is safe here.
    unsafe { std::env::set_var("cuvs_DIR", &cmake_dir) };
    try_find_cuvs_package()
}

// ---------------------------------------------------------------------------
// Pip prefix detection helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Vendored build
// ---------------------------------------------------------------------------

/// Build cuVS from source, then discover it via CMake against the install prefix.
#[cfg(feature = "vendored")]
fn try_build_and_discover() -> Option<String> {
    let cpp_source =
        std::env::var("CUVS_CPP_SOURCE").unwrap_or_else(|_| DEFAULT_CPP_SOURCE.to_owned());

    // Tell Cargo to rerun build.rs when the C++ source tree changes.
    println!("cargo:rerun-if-changed={cpp_source}");

    let install_prefix = cmake::Config::new(&cpp_source)
        .generator("Ninja")
        .profile("Release")
        .define("BUILD_C_LIBRARY", "ON")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_C_TESTS", "OFF")
        .define("BUILD_CUVS_BENCH", "OFF")
        .define("BUILD_SHARED_LIBS", "ON")
        .define("BUILD_MG_ALGOS", "OFF")
        .define("CUDA_LOG_COMPILE_TIME", "OFF")
        .define("CUVS_NVTX", "ON")
        .define("CMAKE_CUDA_ARCHITECTURES", "NATIVE")
        .build();

    // Point CMake at the freshly-built install and discover it like any other.
    let cmake_dir = ["lib/cmake/cuvs", "lib64/cmake/cuvs"]
        .iter()
        .map(|p| install_prefix.join(p))
        .find(|p| p.is_dir())
        .expect("vendored build did not produce cmake config files");

    // SAFETY: build scripts are single-threaded.
    unsafe { std::env::set_var("cuvs_DIR", &cmake_dir) };
    try_find_cuvs_package()
}

#[cfg(feature = "vendored")]
fn try_vendored() -> Option<String> {
    try_build_and_discover()
}

#[cfg(not(feature = "vendored"))]
fn try_vendored() -> Option<String> {
    None
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Locate cuVS and emit all cargo link directives.
/// Returns the include directory for bindgen.
fn locate_cuvs() -> String {
    // If the vendored feature is explicitly enabled, build from source immediately.
    if cfg!(feature = "vendored") {
        return try_vendored().expect("vendored feature enabled but build from source failed");
    }

    // If the user explicitly set cuvs_DIR, honor it and fail fast if it doesn't work.
    if std::env::var("cuvs_DIR").is_ok() {
        if let Some(dir) = try_find_cuvs_package() {
            return dir;
        }
        eprintln!("error: cuvs_DIR is set but CMake could not find cuVS at that location.");
        std::process::exit(1);
    }

    // Try system discovery: conda, system install, tarball, CMAKE_PREFIX_PATH.
    if let Some(dir) = try_find_cuvs_package() {
        return dir;
    }

    // Try pip-installed cuVS.
    if let Some(dir) = try_discover_pip() {
        return dir;
    }

    eprintln!(
        "error: Could not find the cuVS CMake package.\n\
         Install cuVS via one of:\n\
         - conda: conda install -c rapidsai libcuvs\n\
         - pip:   pip install cuvs-cu13\n\
         Or set CMAKE_PREFIX_PATH to point to your cuVS installation.\n\
         Or enable the 'vendored' feature to build from source."
    );
    std::process::exit(1);
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings(include_dir: &str) {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set by Cargo"));

    bindgen::Builder::default()
        .header("cuvs_c_wrapper.h")
        .must_use_type("cuvsError_t")
        .allowlist_function("cuvs.*")
        .allowlist_type("(cuvs|DL).*")
        .rustified_enum("(cuvs|DL).*")
        .clang_arg(format!("-I{include_dir}"))
        .clang_arg("-Ivendor") // vendored dlpack header
        .generate()
        .expect("bindgen failed to generate cuvs bindings")
        .write_to_file(out_dir.join("cuvs_bindings.rs"))
        .expect("failed to write cuvs_bindings.rs");
}

#[cfg(not(feature = "generate-bindings"))]
fn generate_bindings(_include_dir: &str) {
    // Pre-generated bindings are used from src/bindings.rs.
}

fn main() {
    // docs.rs builds have no CUDA/cuVS. Skip discovery and linking entirely;
    // the pre-generated bindings in src/bindings.rs are sufficient for docs.
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rerun-if-changed=cuvs_c_wrapper.h");
    println!("cargo:rerun-if-changed=vendor/dlpack/dlpack.h");
    println!("cargo:rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=cuvs_DIR");
    println!("cargo:rerun-if-env-changed=CUVS_CPP_SOURCE");

    let include_dir = locate_cuvs();

    // Expose include path to downstream crates via DEP_CUVS_C_INCLUDE.
    println!("cargo:include={include_dir}");

    generate_bindings(&include_dir);
}
