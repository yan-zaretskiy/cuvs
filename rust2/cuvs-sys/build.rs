/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#[cfg(not(feature = "vendored"))]
use std::process::Command;
use std::path::Path;
use std::path::PathBuf;

use cmake_package::find_package;
use cmake_package::{Error as CmakeError, VersionError};
use thiserror::Error;

const CUVS_COMPONENT: &str = "c_api";
#[cfg(feature = "vendored")]
const DEFAULT_CPP_SOURCE: &str = "../../cpp";
const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

struct CuvsMetadata {
    include_dir: String,
    lib_dir: String,
}

// ---------------------------------------------------------------------------
// CMake package discovery
// ---------------------------------------------------------------------------

/// Discovery error with enough context for actionable error messages.
#[derive(Debug, Error)]
enum DiscoveryError {
    /// A cuVS library was found but with an incompatible version.
    #[error("Found cuVS {found}, but cuvs-sys requires {required}.")]
    VersionMismatch {
        found: String,
        required: &'static str,
    },
    /// CMake is not installed or too old.
    #[error("CMake is not installed or too old (3.19+ required). Install CMake and try again.")]
    CmakeUnavailable,
    /// No cuVS library was found.
    #[error(
        "Could not find cuVS CMake package.\n\
         \n\
         Install cuVS via one of:\n\
         - conda: conda install -c rapidsai libcuvs\n\
         - pip:   pip install libcuvs-cu<CUDA_VERSION>\n\
         Or set CMAKE_PREFIX_PATH to point to your cuVS installation.\n\
         Or enable the 'vendored' feature to build from source."
    )]
    NotFound,
}

/// Run CMake `find_package(cuvs)` and extract the include and library directories.
/// Calls `CMakeTarget::link()` to emit the full set of cargo link directives,
/// preserving all link libraries, directories, and options from the CMake target.
///
/// When `prefix` is provided, it is passed as `CMAKE_PREFIX_PATH` so CMake
/// searches under that installation root (e.g. `<prefix>/lib/cmake/cuvs`).
fn try_find_cuvs_package(prefix: Option<PathBuf>) -> Result<CuvsMetadata, DiscoveryError> {
    let mut builder = find_package("cuvs")
        .version(PACKAGE_VERSION)
        .components([CUVS_COMPONENT.to_owned()]);
    if let Some(path) = prefix {
        builder = builder.prefix_paths(vec![path]);
    }
    let package = builder.find().map_err(|e| match e {
        CmakeError::Version(VersionError::VersionTooOld(v)) => {
            DiscoveryError::VersionMismatch {
                found: format!("{}.{}.{}", v.major, v.minor, v.patch),
                required: PACKAGE_VERSION,
            }
        }
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            DiscoveryError::CmakeUnavailable
        }
        _ => DiscoveryError::NotFound,
    })?;

    let target = package
        .target("cuvs::c_api")
        .ok_or(DiscoveryError::NotFound)?;

    let include_dir = target
        .include_directories
        .first()
        .cloned()
        .ok_or(DiscoveryError::NotFound)?;

    let lib_dir = target
        .location
        .as_deref()
        .and_then(|location| Path::new(location).parent())
        .and_then(|path| path.to_str())
        .map(str::to_owned)
        .or_else(|| target.link_directories.first().cloned())
        .ok_or(DiscoveryError::NotFound)?;

    target.link();

    Ok(CuvsMetadata {
        include_dir,
        lib_dir,
    })
}

/// Try to discover a pip-installed cuVS.
/// The pip package places files under `site-packages/libcuvs/lib64/cmake/cuvs/`,
/// which CMake does not locate from a generic prefix path. Point `cuvs_DIR`
/// directly at the package config directory for this discovery attempt.
#[cfg(not(feature = "vendored"))]
fn try_discover_pip() -> Result<CuvsMetadata, DiscoveryError> {
    let cmake_dir = pip_cuvs_cmake_dir().ok_or(DiscoveryError::NotFound)?;
    // SAFETY: build scripts are single-threaded, so mutating the process
    // environment is safe here.
    unsafe { std::env::set_var("cuvs_DIR", &cmake_dir) };
    try_find_cuvs_package(None)
}

/// Try to find a pip-installed cuVS by locating `site-packages/libcuvs`.
/// Checks both the active venv (via VIRTUAL_ENV) and the system python.
#[cfg(not(feature = "vendored"))]
fn pip_cuvs_cmake_dir() -> Option<PathBuf> {
    // If a venv is active, check its site-packages first.
    // Venvs have a single lib/python3.XX/ directory.
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let lib_dir = Path::new(&venv).join("lib");
        if let Ok(entries) = std::fs::read_dir(&lib_dir)
            && let Some(prefix) = entries
                .filter_map(|e| e.ok())
                .map(|entry| entry.path().join("site-packages/libcuvs/lib64/cmake/cuvs"))
                .find(|path| path.is_dir())
        {
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
    let prefix = site_packages.join("libcuvs/lib64/cmake/cuvs");
    prefix.is_dir().then_some(prefix)
}

// ---------------------------------------------------------------------------
// Vendored build
// ---------------------------------------------------------------------------

/// Build cuVS from source, then discover it via CMake against the install prefix.
#[cfg(feature = "vendored")]
fn locate_cuvs() -> Result<CuvsMetadata, DiscoveryError> {
    let cpp_source =
        std::env::var("CUVS_CPP_SOURCE").unwrap_or_else(|_| DEFAULT_CPP_SOURCE.to_owned());

    // Tell Cargo to rerun build.rs when the C++ source tree changes.
    println!("cargo::rerun-if-changed={cpp_source}");

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
    try_find_cuvs_package(Some(install_prefix))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Locate cuVS and emit all cargo link directives.
/// Returns the include directory for bindgen or a typed discovery error.
#[cfg(not(feature = "vendored"))]
fn locate_cuvs() -> Result<CuvsMetadata, DiscoveryError> {
    match try_find_cuvs_package(None) {
        Ok(metadata) => Ok(metadata),
        Err(DiscoveryError::NotFound) => try_discover_pip(),
        Err(DiscoveryError::VersionMismatch { found, .. }) => match try_discover_pip() {
            Err(DiscoveryError::NotFound) => Err(DiscoveryError::VersionMismatch {
                found,
                required: PACKAGE_VERSION,
            }),
            result => result,
        },
        Err(error) => Err(error),
    }
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
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
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
    println!("cargo::rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo::rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo::rerun-if-env-changed=cuvs_DIR");
    #[cfg(feature = "vendored")]
    println!("cargo::rerun-if-env-changed=CUVS_CPP_SOURCE");

    // docs.rs builds have no CUDA/cuVS. Skip discovery and linking entirely;
    // the pre-generated bindings in src/bindings.rs are sufficient for docs.
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    let metadata = match locate_cuvs() {
        Ok(metadata) => metadata,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    };

    // Expose include path to downstream crates via DEP_CUVS_C_INCLUDE.
    println!("cargo::metadata=include={}", metadata.include_dir);
    // Expose the directory containing libcuvs_c.so via DEP_CUVS_C_LIB.
    println!("cargo::metadata=lib={}", metadata.lib_dir);

    generate_bindings(&metadata.include_dir);
}
