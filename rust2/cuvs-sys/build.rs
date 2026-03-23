/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use cmake_package::find_package;
use cmake_package::{Error as CmakeError, VersionError};

const CUVS_COMPONENT: &str = "c_api";
#[cfg(feature = "vendored")]
const DEFAULT_CPP_SOURCE: &str = "../../cpp";

// ---------------------------------------------------------------------------
// CMake package discovery
// ---------------------------------------------------------------------------

/// Discovery error with enough context for actionable error messages.
enum DiscoveryError {
    /// A cuVS library was found but with an incompatible version.
    VersionMismatch { found: String },
    /// CMake is not installed or too old.
    CmakeUnavailable,
    /// No cuVS library was found.
    NotFound,
}

/// Run CMake `find_package(cuvs)` and extract the include directory.
/// Calls `CMakeTarget::link()` to emit the full set of cargo link directives,
/// preserving all link libraries, directories, and options from the CMake target.
///
/// When `prefixes` is provided, they are passed as `CMAKE_PREFIX_PATH` so CMake
/// searches under those installation roots (e.g. `<prefix>/lib/cmake/cuvs`).
fn try_find_cuvs_package(prefixes: Option<Vec<PathBuf>>) -> Result<String, DiscoveryError> {
    let mut builder = find_package("cuvs")
        .version(std::env::var("CARGO_PKG_VERSION").unwrap())
        .components([CUVS_COMPONENT.to_owned()]);
    if let Some(paths) = prefixes {
        builder = builder.prefix_paths(paths);
    }
    let package = builder.find().map_err(|e| match e {
        CmakeError::Version(VersionError::VersionTooOld(v)) => DiscoveryError::VersionMismatch {
            found: format!("{}.{}.{}", v.major, v.minor, v.patch),
        },
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

    target.link();

    Ok(include_dir)
}

/// Try to discover a pip-installed cuVS.
/// The pip package places files under `site-packages/libcuvs/`, which CMake
/// doesn't search by default. We pass it as a prefix path so CMake looks for
/// config files under `<prefix>/lib[64]/cmake/cuvs/`.
fn try_discover_pip() -> Result<String, DiscoveryError> {
    let prefix = pip_cuvs_prefix().ok_or(DiscoveryError::NotFound)?;
    try_find_cuvs_package(Some(vec![prefix]))
}

/// Try to find a pip-installed cuVS by locating `site-packages/libcuvs`.
/// Checks both the active venv (via VIRTUAL_ENV) and the system python.
fn pip_cuvs_prefix() -> Option<PathBuf> {
    // If a venv is active, check its site-packages first.
    // Venvs have a single lib/python3.XX/ directory.
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let lib_dir = Path::new(&venv).join("lib");
        if let Ok(entries) = std::fs::read_dir(&lib_dir) {
            let candidate = entries
                .filter_map(|e| e.ok())
                .find(|e| e.file_name().as_encoded_bytes().starts_with(b"python"))
                .map(|e| e.path().join("site-packages/libcuvs"));
            if let Some(prefix) = candidate.filter(|p| p.is_dir()) {
                return Some(prefix);
            }
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

// ---------------------------------------------------------------------------
// Vendored build
// ---------------------------------------------------------------------------

/// Build cuVS from source, then discover it via CMake against the install prefix.
#[cfg(feature = "vendored")]
fn try_vendored() -> Option<String> {
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
    try_find_cuvs_package(Some(vec![install_prefix])).ok()
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

    let version = std::env::var("CARGO_PKG_VERSION").unwrap();

    // If the user explicitly set cuvs_DIR, honor it and fail fast if it doesn't work.
    if let Some(cuvs_dir) = std::env::var("cuvs_DIR").ok().filter(|s| !s.is_empty()) {
        match try_find_cuvs_package(None) {
            Ok(dir) => return dir,
            Err(DiscoveryError::VersionMismatch { found }) => {
                eprintln!(
                    "error: cuvs_DIR is set to '{cuvs_dir}' which contains cuVS {found}, \
                     but cuvs-sys requires {version}."
                );
            }
            Err(DiscoveryError::CmakeUnavailable) => {
                eprintln!(
                    "error: CMake is not installed or too old (3.19+ required). \
                     Install CMake and try again."
                );
            }
            Err(DiscoveryError::NotFound) => {
                eprintln!(
                    "error: cuvs_DIR is set to '{cuvs_dir}' but CMake could not find cuVS \
                     at that location."
                );
            }
        }
        std::process::exit(1);
    }

    // Track version mismatches across discovery attempts for a better error message.
    let mut found_version: Option<String> = None;

    let strategies: [&dyn Fn() -> Result<String, DiscoveryError>; 2] = [
        &|| try_find_cuvs_package(None), // system/conda/CMAKE_PREFIX_PATH
        &try_discover_pip,               // pip site-packages
    ];

    for discover in strategies {
        match discover() {
            Ok(dir) => return dir,
            Err(DiscoveryError::VersionMismatch { found }) => {
                found_version = Some(found);
            }
            Err(DiscoveryError::CmakeUnavailable) => {
                eprintln!(
                    "error: CMake is not installed or too old (3.19+ required). \
                     Install CMake and try again."
                );
                std::process::exit(1);
            }
            Err(DiscoveryError::NotFound) => continue,
        }
    }

    if let Some(found) = found_version {
        eprintln!(
            "error: Found cuVS {found}, but cuvs-sys requires {version}.\n\
             \n\
             Install cuVS {version} via one of:\n\
             - conda: conda install -c rapidsai libcuvs={version}\n\
             - pip:   pip install libcuvs-cu<CUDA_VERSION>\n\
             Or update the cuvs-sys crate to match your installed library."
        );
    } else {
        eprintln!(
            "error: Could not find cuVS {version} CMake package.\n\
             \n\
             Searched:\n\
             - CMake default search paths (CMAKE_PREFIX_PATH, system paths)\n\
             - pip site-packages (VIRTUAL_ENV, python3 sysconfig)\n\
             \n\
             Install cuVS {version} via one of:\n\
             - conda: conda install -c rapidsai libcuvs={version}\n\
             - pip:   pip install libcuvs-cu<CUDA_VERSION>\n\
             Or set cuvs_DIR or CMAKE_PREFIX_PATH to point to your cuVS {version} installation.\n\
             Or enable the 'vendored' feature to build from source."
        );
    }
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
    // docs.rs builds have no CUDA/cuVS. Skip discovery and linking entirely;
    // the pre-generated bindings in src/bindings.rs are sufficient for docs.
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    let include_dir = locate_cuvs();

    println!("cargo::rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo::rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo::rerun-if-env-changed=cuvs_DIR");
    #[cfg(feature = "vendored")]
    println!("cargo::rerun-if-env-changed=CUVS_CPP_SOURCE");

    // Expose include path to downstream crates via DEP_CUVS_C_INCLUDE.
    println!("cargo::metadata=include={include_dir}");

    generate_bindings(&include_dir);
}
