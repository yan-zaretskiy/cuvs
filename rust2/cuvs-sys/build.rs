/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use cmake_package::find_package;
use cmake_package::{Error as CmakeError, Version, VersionError};
use thiserror::Error;

const CUVS_COMPONENT: &str = "c_api";
const CUVS_C_API_TARGET: &str = "cuvs::c_api";
const CUDA_TOOLKIT_TARGET: &str = "CUDA::toolkit";
const DLPACK_TARGET: &str = "dlpack::dlpack";
const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

struct CuvsMetadata {
    include_dir: PathBuf,
    bindgen_include_dirs: Vec<PathBuf>,
    lib_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// CMake package discovery
// ---------------------------------------------------------------------------

/// Discovery error with enough context for actionable error messages.
#[derive(Debug, Error)]
enum DiscoveryError {
    /// A cuVS library was found but with an incompatible version.
    #[error("Found cuVS {found}, but cuvs-sys requires exact version {required}.")]
    VersionMismatch {
        found: String,
        required: &'static str,
    },
    /// cuVS did not report a package version that we can validate.
    #[error("Found cuVS, but it did not report a parseable package version.")]
    VersionUnavailable,
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
         Or set CMAKE_PREFIX_PATH to point to your cuVS build/install directory."
    )]
    NotFound,
    /// The discovered package did not export the expected target.
    #[error("Found CMake package {package}, but target {target} was not exported.")]
    MissingTarget {
        package: &'static str,
        target: &'static str,
    },
    /// No DLPack package was found.
    #[error(
        "Could not find DLPack CMake package.\n\
         \n\
         Install DLPack so that `find_package(dlpack)` succeeds."
    )]
    DlpackNotFound,
    /// No CUDA Toolkit package was found.
    #[error(
        "Could not find CUDA Toolkit CMake package.\n\
         \n\
         Install CUDA Toolkit so that `find_package(CUDAToolkit)` succeeds."
    )]
    CudaToolkitNotFound,
}

fn requested_version() -> Version {
    PACKAGE_VERSION
        .try_into()
        .expect("workspace package version must be a valid semantic version")
}

fn ensure_exact_cuvs_version(package: &cmake_package::CMakePackage) -> Result<(), DiscoveryError> {
    let found = package.version.ok_or(DiscoveryError::VersionUnavailable)?;
    if found != requested_version() {
        return Err(DiscoveryError::VersionMismatch {
            found: found.to_string(),
            required: PACKAGE_VERSION,
        });
    }
    Ok(())
}

fn find_target(
    package: &cmake_package::CMakePackage,
    package_name: &'static str,
    target_name: &'static str,
) -> Result<cmake_package::CMakeTarget, DiscoveryError> {
    package
        .target(target_name)
        .ok_or(DiscoveryError::MissingTarget {
            package: package_name,
            target: target_name,
        })
}

fn find_cuvs_package(
    prefix: Option<PathBuf>,
) -> Result<cmake_package::CMakePackage, DiscoveryError> {
    let mut builder = find_package("cuvs").components([CUVS_COMPONENT.to_owned()]);
    if let Some(ref path) = prefix {
        builder = builder.prefix_paths(vec![path.to_path_buf()]);
    }
    let package = builder.find().map_err(|e| match e {
        CmakeError::Version(VersionError::InvalidVersion) => DiscoveryError::VersionUnavailable,
        CmakeError::Version(VersionError::VersionTooOld(v)) => DiscoveryError::VersionMismatch {
            found: v.to_string(),
            required: PACKAGE_VERSION,
        },
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            DiscoveryError::CmakeUnavailable
        }
        _ => DiscoveryError::NotFound,
    })?;
    ensure_exact_cuvs_version(&package)?;
    Ok(package)
}

fn find_cudatoolkit_package() -> Result<cmake_package::CMakePackage, DiscoveryError> {
    find_package("CUDAToolkit").find().map_err(|e| match e {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            DiscoveryError::CmakeUnavailable
        }
        _ => DiscoveryError::CudaToolkitNotFound,
    })
}

fn find_dlpack_package() -> Result<cmake_package::CMakePackage, DiscoveryError> {
    find_package("dlpack").find().map_err(|e| match e {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            DiscoveryError::CmakeUnavailable
        }
        _ => DiscoveryError::DlpackNotFound,
    })
}

/// Run CMake `find_package(cuvs)` and extract the include and library directories.
/// Calls `CMakeTarget::link()` to emit the full set of cargo link directives,
/// preserving all link libraries, directories, and options from the CMake target.
///
/// When `prefix` is provided, it is passed as `CMAKE_PREFIX_PATH` so CMake
/// searches under that installation root (e.g. `<prefix>/lib/cmake/cuvs`).
fn try_find_cuvs_package(prefix: Option<PathBuf>) -> Result<CuvsMetadata, DiscoveryError> {
    let package = find_cuvs_package(prefix)?;
    let target = find_target(&package, "cuvs", CUVS_C_API_TARGET)?;

    let include_dir = target
        .include_directories
        .first()
        .map(PathBuf::from)
        .ok_or(DiscoveryError::NotFound)?;

    // CUDAToolkit and DLPack include directories are only needed for bindgen.
    // When using pre-generated bindings (the default), skip their discovery —
    // the cuvs CMake target already carries the transitive link flags we need.
    let bindgen_include_dirs = if cfg!(feature = "generate-bindings") {
        let cudatoolkit = find_cudatoolkit_package()?;
        let cudatoolkit_target =
            find_target(&cudatoolkit, "CUDAToolkit", CUDA_TOOLKIT_TARGET)?;
        let dlpack = find_dlpack_package()?;
        let dlpack_target = find_target(&dlpack, "dlpack", DLPACK_TARGET)?;
        target
            .include_directories
            .iter()
            .chain(cudatoolkit_target.include_directories.iter())
            .chain(dlpack_target.include_directories.iter())
            .map(PathBuf::from)
            .filter(|dir| dir.is_dir())
            .collect()
    } else {
        vec![]
    };

    let lib_dir = target
        .location
        .as_deref()
        .and_then(|location| Path::new(location).parent())
        .map(Path::to_path_buf)
        .or_else(|| target.link_directories.first().map(PathBuf::from))
        .ok_or(DiscoveryError::NotFound)?;

    target.link();

    Ok(CuvsMetadata {
        include_dir,
        bindgen_include_dirs,
        lib_dir,
    })
}

/// Try to discover a pip-installed cuVS.
/// The pip package places files under `site-packages/libcuvs/lib64/cmake/cuvs/`,
/// which CMake does not locate from a generic prefix path. Point `cuvs_DIR`
/// directly at the package config directory for this discovery attempt.
fn try_discover_pip() -> Result<CuvsMetadata, DiscoveryError> {
    let cmake_dir = pip_cuvs_cmake_dir().ok_or(DiscoveryError::NotFound)?;
    // SAFETY: build scripts are single-threaded, so mutating the process
    // environment is safe here.
    unsafe { std::env::set_var("cuvs_DIR", &cmake_dir) };
    try_find_cuvs_package(None)
}

/// Try to find a pip-installed cuVS by locating `site-packages/libcuvs`.
/// Checks both the active venv (via VIRTUAL_ENV) and the system python.
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
// Main
// ---------------------------------------------------------------------------

/// Locate cuVS: try CMake find_package first, fall back to pip-installed package.
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
fn generate_bindings(include_dirs: &[PathBuf]) {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set by Cargo"));

    let mut builder = bindgen::Builder::default()
        .header("cuvs_c_wrapper.h")
        .must_use_type("cuvsError_t")
        .allowlist_function("cuvs.*")
        .allowlist_type("(cuvs|DL).*")
        .rustified_enum("(cuvs|DL).*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    for include_dir in include_dirs {
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    builder
        .generate()
        .expect("bindgen failed to generate cuvs bindings")
        .write_to_file(out_dir.join("cuvs_bindings.rs"))
        .expect("failed to write cuvs_bindings.rs");
}

#[cfg(not(feature = "generate-bindings"))]
fn generate_bindings(_include_dirs: &[PathBuf]) {
    // Pre-generated bindings are used from src/bindings.rs.
}

fn main() {
    println!("cargo::rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo::rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo::rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo::rerun-if-env-changed=cuvs_DIR");

    // doc-only: skip native library discovery and linking entirely.
    // The pre-generated bindings in src/bindings.rs are sufficient for
    // building documentation without a GPU or cuVS install.
    if cfg!(feature = "doc-only") {
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
    println!("cargo::metadata=include={}", metadata.include_dir.display());
    // Expose the directory containing libcuvs_c.so via DEP_CUVS_C_LIB.
    println!("cargo::metadata=lib={}", metadata.lib_dir.display());

    generate_bindings(&metadata.bindgen_include_dirs);
}
