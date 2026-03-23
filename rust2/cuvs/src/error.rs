/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error types for the cuVS safe bindings.

use std::ffi::CStr;

use crate::ffi;

/// An error reported by the cuVS C library.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct LibraryError(pub(crate) String);

/// Check the return status of a cuVS C API call.
///
/// On failure, the thread-local error text is captured immediately before any
/// subsequent FFI call can overwrite it.
pub(crate) fn check_cuvs(status: ffi::cuvsError_t) -> Result<(), LibraryError> {
    if status == ffi::cuvsError_t::CUVS_SUCCESS {
        return Ok(());
    }

    // SAFETY:
    // - cuvsGetLastErrorText() returns a pointer to thread-local storage
    //   that is valid until the next FFI call on this thread.
    // - We copy the string immediately, so the pointer is not held past
    //   any subsequent FFI call.
    let msg = unsafe { ffi::cuvsGetLastErrorText() };
    let msg = if msg.is_null() {
        "unknown cuVS error".to_owned()
    } else {
        // SAFETY: The pointer is non-null (checked above).
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    };

    Err(LibraryError(msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_cuvs_success() {
        assert!(check_cuvs(ffi::cuvsError_t::CUVS_SUCCESS).is_ok());
    }

    #[test]
    fn check_cuvs_error_without_message() {
        let err = check_cuvs(ffi::cuvsError_t::CUVS_ERROR).unwrap_err();
        // No prior FFI call set error text, so we get either null or empty.
        assert!(!err.0.is_empty());
    }
}
