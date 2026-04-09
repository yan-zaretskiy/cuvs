/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuVS library version query.

use crate::error::{LibraryError, check_cuvs};
use crate::ffi;

/// Returns the cuVS library version as `(major, minor, patch)`.
pub fn version() -> Result<(u16, u16, u16), LibraryError> {
    let mut major: u16 = 0;
    let mut minor: u16 = 0;
    let mut patch: u16 = 0;

    // SAFETY:
    // - All three pointers are valid, aligned `u16` locals.
    let status = unsafe { ffi::cuvsVersionGet(&mut major, &mut minor, &mut patch) };
    check_cuvs(status)?;
    Ok((major, minor, patch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_crate() {
        let (major, minor, patch) = version().expect("failed to get version");

        let parts: [u16; 3] = env!("CARGO_PKG_VERSION")
            .split('.')
            .map(|s| s.parse().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        assert_eq!([major, minor, patch], parts);
    }
}
