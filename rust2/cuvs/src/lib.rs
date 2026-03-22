/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuVS: Safe Rust bindings for GPU-accelerated vector search.

#[cfg(test)]
mod tests {
    #[test]
    fn test_build_script_receives_cuvs_sys_include_metadata() {
        assert!(
            option_env!("CUVS_SYS_INCLUDE").is_some(),
            "cuvs build.rs should receive DEP_CUVS_C_INCLUDE metadata from cuvs-sys"
        );
    }
}
