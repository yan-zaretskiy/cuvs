/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuVS C library log level control.
//!
//! This controls the verbosity of the C library's internal logging, which
//! goes directly to stderr via `rapids_logger`. It is **global process state**
//! and is **not thread-safe**.

use crate::ffi;

/// Re-export of `cuvsLogLevel_t` — the C library's log verbosity levels.
pub use ffi::cuvsLogLevel_t as LogLevel;

/// Returns the current cuVS C library log level.
pub fn get_log_level() -> LogLevel {
    // SAFETY: cuvsGetLogLevel has no preconditions and cannot fail.
    unsafe { ffi::cuvsGetLogLevel() }
}

/// Sets the cuVS C library log level.
pub fn set_log_level(level: LogLevel) {
    // SAFETY: cuvsSetLogLevel has no preconditions and cannot fail.
    unsafe { ffi::cuvsSetLogLevel(level) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_and_set_log_level() {
        let original = get_log_level();

        set_log_level(LogLevel::CUVS_LOG_LEVEL_WARN);
        assert_eq!(get_log_level(), LogLevel::CUVS_LOG_LEVEL_WARN);

        // Restore original level for other tests.
        set_log_level(original);
    }
}
