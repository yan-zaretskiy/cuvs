/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::env;

fn main() {
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo::rerun-if-env-changed=DEP_CUVS_C_INCLUDE");

    match env::var_os("DEP_CUVS_C_INCLUDE") {
        Some(include_paths) => {
            println!(
                "cargo:rustc-env=CUVS_SYS_INCLUDE={}",
                include_paths.to_string_lossy()
            );
        }
        None => {
            eprintln!(
                "error: expected DEP_CUVS_C_INCLUDE from cuvs-sys (links = \"cuvs_c\"), but it was not set"
            );
            std::process::exit(1);
        }
    }
}
