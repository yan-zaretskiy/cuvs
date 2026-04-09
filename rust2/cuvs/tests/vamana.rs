#![cfg(feature = "torch")]

/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use cuvs::distance::DistanceType;
use cuvs::dlpack::BorrowedDLTensor;
use cuvs::neighbors::vamana::{Index, IndexParams, VamanaError};
use cuvs::resources::Resources;

const N_ROWS: i64 = 256;
const DIM: i64 = 16;

fn temp_prefix(name: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cuvs-rust-{name}-{}-{unique}",
        std::process::id()
    ))
}

#[test]
fn index_params_reject_invalid_values() {
    let err = IndexParams::builder()
        .metric(DistanceType::InnerProduct)
        .build()
        .unwrap_err();
    assert!(matches!(err, VamanaError::Validation(_)));

    let err = IndexParams::builder().graph_degree(16).build().unwrap_err();
    assert!(matches!(err, VamanaError::Validation(_)));

    let err = IndexParams::builder()
        .graph_degree(32)
        .visited_size(32)
        .build()
        .unwrap_err();
    assert!(matches!(err, VamanaError::Validation(_)));

    let err = IndexParams::builder().vamana_iters(0.5).build().unwrap_err();
    assert!(matches!(err, VamanaError::Validation(_)));
}

#[test]
fn build_exposes_dims() {
    let res = Resources::new().unwrap();
    let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
    let dataset_dl = BorrowedDLTensor::try_from(&dataset).unwrap();

    let params = IndexParams::builder()
        .graph_degree(32)
        .visited_size(64)
        .build()
        .unwrap();
    let index = Index::build(&res, &params, &dataset_dl).unwrap();

    assert_eq!(index.dims().unwrap(), DIM);
}

#[test]
fn serialize_writes_diskann_files() {
    let res = Resources::new().unwrap();
    let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
    let dataset_dl = BorrowedDLTensor::try_from(&dataset).unwrap();

    let params = IndexParams::builder()
        .graph_degree(32)
        .visited_size(64)
        .build()
        .unwrap();
    let index = Index::build(&res, &params, &dataset_dl).unwrap();

    let prefix = temp_prefix("vamana");
    index.serialize(&res, &prefix, true).unwrap();

    assert!(prefix.exists());
    assert!(prefix.with_extension("data").exists());

    let _ = std::fs::remove_file(&prefix);
    let _ = std::fs::remove_file(prefix.with_extension("data"));
}
