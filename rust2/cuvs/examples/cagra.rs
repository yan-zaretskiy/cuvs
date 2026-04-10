/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::convert::TryFrom;

use cuvs::neighbors::cagra::{Index, IndexParams, SearchParams};
use cuvs::resources::Resources;

/// Example showing how to index and search data with CAGRA.
fn cagra_example() -> Result<(), Box<dyn std::error::Error>> {
    let res = Resources::new()?;

    // Create a new random dataset to index
    let n_datapoints: i64 = 65536;
    let n_features: i64 = 512;
    let dataset = tch::Tensor::randn(
        [n_datapoints, n_features],
        (tch::Kind::Float, tch::Device::Cuda(0)),
    );

    // Build the CAGRA index
    let build_params = IndexParams::try_new()?;
    let index = Index::build(&res, &build_params, &dataset)?;
    println!(
        "Indexed {}x{} datapoints into CAGRA index",
        n_datapoints, n_features
    );

    // Use the first 4 points from the dataset as queries — will test that we
    // get them back as their own nearest neighbor.
    let n_queries = 4;
    let queries = dataset.narrow(0, 0, n_queries);
    let k = 10;

    // Allocate device memory for search outputs
    let mut neighbors =
        tch::Tensor::zeros([n_queries, k], (tch::Kind::Int64, tch::Device::Cuda(0)));
    let mut distances =
        tch::Tensor::zeros([n_queries, k], (tch::Kind::Float, tch::Device::Cuda(0)));

    let search_params = SearchParams::try_new()?;
    index.search(
        &res,
        &search_params,
        &queries,
        &mut neighbors,
        &mut distances,
    )?;

    println!("Neighbors {:?}", Vec::<Vec<i64>>::try_from(&neighbors)?);
    println!("Distances {:?}", Vec::<Vec<f32>>::try_from(&distances)?);
    Ok(())
}

fn main() {
    if let Err(e) = cagra_example() {
        println!("Failed to run CAGRA: {e:?}");
    }
}
