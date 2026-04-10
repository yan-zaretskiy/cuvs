/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! K-means clustering (`cuvsKMeans*`).
//!
//! # Data placement
//!
//! - [`fit`]: `X` may be host or device row-major `f32` / `f64`. When `X` is on the host,
//!   samples are streamed to the GPU in batches controlled by
//!   [`Params::builder`] `streaming_batch_size` (`0` means all samples at once, matching the
//!   C default after `cuvsKMeansParamsCreate`). **`centroids` must be device-accessible** in
//!   all cases. When `X` is host-resident, non-null `sample_weight` must also be host-accessible.
//!   With [`KMeansInitMethod::Array`], initial centers are read from `centroids`.
//! - [`predict`]: **`X` must be device-accessible**; the C++ API rejects host `X`.
//! - [`cluster_cost`]: **`X` and `centroids` must be device-accessible**.
//!
//! # Hierarchical (balanced) k-means
//!
//! When `hierarchical` is true, the implementation switches to the balanced k-means path.
//! The C++ layer then requires **device `X`**, rejects **non-null `sample_weight`**, and rejects
//! **`f64`** tensors.

pub mod params;

pub use params::{KMeansInitMethod, Params};

use crate::dlpack::{DLPackError, IntoDlTensor, IntoDlTensorMut};
use crate::error::check_cuvs;
use crate::ffi;
use crate::resources::Resources;

/// Error type for k-means operations.
#[derive(Debug, thiserror::Error)]
pub enum KMeansError {
    /// The C library reported a failure.
    #[error(transparent)]
    Library(#[from] crate::error::LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
    /// A parameter value failed validation.
    #[error("invalid parameter: {0}")]
    Validation(String),
}

/// Run k-means training (`cuvsKMeansFit`).
///
/// Writes learned centers into `centroids` and returns `(inertia, n_iter)` as reported by C.
pub fn fit<'x, 'w, 'c, X, W, C>(
    res: &Resources,
    params: &Params,
    x: X,
    sample_weight: Option<W>,
    centroids: C,
) -> Result<(f64, i32), KMeansError>
where
    X: IntoDlTensor<'x>,
    W: IntoDlTensor<'w>,
    C: IntoDlTensorMut<'c>,
{
    let x = x.into_dl_tensor()?;
    let sample_weight = sample_weight.map(|w| w.into_dl_tensor()).transpose()?;
    let centroids = centroids.into_dl_tensor_mut()?;
    let mut inertia = 0.0f64;
    let mut n_iter = 0i32;
    let sw_ptr = sample_weight
        .as_ref()
        .map(|w| w.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let status = unsafe {
        ffi::cuvsKMeansFit(
            res.handle(),
            params.as_ptr(),
            x.as_ptr(),
            sw_ptr,
            centroids.as_ptr(),
            &mut inertia,
            &mut n_iter,
        )
    };
    check_cuvs(status)?;
    Ok((inertia, n_iter))
}

/// Assign each row of `X` to a cluster (`cuvsKMeansPredict`).
///
/// For non-hierarchical k-means, returns inertia as a `f64` promoted from the internal
/// floating-point type. For hierarchical k-means, the C++ wrapper sets inertia to `0.0`.
pub fn predict<'x, 'w, 'cent, 'l, X, W, Cent, L>(
    res: &Resources,
    params: &Params,
    x: X,
    sample_weight: Option<W>,
    centroids: Cent,
    labels: L,
    normalize_weight: bool,
) -> Result<f64, KMeansError>
where
    X: IntoDlTensor<'x>,
    W: IntoDlTensor<'w>,
    Cent: IntoDlTensor<'cent>,
    L: IntoDlTensorMut<'l>,
{
    let x = x.into_dl_tensor()?;
    let sample_weight = sample_weight.map(|w| w.into_dl_tensor()).transpose()?;
    let centroids = centroids.into_dl_tensor()?;
    let labels = labels.into_dl_tensor_mut()?;
    let mut inertia = 0.0f64;
    let sw_ptr = sample_weight
        .as_ref()
        .map(|w| w.as_ptr())
        .unwrap_or(std::ptr::null_mut());
    let status = unsafe {
        ffi::cuvsKMeansPredict(
            res.handle(),
            params.as_ptr(),
            x.as_ptr(),
            sw_ptr,
            centroids.as_ptr(),
            labels.as_ptr(),
            normalize_weight,
            &mut inertia,
        )
    };
    check_cuvs(status)?;
    Ok(inertia)
}

/// Sum of per-sample squared distances to the nearest centroid (`cuvsKMeansClusterCost`).
///
/// Both tensors must be device-accessible in the current C++ implementation.
pub fn cluster_cost<'x, 'c, X, C>(res: &Resources, x: X, centroids: C) -> Result<f64, KMeansError>
where
    X: IntoDlTensor<'x>,
    C: IntoDlTensor<'c>,
{
    let x = x.into_dl_tensor()?;
    let centroids = centroids.into_dl_tensor()?;
    let mut cost = 0.0f64;
    let status = unsafe {
        ffi::cuvsKMeansClusterCost(res.handle(), x.as_ptr(), centroids.as_ptr(), &mut cost)
    };
    check_cuvs(status)?;
    Ok(cost)
}

#[cfg(all(test, feature = "torch"))]
mod torch_tests {
    use super::*;

    const K: i64 = 4;
    const N: i64 = 200;
    const D: i64 = 8;

    #[test]
    fn fit_predict_cluster_cost_device_f32() {
        let res = Resources::new().unwrap();
        let dev = tch::Device::Cuda(0);

        let x = tch::Tensor::randn([N, D], (tch::Kind::Float, dev));

        let mut centroids = tch::Tensor::zeros([K, D], (tch::Kind::Float, dev));

        let params = Params::builder()
            .n_clusters(K as i32)
            .max_iter(50)
            .inertia_check(true)
            .build()
            .unwrap();

        let (inertia, n_iter) =
            fit(&res, &params, &x, None::<&tch::Tensor>, &mut centroids).unwrap();
        assert!(n_iter >= 0);
        assert!(inertia.is_finite());

        let mut labels = tch::Tensor::zeros([N], (tch::Kind::Int, dev));

        let pred_inertia = predict(
            &res,
            &params,
            &x,
            None::<&tch::Tensor>,
            &centroids,
            &mut labels,
            false,
        )
        .unwrap();
        assert!(pred_inertia.is_finite());

        let cost = cluster_cost(&res, &x, &centroids).unwrap();
        assert!(cost.is_finite());
    }

    #[test]
    fn host_x_fit_with_streaming_batch_size() {
        let res = Resources::new().unwrap();
        let dev = tch::Device::Cuda(0);

        let x_cpu = tch::Tensor::randn([N, D], (tch::Kind::Float, tch::Device::Cpu));

        let mut centroids = tch::Tensor::zeros([K, D], (tch::Kind::Float, dev));

        let params = Params::builder()
            .n_clusters(K as i32)
            .streaming_batch_size(32)
            .build()
            .unwrap();

        let (_inertia, _n_iter) =
            fit(&res, &params, &x_cpu, None::<&tch::Tensor>, &mut centroids).unwrap();
    }

    #[test]
    fn init_array_uses_initial_centroids() {
        let res = Resources::new().unwrap();
        let dev = tch::Device::Cuda(0);

        let x = tch::Tensor::randn([N, D], (tch::Kind::Float, dev));

        let seed = tch::Tensor::randn([K, D], (tch::Kind::Float, dev));
        let mut centroids = seed.shallow_clone();

        let params = Params::builder()
            .n_clusters(K as i32)
            .init(KMeansInitMethod::Array)
            .max_iter(0)
            .build()
            .unwrap();

        let (_inertia, _n_iter) =
            fit(&res, &params, &x, None::<&tch::Tensor>, &mut centroids).unwrap();
        assert!(centroids.allclose(&seed, 1e-5, 1e-5, false));
    }

    #[test]
    fn hierarchical_fit_predict_f32() {
        let res = Resources::new().unwrap();
        let dev = tch::Device::Cuda(0);

        let x = tch::Tensor::randn([N, D], (tch::Kind::Float, dev));

        let mut centroids = tch::Tensor::zeros([K, D], (tch::Kind::Float, dev));

        let params = Params::builder()
            .n_clusters(K as i32)
            .hierarchical(true)
            .hierarchical_n_iters(5)
            .build()
            .unwrap();

        fit(&res, &params, &x, None::<&tch::Tensor>, &mut centroids).unwrap();

        let mut labels = tch::Tensor::zeros([N], (tch::Kind::Int, dev));
        let inertia = predict(
            &res,
            &params,
            &x,
            None::<&tch::Tensor>,
            &centroids,
            &mut labels,
            false,
        )
        .unwrap();
        assert_eq!(inertia, 0.0);
    }
}
