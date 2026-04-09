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

use crate::dlpack::{AsDLTensor, AsMutDLTensor, DLTensorFfi};
use crate::error::check_cuvs;
use crate::ffi;
use crate::resources::Resources;

/// Error type for k-means operations.
#[derive(Debug, thiserror::Error)]
pub enum KMeansError {
    /// The C library reported a failure.
    #[error(transparent)]
    Library(#[from] crate::error::LibraryError),
    /// A parameter value failed validation.
    #[error("invalid parameter: {0}")]
    Validation(String),
}

/// Run k-means training (`cuvsKMeansFit`).
///
/// Writes learned centers into `centroids` and returns `(inertia, n_iter)` as reported by C.
pub fn fit<X, W, C>(
    res: &Resources,
    params: &Params,
    x: &X,
    sample_weight: Option<&W>,
    centroids: &C,
) -> Result<(f64, i32), KMeansError>
where
    X: AsDLTensor + ?Sized,
    W: AsDLTensor + ?Sized,
    C: AsMutDLTensor + ?Sized,
{
    let mut inertia = 0.0f64;
    let mut n_iter = 0i32;
    let sw_ptr = sample_weight
        .map(|w| w.ffi_ptr())
        .unwrap_or(std::ptr::null_mut());
    let status = unsafe {
        ffi::cuvsKMeansFit(
            res.handle(),
            params.as_ptr(),
            x.ffi_ptr(),
            sw_ptr,
            centroids.as_mut_dl_tensor(),
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
pub fn predict<X, W, Cent, L>(
    res: &Resources,
    params: &Params,
    x: &X,
    sample_weight: Option<&W>,
    centroids: &Cent,
    labels: &L,
    normalize_weight: bool,
) -> Result<f64, KMeansError>
where
    X: AsDLTensor + ?Sized,
    W: AsDLTensor + ?Sized,
    Cent: AsDLTensor + ?Sized,
    L: AsMutDLTensor + ?Sized,
{
    let mut inertia = 0.0f64;
    let sw_ptr = sample_weight
        .map(|w| w.ffi_ptr())
        .unwrap_or(std::ptr::null_mut());
    let status = unsafe {
        ffi::cuvsKMeansPredict(
            res.handle(),
            params.as_ptr(),
            x.ffi_ptr(),
            sw_ptr,
            centroids.ffi_ptr(),
            labels.as_mut_dl_tensor(),
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
pub fn cluster_cost<X, C>(res: &Resources, x: &X, centroids: &C) -> Result<f64, KMeansError>
where
    X: AsDLTensor + ?Sized,
    C: AsDLTensor + ?Sized,
{
    let mut cost = 0.0f64;
    let status = unsafe {
        ffi::cuvsKMeansClusterCost(res.handle(), x.ffi_ptr(), centroids.ffi_ptr(), &mut cost)
    };
    check_cuvs(status)?;
    Ok(cost)
}

#[cfg(all(test, feature = "torch"))]
mod torch_tests {
    use super::*;
    use crate::dlpack::{BorrowedDLTensor, MutBorrowedDLTensor};

    const K: i64 = 4;
    const N: i64 = 200;
    const D: i64 = 8;

    #[test]
    fn fit_predict_cluster_cost_device_f32() {
        let res = Resources::new().unwrap();
        let dev = tch::Device::Cuda(0);

        let x = tch::Tensor::randn([N, D], (tch::Kind::Float, dev));
        let x_v = BorrowedDLTensor::try_from(&x).unwrap();

        let centroids = tch::Tensor::zeros([K, D], (tch::Kind::Float, dev));
        let c_v = MutBorrowedDLTensor::try_from(&centroids).unwrap();

        let params = Params::builder()
            .n_clusters(K as i32)
            .max_iter(50)
            .inertia_check(true)
            .build()
            .unwrap();

        let (inertia, n_iter) = fit(&res, &params, &x_v, None::<&BorrowedDLTensor>, &c_v).unwrap();
        assert!(n_iter >= 0);
        assert!(inertia.is_finite());

        let labels = tch::Tensor::zeros([N], (tch::Kind::Int, dev));
        let l_v = MutBorrowedDLTensor::try_from(&labels).unwrap();
        let x_pred = BorrowedDLTensor::try_from(&x).unwrap();
        let c_pred = BorrowedDLTensor::try_from(&centroids).unwrap();

        let pred_inertia = predict(
            &res,
            &params,
            &x_pred,
            None::<&BorrowedDLTensor>,
            &c_pred,
            &l_v,
            false,
        )
        .unwrap();
        assert!(pred_inertia.is_finite());

        let cost = cluster_cost(&res, &x_pred, &c_pred).unwrap();
        assert!(cost.is_finite());
    }

    #[test]
    fn host_x_fit_with_streaming_batch_size() {
        let res = Resources::new().unwrap();
        let dev = tch::Device::Cuda(0);

        let x_cpu = tch::Tensor::randn([N, D], (tch::Kind::Float, tch::Device::Cpu));
        let x_v = BorrowedDLTensor::try_from(&x_cpu).unwrap();

        let centroids = tch::Tensor::zeros([K, D], (tch::Kind::Float, dev));
        let c_v = MutBorrowedDLTensor::try_from(&centroids).unwrap();

        let params = Params::builder()
            .n_clusters(K as i32)
            .streaming_batch_size(32)
            .build()
            .unwrap();

        let (_inertia, _n_iter) = fit(&res, &params, &x_v, None::<&BorrowedDLTensor>, &c_v).unwrap();
    }

    #[test]
    fn init_array_uses_initial_centroids() {
        let res = Resources::new().unwrap();
        let dev = tch::Device::Cuda(0);

        let x = tch::Tensor::randn([N, D], (tch::Kind::Float, dev));
        let x_v = BorrowedDLTensor::try_from(&x).unwrap();

        let seed = tch::Tensor::randn([K, D], (tch::Kind::Float, dev));
        let centroids = seed.shallow_clone();
        let c_v = MutBorrowedDLTensor::try_from(&centroids).unwrap();

        let params = Params::builder()
            .n_clusters(K as i32)
            .init(KMeansInitMethod::Array)
            .max_iter(0)
            .build()
            .unwrap();

        let (_inertia, _n_iter) =
            fit(&res, &params, &x_v, None::<&BorrowedDLTensor>, &c_v).unwrap();
        assert!(centroids.allclose(&seed, 1e-5, 1e-5, false));
    }

    #[test]
    fn hierarchical_fit_predict_f32() {
        let res = Resources::new().unwrap();
        let dev = tch::Device::Cuda(0);

        let x = tch::Tensor::randn([N, D], (tch::Kind::Float, dev));
        let x_v = BorrowedDLTensor::try_from(&x).unwrap();

        let centroids = tch::Tensor::zeros([K, D], (tch::Kind::Float, dev));
        let c_v = MutBorrowedDLTensor::try_from(&centroids).unwrap();

        let params = Params::builder()
            .n_clusters(K as i32)
            .hierarchical(true)
            .hierarchical_n_iters(5)
            .build()
            .unwrap();

        fit(&res, &params, &x_v, None::<&BorrowedDLTensor>, &c_v).unwrap();

        let labels = tch::Tensor::zeros([N], (tch::Kind::Int, dev));
        let l_v = MutBorrowedDLTensor::try_from(&labels).unwrap();
        let c_pred = BorrowedDLTensor::try_from(&centroids).unwrap();
        let inertia = predict(
            &res,
            &params,
            &x_v,
            None::<&BorrowedDLTensor>,
            &c_pred,
            &l_v,
            false,
        )
        .unwrap();
        assert_eq!(inertia, 0.0);
    }
}
