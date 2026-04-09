/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! K-means hyperparameters backed by `cuvsKMeansParams`.

use std::ptr;

use bon::bon;

use crate::distance::DistanceType;
use crate::error::check_cuvs;
use crate::ffi;

use super::KMeansError;

/// How initial cluster centers are chosen (maps to `cuvsKMeansInitMethod`).
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum KMeansInitMethod {
    /// k-means++ (scalable k-means++) — library default when params are created.
    KMeansPlusPlus = 0,
    /// Pick `n_clusters` random rows from `X` as starting centers.
    Random = 1,
    /// Use the `centroids` tensor passed to [`super::fit`] as the starting centers.
    Array = 2,
}

impl From<KMeansInitMethod> for ffi::cuvsKMeansInitMethod {
    fn from(v: KMeansInitMethod) -> Self {
        match v {
            KMeansInitMethod::KMeansPlusPlus => Self::KMeansPlusPlus,
            KMeansInitMethod::Random => Self::Random,
            KMeansInitMethod::Array => Self::Array,
        }
    }
}

/// Owned handle to default-populated k-means parameters.
///
/// Unset builder fields keep the values assigned by `cuvsKMeansParamsCreate`.
pub struct Params {
    handle: ffi::cuvsKMeansParams_t,
}

impl Params {
    pub(crate) fn as_ptr(&self) -> ffi::cuvsKMeansParams_t {
        self.handle
    }
}

impl Drop for Params {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = unsafe { ffi::cuvsKMeansParamsDestroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[bon]
impl Params {
    /// Allocate params via the C API and optionally override fields.
    ///
    /// See `c/include/cuvs/cluster/kmeans.h` and `c/src/cluster/kmeans.cpp` for
    /// runtime restrictions (e.g. hierarchical mode, host vs device data).
    #[builder]
    pub fn new(
        metric: Option<DistanceType>,
        n_clusters: Option<i32>,
        init: Option<KMeansInitMethod>,
        max_iter: Option<i32>,
        tol: Option<f64>,
        n_init: Option<i32>,
        oversampling_factor: Option<f64>,
        batch_samples: Option<i32>,
        batch_centroids: Option<i32>,
        inertia_check: Option<bool>,
        hierarchical: Option<bool>,
        hierarchical_n_iters: Option<i32>,
        streaming_batch_size: Option<i64>,
    ) -> Result<Self, KMeansError> {
        if let Some(n) = n_clusters
            && n <= 0
        {
            return Err(KMeansError::Validation("n_clusters must be > 0".into()));
        }
        if let Some(n) = max_iter
            && n < 0
        {
            return Err(KMeansError::Validation("max_iter must be >= 0".into()));
        }
        if let Some(n) = n_init
            && n <= 0
        {
            return Err(KMeansError::Validation("n_init must be > 0".into()));
        }
        if let Some(n) = hierarchical_n_iters
            && n < 0
        {
            return Err(KMeansError::Validation(
                "hierarchical_n_iters must be >= 0".into(),
            ));
        }
        if let Some(b) = streaming_batch_size
            && b < 0
        {
            return Err(KMeansError::Validation(
                "streaming_batch_size must be >= 0".into(),
            ));
        }

        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsKMeansParamsCreate(&mut handle) })?;

        let params = Self { handle };
        unsafe {
            if let Some(v) = metric {
                (*params.handle).metric = v.into();
            }
            if let Some(v) = n_clusters {
                (*params.handle).n_clusters = v;
            }
            if let Some(v) = init {
                (*params.handle).init = v.into();
            }
            if let Some(v) = max_iter {
                (*params.handle).max_iter = v;
            }
            if let Some(v) = tol {
                (*params.handle).tol = v;
            }
            if let Some(v) = n_init {
                (*params.handle).n_init = v;
            }
            if let Some(v) = oversampling_factor {
                (*params.handle).oversampling_factor = v;
            }
            if let Some(v) = batch_samples {
                (*params.handle).batch_samples = v;
            }
            if let Some(v) = batch_centroids {
                (*params.handle).batch_centroids = v;
            }
            if let Some(v) = inertia_check {
                (*params.handle).inertia_check = v;
            }
            if let Some(v) = hierarchical {
                (*params.handle).hierarchical = v;
            }
            if let Some(v) = hierarchical_n_iters {
                (*params.handle).hierarchical_n_iters = v;
            }
            if let Some(v) = streaming_batch_size {
                (*params.handle).streaming_batch_size = v;
            }
        }

        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_positive_n_clusters() {
        let res = Params::builder().n_clusters(0).build();
        assert!(matches!(res, Err(KMeansError::Validation(_))));
    }

    #[test]
    fn builder_sets_streaming_batch_size() {
        let p = Params::builder().streaming_batch_size(128).build().unwrap();
        unsafe {
            assert_eq!((*p.as_ptr()).streaming_batch_size, 128);
        }
    }
}
