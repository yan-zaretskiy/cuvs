/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder-pattern parameter types for IVF-PQ index build and search.
//!
//! Each parameter type owns the corresponding C params handle directly. The
//! generated `bon` builder configures that handle in the constructor, so there
//! is no duplicate Rust field-bag to keep in sync with the FFI state. All
//! setters are optional; unset values retain the library defaults from the
//! underlying C `*ParamsCreate` functions.

use std::{fmt, ptr};

use bon::bon;

use crate::distance::DistanceType;
use crate::error::check_cuvs;
use crate::ffi;

use super::{
    CodebookGen, CoarseSearchDType, InternalDistanceDType, IvfPqError, ListLayout, LutDType,
};

/// Parameters for building an IVF-PQ index.
///
/// ```ignore
/// use cuvs::neighbors::ivf_pq::IndexParams;
/// use cuvs::distance::DistanceType;
///
/// let params = IndexParams::builder()
///     .n_lists(100)
///     .pq_dim(16)
///     .metric(DistanceType::L2Expanded)
///     .build()?;
/// ```
pub struct IndexParams {
    handle: ffi::cuvsIvfPqIndexParams_t,
}

#[bon]
impl IndexParams {
    #[builder]
    pub fn new(
        metric: Option<DistanceType>,
        n_lists: Option<u32>,
        kmeans_n_iters: Option<u32>,
        kmeans_trainset_fraction: Option<f64>,
        pq_bits: Option<u32>,
        pq_dim: Option<u32>,
        codebook_kind: Option<CodebookGen>,
        codes_layout: Option<ListLayout>,
        force_random_rotation: Option<bool>,
        conservative_memory_allocation: Option<bool>,
        max_train_points_per_pq_code: Option<u32>,
        add_data_on_build: Option<bool>,
    ) -> Result<Self, IvfPqError> {
        if let Some(n) = n_lists
            && n == 0
        {
            return Err(IvfPqError::Validation("n_lists must be > 0".into()));
        }

        if let Some(frac) = kmeans_trainset_fraction
            && (!(0.0 < frac && frac <= 1.0))
        {
            return Err(IvfPqError::Validation(format!(
                "kmeans_trainset_fraction must be in (0, 1], got {frac}"
            )));
        }

        if let Some(bits) = pq_bits
            && !(4..=8).contains(&bits)
        {
            return Err(IvfPqError::Validation(format!(
                "pq_bits must be within [4, 8], got {bits}"
            )));
        }

        let effective_pq_bits = pq_bits.unwrap_or(8);
        if let Some(dim) = pq_dim
            && dim != 0
            && (u64::from(dim) * u64::from(effective_pq_bits)) % 8 != 0
        {
            return Err(IvfPqError::Validation(format!(
                "pq_dim * pq_bits must be a multiple of 8, got pq_dim={dim}, pq_bits={effective_pq_bits}"
            )));
        }

        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsIvfPqIndexParamsCreate(&mut handle) })?;

        let params = Self { handle };
        unsafe {
            if let Some(v) = metric {
                (*params.handle).metric = v.into();
                (*params.handle).metric_arg = v.metric_arg();
            }
            if let Some(v) = n_lists {
                (*params.handle).n_lists = v;
            }
            if let Some(v) = kmeans_n_iters {
                (*params.handle).kmeans_n_iters = v;
            }
            if let Some(v) = kmeans_trainset_fraction {
                (*params.handle).kmeans_trainset_fraction = v;
            }
            if let Some(v) = pq_bits {
                (*params.handle).pq_bits = v;
            }
            if let Some(v) = pq_dim {
                (*params.handle).pq_dim = v;
            }
            if let Some(v) = codebook_kind {
                (*params.handle).codebook_kind = v.into();
            }
            if let Some(v) = codes_layout {
                (*params.handle).codes_layout = v.into();
            }
            if let Some(v) = force_random_rotation {
                (*params.handle).force_random_rotation = v;
            }
            if let Some(v) = conservative_memory_allocation {
                (*params.handle).conservative_memory_allocation = v;
            }
            if let Some(v) = max_train_points_per_pq_code {
                (*params.handle).max_train_points_per_pq_code = v;
            }
            if let Some(v) = add_data_on_build {
                (*params.handle).add_data_on_build = v;
            }
        }

        Ok(params)
    }
}

impl IndexParams {
    pub(super) fn handle(&self) -> ffi::cuvsIvfPqIndexParams_t {
        self.handle
    }
}

impl fmt::Debug for IndexParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IndexParams")
            .field(unsafe { &*self.handle })
            .finish()
    }
}

impl Drop for IndexParams {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsIvfPqIndexParamsDestroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// SearchParams
// ---------------------------------------------------------------------------

/// Parameters for searching an IVF-PQ index.
///
/// ```ignore
/// use cuvs::neighbors::ivf_pq::SearchParams;
///
/// let params = SearchParams::builder()
///     .n_probes(20)
///     .build()?;
/// ```
pub struct SearchParams {
    handle: ffi::cuvsIvfPqSearchParams_t,
}

#[bon]
impl SearchParams {
    #[builder]
    pub fn new(
        n_probes: Option<u32>,
        lut_dtype: Option<LutDType>,
        internal_distance_dtype: Option<InternalDistanceDType>,
        coarse_search_dtype: Option<CoarseSearchDType>,
        max_internal_batch_size: Option<u32>,
        preferred_shmem_carveout: Option<f64>,
    ) -> Result<Self, IvfPqError> {
        if let Some(n) = n_probes
            && n == 0
        {
            return Err(IvfPqError::Validation("n_probes must be > 0".into()));
        }

        if let Some(carveout) = preferred_shmem_carveout
            && !(0.0..=1.0).contains(&carveout)
        {
            return Err(IvfPqError::Validation(format!(
                "preferred_shmem_carveout must be in [0, 1], got {carveout}"
            )));
        }

        if matches!(
            (lut_dtype, internal_distance_dtype),
            (Some(LutDType::F32), Some(InternalDistanceDType::F16))
        ) {
            return Err(IvfPqError::Validation(
                "internal_distance_dtype must be at least as wide as lut_dtype".into(),
            ));
        }

        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsIvfPqSearchParamsCreate(&mut handle) })?;

        let params = Self { handle };
        unsafe {
            if let Some(v) = n_probes {
                (*params.handle).n_probes = v;
            }
            if let Some(v) = lut_dtype {
                (*params.handle).lut_dtype = v.into();
            }
            if let Some(v) = internal_distance_dtype {
                (*params.handle).internal_distance_dtype = v.into();
            }
            if let Some(v) = coarse_search_dtype {
                (*params.handle).coarse_search_dtype = v.into();
            }
            if let Some(v) = max_internal_batch_size {
                (*params.handle).max_internal_batch_size = v;
            }
            if let Some(v) = preferred_shmem_carveout {
                (*params.handle).preferred_shmem_carveout = v;
            }
        }

        Ok(params)
    }
}

impl SearchParams {
    pub(super) fn handle(&self) -> ffi::cuvsIvfPqSearchParams_t {
        self.handle
    }
}

impl fmt::Debug for SearchParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SearchParams")
            .field(unsafe { &*self.handle })
            .finish()
    }
}

impl Drop for SearchParams {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsIvfPqSearchParamsDestroy(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;
    use crate::neighbors::ivf_pq::{CoarseSearchDType, InternalDistanceDType, LutDType};

    #[test]
    fn index_params_reject_zero_n_lists() {
        let err = IndexParams::builder().n_lists(0).build().unwrap_err();
        assert!(matches!(err, IvfPqError::Validation(_)));
    }

    #[test]
    fn index_params_reject_invalid_pq_bits() {
        let err = IndexParams::builder().pq_bits(3).build().unwrap_err();
        assert!(matches!(err, IvfPqError::Validation(_)));
    }

    #[test]
    fn index_params_reject_invalid_pq_dim_alignment() {
        let err = IndexParams::builder()
            .pq_bits(5)
            .pq_dim(3)
            .build()
            .unwrap_err();
        assert!(matches!(err, IvfPqError::Validation(_)));
    }

    #[test]
    fn search_params_reject_zero_n_probes() {
        let err = SearchParams::builder().n_probes(0).build().unwrap_err();
        assert!(matches!(err, IvfPqError::Validation(_)));
    }

    #[test]
    fn search_params_reject_out_of_range_carveout() {
        let err = SearchParams::builder()
            .preferred_shmem_carveout(1.5)
            .build()
            .unwrap_err();
        assert!(matches!(err, IvfPqError::Validation(_)));
    }

    #[test]
    fn search_params_reject_invalid_dtype_combinations() {
        let err = SearchParams::builder()
            .lut_dtype(LutDType::F32)
            .internal_distance_dtype(InternalDistanceDType::F16)
            .build()
            .unwrap_err();
        assert!(matches!(err, IvfPqError::Validation(_)));

        let params = SearchParams::builder()
            .lut_dtype(LutDType::U8)
            .internal_distance_dtype(InternalDistanceDType::F16)
            .coarse_search_dtype(CoarseSearchDType::I8)
            .build()
            .unwrap();
        let _ = params;
    }

    #[test]
    fn lp_metric_sets_metric_arg_from_distance_type() {
        let params = IndexParams::builder()
            .metric(DistanceType::LpUnexpanded(OrderedFloat(4.0)))
            .build()
            .unwrap();
        unsafe {
            assert_eq!((*params.handle()).metric, ffi::cuvsDistanceType::LpUnexpanded);
            assert_eq!((*params.handle()).metric_arg, 4.0);
        }
    }
}
