/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder-pattern parameter types for IVF-Flat index build and search.
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

use super::IvfFlatError;

/// Parameters for building an IVF-Flat index.
///
/// ```ignore
/// use cuvs::neighbors::ivf_flat::IndexParams;
/// use cuvs::distance::DistanceType;
///
/// let params = IndexParams::builder()
///     .n_lists(100)
///     .metric(DistanceType::L2Expanded)
///     .build()?;
/// ```
pub struct IndexParams {
    handle: ffi::cuvsIvfFlatIndexParams_t,
}

#[bon]
impl IndexParams {
    #[builder]
    pub fn new(
        metric: Option<DistanceType>,
        n_lists: Option<u32>,
        kmeans_n_iters: Option<u32>,
        kmeans_trainset_fraction: Option<f64>,
        add_data_on_build: Option<bool>,
        adaptive_centers: Option<bool>,
        conservative_memory_allocation: Option<bool>,
    ) -> Result<Self, IvfFlatError> {
        if let Some(n) = n_lists
            && n == 0
        {
            return Err(IvfFlatError::Validation("n_lists must be > 0".into()));
        }

        if let Some(frac) = kmeans_trainset_fraction
            && (!(0.0 < frac && frac <= 1.0))
        {
            return Err(IvfFlatError::Validation(format!(
                "kmeans_trainset_fraction must be in (0, 1], got {frac}"
            )));
        }

        let params = Self::try_new()?;
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
            if let Some(v) = add_data_on_build {
                (*params.handle).add_data_on_build = v;
            }
            if let Some(v) = adaptive_centers {
                (*params.handle).adaptive_centers = v;
            }
            if let Some(v) = conservative_memory_allocation {
                (*params.handle).conservative_memory_allocation = v;
            }
        }

        Ok(params)
    }
}

impl IndexParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, IvfFlatError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsIvfFlatIndexParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> ffi::cuvsIvfFlatIndexParams_t {
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
        let _ = unsafe { ffi::cuvsIvfFlatIndexParamsDestroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// SearchParams
// ---------------------------------------------------------------------------

/// Parameters for searching an IVF-Flat index.
///
/// ```ignore
/// use cuvs::neighbors::ivf_flat::SearchParams;
///
/// let params = SearchParams::builder()
///     .n_probes(20)
///     .build()?;
/// ```
pub struct SearchParams {
    handle: ffi::cuvsIvfFlatSearchParams_t,
}

#[bon]
impl SearchParams {
    #[builder]
    pub fn new(n_probes: Option<u32>) -> Result<Self, IvfFlatError> {
        if let Some(n) = n_probes
            && n == 0
        {
            return Err(IvfFlatError::Validation("n_probes must be > 0".into()));
        }

        let params = Self::try_new()?;
        unsafe {
            if let Some(v) = n_probes {
                (*params.handle).n_probes = v;
            }
        }

        Ok(params)
    }
}

impl SearchParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, IvfFlatError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsIvfFlatSearchParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> ffi::cuvsIvfFlatSearchParams_t {
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
        let _ = unsafe { ffi::cuvsIvfFlatSearchParamsDestroy(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;

    #[test]
    fn index_params_reject_zero_n_lists() {
        let err = IndexParams::builder().n_lists(0).build().unwrap_err();
        assert!(matches!(err, IvfFlatError::Validation(_)));
    }

    #[test]
    fn index_params_try_new_uses_library_defaults() {
        let params = IndexParams::try_new().unwrap();
        unsafe {
            assert_eq!((*params.handle()).metric, ffi::cuvsDistanceType::L2Expanded);
            assert!((*params.handle()).n_lists > 0);
        }
    }

    #[test]
    fn index_params_reject_out_of_range_trainset_fraction() {
        let err = IndexParams::builder()
            .kmeans_trainset_fraction(0.0)
            .build()
            .unwrap_err();
        assert!(matches!(err, IvfFlatError::Validation(_)));

        let err = IndexParams::builder()
            .kmeans_trainset_fraction(1.5)
            .build()
            .unwrap_err();
        assert!(matches!(err, IvfFlatError::Validation(_)));
    }

    #[test]
    fn search_params_reject_zero_n_probes() {
        let err = SearchParams::builder().n_probes(0).build().unwrap_err();
        assert!(matches!(err, IvfFlatError::Validation(_)));
    }

    #[test]
    fn search_params_try_new_uses_library_defaults() {
        let params = SearchParams::try_new().unwrap();
        unsafe {
            assert!((*params.handle()).n_probes > 0);
        }
    }

    #[test]
    fn lp_metric_sets_metric_arg_from_distance_type() {
        let params = IndexParams::builder()
            .metric(DistanceType::LpUnexpanded(OrderedFloat(3.0)))
            .build()
            .unwrap();
        unsafe {
            assert_eq!(
                (*params.handle()).metric,
                ffi::cuvsDistanceType::LpUnexpanded
            );
            assert_eq!((*params.handle()).metric_arg, 3.0);
        }
    }
}
