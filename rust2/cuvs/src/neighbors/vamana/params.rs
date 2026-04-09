/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder-pattern parameter type for Vamana index build.
//!
//! The parameter type owns the corresponding C params handle directly. The
//! generated `bon` builder configures that handle in the constructor, so there
//! is no duplicate Rust field-bag to keep in sync with the FFI state. All
//! setters are optional; unset values retain the library defaults from the
//! underlying C `*ParamsCreate` function.

use std::{fmt, ptr};

use bon::bon;

use crate::distance::DistanceType;
use crate::error::check_cuvs;
use crate::ffi;

use super::VamanaError;

const SUPPORTED_GRAPH_DEGREES: [u32; 4] = [32, 64, 128, 256];

/// Parameters for building a Vamana index.
///
/// ```ignore
/// use cuvs::distance::DistanceType;
/// use cuvs::neighbors::vamana::IndexParams;
///
/// let params = IndexParams::builder()
///     .metric(DistanceType::L2Expanded)
///     .graph_degree(32)
///     .visited_size(64)
///     .build()?;
/// ```
///
/// Note: the underlying Vamana implementation grows non-power-of-two
/// `visited_size` values by repeated doubling from the compiled
/// `graph_degree` until the effective width reaches or exceeds the requested
/// value, so the actual search breadth may be larger than the configured
/// input.
pub struct IndexParams {
    handle: ffi::cuvsVamanaIndexParams_t,
}

#[bon]
impl IndexParams {
    #[builder]
    pub fn new(
        metric: Option<DistanceType>,
        graph_degree: Option<u32>,
        visited_size: Option<u32>,
        vamana_iters: Option<f32>,
        alpha: Option<f32>,
        max_fraction: Option<f32>,
        batch_base: Option<f32>,
        queue_size: Option<u32>,
        reverse_batchsize: Option<u32>,
    ) -> Result<Self, VamanaError> {
        let params = Self::try_new()?;
        let effective_metric = metric.unwrap_or_else(|| unsafe { (*params.handle).metric.into() });
        let effective_graph_degree =
            graph_degree.unwrap_or_else(|| unsafe { (*params.handle).graph_degree });
        let effective_visited_size =
            visited_size.unwrap_or_else(|| unsafe { (*params.handle).visited_size });
        let effective_vamana_iters =
            vamana_iters.unwrap_or_else(|| unsafe { (*params.handle).vamana_iters });

        if effective_metric != DistanceType::L2Expanded {
            return Err(VamanaError::Validation(
                "Vamana currently only supports L2Expanded metric".into(),
            ));
        }

        if !SUPPORTED_GRAPH_DEGREES.contains(&effective_graph_degree) {
            return Err(VamanaError::Validation(format!(
                "graph_degree must be one of {:?}, got {effective_graph_degree}",
                SUPPORTED_GRAPH_DEGREES
            )));
        }

        if effective_visited_size <= effective_graph_degree {
            return Err(VamanaError::Validation(format!(
                "visited_size must be > graph_degree ({effective_graph_degree}), got {effective_visited_size}"
            )));
        }

        if !effective_vamana_iters.is_finite() {
            return Err(VamanaError::Validation(format!(
                "vamana_iters must be finite, got {effective_vamana_iters}"
            )));
        }

        if effective_vamana_iters < 1.0 {
            return Err(VamanaError::Validation(format!(
                "vamana_iters must be >= 1.0, got {effective_vamana_iters}"
            )));
        }

        unsafe {
            if let Some(v) = metric {
                (*params.handle).metric = v.into();
            }
            if let Some(v) = graph_degree {
                (*params.handle).graph_degree = v;
            }
            if let Some(v) = visited_size {
                (*params.handle).visited_size = v;
            }
            if let Some(v) = vamana_iters {
                (*params.handle).vamana_iters = v;
            }
            if let Some(v) = alpha {
                (*params.handle).alpha = v;
            }
            if let Some(v) = max_fraction {
                (*params.handle).max_fraction = v;
            }
            if let Some(v) = batch_base {
                (*params.handle).batch_base = v;
            }
            if let Some(v) = queue_size {
                (*params.handle).queue_size = v;
            }
            if let Some(v) = reverse_batchsize {
                (*params.handle).reverse_batchsize = v;
            }
        }

        Ok(params)
    }
}

impl IndexParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, VamanaError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsVamanaIndexParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> ffi::cuvsVamanaIndexParams_t {
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
        let _ = unsafe { ffi::cuvsVamanaIndexParamsDestroy(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_l2_metric() {
        let err = IndexParams::builder()
            .metric(DistanceType::InnerProduct)
            .build()
            .unwrap_err();
        assert!(matches!(err, VamanaError::Validation(_)));
    }

    #[test]
    fn try_new_uses_library_defaults() {
        let params = IndexParams::try_new().unwrap();
        unsafe {
            assert_eq!((*params.handle()).metric, ffi::cuvsDistanceType::L2Expanded);
            assert!(SUPPORTED_GRAPH_DEGREES.contains(&(*params.handle()).graph_degree));
        }
    }

    #[test]
    fn rejects_unsupported_graph_degree() {
        let err = IndexParams::builder().graph_degree(16).build().unwrap_err();
        assert!(matches!(err, VamanaError::Validation(_)));
    }

    #[test]
    fn rejects_visited_size_not_exceeding_graph_degree() {
        let err = IndexParams::builder()
            .graph_degree(32)
            .visited_size(32)
            .build()
            .unwrap_err();
        assert!(matches!(err, VamanaError::Validation(_)));
    }

    #[test]
    fn rejects_vamana_iters_below_one() {
        let err = IndexParams::builder()
            .vamana_iters(0.5)
            .build()
            .unwrap_err();
        assert!(matches!(err, VamanaError::Validation(_)));
    }

    #[test]
    fn rejects_non_finite_vamana_iters() {
        let err = IndexParams::builder()
            .vamana_iters(f32::NAN)
            .build()
            .unwrap_err();
        assert!(matches!(err, VamanaError::Validation(_)));

        let err = IndexParams::builder()
            .vamana_iters(f32::INFINITY)
            .build()
            .unwrap_err();
        assert!(matches!(err, VamanaError::Validation(_)));
    }
}
