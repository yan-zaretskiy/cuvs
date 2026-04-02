/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder-pattern parameter types for CAGRA index build, search, and extend.
//!
//! Each parameter type owns the corresponding C params handle directly. The
//! generated `bon` builder configures that handle in the constructor, so there
//! is no duplicate Rust field-bag to keep in sync with the FFI state. All
//! setters are optional; unset values retain the library defaults from the
//! underlying C `*ParamsCreate` functions.

use std::{ffi::c_void, fmt, ptr};

use bon::bon;

use crate::distance::DistanceType;
use crate::error::check_cuvs;
use crate::ffi;

use super::{CagraError, GraphBuildAlgo, HashMode, SearchAlgo};

// ---------------------------------------------------------------------------
// CompressionParams
// ---------------------------------------------------------------------------

/// VPQ (Vector-Product Quantization) compression parameters.
///
/// Attach to [`IndexParams`] to enable compressed dataset storage.
///
/// ```ignore
/// use cuvs::neighbors::cagra::CompressionParams;
///
/// let compression = CompressionParams::builder()
///     .pq_bits(4)
///     .pq_dim(8)
///     .build()?;
/// ```
pub struct CompressionParams {
    handle: ffi::cuvsCagraCompressionParams_t,
}

#[bon]
impl CompressionParams {
    #[builder]
    pub fn new(
        pq_bits: Option<u32>,
        pq_dim: Option<u32>,
        vq_n_centers: Option<u32>,
        kmeans_n_iters: Option<u32>,
        vq_kmeans_trainset_fraction: Option<f64>,
        pq_kmeans_trainset_fraction: Option<f64>,
    ) -> Result<Self, CagraError> {
        if let Some(bits) = pq_bits
            && !(4..=16).contains(&bits) {
                return Err(CagraError::Validation(format!(
                    "pq_bits must be within [4, 16], got {bits}"
                )));
            }

        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraCompressionParamsCreate(&mut handle) })?;

        let params = Self { handle };
        unsafe {
            if let Some(v) = pq_bits {
                (*params.handle).pq_bits = v;
            }
            if let Some(v) = pq_dim {
                (*params.handle).pq_dim = v;
            }
            if let Some(v) = vq_n_centers {
                (*params.handle).vq_n_centers = v;
            }
            if let Some(v) = kmeans_n_iters {
                (*params.handle).kmeans_n_iters = v;
            }
            if let Some(v) = vq_kmeans_trainset_fraction {
                (*params.handle).vq_kmeans_trainset_fraction = v;
            }
            if let Some(v) = pq_kmeans_trainset_fraction {
                (*params.handle).pq_kmeans_trainset_fraction = v;
            }
        }

        Ok(params)
    }
}

impl CompressionParams {
    fn handle(&self) -> ffi::cuvsCagraCompressionParams_t {
        self.handle
    }
}

impl fmt::Debug for CompressionParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CompressionParams")
            .field(unsafe { &*self.handle })
            .finish()
    }
}

impl Drop for CompressionParams {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsCagraCompressionParamsDestroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// IndexParams
// ---------------------------------------------------------------------------

/// Parameters for building a CAGRA index.
///
/// ```ignore
/// use cuvs::neighbors::cagra::{IndexParams, GraphBuildAlgo};
/// use cuvs::distance::DistanceType;
///
/// let params = IndexParams::builder()
///     .metric(DistanceType::InnerProduct)
///     .graph_degree(64)
///     .build_algo(GraphBuildAlgo::NnDescent)
///     .build()?;
/// ```
pub struct IndexParams {
    handle: ffi::cuvsCagraIndexParams_t,
    default_graph_build_params: *mut c_void,
    ace_graph_build_params: Option<ffi::cuvsAceParams_t>,
    compression: Option<CompressionParams>,
}

#[bon]
impl IndexParams {
    #[builder]
    pub fn new(
        metric: Option<DistanceType>,
        intermediate_graph_degree: Option<usize>,
        graph_degree: Option<usize>,
        build_algo: Option<GraphBuildAlgo>,
        nn_descent_niter: Option<usize>,
        compression: Option<CompressionParams>,
    ) -> Result<Self, CagraError> {
        if let Some(d) = graph_degree
            && d == 0 {
                return Err(CagraError::Validation("graph_degree must be > 0".into()));
            }

        if let (Some(inter), Some(graph)) = (intermediate_graph_degree, graph_degree)
            && inter < graph {
                return Err(CagraError::Validation(format!(
                    "intermediate_graph_degree ({inter}) must be >= graph_degree ({graph})"
                )));
            }

        if let Some(n) = nn_descent_niter
            && n == 0 {
                return Err(CagraError::Validation(
                    "nn_descent_niter must be > 0".into(),
                ));
            }

        let metric_supports_compression = metric.is_none_or(|v| v == DistanceType::L2Expanded);
        if compression.is_some() && !metric_supports_compression {
            return Err(CagraError::Validation(
                "VPQ compression is only supported with L2Expanded distance metric".into(),
            ));
        }

        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraIndexParamsCreate(&mut handle) })?;

        let default_graph_build_params = unsafe { (*handle).graph_build_params };
        let mut params = Self {
            handle,
            default_graph_build_params,
            ace_graph_build_params: None,
            compression: None,
        };

        unsafe {
            if let Some(v) = metric {
                (*params.handle).metric = v.into();
            }
            if let Some(v) = intermediate_graph_degree {
                (*params.handle).intermediate_graph_degree = v;
            }
            if let Some(v) = graph_degree {
                (*params.handle).graph_degree = v;
            }
            if let Some(v) = nn_descent_niter {
                (*params.handle).nn_descent_niter = v;
            }
        }

        if let Some(compression) = compression {
            unsafe {
                (*params.handle).compression = compression.handle();
            }
            params.compression = Some(compression);
        }

        params.apply_build_algo(build_algo)?;
        Ok(params)
    }
}

impl IndexParams {
    pub(super) fn handle(&self) -> ffi::cuvsCagraIndexParams_t {
        self.handle
    }

    fn apply_build_algo(&mut self, build_algo: Option<GraphBuildAlgo>) -> Result<(), CagraError> {
        let Some(build_algo) = build_algo else {
            return Ok(());
        };

        match build_algo {
            GraphBuildAlgo::Ace => {
                let mut ace_params = ptr::null_mut();
                check_cuvs(unsafe { ffi::cuvsAceParamsCreate(&mut ace_params) })?;
                unsafe {
                    (*self.handle).build_algo = build_algo.into();
                    (*self.handle).graph_build_params = ace_params.cast::<c_void>();
                }
                if let Some(prev) = self.ace_graph_build_params.replace(ace_params) {
                    let _ = unsafe { ffi::cuvsAceParamsDestroy(prev) };
                }
            }
            GraphBuildAlgo::Auto
            | GraphBuildAlgo::NnDescent
            | GraphBuildAlgo::IterativeCagraSearch => unsafe {
                (*self.handle).build_algo = build_algo.into();
                (*self.handle).graph_build_params = ptr::null_mut();
            },
            GraphBuildAlgo::IvfPq => unsafe {
                (*self.handle).build_algo = build_algo.into();
                (*self.handle).graph_build_params = self.default_graph_build_params;
            },
        }

        Ok(())
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
        // Ownership contract with the C layer:
        // - `cuvsCagraIndexParamsCreate()` installs the default IVF-PQ
        //   `graph_build_params` allocation.
        // - `cuvsCagraIndexParamsDestroy()` frees whatever
        //   `graph_build_params` currently points to based on `build_algo`.
        // - Rust swaps that field between the original C-owned IVF-PQ payload,
        //   a Rust-owned ACE payload, and `null` for algorithms that do not use
        //   graph-build params.
        //
        // Before delegating to the C destructor we therefore restore the
        // original C-owned IVF-PQ pointer whenever Rust has either taken
        // ownership of ACE params or nulled the field out, so the C side frees
        // the correct allocation exactly once.
        unsafe {
            if let Some(ace_params) = self.ace_graph_build_params.take() {
                let _ = ffi::cuvsAceParamsDestroy(ace_params);
                (*self.handle).graph_build_params = self.default_graph_build_params;
                (*self.handle).build_algo = ffi::cuvsCagraGraphBuildAlgo::IVF_PQ;
            } else if (*self.handle).graph_build_params.is_null()
                && !self.default_graph_build_params.is_null()
            {
                (*self.handle).graph_build_params = self.default_graph_build_params;
                (*self.handle).build_algo = ffi::cuvsCagraGraphBuildAlgo::IVF_PQ;
            }

            let _ = ffi::cuvsCagraIndexParamsDestroy(self.handle);
        }
    }
}

// ---------------------------------------------------------------------------
// SearchParams
// ---------------------------------------------------------------------------

/// Parameters for searching a CAGRA index.
///
/// ```ignore
/// use cuvs::neighbors::cagra::SearchParams;
///
/// let params = SearchParams::builder()
///     .itopk_size(128)
///     .build()?;
/// ```
pub struct SearchParams {
    handle: ffi::cuvsCagraSearchParams_t,
}

#[bon]
impl SearchParams {
    #[builder]
    pub fn new(
        max_queries: Option<usize>,
        itopk_size: Option<usize>,
        max_iterations: Option<usize>,
        algo: Option<SearchAlgo>,
        team_size: Option<usize>,
        search_width: Option<usize>,
        min_iterations: Option<usize>,
        thread_block_size: Option<usize>,
        hashmap_mode: Option<HashMode>,
        hashmap_min_bitlen: Option<usize>,
        hashmap_max_fill_rate: Option<f32>,
        num_random_samplings: Option<u32>,
        rand_xor_mask: Option<u64>,
        persistent: Option<bool>,
        persistent_lifetime: Option<f32>,
        persistent_device_usage: Option<f32>,
    ) -> Result<Self, CagraError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraSearchParamsCreate(&mut handle) })?;
        let params = Self { handle };

        let effective_algo = algo.unwrap_or(unsafe { (*params.handle).algo.into() });
        let effective_hashmap_mode =
            hashmap_mode.unwrap_or(unsafe { (*params.handle).hashmap_mode.into() });

        if let Some(n) = itopk_size
            && effective_algo == SearchAlgo::SingleCta && n > 512 {
                return Err(CagraError::Validation(format!(
                    "itopk_size cannot be larger than 512 for SingleCta, got {n}"
                )));
            }

        if let Some(n) = team_size
            && !matches!(n, 0 | 8 | 16 | 32) {
                return Err(CagraError::Validation(format!(
                    "team_size must be 0 (auto), 8, 16, or 32, got {n}"
                )));
            }

        if let Some(n) = thread_block_size
            && !matches!(n, 0 | 64 | 128 | 256 | 512 | 1024) {
                return Err(CagraError::Validation(format!(
                    "thread_block_size must be 0, 64, 128, 256, 512, or 1024, got {n}"
                )));
            }

        if let Some(bitlen) = hashmap_min_bitlen
            && bitlen > 20 {
                return Err(CagraError::Validation(format!(
                    "hashmap_min_bitlen must be <= 20, got {bitlen}"
                )));
            }

        if let Some(rate) = hashmap_max_fill_rate
            && (!(0.1..0.9).contains(&rate)) {
                return Err(CagraError::Validation(format!(
                    "hashmap_max_fill_rate must be in [0.1, 0.9), got {rate}"
                )));
            }

        if effective_algo == SearchAlgo::MultiCta && effective_hashmap_mode == HashMode::Small {
            return Err(CagraError::Validation(
                "`small_hash` is not available when 'search_mode' is \"multi-cta\"".into(),
            ));
        }
        unsafe {
            if let Some(v) = max_queries {
                (*params.handle).max_queries = v;
            }
            if let Some(v) = itopk_size {
                (*params.handle).itopk_size = v;
            }
            if let Some(v) = max_iterations {
                (*params.handle).max_iterations = v;
            }
            if let Some(v) = algo {
                (*params.handle).algo = v.into();
            }
            if let Some(v) = team_size {
                (*params.handle).team_size = v;
            }
            if let Some(v) = search_width {
                (*params.handle).search_width = v;
            }
            if let Some(v) = min_iterations {
                (*params.handle).min_iterations = v;
            }
            if let Some(v) = thread_block_size {
                (*params.handle).thread_block_size = v;
            }
            if let Some(v) = hashmap_mode {
                (*params.handle).hashmap_mode = v.into();
            }
            if let Some(v) = hashmap_min_bitlen {
                (*params.handle).hashmap_min_bitlen = v;
            }
            if let Some(v) = hashmap_max_fill_rate {
                (*params.handle).hashmap_max_fill_rate = v;
            }
            if let Some(v) = num_random_samplings {
                (*params.handle).num_random_samplings = v;
            }
            if let Some(v) = rand_xor_mask {
                (*params.handle).rand_xor_mask = v;
            }
            if let Some(v) = persistent {
                (*params.handle).persistent = v;
            }
            if let Some(v) = persistent_lifetime {
                (*params.handle).persistent_lifetime = v;
            }
            if let Some(v) = persistent_device_usage {
                (*params.handle).persistent_device_usage = v;
            }
        }

        Ok(params)
    }
}

impl SearchParams {
    pub(super) fn handle(&self) -> ffi::cuvsCagraSearchParams_t {
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
        let _ = unsafe { ffi::cuvsCagraSearchParamsDestroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// ExtendParams
// ---------------------------------------------------------------------------

/// Parameters for extending a CAGRA index with additional vectors.
///
/// ```ignore
/// use cuvs::neighbors::cagra::ExtendParams;
///
/// let params = ExtendParams::builder()
///     .max_chunk_size(1024)
///     .build()?;
/// ```
pub struct ExtendParams {
    handle: ffi::cuvsCagraExtendParams_t,
}

#[bon]
impl ExtendParams {
    #[builder]
    pub fn new(max_chunk_size: Option<u32>) -> Result<Self, CagraError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraExtendParamsCreate(&mut handle) })?;

        let params = Self { handle };
        unsafe {
            if let Some(v) = max_chunk_size {
                (*params.handle).max_chunk_size = v;
            }
        }

        Ok(params)
    }
}

impl ExtendParams {
    pub(super) fn handle(&self) -> ffi::cuvsCagraExtendParams_t {
        self.handle
    }
}

impl fmt::Debug for ExtendParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ExtendParams")
            .field(unsafe { &*self.handle })
            .finish()
    }
}

impl Drop for ExtendParams {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsCagraExtendParamsDestroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi;

    // -- IndexParams -------------------------------------------------------

    #[test]
    fn index_params_all_defaults() {
        let params = IndexParams::builder().build().unwrap();
        unsafe {
            assert_eq!((*params.handle).metric, ffi::cuvsDistanceType::L2Expanded);
            assert_eq!((*params.handle).graph_degree, 64);
        }
    }

    #[test]
    fn index_params_with_values() {
        let params = IndexParams::builder()
            .metric(DistanceType::InnerProduct)
            .graph_degree(64)
            .intermediate_graph_degree(128)
            .build_algo(GraphBuildAlgo::NnDescent)
            .nn_descent_niter(10)
            .build()
            .unwrap();

        unsafe {
            assert_eq!((*params.handle).metric, ffi::cuvsDistanceType::InnerProduct);
            assert_eq!((*params.handle).graph_degree, 64);
            assert_eq!((*params.handle).intermediate_graph_degree, 128);
            assert_eq!(
                (*params.handle).build_algo,
                ffi::cuvsCagraGraphBuildAlgo::NN_DESCENT
            );
            assert_eq!((*params.handle).nn_descent_niter, 10);
        }
    }

    #[test]
    fn index_params_rejects_zero_graph_degree() {
        let err = IndexParams::builder().graph_degree(0).build().unwrap_err();
        assert!(err.to_string().contains("graph_degree must be > 0"));
    }

    #[test]
    fn index_params_rejects_invalid_intermediate_degree() {
        let err = IndexParams::builder()
            .graph_degree(64)
            .intermediate_graph_degree(32)
            .build()
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("intermediate_graph_degree (32) must be >= graph_degree (64)")
        );
    }

    #[test]
    fn index_params_rejects_zero_niter() {
        let err = IndexParams::builder()
            .nn_descent_niter(0)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("nn_descent_niter must be > 0"));
    }

    #[test]
    fn index_params_switching_to_nn_descent_clears_default_graph_build_params() {
        let params = IndexParams::builder()
            .build_algo(GraphBuildAlgo::NnDescent)
            .build()
            .unwrap();

        unsafe {
            assert_eq!(
                (*params.handle).build_algo,
                ffi::cuvsCagraGraphBuildAlgo::NN_DESCENT
            );
            assert!((*params.handle).graph_build_params.is_null());
        }
    }

    #[test]
    fn index_params_rejects_non_l2_metric_with_compression() {
        let compression = CompressionParams::builder().pq_bits(8).build().unwrap();
        let err = IndexParams::builder()
            .metric(DistanceType::InnerProduct)
            .compression(compression)
            .build()
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("VPQ compression is only supported with L2Expanded")
        );
    }

    #[test]
    fn index_params_with_compression() {
        let params = IndexParams::builder()
            .compression(
                CompressionParams::builder()
                    .pq_bits(4)
                    .pq_dim(8)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();

        unsafe {
            let c = (*params.handle).compression;
            assert!(!c.is_null());
            assert_eq!((*c).pq_bits, 4);
            assert_eq!((*c).pq_dim, 8);
        }
    }

    // -- CompressionParams -------------------------------------------------

    #[test]
    fn compression_params_rejects_pq_bits_below_cpp_range() {
        let err = CompressionParams::builder().pq_bits(3).build().unwrap_err();
        assert!(err.to_string().contains("pq_bits"));
    }

    #[test]
    fn compression_params_accepts_pq_bits_at_cpp_upper_bound() {
        CompressionParams::builder().pq_bits(16).build().unwrap();
    }

    // -- SearchParams ------------------------------------------------------

    #[test]
    fn search_params_all_defaults() {
        let params = SearchParams::builder().build().unwrap();
        unsafe {
            assert_eq!((*params.handle).itopk_size, 64);
            assert_eq!((*params.handle).algo, ffi::cuvsCagraSearchAlgo::SINGLE_CTA);
            assert_eq!((*params.handle).hashmap_mode, ffi::cuvsCagraHashMode::HASH);
        }
    }

    #[test]
    fn search_params_accepts_non_power_of_two_itopk() {
        SearchParams::builder().itopk_size(100).build().unwrap();
    }

    #[test]
    fn search_params_accepts_zero_itopk() {
        SearchParams::builder().itopk_size(0).build().unwrap();
    }

    #[test]
    fn search_params_rejects_invalid_team_size_not_in_compiled_descriptor_table() {
        let err = SearchParams::builder().team_size(4).build().unwrap_err();
        assert!(err.to_string().contains("team_size must be"));
    }

    #[test]
    fn search_params_accepts_compiled_team_sizes() {
        for team_size in [8, 16, 32] {
            let params = SearchParams::builder().team_size(team_size).build().unwrap();
            unsafe {
                assert_eq!((*params.handle).team_size, team_size);
            }
        }
    }

    #[test]
    fn search_params_rejects_fill_rate_out_of_range() {
        let err = SearchParams::builder()
            .hashmap_max_fill_rate(0.95)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("hashmap_max_fill_rate"));
    }

    #[test]
    fn search_params_accepts_fill_rate_at_cpp_lower_bound() {
        SearchParams::builder()
            .hashmap_max_fill_rate(0.1)
            .build()
            .unwrap();
    }

    #[test]
    fn search_params_rejects_invalid_thread_block_size() {
        let err = SearchParams::builder()
            .thread_block_size(33)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("thread_block_size"));
    }

    #[test]
    fn search_params_accepts_supported_thread_block_sizes() {
        for thread_block_size in [64, 128, 256, 512, 1024] {
            let params = SearchParams::builder()
                .thread_block_size(thread_block_size)
                .build()
                .unwrap();
            unsafe {
                assert_eq!((*params.handle).thread_block_size, thread_block_size);
            }
        }
    }

    #[test]
    fn search_params_rejects_hashmap_min_bitlen_above_cpp_limit() {
        let err = SearchParams::builder()
            .hashmap_min_bitlen(21)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("hashmap_min_bitlen"));
    }

    #[test]
    fn search_params_accepts_hashmap_min_bitlen_at_cpp_upper_limit() {
        let params = SearchParams::builder()
            .hashmap_min_bitlen(20)
            .build()
            .unwrap();
        unsafe {
            assert_eq!((*params.handle).hashmap_min_bitlen, 20);
        }
    }

    #[test]
    fn search_params_rejects_small_hash_with_multi_cta() {
        let err = SearchParams::builder()
            .algo(SearchAlgo::MultiCta)
            .hashmap_mode(HashMode::Small)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("small_hash"));
    }

    #[test]
    fn search_params_accepts_small_hash_with_single_cta() {
        let params = SearchParams::builder()
            .algo(SearchAlgo::SingleCta)
            .hashmap_mode(HashMode::Small)
            .build()
            .unwrap();
        unsafe {
            assert_eq!((*params.handle).hashmap_mode, ffi::cuvsCagraHashMode::SMALL);
        }
    }

    #[test]
    fn search_params_rejects_single_cta_itopk_above_cpp_limit() {
        let err = SearchParams::builder()
            .algo(SearchAlgo::SingleCta)
            .itopk_size(513)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn search_params_accepts_multi_cta_itopk_at_cpp_upper_limit() {
        let params = SearchParams::builder()
            .algo(SearchAlgo::MultiCta)
            .itopk_size(1024)
            .build()
            .unwrap();
        unsafe {
            assert_eq!((*params.handle).algo, ffi::cuvsCagraSearchAlgo::MULTI_CTA);
            assert_eq!((*params.handle).itopk_size, 1024);
        }
    }

    // -- ExtendParams ------------------------------------------------------

    #[test]
    fn extend_params_builder() {
        let params = ExtendParams::builder()
            .max_chunk_size(1024)
            .build()
            .unwrap();

        unsafe {
            assert_eq!((*params.handle).max_chunk_size, 1024);
        }
    }
}
