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

use std::ffi::{CString, c_void};
use std::path::Path;
use std::{fmt, ptr};

use bon::bon;

use crate::distance::DistanceType;
use crate::error::check_cuvs;
use crate::ffi;

use super::{CagraError, GraphBuildAlgo, HashMode, HnswHeuristicType, SearchAlgo};

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
            && !(4..=16).contains(&bits)
        {
            return Err(CagraError::Validation(format!(
                "pq_bits must be within [4, 16], got {bits}"
            )));
        }

        let params = Self::try_new()?;
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
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, CagraError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraCompressionParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

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
// AceParams
// ---------------------------------------------------------------------------

/// Parameters for ACE (Augmented Core Extraction) graph build.
///
/// ACE enables building indices for datasets too large to fit in GPU memory
/// by partitioning the data, building sub-indices, and merging the graphs.
///
/// ```ignore
/// use cuvs::neighbors::cagra::AceParams;
///
/// let ace = AceParams::builder()
///     .npartitions(4)
///     .use_disk(true)
///     .build_dir("/tmp/ace")
///     .build()?;
/// ```
pub struct AceParams {
    handle: ffi::cuvsAceParams_t,
}

#[bon]
impl AceParams {
    #[builder]
    pub fn new(
        npartitions: Option<usize>,
        ef_construction: Option<usize>,
        build_dir: Option<&Path>,
        use_disk: Option<bool>,
        max_host_memory_gb: Option<f64>,
        max_gpu_memory_gb: Option<f64>,
    ) -> Result<Self, CagraError> {
        let params = Self::try_new()?;

        unsafe {
            if let Some(v) = npartitions {
                (*params.handle).npartitions = v;
            }
            if let Some(v) = ef_construction {
                (*params.handle).ef_construction = v;
            }
            if let Some(v) = use_disk {
                (*params.handle).use_disk = v;
            }
            if let Some(v) = max_host_memory_gb {
                (*params.handle).max_host_memory_gb = v;
            }
            if let Some(v) = max_gpu_memory_gb {
                (*params.handle).max_gpu_memory_gb = v;
            }
        }

        if let Some(dir) = build_dir {
            let c_str = CString::new(dir.as_os_str().as_encoded_bytes())?;
            check_cuvs(unsafe { ffi::cuvsAceParamsSetBuildDir(params.handle, c_str.as_ptr()) })?;
        }

        Ok(params)
    }
}

impl AceParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, CagraError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsAceParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    fn handle(&self) -> ffi::cuvsAceParams_t {
        self.handle
    }
}

impl fmt::Debug for AceParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AceParams")
            .field(unsafe { &*self.handle })
            .finish()
    }
}

impl Drop for AceParams {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsAceParamsDestroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// IvfPqGraphBuildParams
// ---------------------------------------------------------------------------

/// IVF-PQ parameters used for CAGRA graph construction.
///
/// This wraps the composite `cuvsIvfPqParams` struct that bundles IVF-PQ
/// build params, search params, and a refinement rate.
///
/// ```ignore
/// use cuvs::neighbors::cagra::IvfPqGraphBuildParams;
///
/// let ivf_pq = IvfPqGraphBuildParams::builder()
///     .n_lists(100)
///     .pq_dim(16)
///     .refinement_rate(2.0)
///     .build()?;
/// ```
// Box-allocated because the C API has no `cuvsIvfPqParamsCreate` — only the
// nested build/search params have C allocators.  We need a stable heap
// address so `graph_build_params` can point into it across moves of the
// owning `IndexParams`.
pub struct IvfPqGraphBuildParams {
    inner: Box<ffi::cuvsIvfPqParams>,
}

#[bon]
impl IvfPqGraphBuildParams {
    #[builder]
    pub fn new(
        n_lists: Option<u32>,
        kmeans_n_iters: Option<u32>,
        kmeans_trainset_fraction: Option<f64>,
        pq_bits: Option<u32>,
        pq_dim: Option<u32>,
        n_probes: Option<u32>,
        refinement_rate: Option<f32>,
    ) -> Result<Self, CagraError> {
        let mut params = Self::try_new()?;
        params.inner.refinement_rate = refinement_rate.unwrap_or(params.inner.refinement_rate);
        let build_handle = params.inner.ivf_pq_build_params;
        let search_handle = params.inner.ivf_pq_search_params;

        unsafe {
            if let Some(v) = n_lists {
                (*build_handle).n_lists = v;
            }
            if let Some(v) = kmeans_n_iters {
                (*build_handle).kmeans_n_iters = v;
            }
            if let Some(v) = kmeans_trainset_fraction {
                (*build_handle).kmeans_trainset_fraction = v;
            }
            if let Some(v) = pq_bits {
                (*build_handle).pq_bits = v;
            }
            if let Some(v) = pq_dim {
                (*build_handle).pq_dim = v;
            }
            if let Some(v) = n_probes {
                (*search_handle).n_probes = v;
            }
        }

        Ok(params)
    }
}

impl IvfPqGraphBuildParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, CagraError> {
        let mut build_handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsIvfPqIndexParamsCreate(&mut build_handle) })?;

        let mut params = Self {
            inner: Box::new(ffi::cuvsIvfPqParams {
                ivf_pq_build_params: build_handle,
                ivf_pq_search_params: ptr::null_mut(),
                refinement_rate: 2.0,
            }),
        };

        let mut search_handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsIvfPqSearchParamsCreate(&mut search_handle) })?;
        params.inner.ivf_pq_search_params = search_handle;

        Ok(params)
    }

    fn as_mut_ptr(&mut self) -> *mut ffi::cuvsIvfPqParams {
        &mut *self.inner as *mut _
    }
}

impl fmt::Debug for IvfPqGraphBuildParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IvfPqGraphBuildParams")
            .field("refinement_rate", &self.inner.refinement_rate)
            .finish()
    }
}

impl Drop for IvfPqGraphBuildParams {
    fn drop(&mut self) {
        let bp = self.inner.ivf_pq_build_params;
        let sp = self.inner.ivf_pq_search_params;
        if !bp.is_null() {
            let _ = unsafe { ffi::cuvsIvfPqIndexParamsDestroy(bp) };
        }
        if !sp.is_null() {
            let _ = unsafe { ffi::cuvsIvfPqSearchParamsDestroy(sp) };
        }
    }
}

// ---------------------------------------------------------------------------
// RequestedGraphBuild / GraphBuildOwner
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum RequestedGraphBuild {
    Auto,
    NnDescent { nn_descent_niter: Option<usize> },
    IterativeCagraSearch,
    AceDefault,
    Ace(AceParams),
    IvfPqDefault,
    IvfPq(IvfPqGraphBuildParams),
}

enum GraphBuildOwner {
    /// Rust-owned ACE params.
    Ace(AceParams),
    /// Rust-owned custom IVF-PQ params.
    IvfPq(IvfPqGraphBuildParams),
}

impl GraphBuildOwner {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        match self {
            Self::Ace(ace) => ace.handle().cast::<c_void>(),
            Self::IvfPq(ivf_pq) => ivf_pq.as_mut_ptr().cast::<c_void>(),
        }
    }
}

// ---------------------------------------------------------------------------
// IndexParams
// ---------------------------------------------------------------------------

/// Parameters for building a CAGRA index.
///
/// ```ignore
/// use cuvs::neighbors::cagra::IndexParams;
/// use cuvs::distance::DistanceType;
///
/// let params = IndexParams::builder()
///     .metric(DistanceType::InnerProduct)
///     .graph_degree(64)
///     .nn_descent_with(20)
///     .build()?;
/// ```
///
/// Graph-build strategy setters should be mutually exclusive.
///
/// ```compile_fail
/// use cuvs::neighbors::cagra::{AceParams, IndexParams, IvfPqGraphBuildParams};
///
/// let ace = AceParams::builder().build().unwrap();
/// let ivf_pq = IvfPqGraphBuildParams::builder().build().unwrap();
///
/// let _params = IndexParams::builder()
///     .ace_with(ace)
///     .ivf_pq_with(ivf_pq);
/// ```
pub struct IndexParams {
    handle: ffi::cuvsCagraIndexParams_t,
    default_graph_build_params: *mut c_void,
    graph_build_owner: Option<GraphBuildOwner>,
    compression: Option<CompressionParams>,
}

#[bon]
impl IndexParams {
    #[builder]
    pub fn new(
        metric: Option<DistanceType>,
        intermediate_graph_degree: Option<usize>,
        graph_degree: Option<usize>,
        compression: Option<CompressionParams>,
        #[builder(setters(vis = "", some_fn = graph_build_internal))] graph_build: Option<
            RequestedGraphBuild,
        >,
    ) -> Result<Self, CagraError> {
        if let Some(d) = graph_degree
            && d == 0
        {
            return Err(CagraError::Validation("graph_degree must be > 0".into()));
        }

        if let (Some(inter), Some(graph)) = (intermediate_graph_degree, graph_degree)
            && inter < graph
        {
            return Err(CagraError::Validation(format!(
                "intermediate_graph_degree ({inter}) must be >= graph_degree ({graph})"
            )));
        }

        if let Some(RequestedGraphBuild::NnDescent {
            nn_descent_niter: Some(n),
        }) = &graph_build
            && *n == 0
        {
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

        let mut params = Self::try_new()?;

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
        }

        if let Some(compression) = compression {
            unsafe { (*params.handle).compression = compression.handle() };
            params.compression = Some(compression);
        }

        params.apply_graph_build(graph_build)?;
        Ok(params)
    }
}

use index_params_builder::{IsUnset, SetGraphBuild, State};

impl<S: State> IndexParamsBuilder<S> {
    /// Use the library's automatic graph-build selection.
    pub fn auto(self) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::Auto)
    }

    /// Build the graph with NN-Descent using the library defaults.
    pub fn nn_descent(self) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::NnDescent {
            nn_descent_niter: None,
        })
    }

    /// Build the graph with NN-Descent and an explicit iteration count.
    pub fn nn_descent_with(self, iterations: usize) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::NnDescent {
            nn_descent_niter: Some(iterations),
        })
    }

    /// Build the graph using iterative CAGRA search.
    pub fn iterative_cagra_search(self) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::IterativeCagraSearch)
    }

    /// Build the graph with ACE using the library defaults.
    pub fn ace(self) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::AceDefault)
    }

    /// Build the graph with explicit ACE parameters.
    pub fn ace_with(self, params: AceParams) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::Ace(params))
    }

    /// Build the graph with IVF-PQ using the library defaults.
    pub fn ivf_pq(self) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::IvfPqDefault)
    }

    /// Build the graph with explicit IVF-PQ graph-build parameters.
    pub fn ivf_pq_with(self, params: IvfPqGraphBuildParams) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::IvfPq(params))
    }
}

impl IndexParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, CagraError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraIndexParamsCreate(&mut handle) })?;
        let default_graph_build_params = unsafe { (*handle).graph_build_params };
        Ok(Self {
            handle,
            default_graph_build_params,
            graph_build_owner: None,
            compression: None,
        })
    }

    pub(super) fn handle(&self) -> ffi::cuvsCagraIndexParams_t {
        self.handle
    }

    /// Populate `graph_build_params` from HNSW-compatible parameters.
    ///
    /// The C factory sets `build_algo`, `graph_degree`,
    /// `intermediate_graph_degree`, and `graph_build_params` on the handle.
    pub fn from_hnsw_params(
        n_rows: i64,
        dim: i64,
        m: i32,
        ef_construction: i32,
        heuristic: HnswHeuristicType,
        metric: DistanceType,
    ) -> Result<Self, CagraError> {
        if n_rows <= 0 {
            return Err(CagraError::Validation("n_rows must be > 0".into()));
        }
        if dim <= 0 {
            return Err(CagraError::Validation("dim must be > 0".into()));
        }
        if m <= 0 {
            return Err(CagraError::Validation("m must be > 0".into()));
        }
        if ef_construction <= 0 {
            return Err(CagraError::Validation("ef_construction must be > 0".into()));
        }

        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraIndexParamsCreate(&mut handle) })?;
        let default_graph_build_params = unsafe { (*handle).graph_build_params };

        // Wrap early so Drop cleans up if the factory call fails.
        let params = Self {
            handle,
            default_graph_build_params,
            graph_build_owner: None,
            compression: None,
        };

        check_cuvs(unsafe {
            ffi::cuvsCagraIndexParamsFromHnswParams(
                params.handle,
                n_rows,
                dim,
                m,
                ef_construction,
                heuristic.into(),
                metric.into(),
            )
        })?;

        Ok(params)
    }

    fn apply_graph_build(
        &mut self,
        graph_build: Option<RequestedGraphBuild>,
    ) -> Result<(), CagraError> {
        let Some(graph_build) = graph_build else {
            return Ok(());
        };

        match graph_build {
            RequestedGraphBuild::Auto => {
                unsafe {
                    (*self.handle).build_algo = GraphBuildAlgo::Auto.into();
                    (*self.handle).graph_build_params = ptr::null_mut();
                }
                self.graph_build_owner = None;
            }
            RequestedGraphBuild::NnDescent { nn_descent_niter } => {
                unsafe {
                    (*self.handle).build_algo = GraphBuildAlgo::NnDescent.into();
                    (*self.handle).graph_build_params = ptr::null_mut();
                    if let Some(v) = nn_descent_niter {
                        (*self.handle).nn_descent_niter = v;
                    }
                }
                self.graph_build_owner = None;
            }
            RequestedGraphBuild::IterativeCagraSearch => {
                unsafe {
                    (*self.handle).build_algo = GraphBuildAlgo::IterativeCagraSearch.into();
                    (*self.handle).graph_build_params = ptr::null_mut();
                }
                self.graph_build_owner = None;
            }
            RequestedGraphBuild::AceDefault => {
                let ace = AceParams::try_new()?;
                self.graph_build_owner = Some(GraphBuildOwner::Ace(ace));
                unsafe {
                    (*self.handle).build_algo = GraphBuildAlgo::Ace.into();
                    (*self.handle).graph_build_params =
                        self.graph_build_owner.as_mut().unwrap().as_mut_ptr();
                }
            }
            RequestedGraphBuild::Ace(ace) => {
                self.graph_build_owner = Some(GraphBuildOwner::Ace(ace));
                unsafe {
                    (*self.handle).build_algo = GraphBuildAlgo::Ace.into();
                    (*self.handle).graph_build_params =
                        self.graph_build_owner.as_mut().unwrap().as_mut_ptr();
                }
            }
            RequestedGraphBuild::IvfPqDefault => {
                unsafe {
                    (*self.handle).build_algo = GraphBuildAlgo::IvfPq.into();
                    (*self.handle).graph_build_params = self.default_graph_build_params;
                }
                self.graph_build_owner = None;
            }
            RequestedGraphBuild::IvfPq(ivf_pq) => {
                self.graph_build_owner = Some(GraphBuildOwner::IvfPq(ivf_pq));
                unsafe {
                    (*self.handle).build_algo = GraphBuildAlgo::IvfPq.into();
                    (*self.handle).graph_build_params =
                        self.graph_build_owner.as_mut().unwrap().as_mut_ptr();
                }
            }
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
        // Restore the C-owned default IVF-PQ graph_build_params so the C
        // destructor frees its own allocation exactly once. Rust-owned graph
        // build params are dropped immediately afterwards.
        unsafe {
            (*self.handle).graph_build_params = self.default_graph_build_params;
            (*self.handle).build_algo = ffi::cuvsCagraGraphBuildAlgo::IVF_PQ;
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
        let params = Self::try_new()?;

        let effective_algo = algo.unwrap_or(unsafe { (*params.handle).algo.into() });
        let effective_hashmap_mode =
            hashmap_mode.unwrap_or(unsafe { (*params.handle).hashmap_mode.into() });

        if let Some(n) = itopk_size
            && effective_algo == SearchAlgo::SingleCta
            && n > 512
        {
            return Err(CagraError::Validation(format!(
                "itopk_size cannot be larger than 512 for SingleCta, got {n}"
            )));
        }

        if let Some(n) = team_size
            && !matches!(n, 0 | 8 | 16 | 32)
        {
            return Err(CagraError::Validation(format!(
                "team_size must be 0 (auto), 8, 16, or 32, got {n}"
            )));
        }

        if let Some(n) = thread_block_size
            && !matches!(n, 0 | 64 | 128 | 256 | 512 | 1024)
        {
            return Err(CagraError::Validation(format!(
                "thread_block_size must be 0, 64, 128, 256, 512, or 1024, got {n}"
            )));
        }

        if let Some(bitlen) = hashmap_min_bitlen
            && bitlen > 20
        {
            return Err(CagraError::Validation(format!(
                "hashmap_min_bitlen must be <= 20, got {bitlen}"
            )));
        }

        if let Some(rate) = hashmap_max_fill_rate
            && (!(0.1..0.9).contains(&rate))
        {
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
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, CagraError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraSearchParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

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
        let params = Self::try_new()?;
        unsafe {
            if let Some(v) = max_chunk_size {
                (*params.handle).max_chunk_size = v;
            }
        }

        Ok(params)
    }
}

impl ExtendParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, CagraError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraExtendParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

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
    use std::ffi::{CStr, CString};
    use std::path::Path;

    use super::*;
    use crate::ffi;

    // -- IndexParams -------------------------------------------------------

    #[test]
    fn index_params_all_defaults() {
        let params = IndexParams::try_new().unwrap();
        unsafe {
            assert_eq!((*params.handle).metric, ffi::cuvsDistanceType::L2Expanded);
            assert_eq!((*params.handle).graph_degree, 64);
        }
        assert!(params.graph_build_owner.is_none());
    }

    #[test]
    fn index_params_with_values() {
        let params = IndexParams::builder()
            .metric(DistanceType::InnerProduct)
            .graph_degree(64)
            .intermediate_graph_degree(128)
            .nn_descent_with(10)
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
            .nn_descent_with(0)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("nn_descent_niter must be > 0"));
    }

    #[test]
    fn index_params_switching_to_nn_descent_clears_default_graph_build_params() {
        let params = IndexParams::builder().nn_descent().build().unwrap();

        unsafe {
            assert_eq!(
                (*params.handle).build_algo,
                ffi::cuvsCagraGraphBuildAlgo::NN_DESCENT
            );
            assert!((*params.handle).graph_build_params.is_null());
        }
        assert!(params.graph_build_owner.is_none());
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

    #[test]
    fn index_params_with_ace_params() {
        let params = IndexParams::builder()
            .ace_with(AceParams::builder().npartitions(4).build().unwrap())
            .build()
            .unwrap();

        unsafe {
            assert_eq!(
                (*params.handle).build_algo,
                ffi::cuvsCagraGraphBuildAlgo::ACE
            );
            assert!(!(*params.handle).graph_build_params.is_null());
            let ace = (*params.handle).graph_build_params as ffi::cuvsAceParams_t;
            assert_eq!((*ace).npartitions, 4);
        }
        assert!(params.graph_build_owner.is_some());
    }

    #[test]
    fn index_params_ace_implied_by_ace_with() {
        let params = IndexParams::builder()
            .ace_with(AceParams::builder().build().unwrap())
            .build()
            .unwrap();

        unsafe {
            assert_eq!(
                (*params.handle).build_algo,
                ffi::cuvsCagraGraphBuildAlgo::ACE
            );
        }
    }

    #[test]
    fn index_params_with_default_ace() {
        let params = IndexParams::builder().ace().build().unwrap();

        unsafe {
            assert_eq!(
                (*params.handle).build_algo,
                ffi::cuvsCagraGraphBuildAlgo::ACE
            );
            assert!(!(*params.handle).graph_build_params.is_null());
        }
    }

    #[test]
    fn index_params_with_ivf_pq_params() {
        let params = IndexParams::builder()
            .ivf_pq_with(
                IvfPqGraphBuildParams::builder()
                    .n_lists(100)
                    .refinement_rate(3.0)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();

        unsafe {
            assert_eq!(
                (*params.handle).build_algo,
                ffi::cuvsCagraGraphBuildAlgo::IVF_PQ
            );
            assert!(!(*params.handle).graph_build_params.is_null());
        }
        assert!(params.graph_build_owner.is_some());
    }

    #[test]
    fn index_params_with_default_ivf_pq() {
        let params = IndexParams::builder().ivf_pq().build().unwrap();

        unsafe {
            assert_eq!(
                (*params.handle).build_algo,
                ffi::cuvsCagraGraphBuildAlgo::IVF_PQ
            );
            assert!(!(*params.handle).graph_build_params.is_null());
        }
    }

    #[test]
    fn index_params_from_hnsw_params() {
        let params = IndexParams::from_hnsw_params(
            10000,
            128,
            16,
            200,
            HnswHeuristicType::SimilarSearchPerformance,
            DistanceType::L2Expanded,
        )
        .unwrap();

        unsafe {
            assert!((*params.handle).graph_degree > 0);
        }
        assert!(params.graph_build_owner.is_none());
    }

    #[test]
    fn index_params_from_hnsw_params_rejects_non_positive_n_rows() {
        let err = IndexParams::from_hnsw_params(
            0,
            128,
            16,
            200,
            HnswHeuristicType::SimilarSearchPerformance,
            DistanceType::L2Expanded,
        )
        .unwrap_err();

        assert!(err.to_string().contains("n_rows must be > 0"));
    }

    #[test]
    fn index_params_from_hnsw_params_rejects_non_positive_dim() {
        let err = IndexParams::from_hnsw_params(
            10000,
            0,
            16,
            200,
            HnswHeuristicType::SimilarSearchPerformance,
            DistanceType::L2Expanded,
        )
        .unwrap_err();

        assert!(err.to_string().contains("dim must be > 0"));
    }

    #[test]
    fn index_params_from_hnsw_params_rejects_non_positive_m() {
        let err = IndexParams::from_hnsw_params(
            10000,
            128,
            0,
            200,
            HnswHeuristicType::SimilarSearchPerformance,
            DistanceType::L2Expanded,
        )
        .unwrap_err();

        assert!(err.to_string().contains("m must be > 0"));
    }

    #[test]
    fn index_params_from_hnsw_params_rejects_non_positive_ef_construction() {
        let err = IndexParams::from_hnsw_params(
            10000,
            128,
            16,
            0,
            HnswHeuristicType::SimilarSearchPerformance,
            DistanceType::L2Expanded,
        )
        .unwrap_err();

        assert!(err.to_string().contains("ef_construction must be > 0"));
    }

    // -- AceParams ---------------------------------------------------------

    #[test]
    fn ace_params_all_defaults() {
        let params = AceParams::try_new().unwrap();
        unsafe {
            assert!((*params.handle).ef_construction > 0);
            assert!(!(*params.handle).build_dir.is_null());
            assert!(!(*params.handle).use_disk);
        }
    }

    #[test]
    fn ace_params_with_values() {
        let params = AceParams::builder()
            .npartitions(8)
            .ef_construction(128)
            .use_disk(true)
            .max_host_memory_gb(16.0)
            .max_gpu_memory_gb(8.0)
            .build()
            .unwrap();

        unsafe {
            assert_eq!((*params.handle).npartitions, 8);
            assert_eq!((*params.handle).ef_construction, 128);
            assert!((*params.handle).use_disk);
            assert_eq!((*params.handle).max_host_memory_gb, 16.0);
            assert_eq!((*params.handle).max_gpu_memory_gb, 8.0);
        }
    }

    #[test]
    fn ace_params_with_build_dir() {
        let params = AceParams::builder()
            .build_dir(Path::new("/tmp/ace"))
            .build()
            .unwrap();

        unsafe {
            assert!(!(*params.handle).build_dir.is_null());
            let build_dir = CStr::from_ptr((*params.handle).build_dir);
            assert_eq!(build_dir.to_bytes(), b"/tmp/ace");
        }
    }

    #[test]
    fn ace_params_c_api_can_replace_build_dir() {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsAceParamsCreate(&mut handle) }).unwrap();

        let first = CString::new("/tmp/ace-one").unwrap();
        let second = CString::new("/tmp/ace-two").unwrap();
        check_cuvs(unsafe { ffi::cuvsAceParamsSetBuildDir(handle, first.as_ptr()) }).unwrap();
        check_cuvs(unsafe { ffi::cuvsAceParamsSetBuildDir(handle, second.as_ptr()) }).unwrap();

        unsafe {
            let build_dir = CStr::from_ptr((*handle).build_dir);
            assert_eq!(build_dir.to_bytes(), b"/tmp/ace-two");
            let _ = ffi::cuvsAceParamsDestroy(handle);
        }
    }

    // -- IvfPqGraphBuildParams ---------------------------------------------

    #[test]
    fn ivf_pq_graph_build_params_defaults() {
        let params = IvfPqGraphBuildParams::try_new().unwrap();
        assert_eq!(params.inner.refinement_rate, 2.0);
    }

    #[test]
    fn ivf_pq_graph_build_params_with_values() {
        let params = IvfPqGraphBuildParams::builder()
            .n_lists(200)
            .pq_bits(4)
            .pq_dim(16)
            .n_probes(20)
            .refinement_rate(3.0)
            .build()
            .unwrap();

        assert_eq!(params.inner.refinement_rate, 3.0);
        unsafe {
            assert_eq!((*params.inner.ivf_pq_build_params).n_lists, 200);
            assert_eq!((*params.inner.ivf_pq_build_params).pq_bits, 4);
            assert_eq!((*params.inner.ivf_pq_build_params).pq_dim, 16);
            assert_eq!((*params.inner.ivf_pq_search_params).n_probes, 20);
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
        let params = SearchParams::try_new().unwrap();
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
    fn extend_params_try_new_uses_library_defaults() {
        let params = ExtendParams::try_new().unwrap();
        let builder_defaults = ExtendParams::builder().build().unwrap();
        unsafe {
            assert_eq!(
                (*params.handle).max_chunk_size,
                (*builder_defaults.handle).max_chunk_size
            );
        }
    }

    #[test]
    fn compression_params_try_new_uses_library_defaults() {
        let params = CompressionParams::try_new().unwrap();
        unsafe {
            assert!((*params.handle).pq_bits > 0);
        }
    }

    #[test]
    fn search_params_rejects_invalid_team_size_not_in_compiled_descriptor_table() {
        let err = SearchParams::builder().team_size(4).build().unwrap_err();
        assert!(err.to_string().contains("team_size must be"));
    }

    #[test]
    fn search_params_accepts_compiled_team_sizes() {
        for team_size in [8, 16, 32] {
            let params = SearchParams::builder()
                .team_size(team_size)
                .build()
                .unwrap();
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
