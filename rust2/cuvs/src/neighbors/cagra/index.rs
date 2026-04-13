/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CAGRA index: build, search, extend, serialize/deserialize, and accessors.

use std::ffi::CString;
use std::marker::PhantomData;
use std::path::Path;

use crate::distance::DistanceType;
use crate::dlpack::{DLTensorView, DLTensorViewMut, IntoDlTensor, IntoDlTensorMut, view_from_ffi};
use crate::error::check_cuvs;
use crate::neighbors::filters::{SearchFilter, prepare_filter};
use crate::resources::Resources;
use crate::{NotSend, ffi};

use super::CagraError;
use super::params::{ExtendParams, IndexParams, SearchParams};

/// A CAGRA approximate nearest neighbor index.
///
/// CAGRA builds a k-NN graph on the GPU and prunes it to the requested
/// `graph_degree`. The resulting graph is used for fast approximate search.
///
/// The lifetime `'d` ties this index to the dataset (and graph) tensors
/// passed at construction time. The C library may store a non-owning view
/// of properly aligned device-resident data, so the dataset must outlive
/// the index. When an index is deserialized from disk or produced by
/// [`Index::merge`], the data is self-contained and the lifetime is
/// `'static`.
pub struct Index<'d> {
    handle: ffi::cuvsCagraIndex_t,
    _dataset: PhantomData<&'d ()>,
    _not_send: NotSend,
}

impl<'d> Index<'d> {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Build a CAGRA index from a dataset tensor.
    ///
    /// The C library may retain a non-owning view of properly-aligned
    /// device-resident data. The returned index borrows `dataset` — it
    /// must outlive the index.
    pub fn build<D>(res: &Resources, params: &IndexParams, dataset: D) -> Result<Self, CagraError>
    where
        D: IntoDlTensor<'d>,
    {
        let dataset = dataset.into_dl_tensor()?;
        let handle = Self::create_raw_handle()?;

        let mut dataset_c = dataset.to_c();
        let status = unsafe {
            ffi::cuvsCagraBuild(
                res.handle(),
                params.handle(),
                dataset_c.as_mut_ptr(),
                handle,
            )
        };
        check_cuvs(status)?;
        Ok(Self {
            handle,
            _dataset: PhantomData,
            _not_send: PhantomData,
        })
    }

    /// Construct a CAGRA index from an existing graph and dataset.
    ///
    /// Both the graph and dataset may be retained as non-owning views;
    /// both must outlive the returned index.
    pub fn from_args<G, D>(
        res: &Resources,
        metric: DistanceType,
        graph: G,
        dataset: D,
    ) -> Result<Self, CagraError>
    where
        G: IntoDlTensor<'d>,
        D: IntoDlTensor<'d>,
    {
        let graph = graph.into_dl_tensor()?;
        let dataset = dataset.into_dl_tensor()?;
        let handle = Self::create_raw_handle()?;

        let (mut graph_c, mut dataset_c) = (graph.to_c(), dataset.to_c());
        let status = unsafe {
            ffi::cuvsCagraIndexFromArgs(
                res.handle(),
                metric.into(),
                graph_c.as_mut_ptr(),
                dataset_c.as_mut_ptr(),
                handle,
            )
        };
        check_cuvs(status)?;
        Ok(Self {
            handle,
            _dataset: PhantomData,
            _not_send: PhantomData,
        })
    }

    /// Deserialize a CAGRA index from a file previously written by
    /// [`Index::serialize`]. The deserialized index owns its data, so the
    /// returned lifetime is `'static`.
    pub fn deserialize(
        res: &Resources,
        path: impl AsRef<Path>,
    ) -> Result<Index<'static>, CagraError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let handle = Self::create_raw_handle()?;

        let status = unsafe { ffi::cuvsCagraDeserialize(res.handle(), c_path.as_ptr(), handle) };
        check_cuvs(status)?;
        Ok(Index {
            handle,
            _dataset: PhantomData,
            _not_send: PhantomData,
        })
    }

    // -----------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------

    /// Search the index for approximate nearest neighbors.
    ///
    /// The C function writes results into the pre-allocated `neighbors` and
    /// `distances` buffers.
    pub fn search<'q, 'n, 'dist, Q, N, Dist>(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: Q,
        neighbors: N,
        distances: Dist,
    ) -> Result<(), CagraError>
    where
        Q: IntoDlTensor<'q>,
        N: IntoDlTensorMut<'n>,
        Dist: IntoDlTensorMut<'dist>,
    {
        let queries = queries.into_dl_tensor()?;
        let neighbors = neighbors.into_dl_tensor_mut()?;
        let distances = distances.into_dl_tensor_mut()?;
        self.search_impl(res, params, &queries, &neighbors, &distances, None)
    }

    /// Search the index for approximate nearest neighbors with a row filter.
    pub fn search_filtered<'q, 'n, 'dist, Q, N, Dist>(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: Q,
        neighbors: N,
        distances: Dist,
        filter: &SearchFilter<'_>,
    ) -> Result<(), CagraError>
    where
        Q: IntoDlTensor<'q>,
        N: IntoDlTensorMut<'n>,
        Dist: IntoDlTensorMut<'dist>,
    {
        let queries = queries.into_dl_tensor()?;
        let neighbors = neighbors.into_dl_tensor_mut()?;
        let distances = distances.into_dl_tensor_mut()?;
        Self::validate_filter_support(filter)?;
        self.search_impl(res, params, &queries, &neighbors, &distances, Some(filter))
    }

    // -----------------------------------------------------------------
    // Extend
    // -----------------------------------------------------------------

    /// Extend the index with additional vectors.
    ///
    /// The additional dataset is divided into chunks and merged into the
    /// existing graph.
    pub fn extend<'a, D>(
        &mut self,
        res: &Resources,
        params: &ExtendParams,
        additional_dataset: D,
    ) -> Result<(), CagraError>
    where
        D: IntoDlTensor<'a>,
    {
        let additional_dataset = additional_dataset.into_dl_tensor()?;
        let mut additional_dataset_c = additional_dataset.to_c();
        let status = unsafe {
            ffi::cuvsCagraExtend(
                res.handle(),
                params.handle(),
                additional_dataset_c.as_mut_ptr(),
                self.handle,
            )
        };
        check_cuvs(status)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Merge
    // -----------------------------------------------------------------

    /// Merge multiple CAGRA indices into a new index.
    ///
    /// All input indices must share the same dtype and dimensionality.
    /// The merged index owns its data, so the returned lifetime is `'static`.
    pub fn merge(
        res: &Resources,
        params: &IndexParams,
        indices: &[&Index<'_>],
    ) -> Result<Index<'static>, CagraError> {
        Self::merge_impl(res, params, indices, None)
    }

    /// Merge multiple CAGRA indices into a new index using a row filter.
    pub fn merge_filtered(
        res: &Resources,
        params: &IndexParams,
        indices: &[&Index<'_>],
        filter: &SearchFilter<'_>,
    ) -> Result<Index<'static>, CagraError> {
        Self::validate_filter_support(filter)?;
        Self::merge_impl(res, params, indices, Some(filter))
    }

    // -----------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------

    /// Serialize the index to a file.
    ///
    /// When `include_dataset` is false, only the graph is saved; the dataset
    /// must be provided again when loading via [`Index::deserialize`].
    pub fn serialize(
        &self,
        res: &Resources,
        path: impl AsRef<Path>,
        include_dataset: bool,
    ) -> Result<(), CagraError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let status = unsafe {
            ffi::cuvsCagraSerialize(res.handle(), c_path.as_ptr(), self.handle, include_dataset)
        };
        check_cuvs(status)?;
        Ok(())
    }

    /// Serialize the index to a file in hnswlib-compatible format.
    ///
    /// The resulting file can only be read by the cuVS hnswlib wrapper (not
    /// by upstream hnswlib).
    pub fn serialize_to_hnswlib(
        &self,
        res: &Resources,
        path: impl AsRef<Path>,
    ) -> Result<(), CagraError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let status =
            unsafe { ffi::cuvsCagraSerializeToHnswlib(res.handle(), c_path.as_ptr(), self.handle) };
        check_cuvs(status)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Number of dimensions in the indexed vectors.
    pub fn dims(&self) -> Result<i64, CagraError> {
        let mut dim: i64 = 0;
        let status = unsafe { ffi::cuvsCagraIndexGetDims(self.handle, &mut dim) };
        check_cuvs(status)?;
        Ok(dim)
    }

    /// Number of vectors in the index.
    pub fn size(&self) -> Result<i64, CagraError> {
        let mut size: i64 = 0;
        let status = unsafe { ffi::cuvsCagraIndexGetSize(self.handle, &mut size) };
        check_cuvs(status)?;
        Ok(size)
    }

    /// Degree of the pruned graph.
    pub fn graph_degree(&self) -> Result<i64, CagraError> {
        let mut degree: i64 = 0;
        let status = unsafe { ffi::cuvsCagraIndexGetGraphDegree(self.handle, &mut degree) };
        check_cuvs(status)?;
        Ok(degree)
    }

    /// Return a non-owning view of the dataset attached to the index.
    pub fn dataset(&self) -> Result<DLTensorView<'_>, CagraError> {
        // SAFETY: the C function fully initializes the DLManagedTensor and
        // the data pointer is valid for &self's lifetime (index-owned).
        Ok(unsafe {
            view_from_ffi::<CagraError>(|ptr| ffi::cuvsCagraIndexGetDataset(self.handle, ptr))
        }?)
    }

    /// Return a non-owning view of the graph stored inside the index.
    pub fn graph(&self) -> Result<DLTensorView<'_>, CagraError> {
        // SAFETY: the C function fully initializes the DLManagedTensor and
        // the data pointer is valid for &self's lifetime (index-owned).
        Ok(unsafe {
            view_from_ffi::<CagraError>(|ptr| ffi::cuvsCagraIndexGetGraph(self.handle, ptr))
        }?)
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    fn validate_filter_support(filter: &SearchFilter<'_>) -> Result<(), CagraError> {
        if filter.uses_bitmap() {
            return Err(CagraError::Validation(
                "bitmap filters are not supported for CAGRA".into(),
            ));
        }
        Ok(())
    }

    fn search_impl(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &DLTensorView<'_>,
        neighbors: &DLTensorViewMut<'_>,
        distances: &DLTensorViewMut<'_>,
        filter: Option<&SearchFilter<'_>>,
    ) -> Result<(), CagraError> {
        let (mut q, mut n, mut d) = (queries.to_c(), neighbors.to_c(), distances.to_c());
        let mut filter_managed = None;
        let cuvs_filter = prepare_filter(filter, &mut filter_managed);
        let status = unsafe {
            ffi::cuvsCagraSearch(
                res.handle(),
                params.handle(),
                self.handle,
                q.as_mut_ptr(),
                n.as_mut_ptr(),
                d.as_mut_ptr(),
                cuvs_filter,
            )
        };
        check_cuvs(status)?;
        Ok(())
    }

    fn merge_impl(
        res: &Resources,
        params: &IndexParams,
        indices: &[&Index<'_>],
        filter: Option<&SearchFilter<'_>>,
    ) -> Result<Index<'static>, CagraError> {
        let handle = Self::create_raw_handle()?;

        let mut handles: Vec<ffi::cuvsCagraIndex_t> =
            indices.iter().map(|idx| idx.handle).collect();

        let mut filter_managed = None;
        let cuvs_filter = prepare_filter(filter, &mut filter_managed);
        let status = unsafe {
            ffi::cuvsCagraMerge(
                res.handle(),
                params.handle(),
                handles.as_mut_ptr(),
                handles.len(),
                cuvs_filter,
                handle,
            )
        };
        check_cuvs(status)?;
        Ok(Index {
            handle,
            _dataset: PhantomData,
            _not_send: PhantomData,
        })
    }

    fn create_raw_handle() -> Result<ffi::cuvsCagraIndex_t, CagraError> {
        let mut handle: ffi::cuvsCagraIndex_t = std::ptr::null_mut();
        let status = unsafe { ffi::cuvsCagraIndexCreate(&mut handle) };
        check_cuvs(status)?;
        Ok(handle)
    }
}

impl Drop for Index<'_> {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsCagraIndexDestroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "torch"))]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::distance::DistanceType;
    use crate::neighbors::cagra::{ExtendParams, IndexParams, SearchFilter, SearchParams};
    use crate::neighbors::filters::{Bitmap, Bitset, Filter};
    use crate::resources::Resources;

    const N_ROWS: i64 = 256;
    const DIM: i64 = 32;
    const K: i64 = 10;
    const N_QUERIES: i64 = 4;
    const EXTRA_ROWS: i64 = 64;

    fn search_neighbor_indices(
        index: &Index<'_>,
        res: &Resources,
        search_params: &SearchParams,
        queries: &tch::Tensor,
    ) -> Vec<i64> {
        let mut neighbors =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let mut distances =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Float, tch::Device::Cuda(0)));

        index
            .search(res, search_params, queries, &mut neighbors, &mut distances)
            .unwrap();

        Vec::<Vec<i64>>::try_from(&neighbors)
            .unwrap()
            .into_iter()
            .flatten()
            .collect()
    }

    fn search_neighbor_indices_with_filter(
        index: &Index<'_>,
        res: &Resources,
        search_params: &SearchParams,
        queries: &tch::Tensor,
        filter: &SearchFilter<'_>,
    ) -> Vec<i64> {
        let mut neighbors =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let mut distances =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Float, tch::Device::Cuda(0)));

        index
            .search_filtered(
                res,
                search_params,
                queries,
                &mut neighbors,
                &mut distances,
                filter,
            )
            .unwrap();

        Vec::<Vec<i64>>::try_from(&neighbors)
            .unwrap()
            .into_iter()
            .flatten()
            .collect()
    }

    fn assert_neighbor_indices_in_range(indices: &[i64], upper_bound: i64) {
        for &idx in indices {
            assert!(
                idx >= 0 && idx < upper_bound,
                "neighbor index {idx} out of range"
            );
        }
    }

    // Build a packed bitset where the first `allowed_rows` dataset rows are
    // marked as allowed (`1`) and all remaining rows are filtered out (`0`).
    fn bitset_words_with_allowed_prefix(allowed_rows: i64, total_rows: i64) -> Vec<i32> {
        let n_words = ((total_rows + 31) / 32) as usize;
        let mut words = vec![0u32; n_words];
        for row in 0..allowed_rows {
            let word = (row / 32) as usize;
            let bit = (row % 32) as u32;
            words[word] |= 1u32 << bit;
        }
        words.into_iter().map(|word| word as i32).collect()
    }

    fn temp_index_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cuvs-rust-{name}-{}-{unique}.bin",
            std::process::id()
        ))
    }

    #[test]
    fn build_and_search() {
        let res = Resources::new().unwrap();

        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        assert_eq!(index.dims().unwrap(), DIM);
        assert_eq!(index.size().unwrap(), N_ROWS);
        assert!(index.graph_degree().unwrap() > 0);

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_neighbor_indices_in_range(&buf, N_ROWS);
    }

    #[test]
    fn build_with_custom_params() {
        let res = Resources::new().unwrap();

        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .graph_degree(32)
            .nn_descent_with(20)
            .build()
            .unwrap();

        let index = Index::build(&res, &params, &dataset).unwrap();
        assert_eq!(index.dims().unwrap(), DIM);
    }

    #[test]
    fn build_with_compression() {
        use crate::neighbors::cagra::CompressionParams;

        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .compression(CompressionParams::builder().pq_bits(8).build().unwrap())
            .build()
            .unwrap();

        let index = Index::build(&res, &params, &dataset).unwrap();
        assert_eq!(index.dims().unwrap(), DIM);
    }

    #[test]
    fn dataset_and_graph_views_expose_expected_metadata() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let dataset_view = index.dataset().unwrap();
        assert_eq!(dataset_view.shape(), &[N_ROWS, DIM]);
        let dataset_dtype = dataset_view.dtype();
        assert_eq!(dataset_dtype.code, ffi::DLDataTypeCode::kDLFloat as u8);
        assert_eq!(dataset_dtype.bits, 32);
        assert_eq!(dataset_dtype.lanes, 1);

        let graph_view = index.graph().unwrap();
        assert_eq!(graph_view.shape(), &[N_ROWS, index.graph_degree().unwrap()]);
        let graph_dtype = graph_view.dtype();
        assert_eq!(graph_dtype.code, ffi::DLDataTypeCode::kDLUInt as u8);
        assert_eq!(graph_dtype.bits, 32);
        assert_eq!(graph_dtype.lanes, 1);
    }

    #[test]
    fn from_args_rebuilds_index_from_graph_and_dataset_views() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let dataset_view = index.dataset().unwrap();
        let graph_view = index.graph().unwrap();

        let rebuilt =
            Index::from_args(&res, DistanceType::L2Expanded, &graph_view, &dataset_view).unwrap();

        assert_eq!(rebuilt.dims().unwrap(), DIM);
        assert_eq!(rebuilt.size().unwrap(), N_ROWS);
        assert_eq!(
            rebuilt.graph_degree().unwrap(),
            index.graph_degree().unwrap()
        );

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&rebuilt, &res, &search_params, &queries);
        assert_neighbor_indices_in_range(&buf, N_ROWS);
    }

    /// Deserialized indices own their data (`'static`) and can be used
    /// independently of any original dataset.
    #[test]
    fn deserialized_index_is_static() {
        let res = Resources::new().unwrap();
        let params = IndexParams::builder().build().unwrap();

        let path = temp_index_path("cagra-static");
        {
            let dataset =
                tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
            let index = Index::build(&res, &params, &dataset).unwrap();
            index.serialize(&res, &path, true).unwrap();
        }
        // original dataset and index are dropped

        let index: Index<'static> = Index::deserialize(&res, &path).unwrap();

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();

        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_neighbor_indices_in_range(&buf, N_ROWS);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn serialize_and_deserialize_round_trip() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let path = temp_index_path("cagra-roundtrip");
        index.serialize(&res, &path, true).unwrap();

        let round_tripped = Index::deserialize(&res, &path).unwrap();
        assert_eq!(round_tripped.dims().unwrap(), DIM);
        assert_eq!(round_tripped.size().unwrap(), N_ROWS);
        assert_eq!(
            round_tripped.graph_degree().unwrap(),
            index.graph_degree().unwrap()
        );

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&round_tripped, &res, &search_params, &queries);
        assert_neighbor_indices_in_range(&buf, N_ROWS);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn extend_increases_size_and_search_still_works() {
        let res = Resources::new().unwrap();

        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let mut index = Index::build(&res, &params, &dataset).unwrap();

        let additional_dataset =
            tch::Tensor::randn([EXTRA_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let extend_params = ExtendParams::builder().max_chunk_size(32).build().unwrap();
        index
            .extend(&res, &extend_params, &additional_dataset)
            .unwrap();

        assert_eq!(index.size().unwrap(), N_ROWS + EXTRA_ROWS);
        assert_eq!(index.dims().unwrap(), DIM);

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_neighbor_indices_in_range(&buf, N_ROWS + EXTRA_ROWS);
    }

    #[test]
    fn multiple_searches() {
        let res = Resources::new().unwrap();

        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let search_params = SearchParams::builder().itopk_size(64).build().unwrap();

        for _ in 0..3 {
            let queries =
                tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
            let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
            assert_neighbor_indices_in_range(&buf, N_ROWS);
        }
    }

    #[test]
    fn search_with_bitset_filter_excludes_filtered_rows() {
        let res = Resources::new().unwrap();

        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();

        let allowed_rows = N_ROWS / 2;
        let bitset_words = bitset_words_with_allowed_prefix(allowed_rows, N_ROWS);
        let bitset = tch::Tensor::from_slice(&bitset_words).to(tch::Device::Cuda(0));
        let buf = search_neighbor_indices_with_filter(
            &index,
            &res,
            &search_params,
            &queries,
            &SearchFilter::Bitset(Filter::<Bitset>::new(&bitset).unwrap()),
        );
        assert_neighbor_indices_in_range(&buf, allowed_rows);
    }

    #[test]
    fn search_with_bitmap_filter_returns_validation_error() {
        let res = Resources::new().unwrap();

        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let mut neighbors =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let mut distances =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Float, tch::Device::Cuda(0)));
        let bitmap = tch::Tensor::from_slice(&[0b1111i32]).to(tch::Device::Cuda(0));

        let err = index
            .search_filtered(
                &res,
                &SearchParams::builder().build().unwrap(),
                &queries,
                &mut neighbors,
                &mut distances,
                &SearchFilter::Bitmap(Filter::<Bitmap>::new(&bitmap).unwrap()),
            )
            .unwrap_err();
        assert!(matches!(err, CagraError::Validation(_)));
    }

    #[test]
    fn merge_two_indices() {
        let res = Resources::new().unwrap();
        let params = IndexParams::builder().build().unwrap();

        let dataset_a = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let dataset_b = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let index_a = Index::build(&res, &params, &dataset_a).unwrap();
        let index_b = Index::build(&res, &params, &dataset_b).unwrap();

        let merged = Index::merge(&res, &params, &[&index_a, &index_b]).unwrap();

        assert_eq!(merged.dims().unwrap(), DIM);
        assert_eq!(merged.size().unwrap(), N_ROWS * 2);

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&merged, &res, &search_params, &queries);
        assert_neighbor_indices_in_range(&buf, N_ROWS * 2);
    }

    #[test]
    fn merge_with_bitmap_filter_returns_validation_error() {
        let res = Resources::new().unwrap();
        let params = IndexParams::builder().build().unwrap();

        let dataset_a = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let dataset_b = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let index_a = Index::build(&res, &params, &dataset_a).unwrap();
        let index_b = Index::build(&res, &params, &dataset_b).unwrap();
        let bitmap = tch::Tensor::from_slice(&[0b1111i32]).to(tch::Device::Cuda(0));

        let err = match Index::merge_filtered(
            &res,
            &params,
            &[&index_a, &index_b],
            &SearchFilter::Bitmap(Filter::<Bitmap>::new(&bitmap).unwrap()),
        ) {
            Ok(_) => panic!("expected bitmap filter to be rejected for CAGRA merge"),
            Err(err) => err,
        };
        assert!(matches!(err, CagraError::Validation(_)));
    }
}
