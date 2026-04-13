/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! IVF-Flat index: build, search, extend, serialize/deserialize, and accessors.

use std::ffi::CString;
use std::path::Path;

use crate::dlpack::{DLTensorView, DLTensorViewMut, IntoDlTensor, IntoDlTensorMut, view_from_ffi};
use crate::error::check_cuvs;
use crate::neighbors::filters::SearchFilter;
use crate::neighbors::filters::prepare_filter;
use crate::resources::Resources;
use crate::{NotSend, ffi};

use super::IvfFlatError;
use super::params::{IndexParams, SearchParams};

/// An IVF-Flat approximate nearest neighbor index.
///
/// IVF-Flat partitions the dataset into `n_lists` clusters using k-means
/// and stores the raw (uncompressed) vectors in each cluster. At search
/// time, only the `n_probes` closest clusters are scanned.
pub struct Index {
    handle: ffi::cuvsIvfFlatIndex_t,
    _not_send: NotSend,
}

impl Index {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Build an IVF-Flat index from a dataset tensor.
    ///
    /// The dataset is copied into the index; the caller may free it after
    /// this call returns.
    ///
    /// Supported dataset/query dtypes in the current C-backed implementation
    /// are `f32`, `f16`, `i8`, and `u8`.
    pub fn build<'a, D>(
        res: &Resources,
        params: &IndexParams,
        dataset: D,
    ) -> Result<Self, IvfFlatError>
    where
        D: IntoDlTensor<'a>,
    {
        let dataset = dataset.into_dl_tensor()?;
        let idx = Self::create_handle()?;

        let mut dataset_c = dataset.to_c();
        let status = unsafe {
            ffi::cuvsIvfFlatBuild(
                res.handle(),
                params.handle(),
                dataset_c.as_mut_ptr(),
                idx.handle,
            )
        };
        check_cuvs(status)?;
        Ok(idx)
    }

    /// Deserialize an IVF-Flat index from a file previously written by
    /// [`Index::serialize`].
    ///
    /// Experimental: both the API and the on-disk serialization format are
    /// subject to change across cuVS releases.
    pub fn deserialize(res: &Resources, path: impl AsRef<Path>) -> Result<Self, IvfFlatError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let idx = Self::create_handle()?;

        let status =
            unsafe { ffi::cuvsIvfFlatDeserialize(res.handle(), c_path.as_ptr(), idx.handle) };
        check_cuvs(status)?;
        Ok(idx)
    }

    // -----------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------

    /// Search the index for approximate nearest neighbors.
    ///
    /// The C layer writes results into the pre-allocated `neighbors` and
    /// `distances` buffers. Queries must use the same dtype as the dataset
    /// used to build the index. In the current implementation, `neighbors`
    /// must be an `int64` tensor and `distances` must be `float32`.
    pub fn search<'q, 'n, 'dist, Q, N, Dist>(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: Q,
        neighbors: N,
        distances: Dist,
    ) -> Result<(), IvfFlatError>
    where
        Q: IntoDlTensor<'q>,
        N: IntoDlTensorMut<'n>,
        Dist: IntoDlTensorMut<'dist>,
    {
        let queries = queries.into_dl_tensor()?;
        let neighbors = neighbors.into_dl_tensor_mut()?;
        let distances = distances.into_dl_tensor_mut()?;
        let index_dtype = unsafe { (*self.handle).dtype };
        let query_dtype = queries.dtype();
        Self::validate_query_dtype(index_dtype, query_dtype)?;
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
    ) -> Result<(), IvfFlatError>
    where
        Q: IntoDlTensor<'q>,
        N: IntoDlTensorMut<'n>,
        Dist: IntoDlTensorMut<'dist>,
    {
        let queries = queries.into_dl_tensor()?;
        let neighbors = neighbors.into_dl_tensor_mut()?;
        let distances = distances.into_dl_tensor_mut()?;
        let index_dtype = unsafe { (*self.handle).dtype };
        let query_dtype = queries.dtype();
        Self::validate_query_dtype(index_dtype, query_dtype)?;
        Self::validate_filter_support(filter)?;

        self.search_impl(res, params, &queries, &neighbors, &distances, Some(filter))
    }

    // -----------------------------------------------------------------
    // Extend
    // -----------------------------------------------------------------

    /// Add new vectors to the index.
    ///
    /// `new_vectors` must be a device-compatible tensor with shape
    /// `[n_rows, self.dims()]` and the same dtype as the indexed dataset.
    /// `new_indices` must be a device-compatible `int64` tensor with shape
    /// `[n_rows]` containing the row ids to assign to the appended vectors.
    ///
    /// When the index was built with `adaptive_centers = true`, extending the
    /// index updates cluster centers to reflect the newly added data.
    pub fn extend<'vectors, 'indices, V, I>(
        &mut self,
        res: &Resources,
        new_vectors: V,
        new_indices: I,
    ) -> Result<(), IvfFlatError>
    where
        V: IntoDlTensor<'vectors>,
        I: IntoDlTensor<'indices>,
    {
        let new_vectors = new_vectors.into_dl_tensor()?;
        let new_indices = new_indices.into_dl_tensor()?;
        let (mut new_vectors_c, mut new_indices_c) = (new_vectors.to_c(), new_indices.to_c());
        let status = unsafe {
            ffi::cuvsIvfFlatExtend(
                res.handle(),
                new_vectors_c.as_mut_ptr(),
                new_indices_c.as_mut_ptr(),
                self.handle,
            )
        };
        check_cuvs(status)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------

    /// Serialize the index to a file.
    ///
    /// Experimental: both the API and the on-disk serialization format are
    /// subject to change across cuVS releases.
    pub fn serialize(&self, res: &Resources, path: impl AsRef<Path>) -> Result<(), IvfFlatError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let status =
            unsafe { ffi::cuvsIvfFlatSerialize(res.handle(), c_path.as_ptr(), self.handle) };
        check_cuvs(status)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Number of clusters (inverted lists).
    pub fn n_lists(&self) -> Result<i64, IvfFlatError> {
        let mut val: i64 = 0;
        let status = unsafe { ffi::cuvsIvfFlatIndexGetNLists(self.handle, &mut val) };
        check_cuvs(status)?;
        Ok(val)
    }

    /// Dimensionality of the indexed vectors.
    pub fn dims(&self) -> Result<i64, IvfFlatError> {
        let mut val: i64 = 0;
        let status = unsafe { ffi::cuvsIvfFlatIndexGetDim(self.handle, &mut val) };
        check_cuvs(status)?;
        Ok(val)
    }

    /// Return a non-owning view of the cluster centers.
    pub fn centers(&self) -> Result<DLTensorView<'_>, IvfFlatError> {
        // SAFETY: the C function fully initializes the DLManagedTensor and
        // the data pointer is valid for &self's lifetime (index-owned).
        Ok(unsafe {
            view_from_ffi::<IvfFlatError>(|ptr| ffi::cuvsIvfFlatIndexGetCenters(self.handle, ptr))
        }?)
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    fn validate_query_dtype(
        index_dtype: ffi::DLDataType,
        query_dtype: ffi::DLDataType,
    ) -> Result<(), IvfFlatError> {
        if index_dtype.code != query_dtype.code || index_dtype.bits != query_dtype.bits {
            return Err(IvfFlatError::Validation(format!(
                "queries dtype must match the index dtype exactly: index has code={} bits={}, got code={} bits={}",
                index_dtype.code, index_dtype.bits, query_dtype.code, query_dtype.bits
            )));
        }
        Ok(())
    }

    fn validate_filter_support(filter: &SearchFilter<'_>) -> Result<(), IvfFlatError> {
        if filter.uses_bitmap() {
            return Err(IvfFlatError::Validation(
                "bitmap filters are not supported for IVF-Flat".into(),
            ));
        }
        Ok(())
    }

    fn create_handle() -> Result<Self, IvfFlatError> {
        let mut handle: ffi::cuvsIvfFlatIndex_t = std::ptr::null_mut();
        let status = unsafe { ffi::cuvsIvfFlatIndexCreate(&mut handle) };
        check_cuvs(status)?;
        Ok(Self {
            handle,
            _not_send: std::marker::PhantomData,
        })
    }

    fn search_impl(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &DLTensorView<'_>,
        neighbors: &DLTensorViewMut<'_>,
        distances: &DLTensorViewMut<'_>,
        filter: Option<&SearchFilter<'_>>,
    ) -> Result<(), IvfFlatError> {
        let (mut q, mut n, mut d) = (queries.to_c(), neighbors.to_c(), distances.to_c());
        let mut filter_managed = None;
        let cuvs_filter = prepare_filter(filter, &mut filter_managed);
        let status = unsafe {
            ffi::cuvsIvfFlatSearch(
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
}

impl Drop for Index {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsIvfFlatIndexDestroy(self.handle) };
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
    use crate::neighbors::filters::{Bitmap, Bitset, Filter, SearchFilter};
    use crate::resources::Resources;

    const N_ROWS: i64 = 1024;
    const DIM: i64 = 32;
    const K: i64 = 10;
    const N_QUERIES: i64 = 4;
    const EXTRA_ROWS: i64 = 64;

    fn search_neighbor_indices(
        index: &Index,
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

    fn assert_indices_in_range(indices: &[i64], upper: i64) {
        for &idx in indices {
            assert!(idx >= 0 && idx < upper, "index {idx} out of range");
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cuvs-rust-{name}-{}-{unique}.bin",
            std::process::id()
        ))
    }

    fn dl_dtype(code: ffi::DLDataTypeCode, bits: u8) -> ffi::DLDataType {
        ffi::DLDataType {
            code: code as u8,
            bits,
            lanes: 1,
        }
    }

    #[test]
    fn build_and_search() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder().n_lists(16).build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        assert_eq!(index.dims().unwrap(), DIM);
        assert!(index.n_lists().unwrap() > 0);

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().n_probes(16).build().unwrap();
        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_indices_in_range(&buf, N_ROWS);
    }

    #[test]
    fn build_and_search_with_default_builders() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder().build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        assert_eq!(index.dims().unwrap(), DIM);
        assert_eq!(index.n_lists().unwrap(), N_ROWS);

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_indices_in_range(&buf, N_ROWS);
    }

    #[test]
    fn search_after_source_dataset_drop() {
        let res = Resources::new().unwrap();
        let params = IndexParams::builder().n_lists(16).build().unwrap();

        let index = {
            let dataset =
                tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
            Index::build(&res, &params, &dataset).unwrap()
        };

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_indices_in_range(&buf, N_ROWS);
    }

    #[test]
    fn serialize_round_trip() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder().n_lists(16).build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let path = temp_path("ivf-flat-roundtrip");
        index.serialize(&res, &path).unwrap();

        let loaded = Index::deserialize(&res, &path).unwrap();
        assert_eq!(loaded.dims().unwrap(), DIM);

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().n_probes(16).build().unwrap();
        let buf = search_neighbor_indices(&loaded, &res, &search_params, &queries);
        assert_indices_in_range(&buf, N_ROWS);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn centers_accessor() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let n_lists = 16u32;
        let params = IndexParams::builder().n_lists(n_lists).build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let centers = index.centers().unwrap();
        assert_eq!(centers.shape(), &[n_lists as i64, DIM]);
    }

    #[test]
    fn extend_adds_new_vectors() {
        let res = Resources::new().unwrap();

        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().n_lists(16).build().unwrap();
        let mut index = Index::build(&res, &params, &dataset).unwrap();

        let new_vectors =
            tch::Tensor::randn([EXTRA_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let new_indices = tch::Tensor::arange_start(
            N_ROWS,
            N_ROWS + EXTRA_ROWS,
            (tch::Kind::Int64, tch::Device::Cuda(0)),
        );
        index.extend(&res, &new_vectors, &new_indices).unwrap();

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_indices_in_range(&buf, N_ROWS + EXTRA_ROWS);
    }

    #[test]
    fn search_with_bitset_filter() {
        let res = Resources::new().unwrap();

        let dataset = tch::Tensor::from_slice(&[0.0f32, 10.0, 20.0, 30.0])
            .view([4, 1])
            .to(tch::Device::Cuda(0));
        let queries = tch::Tensor::from_slice(&[0.0f32, 30.0])
            .view([2, 1])
            .to(tch::Device::Cuda(0));
        let mut neighbors = tch::Tensor::zeros([2, 1], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let mut distances = tch::Tensor::zeros([2, 1], (tch::Kind::Float, tch::Device::Cuda(0)));
        let bitset = tch::Tensor::from_slice(&[0b0110i32]).to(tch::Device::Cuda(0));

        let params = IndexParams::builder().n_lists(2).build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        index
            .search_filtered(
                &res,
                &SearchParams::builder().n_probes(2).build().unwrap(),
                &queries,
                &mut neighbors,
                &mut distances,
                &SearchFilter::Bitset(Filter::<Bitset>::new(&bitset).unwrap()),
            )
            .unwrap();

        let buf: Vec<Vec<i64>> = Vec::try_from(&neighbors).unwrap();
        assert_eq!(buf, vec![vec![1], vec![2]]);
    }

    #[test]
    fn search_with_bitmap_filter_returns_validation_error() {
        let res = Resources::new().unwrap();

        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let mut neighbors =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let mut distances =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Float, tch::Device::Cuda(0)));
        let bitmap = tch::Tensor::from_slice(&[0b1111i32]).to(tch::Device::Cuda(0));

        let params = IndexParams::builder().n_lists(16).build().unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let err = index
            .search_filtered(
                &res,
                &SearchParams::builder().n_probes(16).build().unwrap(),
                &queries,
                &mut neighbors,
                &mut distances,
                &SearchFilter::Bitmap(Filter::<Bitmap>::new(&bitmap).unwrap()),
            )
            .unwrap_err();
        assert!(matches!(err, IvfFlatError::Validation(_)));
    }

    #[test]
    fn query_dtype_validation_requires_exact_code_and_bits_match() {
        Index::validate_query_dtype(
            dl_dtype(ffi::DLDataTypeCode::kDLFloat, 32),
            dl_dtype(ffi::DLDataTypeCode::kDLFloat, 32),
        )
        .unwrap();

        let err = Index::validate_query_dtype(
            dl_dtype(ffi::DLDataTypeCode::kDLFloat, 32),
            dl_dtype(ffi::DLDataTypeCode::kDLFloat, 16),
        )
        .unwrap_err();
        assert!(matches!(err, IvfFlatError::Validation(_)));
    }
}
