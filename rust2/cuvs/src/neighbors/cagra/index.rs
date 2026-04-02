/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CAGRA index: build, search, extend, serialize/deserialize, and accessors.

use std::ffi::CString;
use std::path::Path;

use crate::dlpack::{BorrowedDLTensor, MutBorrowedDLTensor};
use crate::error::check_cuvs;
use crate::neighbors::filters::{Bitset, Filter};
use crate::resources::Resources;
use crate::{NotSend, ffi};

use super::params::{ExtendParams, IndexParams, SearchParams};
use super::CagraError;

/// Optional row filter applied during CAGRA search.
pub enum SearchFilter<'a> {
    /// Search without filtering.
    None,
    /// Reuse one row-level bitset for every query.
    Bitset(Filter<'a, Bitset>),
}

/// A CAGRA approximate nearest neighbor index.
///
/// CAGRA builds a k-NN graph on the GPU and prunes it to the requested
/// `graph_degree`. The resulting graph is used for fast approximate search.
///
/// Unlike [`crate::neighbors::brute_force::Index`], the CAGRA index does not
/// borrow the input tensor. In the current C API-backed bindings, build attempts
/// to attach an internal copy of the dataset to the index.
pub struct Index {
    handle: ffi::cuvsCagraIndex_t,
    _not_send: NotSend,
}

impl Index {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Build a CAGRA index from a dataset tensor.
    ///
    /// The dataset is copied into the index; the caller may free it after
    /// this call returns.
    pub fn build(
        res: &Resources,
        params: &IndexParams,
        dataset: &BorrowedDLTensor<'_>,
    ) -> Result<Self, CagraError> {
        let idx = Self::create_handle()?;

        let status = unsafe {
            ffi::cuvsCagraBuild(res.handle(), params.handle(), dataset.as_ptr(), idx.handle)
        };
        check_cuvs(status)?;
        Ok(idx)
    }

    /// Deserialize a CAGRA index from a file previously written by
    /// [`Index::serialize`].
    pub fn deserialize(res: &Resources, path: impl AsRef<Path>) -> Result<Self, CagraError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let idx = Self::create_handle()?;

        let status =
            unsafe { ffi::cuvsCagraDeserialize(res.handle(), c_path.as_ptr(), idx.handle) };
        check_cuvs(status)?;
        Ok(idx)
    }

    // -----------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------

    /// Search the index for approximate nearest neighbors.
    ///
    /// `queries` is a read-only input; the C function writes results into the
    /// pre-allocated `neighbors` and `distances` buffers.
    pub fn search(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &BorrowedDLTensor<'_>,
        neighbors: &MutBorrowedDLTensor<'_>,
        distances: &MutBorrowedDLTensor<'_>,
        filter: &SearchFilter<'_>,
    ) -> Result<(), CagraError> {
        let filter = match filter {
            SearchFilter::None => ffi::cuvsFilter {
                addr: 0,
                type_: ffi::cuvsFilterType::NO_FILTER,
            },
            SearchFilter::Bitset(filter) => filter.as_cuvs_filter(),
        };

        let status = unsafe {
            ffi::cuvsCagraSearch(
                res.handle(),
                params.handle(),
                self.handle,
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
                filter,
            )
        };
        check_cuvs(status)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Extend
    // -----------------------------------------------------------------

    /// Extend the index with additional vectors.
    ///
    /// The additional dataset is divided into chunks and merged into the
    /// existing graph.
    pub fn extend(
        &mut self,
        res: &Resources,
        params: &ExtendParams,
        additional_dataset: &BorrowedDLTensor<'_>,
    ) -> Result<(), CagraError> {
        let status = unsafe {
            ffi::cuvsCagraExtend(
                res.handle(),
                params.handle(),
                additional_dataset.as_ptr(),
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
        let status = unsafe {
            ffi::cuvsCagraSerializeToHnswlib(res.handle(), c_path.as_ptr(), self.handle)
        };
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

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    fn create_handle() -> Result<Self, CagraError> {
        let mut handle: ffi::cuvsCagraIndex_t = std::ptr::null_mut();
        let status = unsafe { ffi::cuvsCagraIndexCreate(&mut handle) };
        check_cuvs(status)?;
        Ok(Self {
            handle,
            _not_send: std::marker::PhantomData,
        })
    }
}

impl Drop for Index {
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
    use crate::dlpack::{BorrowedDLTensor, MutBorrowedDLTensor};
    use crate::neighbors::cagra::{ExtendParams, GraphBuildAlgo, IndexParams, SearchParams};
    use crate::neighbors::filters::{Bitset, Filter};
    use crate::resources::Resources;

    const N_ROWS: i64 = 256;
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
        let neighbors =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let distances =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Float, tch::Device::Cuda(0)));

        let queries_dl = BorrowedDLTensor::from(queries);
        let neighbors_dl = MutBorrowedDLTensor::from(&neighbors);
        let distances_dl = MutBorrowedDLTensor::from(&distances);

        index
            .search(
                res,
                search_params,
                &queries_dl,
                &neighbors_dl,
                &distances_dl,
                &SearchFilter::None,
            )
            .unwrap();

        let n_elements = (N_QUERIES * K) as usize;
        let mut buf = vec![0i64; n_elements];
        neighbors.copy_data(&mut buf, n_elements);
        buf
    }

    fn search_neighbor_indices_with_filter(
        index: &Index,
        res: &Resources,
        search_params: &SearchParams,
        queries: &tch::Tensor,
        filter: &SearchFilter<'_>,
    ) -> Vec<i64> {
        let neighbors =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let distances =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Float, tch::Device::Cuda(0)));

        let queries_dl = BorrowedDLTensor::from(queries);
        let neighbors_dl = MutBorrowedDLTensor::from(&neighbors);
        let distances_dl = MutBorrowedDLTensor::from(&distances);

        index
            .search(
                res,
                search_params,
                &queries_dl,
                &neighbors_dl,
                &distances_dl,
                filter,
            )
            .unwrap();

        let n_elements = (N_QUERIES * K) as usize;
        let mut buf = vec![0i64; n_elements];
        neighbors.copy_data(&mut buf, n_elements);
        buf
    }

    fn assert_neighbor_indices_in_range(indices: &[i64], upper_bound: i64) {
        for &idx in indices {
            assert!(idx >= 0 && idx < upper_bound, "neighbor index {idx} out of range");
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

        let dataset =
            tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let dataset_dl = BorrowedDLTensor::from(&dataset);
        let index = Index::build(&res, &params, &dataset_dl).unwrap();

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

        let dataset =
            tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .graph_degree(32)
            .build_algo(GraphBuildAlgo::NnDescent)
            .nn_descent_niter(20)
            .build()
            .unwrap();

        let dataset_dl = BorrowedDLTensor::from(&dataset);
        let index = Index::build(&res, &params, &dataset_dl).unwrap();
        assert_eq!(index.dims().unwrap(), DIM);
    }

    #[test]
    fn build_with_compression() {
        use crate::neighbors::cagra::CompressionParams;

        let res = Resources::new().unwrap();
        let dataset =
            tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .compression(CompressionParams::builder().pq_bits(8).build().unwrap())
            .build()
            .unwrap();

        let dataset_dl = BorrowedDLTensor::from(&dataset);
        let index = Index::build(&res, &params, &dataset_dl).unwrap();
        assert_eq!(index.dims().unwrap(), DIM);
    }

    #[test]
    fn search_after_source_dataset_drop() {
        let res = Resources::new().unwrap();
        let params = IndexParams::builder().build().unwrap();

        let index = {
            let dataset =
                tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
            let dataset_dl = BorrowedDLTensor::from(&dataset);
            Index::build(&res, &params, &dataset_dl).unwrap()
        };

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();

        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_neighbor_indices_in_range(&buf, N_ROWS);
    }

    #[test]
    fn serialize_and_deserialize_round_trip() {
        let res = Resources::new().unwrap();
        let dataset =
            tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let dataset_dl = BorrowedDLTensor::from(&dataset);
        let index = Index::build(&res, &params, &dataset_dl).unwrap();

        let path = temp_index_path("cagra-roundtrip");
        index.serialize(&res, &path, true).unwrap();

        let round_tripped = Index::deserialize(&res, &path).unwrap();
        assert_eq!(round_tripped.dims().unwrap(), DIM);
        assert_eq!(round_tripped.size().unwrap(), N_ROWS);
        assert_eq!(round_tripped.graph_degree().unwrap(), index.graph_degree().unwrap());

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

        let dataset =
            tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let dataset_dl = BorrowedDLTensor::from(&dataset);
        let mut index = Index::build(&res, &params, &dataset_dl).unwrap();

        let additional_dataset =
            tch::Tensor::randn([EXTRA_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let extend_params = ExtendParams::builder().max_chunk_size(32).build().unwrap();
        let additional_dataset_dl = BorrowedDLTensor::from(&additional_dataset);

        index
            .extend(&res, &extend_params, &additional_dataset_dl)
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

        let dataset =
            tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let dataset_dl = BorrowedDLTensor::from(&dataset);
        let index = Index::build(&res, &params, &dataset_dl).unwrap();

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

        let dataset =
            tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let params = IndexParams::builder().build().unwrap();
        let dataset_dl = BorrowedDLTensor::from(&dataset);
        let index = Index::build(&res, &params, &dataset_dl).unwrap();

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
}
