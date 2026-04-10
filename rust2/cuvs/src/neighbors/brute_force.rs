/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Brute force exact nearest neighbor search.
//!
//! The brute force index holds a **non-owning view** of the dataset.
//! The Rust lifetime system enforces that the dataset outlives the index.

use std::ffi::CString;
use std::marker::PhantomData;
use std::path::Path;

use crate::distance::DistanceType;
use crate::dlpack::{DLPackError, DLTensorView, DLTensorViewMut, IntoDlTensor, IntoDlTensorMut};
use crate::error::{LibraryError, check_cuvs};
pub use crate::neighbors::filters::SearchFilter;
use crate::neighbors::filters::no_filter;
use crate::resources::Resources;
use crate::{NotSend, ffi};

/// Error type for brute force operations.
#[derive(Debug, thiserror::Error)]
pub enum BruteForceError {
    /// The C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
    /// A file path contained an interior NUL byte.
    #[error("path contains interior NUL byte")]
    InvalidPath(#[from] std::ffi::NulError),
}

/// A brute force nearest neighbor index.
///
/// The lifetime `'d` ties this index to the dataset tensor used to build it.
/// The C library holds a non-owning view of the dataset's data, so the dataset
/// must outlive the index.
pub struct Index<'d> {
    handle: ffi::cuvsBruteForceIndex_t,
    _dataset: PhantomData<&'d ()>,
    _not_send: NotSend,
}

impl<'d> Index<'d> {
    /// Build a brute force index from a dataset tensor.
    ///
    /// The returned index borrows `dataset` — it must outlive the index.
    pub fn build<D>(
        res: &Resources,
        dataset: D,
        metric: DistanceType,
    ) -> Result<Self, BruteForceError>
    where
        D: IntoDlTensor<'d>,
    {
        let dataset = dataset.into_dl_tensor()?;
        let mut handle: ffi::cuvsBruteForceIndex_t = std::ptr::null_mut();
        let status = unsafe { ffi::cuvsBruteForceIndexCreate(&mut handle) };
        check_cuvs(status)?;

        // Wrap early so Drop cleans up if build fails.
        let idx = Self {
            handle,
            _dataset: PhantomData,
            _not_send: PhantomData,
        };

        let status = unsafe {
            ffi::cuvsBruteForceBuild(
                res.handle(),
                dataset.as_ptr(),
                metric.into(),
                metric.metric_arg(),
                idx.handle,
            )
        };
        check_cuvs(status)?;

        Ok(idx)
    }

    /// Search the index for nearest neighbors.
    ///
    /// The C function writes results into the pre-allocated `neighbors` and
    /// `distances` buffers.
    pub fn search<'q, 'n, 'dist, Q, N, Dist>(
        &self,
        res: &Resources,
        queries: Q,
        neighbors: N,
        distances: Dist,
    ) -> Result<(), BruteForceError>
    where
        Q: IntoDlTensor<'q>,
        N: IntoDlTensorMut<'n>,
        Dist: IntoDlTensorMut<'dist>,
    {
        let queries = queries.into_dl_tensor()?;
        let neighbors = neighbors.into_dl_tensor_mut()?;
        let distances = distances.into_dl_tensor_mut()?;
        self.search_impl(res, &queries, &neighbors, &distances, no_filter())
    }

    /// Search the index for nearest neighbors with a row filter.
    pub fn search_filtered<'q, 'n, 'dist, Q, N, Dist>(
        &self,
        res: &Resources,
        queries: Q,
        neighbors: N,
        distances: Dist,
        filter: &SearchFilter<'_>,
    ) -> Result<(), BruteForceError>
    where
        Q: IntoDlTensor<'q>,
        N: IntoDlTensorMut<'n>,
        Dist: IntoDlTensorMut<'dist>,
    {
        let queries = queries.into_dl_tensor()?;
        let neighbors = neighbors.into_dl_tensor_mut()?;
        let distances = distances.into_dl_tensor_mut()?;
        self.search_impl(
            res,
            &queries,
            &neighbors,
            &distances,
            filter.as_cuvs_filter(),
        )
    }

    /// Save the index to a file.
    pub fn serialize(
        &self,
        res: &Resources,
        path: impl AsRef<Path>,
    ) -> Result<(), BruteForceError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let status =
            unsafe { ffi::cuvsBruteForceSerialize(res.handle(), c_path.as_ptr(), self.handle) };
        check_cuvs(status)?;
        Ok(())
    }

    /// Load an index from a file.
    ///
    /// The returned index does not borrow any external dataset, so the
    /// lifetime is `'static`.
    pub fn deserialize(
        res: &Resources,
        path: impl AsRef<Path>,
    ) -> Result<Index<'static>, BruteForceError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;

        let mut handle: ffi::cuvsBruteForceIndex_t = std::ptr::null_mut();
        let status = unsafe { ffi::cuvsBruteForceIndexCreate(&mut handle) };
        check_cuvs(status)?;

        // Wrap early so Drop cleans up if deserialize fails.
        let idx = Index {
            handle,
            _dataset: PhantomData,
            _not_send: PhantomData,
        };

        let status =
            unsafe { ffi::cuvsBruteForceDeserialize(res.handle(), c_path.as_ptr(), idx.handle) };
        check_cuvs(status)?;

        Ok(idx)
    }

    fn search_impl(
        &self,
        res: &Resources,
        queries: &DLTensorView<'_>,
        neighbors: &DLTensorViewMut<'_>,
        distances: &DLTensorViewMut<'_>,
        filter: ffi::cuvsFilter,
    ) -> Result<(), BruteForceError> {
        let status = unsafe {
            ffi::cuvsBruteForceSearch(
                res.handle(),
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
}

impl Drop for Index<'_> {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsBruteForceIndexDestroy(self.handle) };
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;
    use crate::distance::DistanceType;
    use crate::neighbors::cagra;
    use crate::neighbors::filters::{Bitmap, Bitset, Filter, SearchFilter};
    use crate::resources::Resources;

    const N_ROWS: i64 = 256;
    const DIM: i64 = 32;
    const K: i64 = 4;
    const N_QUERIES: i64 = 8;

    fn exact_filter_dataset() -> tch::Tensor {
        tch::Tensor::from_slice(&[0.0f32, 10.0, 20.0, 30.0])
            .view([4, 1])
            .to(tch::Device::Cuda(0))
    }

    fn exact_filter_queries() -> tch::Tensor {
        tch::Tensor::from_slice(&[0.0f32, 30.0])
            .view([2, 1])
            .to(tch::Device::Cuda(0))
    }

    fn extract_neighbor_indices(neighbors: &tch::Tensor, n_queries: i64, k: i64) -> Vec<i64> {
        let expected_shape = [n_queries as usize, k as usize];
        let rows: Vec<Vec<i64>> = Vec::try_from(neighbors).unwrap();
        assert_eq!(rows.len(), expected_shape[0]);
        rows.into_iter().flatten().collect()
    }

    #[test]
    fn build_and_search() {
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let mut neighbors =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let mut distances =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Float, tch::Device::Cuda(0)));

        let res = Resources::new().unwrap();
        let index = Index::build(&res, &dataset, DistanceType::L2Expanded).unwrap();
        index
            .search(&res, &queries, &mut neighbors, &mut distances)
            .unwrap();

        // Verify: all neighbor indices are in range [0, N_ROWS).
        let buf: Vec<Vec<i64>> = Vec::try_from(&neighbors).unwrap();
        for &idx in buf.iter().flatten() {
            assert!(
                (0..N_ROWS).contains(&idx),
                "neighbor index {idx} out of range"
            );
        }

        // Verify: distances are non-negative for L2.
        let dbuf: Vec<Vec<f32>> = Vec::try_from(&distances).unwrap();
        for &d in dbuf.iter().flatten() {
            assert!(d >= 0.0, "L2 distance {d} should be non-negative");
        }
    }

    #[test]
    fn build_from_returned_cagra_dataset_view() {
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let mut neighbors =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let mut distances =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Float, tch::Device::Cuda(0)));

        let res = Resources::new().unwrap();
        let cagra_index = cagra::Index::build(
            &res,
            &cagra::IndexParams::builder().build().unwrap(),
            &dataset,
        )
        .unwrap();
        let dataset_view = cagra_index.dataset().unwrap();

        let index = Index::build(&res, &dataset_view, DistanceType::L2Expanded).unwrap();
        index
            .search(&res, &queries, &mut neighbors, &mut distances)
            .unwrap();

        let indices = extract_neighbor_indices(&neighbors, N_QUERIES, K);
        for &idx in &indices {
            assert!(
                (0..N_ROWS).contains(&idx),
                "neighbor index {idx} out of range"
            );
        }
    }

    #[test]
    fn search_with_bitset_filter_excludes_filtered_rows() {
        let dataset = exact_filter_dataset();
        let queries = exact_filter_queries();
        let mut neighbors = tch::Tensor::zeros([2, 1], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let mut distances = tch::Tensor::zeros([2, 1], (tch::Kind::Float, tch::Device::Cuda(0)));
        let bitset = tch::Tensor::from_slice(&[0b0110i32]).to(tch::Device::Cuda(0));

        let res = Resources::new().unwrap();
        let index = Index::build(&res, &dataset, DistanceType::L2Expanded).unwrap();
        index
            .search_filtered(
                &res,
                &queries,
                &mut neighbors,
                &mut distances,
                &SearchFilter::Bitset(Filter::<Bitset>::new(&bitset).unwrap()),
            )
            .unwrap();

        let indices = extract_neighbor_indices(&neighbors, 2, 1);
        assert_eq!(indices, vec![1, 2]);
    }

    #[test]
    fn search_with_bitmap_filter_uses_per_query_masks() {
        let dataset = exact_filter_dataset();
        let queries = exact_filter_queries();
        let mut neighbors = tch::Tensor::zeros([2, 1], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let mut distances = tch::Tensor::zeros([2, 1], (tch::Kind::Float, tch::Device::Cuda(0)));
        let bitmap = tch::Tensor::from_slice(&[0b0010_0100i32]).to(tch::Device::Cuda(0));

        let res = Resources::new().unwrap();
        let index = Index::build(&res, &dataset, DistanceType::L2Expanded).unwrap();
        index
            .search_filtered(
                &res,
                &queries,
                &mut neighbors,
                &mut distances,
                &SearchFilter::Bitmap(Filter::<Bitmap>::new(&bitmap).unwrap()),
            )
            .unwrap();

        let indices = extract_neighbor_indices(&neighbors, 2, 1);
        assert_eq!(indices, vec![2, 1]);
    }
}
