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
use crate::dlpack::{BorrowedDLTensor, MutBorrowedDLTensor};
use crate::error::{LibraryError, check_cuvs};
use crate::neighbors::filters::{Bitmap, Bitset, Filter};
use crate::resources::Resources;
use crate::{NotSend, ffi};

/// Error type for brute force operations.
#[derive(Debug, thiserror::Error)]
pub enum BruteForceError {
    /// The C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// A file path contained an interior NUL byte.
    #[error("path contains interior NUL byte")]
    InvalidPath(#[from] std::ffi::NulError),
}

/// Optional prefilter applied during brute-force search.
pub enum SearchFilter<'a> {
    /// Search without filtering.
    None,
    /// Reuse one row-level bitset for every query.
    Bitset(Filter<'a, Bitset>),
    /// Use a per-query bitmap of allowed `(query, row)` pairs.
    Bitmap(Filter<'a, Bitmap>),
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
    /// `res` is only used for the duration of this call.
    pub fn build(
        res: &Resources,
        dataset: &BorrowedDLTensor<'d>,
        metric: DistanceType,
        metric_arg: f32,
    ) -> Result<Self, BruteForceError> {
        let mut handle: ffi::cuvsBruteForceIndex_t = std::ptr::null_mut();
        // SAFETY: handle is a valid pointer to a null cuvsIndex_t.
        let status = unsafe { ffi::cuvsBruteForceIndexCreate(&mut handle) };
        check_cuvs(status)?;

        // Wrap immediately so Drop cleans up if build fails.
        let idx = Self {
            handle,
            _dataset: PhantomData,
            _not_send: PhantomData,
        };

        // SAFETY:
        // - res.handle() is a valid cuvsResources_t.
        // - dataset is a valid DLManagedTensor on the stack.
        // - idx.handle was successfully created above.
        let status = unsafe {
            ffi::cuvsBruteForceBuild(
                res.handle(),
                dataset.as_ptr(),
                metric.into(),
                metric_arg,
                idx.handle,
            )
        };
        check_cuvs(status)?;

        Ok(idx)
    }

    /// Search the index for nearest neighbors.
    ///
    /// `res` is only used for the duration of this call.
    /// `queries` is read-only input; the C function writes results into the
    /// pre-allocated `neighbors` and `distances` buffers.
    pub fn search(
        &self,
        res: &Resources,
        queries: &BorrowedDLTensor<'_>,
        neighbors: &MutBorrowedDLTensor<'_>,
        distances: &MutBorrowedDLTensor<'_>,
        filter: &SearchFilter<'_>,
    ) -> Result<(), BruteForceError> {
        let filter = match filter {
            SearchFilter::None => ffi::cuvsFilter {
                addr: 0,
                type_: ffi::cuvsFilterType::NO_FILTER,
            },
            SearchFilter::Bitset(filter) => filter.as_cuvs_filter(),
            SearchFilter::Bitmap(filter) => filter.as_cuvs_filter(),
        };

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

    /// Save the index to a file.
    pub fn serialize(
        &self,
        res: &Resources,
        path: impl AsRef<Path>,
    ) -> Result<(), BruteForceError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        // SAFETY: res and self.handle are valid; c_path is a valid C string.
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

        // Wrap immediately so Drop cleans up if deserialize fails.
        let idx = Index {
            handle,
            _dataset: PhantomData,
            _not_send: PhantomData,
        };

        // SAFETY: res is valid; idx.handle was just created; c_path is a valid C string.
        let status =
            unsafe { ffi::cuvsBruteForceDeserialize(res.handle(), c_path.as_ptr(), idx.handle) };
        check_cuvs(status)?;

        Ok(idx)
    }
}

impl Drop for Index<'_> {
    fn drop(&mut self) {
        // SAFETY: self.handle was successfully created by cuvsIndexCreate.
        let _ = unsafe { ffi::cuvsBruteForceIndexDestroy(self.handle) };
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;
    use crate::distance::DistanceType;
    use crate::dlpack::MutBorrowedDLTensor;
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
        let n_elements = (n_queries * k) as usize;
        let mut buf = vec![0i64; n_elements];
        neighbors.copy_data(&mut buf, n_elements);
        buf
    }

    #[test]
    fn build_and_search() {
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let neighbors =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let distances =
            tch::Tensor::zeros([N_QUERIES, K], (tch::Kind::Float, tch::Device::Cuda(0)));

        let res = Resources::new().unwrap();

        let dataset_dl = BorrowedDLTensor::from(&dataset);
        let index = Index::build(&res, &dataset_dl, DistanceType::L2Expanded, 0.0).unwrap();

        let queries_dl = BorrowedDLTensor::from(&queries);
        let neighbors_dl = MutBorrowedDLTensor::from(&neighbors);
        let distances_dl = MutBorrowedDLTensor::from(&distances);
        index
            .search(
                &res,
                &queries_dl,
                &neighbors_dl,
                &distances_dl,
                &SearchFilter::None,
            )
            .unwrap();

        // Verify: all neighbor indices are in range [0, N_ROWS).
        let n_elements = (N_QUERIES * K) as usize;
        let mut buf = vec![0i64; n_elements];
        neighbors.copy_data(&mut buf, n_elements);
        for &idx in &buf {
            assert!(
                idx >= 0 && idx < N_ROWS,
                "neighbor index {idx} out of range"
            );
        }

        // Verify: distances are non-negative for L2.
        let mut dbuf = vec![0f32; n_elements];
        distances.copy_data(&mut dbuf, n_elements);
        for &d in &dbuf {
            assert!(d >= 0.0, "L2 distance {d} should be non-negative");
        }
    }

    #[test]
    fn search_with_bitset_filter_excludes_filtered_rows() {
        let dataset = exact_filter_dataset();
        let queries = exact_filter_queries();
        let neighbors = tch::Tensor::zeros([2, 1], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let distances = tch::Tensor::zeros([2, 1], (tch::Kind::Float, tch::Device::Cuda(0)));
        let bitset = tch::Tensor::from_slice(&[0b0110i32]).to(tch::Device::Cuda(0));

        let res = Resources::new().unwrap();
        let dataset_dl = BorrowedDLTensor::from(&dataset);
        let index = Index::build(&res, &dataset_dl, DistanceType::L2Expanded, 0.0).unwrap();

        let queries_dl = BorrowedDLTensor::from(&queries);
        let neighbors_dl = MutBorrowedDLTensor::from(&neighbors);
        let distances_dl = MutBorrowedDLTensor::from(&distances);
        index
            .search(
                &res,
                &queries_dl,
                &neighbors_dl,
                &distances_dl,
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
        let neighbors = tch::Tensor::zeros([2, 1], (tch::Kind::Int64, tch::Device::Cuda(0)));
        let distances = tch::Tensor::zeros([2, 1], (tch::Kind::Float, tch::Device::Cuda(0)));
        let bitmap = tch::Tensor::from_slice(&[0b0010_0100i32]).to(tch::Device::Cuda(0));

        let res = Resources::new().unwrap();
        let dataset_dl = BorrowedDLTensor::from(&dataset);
        let index = Index::build(&res, &dataset_dl, DistanceType::L2Expanded, 0.0).unwrap();

        let queries_dl = BorrowedDLTensor::from(&queries);
        let neighbors_dl = MutBorrowedDLTensor::from(&neighbors);
        let distances_dl = MutBorrowedDLTensor::from(&distances);
        index
            .search(
                &res,
                &queries_dl,
                &neighbors_dl,
                &distances_dl,
                &SearchFilter::Bitmap(Filter::<Bitmap>::new(&bitmap).unwrap()),
            )
            .unwrap();

        let indices = extract_neighbor_indices(&neighbors, 2, 1);
        assert_eq!(indices, vec![2, 1]);
    }
}
