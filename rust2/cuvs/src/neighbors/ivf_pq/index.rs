/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! IVF-PQ index: build, search, extend, serialize/deserialize, and accessors.

use std::ffi::CString;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::path::Path;

use crate::dlpack::{IntoDlTensor, IntoDlTensorMut, ReturnedDLTensor};
use crate::error::check_cuvs;
use crate::resources::Resources;
use crate::{NotSend, ffi};

use super::IvfPqError;
use super::params::{IndexParams, SearchParams};

/// An IVF-PQ approximate nearest neighbor index.
///
/// IVF-PQ partitions the dataset into `n_lists` Voronoi cells and
/// compresses each vector with product quantization. This yields a
/// compact in-memory index with fast approximate search.
pub struct Index {
    handle: ffi::cuvsIvfPqIndex_t,
    _not_send: NotSend,
}

/// A non-owning IVF-PQ index built from precomputed tensors.
///
/// The underlying C index keeps references to the supplied codebook and
/// centroid tensors instead of copying them. As a result, those tensors must
/// outlive this wrapper.
pub struct PrecomputedIndex<'a> {
    index: Index,
    _owner: PhantomData<&'a ()>,
}

impl Deref for PrecomputedIndex<'_> {
    type Target = Index;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl DerefMut for PrecomputedIndex<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.index
    }
}

impl Index {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Build an IVF-PQ index from a dataset tensor.
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
    ) -> Result<Self, IvfPqError>
    where
        D: IntoDlTensor<'a>,
    {
        let dataset = dataset.into_dl_tensor()?;
        let idx = Self::create_handle()?;

        let status = unsafe {
            ffi::cuvsIvfPqBuild(res.handle(), params.handle(), dataset.as_ptr(), idx.handle)
        };
        check_cuvs(status)?;
        Ok(idx)
    }

    /// Build a non-owning IVF-PQ index from precomputed model tensors.
    ///
    /// Unlike [`Index::build`], this constructor does not copy the provided
    /// tensors. The returned [`PrecomputedIndex`] is therefore lifetime-bound
    /// to the borrowed tensor views.
    ///
    /// The precomputed tensors must live on device memory, use `float32`
    /// dtype, and match the extents required by the underlying C API:
    ///
    /// - `pq_centers`: `[pq_dim, pq_len, pq_book_size]` for
    ///   `CodebookGen::PerSubspace`, or `[n_lists, pq_len, pq_book_size]` for
    ///   `CodebookGen::PerCluster`
    /// - `centers_padded`: `[n_lists, dim_ext]`, where `dim_ext = round_up(dim + 1, 8)`
    /// - `centers_rot`: `[n_lists, rot_dim]`, where `rot_dim = pq_len * pq_dim`
    /// - `rotation_matrix`: `[rot_dim, dim]`
    ///
    /// The resulting index stores only the trained model state. To make it
    /// searchable, call [`Index::extend`] to add dataset vectors and establish
    /// the active dataset dtype.
    pub fn build_precomputed<'a, P, CP, CR, RM>(
        res: &Resources,
        params: &IndexParams,
        dim: u32,
        pq_centers: P,
        centers_padded: CP,
        centers_rot: CR,
        rotation_matrix: RM,
    ) -> Result<PrecomputedIndex<'a>, IvfPqError>
    where
        P: IntoDlTensor<'a>,
        CP: IntoDlTensor<'a>,
        CR: IntoDlTensor<'a>,
        RM: IntoDlTensor<'a>,
    {
        let pq_centers = pq_centers.into_dl_tensor()?;
        let centers_padded = centers_padded.into_dl_tensor()?;
        let centers_rot = centers_rot.into_dl_tensor()?;
        let rotation_matrix = rotation_matrix.into_dl_tensor()?;
        let idx = Self::create_handle()?;

        let status = unsafe {
            ffi::cuvsIvfPqBuildPrecomputed(
                res.handle(),
                params.handle(),
                dim,
                pq_centers.as_ptr(),
                centers_padded.as_ptr(),
                centers_rot.as_ptr(),
                rotation_matrix.as_ptr(),
                idx.handle,
            )
        };
        check_cuvs(status)?;
        Ok(PrecomputedIndex {
            index: idx,
            _owner: PhantomData,
        })
    }

    /// Deserialize an IVF-PQ index from a file previously written by
    /// [`Index::serialize`].
    ///
    /// Experimental: both the API and the on-disk serialization format are
    /// subject to change across cuVS releases.
    pub fn deserialize(res: &Resources, path: impl AsRef<Path>) -> Result<Self, IvfPqError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let idx = Self::create_handle()?;

        let status =
            unsafe { ffi::cuvsIvfPqDeserialize(res.handle(), c_path.as_ptr(), idx.handle) };
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
    /// used to build or extend the index when that dtype metadata is available
    /// through the current C API. Deserialized or precomputed IVF-PQ indices
    /// may report an unknown dtype marker until data is added via
    /// [`Index::extend`], in which case the Rust bindings cannot pre-validate
    /// the query dtype and defer to the underlying implementation. In the
    /// current implementation,
    /// `neighbors` must be an `int64` tensor and `distances` must be
    /// `float32`.
    pub fn search<'q, 'n, 'dist, Q, N, Dist>(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: Q,
        neighbors: N,
        distances: Dist,
    ) -> Result<(), IvfPqError>
    where
        Q: IntoDlTensor<'q>,
        N: IntoDlTensorMut<'n>,
        Dist: IntoDlTensorMut<'dist>,
    {
        let queries = queries.into_dl_tensor()?;
        let neighbors = neighbors.into_dl_tensor_mut()?;
        let distances = distances.into_dl_tensor_mut()?;
        let index_dtype = unsafe { (*self.handle).dtype };
        let query_dtype = queries.dl_tensor().dtype;
        Self::validate_query_dtype(index_dtype, query_dtype)?;

        let status = unsafe {
            ffi::cuvsIvfPqSearch(
                res.handle(),
                params.handle(),
                self.handle,
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
            )
        };
        check_cuvs(status)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Extend
    // -----------------------------------------------------------------

    /// Add new vectors to the index.
    ///
    /// `new_vectors` must have shape `[n_rows, self.dims()]` and the same
    /// dtype as the indexed dataset. `new_indices` must be an `int64` vector
    /// with shape `[n_rows]`. The current C-backed implementation accepts both
    /// host and device tensors for `extend`, but both inputs must use the same
    /// memory type.
    pub fn extend<'vectors, 'indices, V, I>(
        &mut self,
        res: &Resources,
        new_vectors: V,
        new_indices: I,
    ) -> Result<(), IvfPqError>
    where
        V: IntoDlTensor<'vectors>,
        I: IntoDlTensor<'indices>,
    {
        let new_vectors = new_vectors.into_dl_tensor()?;
        let new_indices = new_indices.into_dl_tensor()?;
        let status = unsafe {
            ffi::cuvsIvfPqExtend(
                res.handle(),
                new_vectors.as_ptr(),
                new_indices.as_ptr(),
                self.handle,
            )
        };
        check_cuvs(status)?;
        Ok(())
    }

    /// Transform vectors by applying IVF-PQ clustering and PQ encoding.
    ///
    /// `input_dataset` must use one of the supported dataset dtypes (`f32`,
    /// `f16`, `i8`, or `u8`) and reside on device memory. `output_labels`
    /// must be a device `uint32` vector of shape `[n_rows]`. `output_dataset`
    /// must be a device `uint8` matrix with shape
    /// `[n_rows, ceil(self.pq_dim() * self.pq_bits() / 8)]`.
    pub fn transform<'input, 'labels, 'output, In, Labels, Out>(
        &self,
        res: &Resources,
        input_dataset: In,
        output_labels: Labels,
        output_dataset: Out,
    ) -> Result<(), IvfPqError>
    where
        In: IntoDlTensor<'input>,
        Labels: IntoDlTensorMut<'labels>,
        Out: IntoDlTensorMut<'output>,
    {
        let input_dataset = input_dataset.into_dl_tensor()?;
        let output_labels = output_labels.into_dl_tensor_mut()?;
        let output_dataset = output_dataset.into_dl_tensor_mut()?;
        let labels_dtype = output_labels.dl_tensor().dtype;
        let dataset_dtype = output_dataset.dl_tensor().dtype;
        Self::validate_transform_output_dtypes(labels_dtype, dataset_dtype)?;

        let status = unsafe {
            ffi::cuvsIvfPqTransform(
                res.handle(),
                self.handle,
                input_dataset.as_ptr(),
                output_labels.as_ptr(),
                output_dataset.as_ptr(),
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
    pub fn serialize(&self, res: &Resources, path: impl AsRef<Path>) -> Result<(), IvfPqError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let status = unsafe { ffi::cuvsIvfPqSerialize(res.handle(), c_path.as_ptr(), self.handle) };
        check_cuvs(status)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Scalar accessors
    // -----------------------------------------------------------------

    /// Number of clusters (inverted lists).
    pub fn n_lists(&self) -> Result<i64, IvfPqError> {
        let mut val: i64 = 0;
        check_cuvs(unsafe { ffi::cuvsIvfPqIndexGetNLists(self.handle, &mut val) })?;
        Ok(val)
    }

    /// Dimensionality of the indexed vectors.
    pub fn dims(&self) -> Result<i64, IvfPqError> {
        let mut val: i64 = 0;
        check_cuvs(unsafe { ffi::cuvsIvfPqIndexGetDim(self.handle, &mut val) })?;
        Ok(val)
    }

    /// Number of vectors in the index.
    pub fn size(&self) -> Result<i64, IvfPqError> {
        let mut val: i64 = 0;
        check_cuvs(unsafe { ffi::cuvsIvfPqIndexGetSize(self.handle, &mut val) })?;
        Ok(val)
    }

    /// PQ-encoded dimensionality.
    pub fn pq_dim(&self) -> Result<i64, IvfPqError> {
        let mut val: i64 = 0;
        check_cuvs(unsafe { ffi::cuvsIvfPqIndexGetPqDim(self.handle, &mut val) })?;
        Ok(val)
    }

    /// Bit width of each PQ code.
    pub fn pq_bits(&self) -> Result<i64, IvfPqError> {
        let mut val: i64 = 0;
        check_cuvs(unsafe { ffi::cuvsIvfPqIndexGetPqBits(self.handle, &mut val) })?;
        Ok(val)
    }

    /// Length of each PQ-encoded vector (number of PQ codes).
    pub fn pq_len(&self) -> Result<i64, IvfPqError> {
        let mut val: i64 = 0;
        check_cuvs(unsafe { ffi::cuvsIvfPqIndexGetPqLen(self.handle, &mut val) })?;
        Ok(val)
    }

    // -----------------------------------------------------------------
    // Tensor view accessors
    // -----------------------------------------------------------------

    /// Non-owning view of the cluster centers.
    pub fn centers(&self) -> Result<ReturnedDLTensor<'_>, IvfPqError> {
        Ok(ReturnedDLTensor::from_ffi(|ptr| unsafe {
            ffi::cuvsIvfPqIndexGetCenters(self.handle, ptr)
        })?)
    }

    /// Non-owning view of the padded cluster centers.
    pub fn centers_padded(&self) -> Result<ReturnedDLTensor<'_>, IvfPqError> {
        Ok(ReturnedDLTensor::from_ffi(|ptr| unsafe {
            ffi::cuvsIvfPqIndexGetCentersPadded(self.handle, ptr)
        })?)
    }

    /// Non-owning view of the PQ codebook centers.
    pub fn pq_centers(&self) -> Result<ReturnedDLTensor<'_>, IvfPqError> {
        Ok(ReturnedDLTensor::from_ffi(|ptr| unsafe {
            ffi::cuvsIvfPqIndexGetPqCenters(self.handle, ptr)
        })?)
    }

    /// Non-owning view of the rotated cluster centers.
    pub fn centers_rot(&self) -> Result<ReturnedDLTensor<'_>, IvfPqError> {
        Ok(ReturnedDLTensor::from_ffi(|ptr| unsafe {
            ffi::cuvsIvfPqIndexGetCentersRot(self.handle, ptr)
        })?)
    }

    /// Non-owning view of the rotation matrix.
    pub fn rotation_matrix(&self) -> Result<ReturnedDLTensor<'_>, IvfPqError> {
        Ok(ReturnedDLTensor::from_ffi(|ptr| unsafe {
            ffi::cuvsIvfPqIndexGetRotationMatrix(self.handle, ptr)
        })?)
    }

    /// Non-owning view of the per-list sizes.
    pub fn list_sizes(&self) -> Result<ReturnedDLTensor<'_>, IvfPqError> {
        Ok(ReturnedDLTensor::from_ffi(|ptr| unsafe {
            ffi::cuvsIvfPqIndexGetListSizes(self.handle, ptr)
        })?)
    }

    /// Non-owning view of the indices stored in a single IVF list.
    pub fn list_indices(&self, label: u32) -> Result<ReturnedDLTensor<'_>, IvfPqError> {
        self.validate_label(label)?;
        Ok(ReturnedDLTensor::from_ffi(|ptr| unsafe {
            ffi::cuvsIvfPqIndexGetListIndices(self.handle, label, ptr)
        })?)
    }

    /// Unpack contiguous PQ codes from a single IVF list into `out_codes`.
    ///
    /// `out_codes` must be a device `uint8` tensor with shape
    /// `[n_rows, ceil(self.pq_dim() * self.pq_bits() / 8)]`. The number of
    /// rows in `out_codes` determines how many records are unpacked, starting
    /// at `offset` within `label`. The caller must also ensure that
    /// `offset + n_rows` does not exceed the size of the selected list.
    pub fn unpack_contiguous_list_data<'a, Out>(
        &self,
        res: &Resources,
        out_codes: Out,
        label: u32,
        offset: u32,
    ) -> Result<(), IvfPqError>
    where
        Out: IntoDlTensorMut<'a>,
    {
        let out_codes = out_codes.into_dl_tensor_mut()?;
        self.validate_label(label)?;
        let status = unsafe {
            ffi::cuvsIvfPqIndexUnpackContiguousListData(
                res.handle(),
                self.handle,
                out_codes.as_ptr(),
                label,
                offset,
            )
        };
        check_cuvs(status)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    fn validate_query_dtype(
        index_dtype: ffi::DLDataType,
        query_dtype: ffi::DLDataType,
    ) -> Result<(), IvfPqError> {
        // The current IVF-PQ C API may leave dtype metadata unset for
        // deserialized or precomputed indices until extend() establishes an
        // active dataset dtype. In that case, we cannot validate here.
        if index_dtype.code == 0 && index_dtype.bits == 0 {
            return Ok(());
        }

        if index_dtype.code != query_dtype.code || index_dtype.bits != query_dtype.bits {
            return Err(IvfPqError::Validation(format!(
                "queries dtype must match the index dtype exactly: index has code={} bits={}, got code={} bits={}",
                index_dtype.code, index_dtype.bits, query_dtype.code, query_dtype.bits
            )));
        }
        Ok(())
    }

    fn validate_transform_output_dtypes(
        output_labels_dtype: ffi::DLDataType,
        output_dataset_dtype: ffi::DLDataType,
    ) -> Result<(), IvfPqError> {
        let expected_labels = ffi::DLDataType {
            code: ffi::DLDataTypeCode::kDLUInt as u8,
            bits: 32,
            lanes: 1,
        };
        if output_labels_dtype.code != expected_labels.code
            || output_labels_dtype.bits != expected_labels.bits
        {
            return Err(IvfPqError::Validation(format!(
                "output_labels must use uint32 dtype, got code={} bits={}",
                output_labels_dtype.code, output_labels_dtype.bits
            )));
        }

        let expected_dataset = ffi::DLDataType {
            code: ffi::DLDataTypeCode::kDLUInt as u8,
            bits: 8,
            lanes: 1,
        };
        if output_dataset_dtype.code != expected_dataset.code
            || output_dataset_dtype.bits != expected_dataset.bits
        {
            return Err(IvfPqError::Validation(format!(
                "output_dataset must use uint8 dtype, got code={} bits={}",
                output_dataset_dtype.code, output_dataset_dtype.bits
            )));
        }

        Ok(())
    }

    fn validate_label(&self, label: u32) -> Result<(), IvfPqError> {
        let n_lists = self.n_lists()?;
        if i64::from(label) >= n_lists {
            return Err(IvfPqError::Validation(format!(
                "label must be < n_lists ({n_lists}), got {label}"
            )));
        }
        Ok(())
    }

    fn create_handle() -> Result<Self, IvfPqError> {
        let mut handle: ffi::cuvsIvfPqIndex_t = std::ptr::null_mut();
        let status = unsafe { ffi::cuvsIvfPqIndexCreate(&mut handle) };
        check_cuvs(status)?;
        Ok(Self {
            handle,
            _not_send: std::marker::PhantomData,
        })
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsIvfPqIndexDestroy(self.handle) };
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
    use crate::dlpack::{DLPackError, DLTensorViewMut, IntoDlTensorMut};
    use crate::resources::Resources;

    const N_ROWS: i64 = 1024;
    const DIM: i64 = 32;
    const K: i64 = 10;
    const N_QUERIES: i64 = 4;
    const EXTRA_ROWS: i64 = 64;

    struct U32TensorView<'a> {
        tensor: &'a mut tch::Tensor,
        shape: Vec<i64>,
        strides: Option<Vec<i64>>,
    }

    impl<'a> U32TensorView<'a> {
        fn new(tensor: &'a mut tch::Tensor) -> Self {
            assert_eq!(tensor.kind(), tch::Kind::Int);

            let shape = tensor.size();
            let strides = if tensor.is_contiguous() {
                None
            } else {
                Some(tensor.stride())
            };

            Self {
                tensor,
                shape,
                strides,
            }
        }
    }

    impl<'a> IntoDlTensorMut<'a> for U32TensorView<'a> {
        fn into_dl_tensor_mut(self) -> Result<DLTensorViewMut<'a>, DLPackError> {
            let device = match self.tensor.device() {
                tch::Device::Cuda(id) => ffi::DLDevice {
                    device_type: ffi::DLDeviceType::kDLCUDA,
                    device_id: id as i32,
                },
                other => return Err(DLPackError::UnsupportedDevice(format!("{other:?}"))),
            };

            unsafe {
                DLTensorViewMut::from_raw_parts(
                    self.tensor.data_ptr() as *mut _,
                    device,
                    &self.shape,
                    self.strides.as_deref(),
                    ffi::DLDataType {
                        code: ffi::DLDataTypeCode::kDLUInt as u8,
                        bits: 32,
                        lanes: 1,
                    },
                )
            }
        }
    }

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

    fn encoded_width(index: &Index) -> i64 {
        (index.pq_dim().unwrap() * index.pq_bits().unwrap() + 7) / 8
    }

    fn find_nonempty_label(index: &Index) -> u32 {
        let n_lists = index.n_lists().unwrap() as u32;
        for label in 0..n_lists {
            let list_indices = index.list_indices(label).unwrap();
            if list_indices.shape().first().copied().unwrap_or(0) > 0 {
                return label;
            }
        }
        panic!("expected at least one non-empty IVF-PQ list");
    }

    #[test]
    fn build_and_search() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .n_lists(16)
            .pq_dim(8)
            .build()
            .unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        assert_eq!(index.dims().unwrap(), DIM);
        assert!(index.n_lists().unwrap() > 0);
        assert_eq!(index.size().unwrap(), N_ROWS);
        assert!(index.pq_dim().unwrap() > 0);
        assert!(index.pq_bits().unwrap() > 0);
        assert!(index.pq_len().unwrap() > 0);

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
        assert_eq!(index.size().unwrap(), N_ROWS);

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_indices_in_range(&buf, N_ROWS);
    }

    #[test]
    fn search_after_source_dataset_drop() {
        let res = Resources::new().unwrap();
        let params = IndexParams::builder()
            .n_lists(16)
            .pq_dim(8)
            .build()
            .unwrap();

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

        let params = IndexParams::builder()
            .n_lists(16)
            .pq_dim(8)
            .build()
            .unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let path = temp_path("ivf-pq-roundtrip");
        index.serialize(&res, &path).unwrap();

        let loaded = Index::deserialize(&res, &path).unwrap();
        assert_eq!(loaded.dims().unwrap(), DIM);
        assert_eq!(loaded.size().unwrap(), N_ROWS);

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().n_probes(16).build().unwrap();
        let buf = search_neighbor_indices(&loaded, &res, &search_params, &queries);
        assert_indices_in_range(&buf, N_ROWS);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn tensor_view_accessors() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let n_lists = 16u32;
        let params = IndexParams::builder()
            .n_lists(n_lists)
            .pq_dim(8)
            .build()
            .unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let centers = index.centers().unwrap();
        assert_eq!(centers.ndim(), 2);
        assert_eq!(centers.shape()[0], n_lists as i64);

        let centers_padded = index.centers_padded().unwrap();
        assert_eq!(centers_padded.ndim(), 2);
        assert_eq!(centers_padded.shape()[0], n_lists as i64);

        let pq_centers = index.pq_centers().unwrap();
        assert!(pq_centers.ndim() > 0);

        let centers_rot = index.centers_rot().unwrap();
        assert_eq!(centers_rot.ndim(), 2);
        assert_eq!(centers_rot.shape()[0], n_lists as i64);

        let rotation = index.rotation_matrix().unwrap();
        assert!(rotation.ndim() > 0);

        let sizes = index.list_sizes().unwrap();
        assert_eq!(sizes.ndim(), 1);
        assert_eq!(sizes.shape()[0], n_lists as i64);
    }

    #[test]
    fn extend_increases_size_and_search_still_works() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .n_lists(16)
            .pq_dim(8)
            .build()
            .unwrap();
        let mut index = Index::build(&res, &params, &dataset).unwrap();

        let new_vectors =
            tch::Tensor::randn([EXTRA_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let new_indices = tch::Tensor::arange_start(
            N_ROWS,
            N_ROWS + EXTRA_ROWS,
            (tch::Kind::Int64, tch::Device::Cuda(0)),
        );
        index.extend(&res, &new_vectors, &new_indices).unwrap();

        assert_eq!(index.size().unwrap(), N_ROWS + EXTRA_ROWS);
        assert_eq!(index.dims().unwrap(), DIM);

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_indices_in_range(&buf, N_ROWS + EXTRA_ROWS);
    }

    #[test]
    fn list_indices_and_unpack_contiguous_list_data() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .n_lists(16)
            .pq_dim(8)
            .build()
            .unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let label = find_nonempty_label(&index);
        let list_indices = index.list_indices(label).unwrap();
        assert_eq!(list_indices.ndim(), 1);
        assert!(list_indices.shape()[0] > 0);

        let n_take = list_indices.shape()[0].min(4);
        let mut codes = tch::Tensor::zeros(
            [n_take, encoded_width(&index)],
            (tch::Kind::Uint8, tch::Device::Cuda(0)),
        );
        index
            .unpack_contiguous_list_data(&res, &mut codes, label, 0)
            .unwrap();

        assert_eq!(codes.size(), vec![n_take, encoded_width(&index)]);
    }

    #[test]
    fn list_access_rejects_out_of_range_labels() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .n_lists(16)
            .pq_dim(8)
            .build()
            .unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let invalid_label = index.n_lists().unwrap() as u32;
        let err = index.list_indices(invalid_label).unwrap_err();
        assert!(matches!(err, IvfPqError::Validation(_)));

        let mut codes = tch::Tensor::zeros(
            [1, encoded_width(&index)],
            (tch::Kind::Uint8, tch::Device::Cuda(0)),
        );
        let err = index
            .unpack_contiguous_list_data(&res, &mut codes, invalid_label, 0)
            .unwrap_err();
        assert!(matches!(err, IvfPqError::Validation(_)));
    }

    #[test]
    fn transform_outputs_cluster_labels_and_codes() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .n_lists(16)
            .pq_dim(8)
            .build()
            .unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let mut labels_storage =
            tch::Tensor::zeros([N_ROWS], (tch::Kind::Int, tch::Device::Cuda(0)));
        let labels_dl = U32TensorView::new(&mut labels_storage);
        let mut codes = tch::Tensor::zeros(
            [N_ROWS, encoded_width(&index)],
            (tch::Kind::Uint8, tch::Device::Cuda(0)),
        );
        index
            .transform(&res, &dataset, labels_dl, &mut codes)
            .unwrap();

        let labels: Vec<i32> = Vec::try_from(&labels_storage).unwrap();
        assert!(
            labels
                .iter()
                .all(|&label| label >= 0 && i64::from(label) < index.n_lists().unwrap())
        );
        assert_eq!(codes.size(), vec![N_ROWS, encoded_width(&index)]);
    }

    #[test]
    fn build_precomputed_can_extend_and_search() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .n_lists(16)
            .pq_dim(8)
            .add_data_on_build(false)
            .build()
            .unwrap();
        let template = Index::build(&res, &params, &dataset).unwrap();
        assert_eq!(template.size().unwrap(), 0);

        let pq_centers = template.pq_centers().unwrap();
        let centers = template.centers_padded().unwrap();
        let centers_rot = template.centers_rot().unwrap();
        let rotation = template.rotation_matrix().unwrap();

        let mut index = Index::build_precomputed(
            &res,
            &params,
            DIM as u32,
            &pq_centers,
            &centers,
            &centers_rot,
            &rotation,
        )
        .unwrap();

        let indices = tch::Tensor::arange(N_ROWS, (tch::Kind::Int64, tch::Device::Cuda(0)));
        index.extend(&res, &dataset, &indices).unwrap();

        assert_eq!(index.size().unwrap(), N_ROWS);
        assert_eq!(index.dims().unwrap(), DIM);

        let queries =
            tch::Tensor::randn([N_QUERIES, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
        let search_params = SearchParams::builder().build().unwrap();
        let buf = search_neighbor_indices(&index, &res, &search_params, &queries);
        assert_indices_in_range(&buf, N_ROWS);
    }

    #[test]
    fn query_dtype_validation_requires_exact_match() {
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
        assert!(matches!(err, IvfPqError::Validation(_)));
    }

    #[test]
    fn query_dtype_validation_allows_unknown_index_metadata() {
        Index::validate_query_dtype(
            ffi::DLDataType {
                code: 0,
                bits: 0,
                lanes: 1,
            },
            dl_dtype(ffi::DLDataTypeCode::kDLFloat, 32),
        )
        .unwrap();
    }

    #[test]
    fn transform_output_dtype_validation_requires_u32_and_u8() {
        Index::validate_transform_output_dtypes(
            dl_dtype(ffi::DLDataTypeCode::kDLUInt, 32),
            dl_dtype(ffi::DLDataTypeCode::kDLUInt, 8),
        )
        .unwrap();

        let err = Index::validate_transform_output_dtypes(
            dl_dtype(ffi::DLDataTypeCode::kDLInt, 32),
            dl_dtype(ffi::DLDataTypeCode::kDLUInt, 8),
        )
        .unwrap_err();
        assert!(matches!(err, IvfPqError::Validation(_)));
    }
}
