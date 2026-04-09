/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Vamana index: build, serialize, and accessors.

use std::ffi::CString;
use std::os::raw::c_int;
use std::path::Path;

use crate::dlpack::IntoDlTensor;
use crate::error::check_cuvs;
use crate::resources::Resources;
use crate::{NotSend, ffi};

use super::VamanaError;
use super::params::IndexParams;

/// A Vamana approximate nearest neighbor index.
///
/// Vamana builds a DiskANN-compatible graph index that can be serialized for
/// CPU-side DiskANN search. The current cuVS C API exposes build and
/// serialization, but not search or deserialize.
pub struct Index {
    handle: ffi::cuvsVamanaIndex_t,
    _not_send: NotSend,
}

impl Index {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Build a Vamana index from a dataset tensor.
    ///
    /// The dataset is copied into the index; the caller may free it after
    /// this call returns.
    ///
    /// Supported dataset dtypes in the current C-backed implementation are
    /// `f32`, `i8`, and `u8`. Host and device datasets are both accepted.
    pub fn build<'a, D>(
        res: &Resources,
        params: &IndexParams,
        dataset: D,
    ) -> Result<Self, VamanaError>
    where
        D: IntoDlTensor<'a>,
    {
        let dataset = dataset.into_dl_tensor()?;
        let idx = Self::create_handle()?;
        let status = unsafe {
            ffi::cuvsVamanaBuild(res.handle(), params.handle(), dataset.as_ptr(), idx.handle)
        };
        check_cuvs(status)?;
        Ok(idx)
    }

    // -----------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------

    /// Serialize the index to a DiskANN-compatible file prefix.
    ///
    /// The current C wrapper uses the non-sector-aligned serialization path,
    /// which writes the main graph to `path` itself and, when
    /// `include_dataset` is true, also writes a sibling `path.data` file.
    pub fn serialize(
        &self,
        res: &Resources,
        path: impl AsRef<Path>,
        include_dataset: bool,
    ) -> Result<(), VamanaError> {
        let c_path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let status = unsafe {
            ffi::cuvsVamanaSerialize(res.handle(), c_path.as_ptr(), self.handle, include_dataset)
        };
        check_cuvs(status)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Number of dimensions in the indexed vectors.
    pub fn dims(&self) -> Result<i64, VamanaError> {
        let mut dim: c_int = 0;
        let status = unsafe { ffi::cuvsVamanaIndexGetDims(self.handle, &mut dim) };
        check_cuvs(status)?;
        Ok(i64::from(dim))
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    fn create_handle() -> Result<Self, VamanaError> {
        let mut handle: ffi::cuvsVamanaIndex_t = std::ptr::null_mut();
        let status = unsafe { ffi::cuvsVamanaIndexCreate(&mut handle) };
        check_cuvs(status)?;
        Ok(Self {
            handle,
            _not_send: std::marker::PhantomData,
        })
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsVamanaIndexDestroy(self.handle) };
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    const N_ROWS: i64 = 256;
    const DIM: i64 = 16;

    fn temp_prefix(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cuvs-rust-{name}-{}-{unique}", std::process::id()))
    }

    #[test]
    fn build_and_dims() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .graph_degree(32)
            .visited_size(64)
            .build()
            .unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        assert_eq!(index.dims().unwrap(), DIM);
    }

    #[test]
    fn serialize_writes_prefix_and_dataset_file() {
        let res = Resources::new().unwrap();
        let dataset = tch::Tensor::randn([N_ROWS, DIM], (tch::Kind::Float, tch::Device::Cuda(0)));

        let params = IndexParams::builder()
            .graph_degree(32)
            .visited_size(64)
            .build()
            .unwrap();
        let index = Index::build(&res, &params, &dataset).unwrap();

        let prefix = temp_prefix("vamana");
        index.serialize(&res, &prefix, true).unwrap();

        assert!(prefix.exists());
        assert!(prefix.with_extension("data").exists());

        let _ = std::fs::remove_file(&prefix);
        let _ = std::fs::remove_file(prefix.with_extension("data"));
    }
}
