/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU resource management with RAII semantics.

use std::marker::PhantomData;

use crate::NotSend;
use crate::error::{LibraryError, check_cuvs};
use crate::ffi;

/// Error type for resource operations.
#[derive(Debug, thiserror::Error)]
pub enum ResourcesError {
    /// The C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
}

/// An opaque handle to GPU resources bound to the current CUDA device and thread.
///
/// Resources are objects that are shared between function calls,
/// and includes things like CUDA streams, cuBLAS handles and other
/// resources that are expensive to create.
///
/// This type is `!Send` and `!Sync` — it must be created, used, and dropped
/// on the same thread.
pub struct Resources {
    handle: ffi::cuvsResources_t,
    _not_send: NotSend,
}

impl Resources {
    /// Create a new GPU resource handle bound to the current CUDA device.
    pub fn new() -> Result<Self, ResourcesError> {
        let mut handle: ffi::cuvsResources_t = 0;

        // SAFETY:
        // - `handle` is a valid, aligned pointer to a `cuvsResources_t`.
        // - On success, the C library writes an opaque handle into `handle`.
        let status = unsafe { ffi::cuvsResourcesCreate(&mut handle) };
        check_cuvs(status)?;

        Ok(Self {
            handle,
            _not_send: PhantomData,
        })
    }

    /// Explicitly destroy the GPU resources, returning any error from the C library.
    ///
    /// If you don't call this, `Drop` will destroy the resources silently.
    pub fn close(self) -> Result<(), ResourcesError> {
        let handle = self.handle;
        std::mem::forget(self);

        // SAFETY:
        // - `handle` was successfully created by `cuvsResourcesCreate`.
        // - We are on the same thread that created it (enforced by `!Send`).
        // - `self` is forgotten, so `Drop` will not double-destroy.
        let status = unsafe { ffi::cuvsResourcesDestroy(handle) };
        check_cuvs(status)?;
        Ok(())
    }

    /// Attach a custom CUDA stream to this resource handle.
    ///
    /// All subsequent cuVS operations using this handle will be enqueued on
    /// the given stream instead of the default internal stream.
    ///
    /// # Safety
    ///
    /// - `stream` must be a valid `cudaStream_t` for the same CUDA device
    ///   this resource handle is bound to.
    /// - The stream must remain valid for as long as this resource handle uses it.
    pub unsafe fn set_stream(&self, stream: ffi::cudaStream_t) -> Result<(), ResourcesError> {
        // SAFETY: Caller guarantees `stream` is valid for this device and lifetime.
        let status = unsafe { ffi::cuvsStreamSet(self.handle, stream) };
        check_cuvs(status)?;
        Ok(())
    }

    /// Returns the current CUDA stream associated with this resource handle.
    pub fn stream(&self) -> Result<ffi::cudaStream_t, ResourcesError> {
        let mut stream: ffi::cudaStream_t = std::ptr::null_mut();

        // SAFETY:
        // - `self.handle` is a valid resource handle.
        // - `stream` is a valid, aligned pointer to a `cudaStream_t`.
        let status = unsafe { ffi::cuvsStreamGet(self.handle, &mut stream) };
        check_cuvs(status)?;
        Ok(stream)
    }

    /// Block until all operations on the current CUDA stream have completed.
    pub fn sync_stream(&self) -> Result<(), ResourcesError> {
        // SAFETY: `self.handle` is a valid resource handle.
        let status = unsafe { ffi::cuvsStreamSync(self.handle) };
        check_cuvs(status)?;
        Ok(())
    }

    /// Returns the CUDA device ID this resource handle is bound to.
    pub fn device_id(&self) -> Result<i32, ResourcesError> {
        let mut device_id: i32 = 0;

        // SAFETY:
        // - `self.handle` is a valid resource handle.
        // - `device_id` is a valid, aligned pointer to an `i32`.
        let status = unsafe { ffi::cuvsDeviceIdGet(self.handle, &mut device_id) };
        check_cuvs(status)?;
        Ok(device_id)
    }

    /// Access the raw handle for FFI calls in other modules.
    pub(crate) fn handle(&self) -> ffi::cuvsResources_t {
        self.handle
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        // SAFETY:
        // - `self.handle` was successfully created by `cuvsResourcesCreate`.
        // - We are on the same thread that created it (enforced by `!Send`).
        let _ = unsafe { ffi::cuvsResourcesDestroy(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_drop() {
        let res = Resources::new().expect("failed to create resources");
        drop(res);
    }

    #[test]
    fn close_returns_ok() {
        let res = Resources::new().expect("failed to create resources");
        res.close().expect("close failed");
    }

    #[test]
    fn device_id_is_non_negative() {
        let res = Resources::new().expect("failed to create resources");
        let id = res.device_id().expect("failed to get device id");
        assert!(id >= 0);
    }

    #[test]
    fn stream_get_returns_non_null() {
        let res = Resources::new().expect("failed to create resources");
        let stream = res.stream().expect("failed to get stream");
        assert!(!stream.is_null());
    }

    #[test]
    fn sync_stream_succeeds() {
        let res = Resources::new().expect("failed to create resources");
        res.sync_stream().expect("failed to sync stream");
    }
}
