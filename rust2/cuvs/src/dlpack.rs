/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! DLPack tensor view types.
//!
//! Two tensor view types are provided:
//!
//! * [`DLTensorView`] — a read-only view created from tensor-like inputs.
//!   Use for C API parameters that only *read* data (datasets, queries).
//!
//! * [`DLTensorViewMut`] — a writable view created from mutable tensor handles.
//!   Use for C API parameters that *write* results (neighbors, distances).
//!
//! # Implementing custom tensor adapters
//!
//! Implement [`IntoDlTensor`] (and/or [`IntoDlTensorMut`]) for your tensor
//! type by calling [`DLTensorView::from_raw_parts`] with the tensor's data
//! pointer, device, shape, optional strides, and dtype.  The built-in
//! `ndarray` and `tch` adapters (behind their respective feature flags) use
//! this same public constructor, so they serve as reference implementations.
//! See the [`IntoDlTensor`] trait docs for a complete example.

use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::slice;

use tinyvec::TinyVec;

use crate::error::{LibraryError, check_cuvs};
use crate::ffi;

pub use ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLTensor};

/// Number of dimensions stored inline before `TinyVec` spills to the heap.
///
/// Most cuVS bindings use 1-D/2-D tensors, with IVF-PQ precomputed codebooks
/// requiring a 3-D tensor view.
const INLINE_DIMS: usize = 3;

pub(crate) type TensorDims = TinyVec<[i64; INLINE_DIMS]>;

/// A public conversion trait for read-only tensor inputs.
///
/// Implement this for your tensor type by calling
/// [`DLTensorView::from_raw_parts`] inside a small `unsafe` block.
/// Custom adapters must uphold the safety contract documented on that
/// constructor.
///
/// # Example: custom GPU matrix adapter
///
/// ```rust,ignore
/// use cuvs::dlpack::{DLTensorView, DType, DLPackError, IntoDlTensor};
/// use cuvs::dlpack::{DLDevice, DLDeviceType};
///
/// struct MyGpuMatrix<T> {
///     ptr: *mut T,
///     rows: usize,
///     cols: usize,
///     device_id: i32,
/// }
///
/// impl<'a, T: DType> IntoDlTensor<'a> for &'a MyGpuMatrix<T> {
///     fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError> {
///         let shape = [self.rows as i64, self.cols as i64];
///         // SAFETY: `self.ptr` points to valid device memory for the
///         // full extent of `rows * cols` elements of type `T`, and
///         // the data remains valid for the lifetime `'a`.
///         unsafe {
///             DLTensorView::from_raw_parts(
///                 self.ptr as *mut std::ffi::c_void,
///                 DLDevice {
///                     device_type: DLDeviceType::kDLCUDA,
///                     device_id: self.device_id,
///                 },
///                 &shape,
///                 None, // contiguous, row-major
///                 T::dl_dtype(),
///             )
///         }
///     }
/// }
/// ```
pub trait IntoDlTensor<'a> {
    fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError>;
}

/// A public conversion trait for writable tensor inputs.
///
/// Implement this for your tensor type by calling
/// [`DLTensorViewMut::from_raw_parts`] inside a small `unsafe` block.
/// In addition to the [`DLTensorView::from_raw_parts`] invariants, the data
/// region must be exclusively writable for the lifetime `'a`.
pub trait IntoDlTensorMut<'a> {
    fn into_dl_tensor_mut(self) -> Result<DLTensorViewMut<'a>, DLPackError>;
}

/// Maps a Rust element type to a DLPack `DLDataType`.
pub trait DType {
    fn dl_dtype() -> ffi::DLDataType;
}

macro_rules! impl_dtype {
    ($ty:ty, $code:expr, $bits:expr) => {
        impl DType for $ty {
            fn dl_dtype() -> ffi::DLDataType {
                ffi::DLDataType {
                    code: $code as u8,
                    bits: $bits,
                    lanes: 1,
                }
            }
        }
    };
}

impl_dtype!(f32, ffi::DLDataTypeCode::kDLFloat, 32);
impl_dtype!(f64, ffi::DLDataTypeCode::kDLFloat, 64);
impl_dtype!(i32, ffi::DLDataTypeCode::kDLInt, 32);
impl_dtype!(i64, ffi::DLDataTypeCode::kDLInt, 64);
impl_dtype!(u32, ffi::DLDataTypeCode::kDLUInt, 32);
impl_dtype!(u64, ffi::DLDataTypeCode::kDLUInt, 64);
impl_dtype!(u8, ffi::DLDataTypeCode::kDLUInt, 8);
impl_dtype!(i8, ffi::DLDataTypeCode::kDLInt, 8);
impl_dtype!(u16, ffi::DLDataTypeCode::kDLUInt, 16);
impl_dtype!(i16, ffi::DLDataTypeCode::kDLInt, 16);

/// Error when converting an external tensor to a DLPack view.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum DLPackError {
    /// The tensor resides on a device not supported by cuVS.
    #[error("unsupported tensor device: {0}")]
    UnsupportedDevice(String),
    /// The tensor dtype is not supported by the current adapter.
    #[error("unsupported tensor dtype: {0}")]
    UnsupportedDType(String),
    /// A strides slice did not match the tensor rank.
    #[error("strides length {strides} does not match tensor rank {ndim}")]
    StridesLenMismatch { ndim: usize, strides: usize },
    /// The source tensor reported invalid DLPack metadata.
    #[error("invalid DLPack metadata: {0}")]
    InvalidMetadata(&'static str),
}

// ---------------------------------------------------------------------------
// ManagedTensorRef — lifetime-bound handle for stack-local DLManagedTensor
// ---------------------------------------------------------------------------

/// A stack-local [`DLManagedTensor`](ffi::DLManagedTensor) whose lifetime is
/// tied to the originating tensor view.
///
/// Returned by [`DLTensorView::to_c`]. [`DLTensorViewMut`] reaches the same
/// method via [`Deref`]. The lifetime parameter ensures the view (which owns
/// the shape and strides arrays that the C struct points into) outlives this
/// handle.
pub(crate) struct ManagedTensorRef<'a> {
    pub(crate) inner: ffi::DLManagedTensor,
    _borrow: PhantomData<&'a ()>,
}

impl ManagedTensorRef<'_> {
    /// Return a mutable pointer suitable for C FFI functions that take
    /// `DLManagedTensor*`.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::DLManagedTensor {
        &mut self.inner
    }
}

// ---------------------------------------------------------------------------
// DLTensorView — read-only view for C API inputs
// ---------------------------------------------------------------------------

/// A non-owning, read-only DLPack tensor view.
///
/// Suitable for C API parameters that only *read* the data
/// (e.g. datasets, queries).
#[must_use]
pub struct DLTensorView<'a> {
    data: *mut std::ffi::c_void,
    device: ffi::DLDevice,
    dtype: ffi::DLDataType,
    shape: TensorDims,
    strides: Option<TensorDims>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> DLTensorView<'a> {
    /// Construct a DLPack view from raw tensor metadata.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that:
    /// - `data` points to initialized tensor storage for the full extent
    ///   described by `shape`, `strides`, and `dtype`
    /// - `device`, `shape`, `strides`, and `dtype` accurately describe
    ///   that storage
    /// - the underlying storage remains valid for the lifetime `'a`
    /// - the C API fully consumes the `DLManagedTensor*` and its `shape` /
    ///   `strides` pointers during the FFI call and does not retain them
    ///   after the call returns
    pub unsafe fn from_raw_parts(
        data: *mut std::ffi::c_void,
        device: ffi::DLDevice,
        shape: &[i64],
        strides: Option<&[i64]>,
        dtype: ffi::DLDataType,
    ) -> Result<Self, DLPackError> {
        if let Some(s) = strides {
            if s.len() != shape.len() {
                return Err(DLPackError::StridesLenMismatch {
                    ndim: shape.len(),
                    strides: s.len(),
                });
            }
        }
        Ok(Self {
            data,
            device,
            dtype,
            shape: shape.iter().copied().collect(),
            strides: strides.map(|s| s.iter().copied().collect()),
            _marker: PhantomData,
        })
    }

    /// Build a stack-local [`DLManagedTensor`] for an FFI call.
    ///
    /// The returned [`ManagedTensorRef`] borrows `self`, so the
    /// compiler ensures the view outlives the C struct and its
    /// pointers into the shape/strides arrays.
    pub(crate) fn to_c(&self) -> ManagedTensorRef<'_> {
        ManagedTensorRef {
            inner: ffi::DLManagedTensor {
                dl_tensor: ffi::DLTensor {
                    data: self.data,
                    device: self.device,
                    ndim: self.shape.len() as i32,
                    dtype: self.dtype,
                    shape: self.shape.as_ptr() as *mut _,
                    strides: self
                        .strides
                        .as_ref()
                        .map_or(std::ptr::null_mut(), |s| s.as_ptr() as *mut _),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            },
            _borrow: PhantomData,
        }
    }

    /// Number of dimensions as a Rust `usize`.
    ///
    /// DLPack stores rank as an `i32`, but the safe Rust API exposes it as a
    /// `usize` for ordinary indexing and slice-length comparisons.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape of the tensor (one element per dimension).
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Strides, if non-contiguous. `None` means C-contiguous row-major.
    pub fn strides(&self) -> Option<&[i64]> {
        self.strides.as_deref()
    }

    /// Element data type.
    pub fn dtype(&self) -> ffi::DLDataType {
        self.dtype
    }

    /// Override the dtype metadata without changing the underlying storage.
    pub(crate) fn set_dtype(&mut self, dtype: ffi::DLDataType) {
        self.dtype = dtype;
    }

    /// Device where the data resides.
    pub fn device(&self) -> ffi::DLDevice {
        self.device
    }
}

impl fmt::Debug for DLTensorView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DLTensorView")
            .field("shape", &self.shape.as_slice())
            .field("strides", &self.strides.as_deref())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DLTensorViewMut — writable view for C API outputs
// ---------------------------------------------------------------------------

/// A non-owning, writable DLPack tensor view.
///
/// This wraps a read-only [`DLTensorView`] plus an exclusive borrow marker,
/// which keeps the shared behavior in one place without letting writable views
/// lose their stronger aliasing contract.
#[must_use]
pub struct DLTensorViewMut<'a> {
    base: DLTensorView<'a>,
    _unique: PhantomData<&'a mut ()>,
}

impl<'a> DLTensorViewMut<'a> {
    /// Construct a writable DLPack view from raw tensor metadata.
    ///
    /// # Safety
    ///
    /// In addition to the [`DLTensorView::from_raw_parts`] invariants, the
    /// caller must guarantee the underlying storage is exclusively writable for
    /// the lifetime `'a`.
    pub unsafe fn from_raw_parts(
        data: *mut std::ffi::c_void,
        device: ffi::DLDevice,
        shape: &[i64],
        strides: Option<&[i64]>,
        dtype: ffi::DLDataType,
    ) -> Result<Self, DLPackError> {
        Ok(Self {
            base: unsafe { DLTensorView::from_raw_parts(data, device, shape, strides, dtype)? },
            _unique: PhantomData,
        })
    }
}

impl<'a> Deref for DLTensorViewMut<'a> {
    type Target = DLTensorView<'a>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl fmt::Debug for DLTensorViewMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DLTensorViewMut")
            .field("shape", &self.base.shape.as_slice())
            .field("strides", &self.base.strides.as_deref())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Conversions between view types
// ---------------------------------------------------------------------------

impl<'a> From<DLTensorViewMut<'a>> for DLTensorView<'a> {
    fn from(view: DLTensorViewMut<'a>) -> Self {
        view.base
    }
}

impl<'a> IntoDlTensor<'a> for DLTensorView<'a> {
    fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError> {
        Ok(self)
    }
}

impl<'a> IntoDlTensor<'a> for DLTensorViewMut<'a> {
    fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError> {
        Ok(DLTensorView::from(self))
    }
}

impl<'a> IntoDlTensorMut<'a> for DLTensorViewMut<'a> {
    fn into_dl_tensor_mut(self) -> Result<DLTensorViewMut<'a>, DLPackError> {
        Ok(self)
    }
}

impl<'a, 'b> IntoDlTensor<'a> for &'b DLTensorView<'a> {
    fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError> {
        Ok(DLTensorView {
            data: self.data,
            device: self.device,
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _marker: PhantomData,
        })
    }
}

impl<'a, 'b> IntoDlTensor<'a> for &'b DLTensorViewMut<'a> {
    fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError> {
        self.deref().into_dl_tensor()
    }
}

// ---------------------------------------------------------------------------
// FFI accessor helper
// ---------------------------------------------------------------------------

/// Call a C accessor that fills in a [`DLManagedTensor`], extract its metadata
/// into a read-only [`DLTensorView`], and run the DLPack deleter immediately.
///
/// The returned view's lifetime `'a` must be tied to the object that owns the
/// underlying tensor data (for example `&'a self` on an index accessor).
///
/// # Safety
///
/// The caller must ensure that:
/// - `init` fully initializes the `DLManagedTensor` on success
/// - the returned tensor data pointer stays valid for `'a`
/// - on error, `init` leaves no cleanup obligation in the output slot; this
///   helper only invokes the DLPack deleter after a successful `check_cuvs`
pub(crate) unsafe fn view_from_ffi<'a, E>(
    init: impl FnOnce(*mut ffi::DLManagedTensor) -> ffi::cuvsError_t,
) -> Result<DLTensorView<'a>, E>
where
    E: From<LibraryError> + From<DLPackError>,
{
    let mut managed = MaybeUninit::<ffi::DLManagedTensor>::zeroed();
    check_cuvs(init(managed.as_mut_ptr())).map_err(E::from)?;

    // SAFETY: the caller's contract guarantees the closure fully initialized
    // the struct on the success path.
    let mut managed = unsafe { managed.assume_init() };

    let result = (|| -> Result<DLTensorView<'a>, DLPackError> {
        let ndim = usize::try_from(managed.dl_tensor.ndim)
            .map_err(|_| DLPackError::InvalidMetadata("negative ndim"))?;

        if managed.dl_tensor.byte_offset != 0 {
            return Err(DLPackError::InvalidMetadata(
                "non-zero byte_offset is not supported",
            ));
        }

        let (shape, strides) = if ndim == 0 {
            (&[][..], None)
        } else {
            if managed.dl_tensor.shape.is_null() {
                return Err(DLPackError::InvalidMetadata("shape pointer is null"));
            }

            let shape = unsafe { slice::from_raw_parts(managed.dl_tensor.shape, ndim) };
            let strides = if managed.dl_tensor.strides.is_null() {
                None
            } else {
                Some(unsafe { slice::from_raw_parts(managed.dl_tensor.strides, ndim) })
            };
            (shape, strides)
        };

        unsafe {
            DLTensorView::from_raw_parts(
                managed.dl_tensor.data,
                managed.dl_tensor.device,
                shape,
                strides,
                managed.dl_tensor.dtype,
            )
        }
    })();

    if let Some(deleter) = managed.deleter {
        unsafe { deleter(&mut managed) };
    }

    result.map_err(E::from)
}

#[cfg(test)]
mod ffi_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, thiserror::Error)]
    enum ViewFromFfiTestError {
        #[error(transparent)]
        Library(#[from] LibraryError),
        #[error(transparent)]
        DLPack(#[from] DLPackError),
    }

    static DELETER_CALLS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn free_test_metadata(tensor: *mut ffi::DLManagedTensor) {
        DELETER_CALLS.fetch_add(1, Ordering::SeqCst);

        let tensor = unsafe { &mut *tensor };
        let ndim = tensor.dl_tensor.ndim as usize;

        if !tensor.dl_tensor.shape.is_null() {
            let shape = std::ptr::slice_from_raw_parts_mut(tensor.dl_tensor.shape, ndim);
            unsafe { drop(Box::from_raw(shape)) };
            tensor.dl_tensor.shape = std::ptr::null_mut();
        }

        if !tensor.dl_tensor.strides.is_null() {
            let strides = std::ptr::slice_from_raw_parts_mut(tensor.dl_tensor.strides, ndim);
            unsafe { drop(Box::from_raw(strides)) };
            tensor.dl_tensor.strides = std::ptr::null_mut();
        }
    }

    #[test]
    fn view_from_ffi_copies_metadata_before_running_deleter() {
        DELETER_CALLS.store(0, Ordering::SeqCst);

        let view: Result<DLTensorView<'_>, ViewFromFfiTestError> = unsafe {
            view_from_ffi(|ptr| {
                let shape = Box::into_raw(vec![2_i64, 3].into_boxed_slice()) as *mut i64;
                let strides = Box::into_raw(vec![3_i64, 1].into_boxed_slice()) as *mut i64;

                (*ptr).dl_tensor.data = std::ptr::null_mut();
                (*ptr).dl_tensor.device = ffi::DLDevice {
                    device_type: ffi::DLDeviceType::kDLCPU,
                    device_id: 0,
                };
                (*ptr).dl_tensor.ndim = 2;
                (*ptr).dl_tensor.dtype = ffi::DLDataType {
                    code: ffi::DLDataTypeCode::kDLFloat as u8,
                    bits: 32,
                    lanes: 1,
                };
                (*ptr).dl_tensor.shape = shape;
                (*ptr).dl_tensor.strides = strides;
                (*ptr).dl_tensor.byte_offset = 0;
                (*ptr).manager_ctx = std::ptr::null_mut();
                (*ptr).deleter = Some(free_test_metadata);

                ffi::cuvsError_t::CUVS_SUCCESS
            })
        };
        let view = view.unwrap();

        assert_eq!(DELETER_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(view.ndim(), 2);
        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), Some(&[3, 1][..]));
        assert_eq!(view.dtype().bits, 32);
        assert_eq!(view.device().device_type, ffi::DLDeviceType::kDLCPU);
    }

    #[test]
    fn view_from_ffi_rejects_negative_ndim() {
        let err: Result<DLTensorView<'_>, ViewFromFfiTestError> = unsafe {
            view_from_ffi(|ptr| {
                (*ptr).dl_tensor.data = std::ptr::null_mut();
                (*ptr).dl_tensor.device = ffi::DLDevice {
                    device_type: ffi::DLDeviceType::kDLCPU,
                    device_id: 0,
                };
                (*ptr).dl_tensor.ndim = -1;
                (*ptr).dl_tensor.dtype = ffi::DLDataType {
                    code: ffi::DLDataTypeCode::kDLFloat as u8,
                    bits: 32,
                    lanes: 1,
                };
                (*ptr).dl_tensor.shape = std::ptr::null_mut();
                (*ptr).dl_tensor.strides = std::ptr::null_mut();
                (*ptr).dl_tensor.byte_offset = 0;
                (*ptr).manager_ctx = std::ptr::null_mut();
                (*ptr).deleter = None;

                ffi::cuvsError_t::CUVS_SUCCESS
            })
        };
        let err = err.unwrap_err();

        assert!(err.to_string().contains("negative ndim"));
    }

    #[test]
    fn view_from_ffi_rejects_null_shape_for_positive_rank() {
        let err: Result<DLTensorView<'_>, ViewFromFfiTestError> = unsafe {
            view_from_ffi(|ptr| {
                (*ptr).dl_tensor.data = std::ptr::null_mut();
                (*ptr).dl_tensor.device = ffi::DLDevice {
                    device_type: ffi::DLDeviceType::kDLCPU,
                    device_id: 0,
                };
                (*ptr).dl_tensor.ndim = 1;
                (*ptr).dl_tensor.dtype = ffi::DLDataType {
                    code: ffi::DLDataTypeCode::kDLFloat as u8,
                    bits: 32,
                    lanes: 1,
                };
                (*ptr).dl_tensor.shape = std::ptr::null_mut();
                (*ptr).dl_tensor.strides = std::ptr::null_mut();
                (*ptr).dl_tensor.byte_offset = 0;
                (*ptr).manager_ctx = std::ptr::null_mut();
                (*ptr).deleter = None;

                ffi::cuvsError_t::CUVS_SUCCESS
            })
        };
        let err = err.unwrap_err();

        assert!(err.to_string().contains("shape pointer is null"));
    }

    #[test]
    fn view_from_ffi_rejects_nonzero_byte_offset() {
        DELETER_CALLS.store(0, Ordering::SeqCst);

        let err: Result<DLTensorView<'_>, ViewFromFfiTestError> = unsafe {
            view_from_ffi(|ptr| {
                let shape = Box::into_raw(vec![4_i64].into_boxed_slice()) as *mut i64;

                (*ptr).dl_tensor.data = std::ptr::null_mut();
                (*ptr).dl_tensor.device = ffi::DLDevice {
                    device_type: ffi::DLDeviceType::kDLCPU,
                    device_id: 0,
                };
                (*ptr).dl_tensor.ndim = 1;
                (*ptr).dl_tensor.dtype = ffi::DLDataType {
                    code: ffi::DLDataTypeCode::kDLFloat as u8,
                    bits: 32,
                    lanes: 1,
                };
                (*ptr).dl_tensor.shape = shape;
                (*ptr).dl_tensor.strides = std::ptr::null_mut();
                (*ptr).dl_tensor.byte_offset = 4;
                (*ptr).manager_ctx = std::ptr::null_mut();
                (*ptr).deleter = Some(free_test_metadata);

                ffi::cuvsError_t::CUVS_SUCCESS
            })
        };
        let err = err.unwrap_err();

        assert!(err.to_string().contains("byte_offset"));
        assert_eq!(DELETER_CALLS.load(Ordering::SeqCst), 1);
    }
}

// ---------------------------------------------------------------------------
// ndarray implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::*;

    fn array_layout<A, D>(arr: &ndarray::ArrayRef<A, D>) -> (TensorDims, Option<TensorDims>)
    where
        D: ndarray::Dimension,
    {
        // TensorDims keeps ≤3 dims on the stack (covers the common case).
        let shape: TensorDims = arr.shape().iter().map(|&d| d as i64).collect();
        let strides: Option<TensorDims> = if arr.is_standard_layout() {
            None
        } else {
            Some(arr.strides().iter().map(|&s| s as i64).collect())
        };
        (shape, strides)
    }

    fn array_view<'a, A, D>(
        arr: &'a ndarray::ArrayRef<A, D>,
    ) -> Result<DLTensorView<'a>, DLPackError>
    where
        A: DType,
        D: ndarray::Dimension,
    {
        let (shape, strides) = array_layout(arr);
        // SAFETY: ArrayRef::as_ptr() points to valid, initialized storage
        // for the full extent of shape/strides/dtype for the lifetime 'a.
        unsafe {
            DLTensorView::from_raw_parts(
                arr.as_ptr() as *mut _,
                ffi::DLDevice {
                    device_type: ffi::DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                &shape,
                strides.as_deref(),
                A::dl_dtype(),
            )
        }
    }

    fn array_view_mut<'a, A, D>(
        arr: &'a mut ndarray::ArrayRef<A, D>,
    ) -> Result<DLTensorViewMut<'a>, DLPackError>
    where
        A: DType,
        D: ndarray::Dimension,
    {
        let (shape, strides) = array_layout(arr);
        // SAFETY: ArrayRef::as_mut_ptr() points to valid, exclusively
        // writable storage for the full extent of shape/strides/dtype.
        unsafe {
            DLTensorViewMut::from_raw_parts(
                arr.as_mut_ptr() as *mut _,
                ffi::DLDevice {
                    device_type: ffi::DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                &shape,
                strides.as_deref(),
                A::dl_dtype(),
            )
        }
    }

    impl<'a, A, D> IntoDlTensor<'a> for &'a ndarray::ArrayRef<A, D>
    where
        A: DType,
        D: ndarray::Dimension,
    {
        fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError> {
            array_view(self)
        }
    }

    impl<'a, A, D> IntoDlTensorMut<'a> for &'a mut ndarray::ArrayRef<A, D>
    where
        A: DType,
        D: ndarray::Dimension,
    {
        fn into_dl_tensor_mut(self) -> Result<DLTensorViewMut<'a>, DLPackError> {
            array_view_mut(self)
        }
    }
}

// ---------------------------------------------------------------------------
// torch (tch-rs / PyTorch) implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "torch")]
mod tch_impl {
    use super::*;

    fn kind_to_dl_dtype(kind: tch::Kind) -> Result<ffi::DLDataType, DLPackError> {
        let (code, bits): (ffi::DLDataTypeCode, u8) = match kind {
            tch::Kind::Float => (ffi::DLDataTypeCode::kDLFloat, 32),
            tch::Kind::Double => (ffi::DLDataTypeCode::kDLFloat, 64),
            tch::Kind::Half => (ffi::DLDataTypeCode::kDLFloat, 16),
            tch::Kind::Int => (ffi::DLDataTypeCode::kDLInt, 32),
            tch::Kind::Int64 => (ffi::DLDataTypeCode::kDLInt, 64),
            tch::Kind::Int8 => (ffi::DLDataTypeCode::kDLInt, 8),
            tch::Kind::Int16 => (ffi::DLDataTypeCode::kDLInt, 16),
            tch::Kind::Uint8 => (ffi::DLDataTypeCode::kDLUInt, 8),
            tch::Kind::BFloat16 => (ffi::DLDataTypeCode::kDLBfloat, 16),
            tch::Kind::Bool => (ffi::DLDataTypeCode::kDLBool, 8),
            other => return Err(DLPackError::UnsupportedDType(format!("{other:?}"))),
        };
        Ok(ffi::DLDataType {
            code: code as u8,
            bits,
            lanes: 1,
        })
    }

    fn device_to_dl(device: tch::Device) -> Result<ffi::DLDevice, DLPackError> {
        match device {
            tch::Device::Cpu => Ok(ffi::DLDevice {
                device_type: ffi::DLDeviceType::kDLCPU,
                device_id: 0,
            }),
            tch::Device::Cuda(id) => Ok(ffi::DLDevice {
                device_type: ffi::DLDeviceType::kDLCUDA,
                device_id: id as i32,
            }),
            other => Err(DLPackError::UnsupportedDevice(format!("{other:?}"))),
        }
    }

    fn tensor_layout(tensor: &tch::Tensor) -> (TensorDims, Option<TensorDims>) {
        let shape: TensorDims = tensor.size().into_iter().collect();
        let strides: Option<TensorDims> = if tensor.is_contiguous() {
            None
        } else {
            Some(tensor.stride().into_iter().collect())
        };
        (shape, strides)
    }

    fn tensor_view<'a>(tensor: &'a tch::Tensor) -> Result<DLTensorView<'a>, DLPackError> {
        let (shape, strides) = tensor_layout(tensor);
        // SAFETY: data_ptr() is valid for the tensor's shape/dtype for 'a.
        unsafe {
            DLTensorView::from_raw_parts(
                tensor.data_ptr() as *mut _,
                device_to_dl(tensor.device())?,
                &shape,
                strides.as_deref(),
                kind_to_dl_dtype(tensor.kind())?,
            )
        }
    }

    fn tensor_view_mut<'a>(
        tensor: &'a mut tch::Tensor,
    ) -> Result<DLTensorViewMut<'a>, DLPackError> {
        let (shape, strides) = tensor_layout(tensor);
        // SAFETY: data_ptr() is valid and exclusively writable for 'a.
        unsafe {
            DLTensorViewMut::from_raw_parts(
                tensor.data_ptr() as *mut _,
                device_to_dl(tensor.device())?,
                &shape,
                strides.as_deref(),
                kind_to_dl_dtype(tensor.kind())?,
            )
        }
    }

    impl<'a> IntoDlTensor<'a> for &'a tch::Tensor {
        fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError> {
            tensor_view(self)
        }
    }

    /// Requiring `&mut tch::Tensor` preserves Rust-side exclusivity for normal
    /// call sites, even though `shallow_clone()` can still produce aliasing
    /// handles to the same underlying C++ storage.
    ///
    /// # Caution
    ///
    /// Do not call in-place operations that may reallocate storage
    /// (e.g., `resize_`) on the source tensor while a [`DLTensorViewMut`]
    /// derived from it is alive.
    impl<'a> IntoDlTensorMut<'a> for &'a mut tch::Tensor {
        fn into_dl_tensor_mut(self) -> Result<DLTensorViewMut<'a>, DLPackError> {
            tensor_view_mut(self)
        }
    }
}

#[cfg(all(test, feature = "torch"))]
mod torch_tests {
    use super::*;

    #[test]
    fn torch_f32_shape_device_and_dtype() {
        let tensor = tch::Tensor::zeros([100, 128], (tch::Kind::Float, tch::Device::Cpu));
        let dl = (&tensor).into_dl_tensor().unwrap();

        assert_eq!(dl.shape.len(), 2);
        assert_eq!(dl.shape[..], [100, 128]);
        assert!(dl.strides.is_none());

        let managed = dl.to_c();
        assert_eq!(
            managed.inner.dl_tensor.device.device_type,
            ffi::DLDeviceType::kDLCPU
        );
        assert_eq!(managed.inner.dl_tensor.device.device_id, 0);

        assert_eq!(dl.dtype.code, ffi::DLDataTypeCode::kDLFloat as u8);
        assert_eq!(dl.dtype.bits, 32);
        assert_eq!(dl.dtype.lanes, 1);
    }

    #[test]
    fn torch_transposed_cpu_tensor_has_strides() {
        let tensor = tch::Tensor::zeros([10, 20], (tch::Kind::Float, tch::Device::Cpu));
        let transposed = tensor.transpose(0, 1);
        let dl = (&transposed).into_dl_tensor().unwrap();

        assert_eq!(dl.shape[..], [20, 10]);
        let strides = dl.strides.as_ref().unwrap();
        assert_eq!(strides[..], [1, 20]);
    }

    #[test]
    fn torch_bool_dtype_maps_to_dl_bool() {
        let tensor = tch::Tensor::zeros([2, 2], (tch::Kind::Bool, tch::Device::Cpu));
        let dl = (&tensor).into_dl_tensor().unwrap();

        assert_eq!(dl.dtype.code, ffi::DLDataTypeCode::kDLBool as u8);
        assert_eq!(dl.dtype.bits, 8);
        assert_eq!(dl.dtype.lanes, 1);
    }

    #[test]
    fn torch_to_c_produces_valid_cpu_tensor() {
        let tensor = tch::Tensor::zeros([10, 20], (tch::Kind::Float, tch::Device::Cpu));
        let dl = (&tensor).into_dl_tensor().unwrap();
        let managed = dl.to_c();

        assert_eq!(managed.inner.dl_tensor.ndim, 2);
        assert!(!managed.inner.dl_tensor.data.is_null());
        assert!(!managed.inner.dl_tensor.shape.is_null());
        assert!(managed.inner.dl_tensor.strides.is_null());
        assert_eq!(unsafe { *managed.inner.dl_tensor.shape }, 10);
        assert_eq!(unsafe { *managed.inner.dl_tensor.shape.add(1) }, 20);
        assert_eq!(managed.inner.dl_tensor.byte_offset, 0);
        assert!(managed.inner.manager_ctx.is_null());
        assert!(managed.inner.deleter.is_none());
    }

    #[test]
    fn torch_mut_view_from_tensor() {
        let mut tensor = tch::Tensor::zeros([8, 16], (tch::Kind::Float, tch::Device::Cpu));
        let dl = (&mut tensor).into_dl_tensor_mut().unwrap();

        assert_eq!(dl.shape[..], [8, 16]);
        assert!(dl.strides.is_none());

        let managed = dl.to_c();
        assert_eq!(managed.inner.dl_tensor.ndim, 2);
        assert!(!managed.inner.dl_tensor.data.is_null());
        assert!(!managed.inner.dl_tensor.shape.is_null());
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn ndarray_f32_shape_and_dtype() {
        let arr = Array2::<f32>::zeros((100, 128));
        let dl = (&arr).into_dl_tensor().unwrap();

        assert_eq!(dl.shape.len(), 2);
        assert_eq!(dl.shape[..], [100, 128]);

        assert_eq!(dl.dtype.code, ffi::DLDataTypeCode::kDLFloat as u8);
        assert_eq!(dl.dtype.bits, 32);
        assert_eq!(dl.dtype.lanes, 1);
    }

    #[test]
    fn ndarray_contiguous_has_no_strides() {
        let arr = Array2::<f32>::zeros((10, 20));
        let dl = (&arr).into_dl_tensor().unwrap();
        assert!(dl.strides.is_none());
    }

    #[test]
    fn ndarray_transposed_has_strides() {
        let arr = Array2::<f32>::zeros((10, 20));
        let transposed = arr.t();
        let dl = (&transposed).into_dl_tensor().unwrap();

        assert_eq!(dl.shape[..], [20, 10]);
        let strides = dl.strides.as_ref().unwrap();
        assert_eq!(strides[..], [1, 20]);
    }

    #[test]
    fn ndarray_data_ptr_is_non_null() {
        let arr = Array2::<f64>::zeros((4, 4));
        let dl = (&arr).into_dl_tensor().unwrap();
        assert!(!dl.data.is_null());
    }

    #[test]
    fn ndarray_device_is_cpu() {
        let arr = Array2::<f32>::zeros((2, 2));
        let dl = (&arr).into_dl_tensor().unwrap();
        assert_eq!(dl.device.device_type, ffi::DLDeviceType::kDLCPU);
        assert_eq!(dl.device.device_id, 0);
    }

    #[test]
    fn ndarray_byte_offset_is_zero() {
        let arr = Array2::<f32>::zeros((2, 2));
        let dl = (&arr).into_dl_tensor().unwrap();
        let managed = dl.to_c();
        assert_eq!(managed.inner.dl_tensor.byte_offset, 0);
    }

    #[test]
    fn to_c_produces_valid_tensor() {
        let arr = Array2::<f32>::zeros((10, 20));
        let dl = (&arr).into_dl_tensor().unwrap();
        let managed = dl.to_c();

        assert_eq!(managed.inner.dl_tensor.ndim, 2);
        assert!(!managed.inner.dl_tensor.shape.is_null());
        assert!(managed.inner.dl_tensor.strides.is_null());
        assert_eq!(unsafe { *managed.inner.dl_tensor.shape }, 10);
        assert_eq!(unsafe { *managed.inner.dl_tensor.shape.add(1) }, 20);
        assert_eq!(managed.inner.dl_tensor.dtype.bits, 32);
        assert!(managed.inner.manager_ctx.is_null());
        assert!(managed.inner.deleter.is_none());
    }

    #[test]
    fn ndarray_mut_view_requires_mut_ref() {
        let mut arr = Array2::<f32>::zeros((10, 20));
        let dl = (&mut arr).into_dl_tensor_mut().unwrap();

        assert_eq!(dl.shape[..], [10, 20]);
        assert!(dl.strides.is_none());

        let managed = dl.to_c();
        assert_eq!(managed.inner.dl_tensor.ndim, 2);
        assert!(!managed.inner.dl_tensor.data.is_null());
        assert!(!managed.inner.dl_tensor.shape.is_null());
    }

    #[test]
    fn ndarray_mut_view_coerces_to_read_only_view_ref() {
        let mut arr = Array2::<f32>::zeros((4, 5));
        let dl = (&mut arr).into_dl_tensor_mut().unwrap();

        let read_only: &DLTensorView<'_> = &dl;
        assert_eq!(read_only.shape(), &[4, 5]);
    }

    #[test]
    fn borrowed_mut_view_can_convert_into_read_only_view() {
        let mut arr = Array2::<f32>::zeros((6, 7));
        let dl = (&mut arr).into_dl_tensor_mut().unwrap();

        let read_only = (&dl).into_dl_tensor().unwrap();
        assert_eq!(read_only.shape(), &[6, 7]);
    }
}
