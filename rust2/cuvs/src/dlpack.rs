/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! DLPack tensor view types.
//!
//! Three tensor view types are provided:
//!
//! * [`BorrowedDLTensor`] — a read-only view created from `&T`.
//!   Use for C API parameters that only *read* data (datasets, queries).
//!
//! * [`MutBorrowedDLTensor`] — a writable view created from `&mut T` (ndarray)
//!   or `&T` (PyTorch, which has interior mutability).
//!   Use for C API parameters that *write* results (neighbors, distances).
//!
//! * [`ReturnedDLTensor`] — a non-owning view returned by the cuVS C API that
//!   owns the returned DLPack metadata and runs its deleter on drop.
//!
//! The traits [`AsDLTensor`] and [`AsMutDLTensor`] describe which wrappers are
//! valid in read-only vs writable C API positions.

use std::cell::UnsafeCell;
use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::slice;

use tinyvec::ArrayVec;

use crate::error::{LibraryError, check_cuvs};
use crate::ffi;

pub use ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLTensor};

/// Maximum tensor dimensions for stack-allocated shape/strides buffers.
///
/// The cuVS C API only uses 1-D vectors and 2-D matrices.
const MAX_DIMS: usize = 2;

pub(crate) type TensorDims = ArrayVec<[i64; MAX_DIMS]>;

/// A read-only DLPack tensor source that can be passed to the cuVS C API.
///
/// # Safety
///
/// The returned pointer must point to a valid [`DLManagedTensor`] whose:
/// - `data` pointer is valid, properly aligned for the declared dtype, and
///   points to initialised memory for the full extent described by `shape`,
///   `strides`, and `dtype`
/// - `shape` (and optional `strides`) arrays have `ndim` elements and remain
///   valid for the lifetime of the `&self` borrow
/// - `device` accurately describes where the data resides
///
/// All of the above — including the `data` pointer — must remain valid and
/// unmodified for the lifetime of the `&self` borrow.  Callers must not
/// reallocate, move, or free the underlying memory during this period.
pub unsafe trait AsDLTensor {
    fn as_dl_tensor(&self) -> *const ffi::DLManagedTensor;
}

/// A writable DLPack tensor source that can be used for cuVS C API outputs.
///
/// # Safety
///
/// In addition to the [`AsDLTensor`] invariants, the data region described
/// by the tensor must be exclusively writable for the duration of the
/// C API call.
pub unsafe trait AsMutDLTensor: AsDLTensor {
    fn as_mut_dl_tensor(&self) -> *mut ffi::DLManagedTensor {
        self.as_dl_tensor().cast_mut()
    }
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
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Uniform `.ffi_ptr()` for passing any tensor to the C API (which takes
/// non-const `DLManagedTensor*` everywhere).
pub(crate) trait DLTensorFfi: AsDLTensor {
    fn ffi_ptr(&self) -> *mut ffi::DLManagedTensor {
        self.as_dl_tensor().cast_mut()
    }
}

impl<T: AsDLTensor + ?Sized> DLTensorFfi for T {}

fn new_managed_tensor(
    data: *mut std::ffi::c_void,
    device: ffi::DLDevice,
    ndim: i32,
    dtype: ffi::DLDataType,
) -> UnsafeCell<ffi::DLManagedTensor> {
    UnsafeCell::new(ffi::DLManagedTensor {
        dl_tensor: ffi::DLTensor {
            data,
            device,
            ndim,
            dtype,
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    })
}

/// Bind shape/strides into the managed tensor and return the raw pointer.
///
/// Called each time before passing to C because the struct's address may have
/// changed since construction (the shape/strides arrays are self-referential).
fn bind_dl_managed_ptr(
    managed: &UnsafeCell<ffi::DLManagedTensor>,
    shape: &TensorDims,
    strides: &Option<TensorDims>,
) -> *mut ffi::DLManagedTensor {
    let ptr = managed.get();
    // SAFETY: UnsafeCell permits interior mutation.  Only the shape/strides
    // *pointer* fields are written; the arrays they point into are owned by
    // the enclosing struct and are not modified.
    unsafe {
        (*ptr).dl_tensor.shape = shape.as_ptr() as *mut _;
        (*ptr).dl_tensor.strides = match strides {
            Some(s) => s.as_ptr() as *mut _,
            None => std::ptr::null_mut(),
        };
    }
    ptr
}

// ---------------------------------------------------------------------------
// BorrowedDLTensor — read-only view for C API inputs
// ---------------------------------------------------------------------------

/// A non-owning, read-only DLPack tensor view.
///
/// Suitable for C API parameters that only *read* the data
/// (e.g. datasets, queries).
pub struct BorrowedDLTensor<'a> {
    shape: TensorDims,
    strides: Option<TensorDims>,
    managed: UnsafeCell<ffi::DLManagedTensor>,
    _marker: PhantomData<&'a ()>,
}

impl BorrowedDLTensor<'_> {
    /// Return a raw pointer suitable for passing to the C API.
    pub(crate) fn as_ptr(&self) -> *mut ffi::DLManagedTensor {
        bind_dl_managed_ptr(&self.managed, &self.shape, &self.strides)
    }

    #[cfg(test)]
    fn managed_ref(&self) -> &ffi::DLManagedTensor {
        unsafe { &*self.managed.get() }
    }
}

// SAFETY: `as_ptr()` returns a valid pointer into this struct's `UnsafeCell`
// with correctly initialised DLPack metadata. The pointee lives as long as
// `&self` because `BorrowedDLTensor` owns the `UnsafeCell`.
unsafe impl AsDLTensor for BorrowedDLTensor<'_> {
    fn as_dl_tensor(&self) -> *const ffi::DLManagedTensor {
        self.as_ptr()
    }
}

impl fmt::Debug for BorrowedDLTensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedDLTensor")
            .field("shape", &self.shape.as_slice())
            .field("strides", &self.strides.as_ref().map(|s| s.as_slice()))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// MutBorrowedDLTensor — writable view for C API outputs
// ---------------------------------------------------------------------------

/// A non-owning, writable DLPack tensor view.
///
/// Constructed from a mutable reference (`&mut ndarray::ArrayRef`) or from a
/// `&tch::Tensor` (PyTorch tensors have interior mutability).  Suitable for
/// C API parameters that *write* results (e.g. neighbors, distances).
pub struct MutBorrowedDLTensor<'a> {
    shape: TensorDims,
    strides: Option<TensorDims>,
    managed: UnsafeCell<ffi::DLManagedTensor>,
    _marker: PhantomData<&'a mut ()>,
}

impl MutBorrowedDLTensor<'_> {
    /// Return a raw pointer suitable for passing to the C API.
    pub(crate) fn as_ptr(&self) -> *mut ffi::DLManagedTensor {
        bind_dl_managed_ptr(&self.managed, &self.shape, &self.strides)
    }
}

// SAFETY: same as BorrowedDLTensor — valid pointer from owned UnsafeCell.
unsafe impl AsDLTensor for MutBorrowedDLTensor<'_> {
    fn as_dl_tensor(&self) -> *const ffi::DLManagedTensor {
        self.as_ptr()
    }
}

// SAFETY: MutBorrowedDLTensor is only constructed from exclusively-borrowed
// (&mut ndarray) or interior-mutable (&tch::Tensor) data, so writing
// through the data pointer is sound.
unsafe impl AsMutDLTensor for MutBorrowedDLTensor<'_> {}

impl fmt::Debug for MutBorrowedDLTensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MutBorrowedDLTensor")
            .field("shape", &self.shape.as_slice())
            .field("strides", &self.strides.as_ref().map(|s| s.as_slice()))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ReturnedDLTensor — non-owning view returned from the C API
// ---------------------------------------------------------------------------

/// A non-owning DLPack tensor view returned by the C API.
///
/// The underlying data is owned elsewhere (for example, by an index), while
/// this wrapper owns only the returned `DLManagedTensor` metadata and calls the
/// provided DLPack deleter on drop.
pub struct ReturnedDLTensor<'a> {
    managed: ffi::DLManagedTensor,
    _owner: PhantomData<&'a ()>,
}

// SAFETY: the managed tensor was initialised by the C API in `from_ffi`.
// `addr_of!` produces a valid, properly-aligned pointer that lives as long
// as `&self`.
unsafe impl AsDLTensor for ReturnedDLTensor<'_> {
    fn as_dl_tensor(&self) -> *const ffi::DLManagedTensor {
        std::ptr::addr_of!(self.managed)
    }
}

impl<'a> ReturnedDLTensor<'a> {
    pub(crate) fn from_ffi(
        init: impl FnOnce(*mut ffi::DLManagedTensor) -> ffi::cuvsError_t,
    ) -> Result<Self, LibraryError> {
        let mut managed = MaybeUninit::<ffi::DLManagedTensor>::uninit();
        check_cuvs(init(managed.as_mut_ptr()))?;
        Ok(Self {
            managed: unsafe { managed.assume_init() },
            _owner: PhantomData,
        })
    }

    pub fn ndim(&self) -> i32 {
        self.managed.dl_tensor.ndim
    }

    pub fn shape(&self) -> &[i64] {
        let n = self.ndim();
        if n <= 0 {
            return &[];
        }
        unsafe { slice::from_raw_parts(self.managed.dl_tensor.shape, n as usize) }
    }

    pub fn strides(&self) -> Option<&[i64]> {
        let n = self.ndim();
        if self.managed.dl_tensor.strides.is_null() || n <= 0 {
            return None;
        }
        Some(unsafe { slice::from_raw_parts(self.managed.dl_tensor.strides, n as usize) })
    }

    pub fn dtype(&self) -> ffi::DLDataType {
        self.managed.dl_tensor.dtype
    }

    pub fn device(&self) -> ffi::DLDevice {
        self.managed.dl_tensor.device
    }
}

impl fmt::Debug for ReturnedDLTensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReturnedDLTensor")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("ndim", &self.ndim())
            .finish()
    }
}

impl Drop for ReturnedDLTensor<'_> {
    fn drop(&mut self) {
        if let Some(deleter) = self.managed.deleter {
            unsafe { deleter(&mut self.managed) };
        }
    }
}

// ---------------------------------------------------------------------------
// ndarray implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::*;

    fn array_metadata<A: DType, D: ndarray::Dimension>(
        arr: &ndarray::ArrayRef<A, D>,
    ) -> (TensorDims, Option<TensorDims>, i32, ffi::DLDataType) {
        let shape: TensorDims = arr.shape().iter().map(|&d| d as i64).collect();
        let strides: Option<TensorDims> = if arr.is_standard_layout() {
            None
        } else {
            Some(arr.strides().iter().map(|&s| s as i64).collect())
        };
        let ndim = shape.len() as i32;
        (shape, strides, ndim, A::dl_dtype())
    }

    impl<'a, A, D> From<&'a ndarray::ArrayRef<A, D>> for BorrowedDLTensor<'a>
    where
        A: DType,
        D: ndarray::Dimension,
    {
        fn from(arr: &'a ndarray::ArrayRef<A, D>) -> Self {
            let (shape, strides, ndim, dtype) = array_metadata(arr);
            let managed = new_managed_tensor(
                arr.as_ptr() as *mut _,
                ffi::DLDevice {
                    device_type: ffi::DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                ndim,
                dtype,
            );
            BorrowedDLTensor {
                shape,
                strides,
                managed,
                _marker: PhantomData,
            }
        }
    }

    impl<'a, A, D> From<&'a mut ndarray::ArrayRef<A, D>> for MutBorrowedDLTensor<'a>
    where
        A: DType,
        D: ndarray::Dimension,
    {
        fn from(arr: &'a mut ndarray::ArrayRef<A, D>) -> Self {
            let (shape, strides, ndim, dtype) = array_metadata(arr);
            let managed = new_managed_tensor(
                arr.as_mut_ptr() as *mut _,
                ffi::DLDevice {
                    device_type: ffi::DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                ndim,
                dtype,
            );
            MutBorrowedDLTensor {
                shape,
                strides,
                managed,
                _marker: PhantomData,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// torch (tch-rs / PyTorch) implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "torch")]
mod tch_impl {
    use super::*;

    fn kind_to_dl_dtype(kind: tch::Kind) -> ffi::DLDataType {
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
            _ => (ffi::DLDataTypeCode::kDLOpaqueHandle, 0),
        };
        ffi::DLDataType {
            code: code as u8,
            bits,
            lanes: 1,
        }
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

    fn tensor_to_parts(
        tensor: &tch::Tensor,
    ) -> Result<(TensorDims, Option<TensorDims>, UnsafeCell<ffi::DLManagedTensor>), DLPackError> {
        let shape: TensorDims = tensor.size().into_iter().collect();
        let strides: Option<TensorDims> = if tensor.is_contiguous() {
            None
        } else {
            Some(tensor.stride().into_iter().collect())
        };
        let ndim = shape.len() as i32;
        let managed = new_managed_tensor(
            tensor.data_ptr() as *mut _,
            device_to_dl(tensor.device())?,
            ndim,
            kind_to_dl_dtype(tensor.kind()),
        );
        Ok((shape, strides, managed))
    }

    /// # Caution
    ///
    /// `tch::Tensor` has interior mutability.  Do not call in-place operations
    /// that may reallocate storage (e.g., `resize_`) on the source tensor
    /// while a [`BorrowedDLTensor`] derived from it is alive.
    impl<'a> TryFrom<&'a tch::Tensor> for BorrowedDLTensor<'a> {
        type Error = DLPackError;

        fn try_from(tensor: &'a tch::Tensor) -> Result<Self, Self::Error> {
            let (shape, strides, managed) = tensor_to_parts(tensor)?;
            Ok(BorrowedDLTensor {
                shape,
                strides,
                managed,
                _marker: PhantomData,
            })
        }
    }

    /// PyTorch tensors use refcounted C++ storage with interior mutability,
    /// so a shared `&tch::Tensor` is sufficient for writable views.
    ///
    /// # Caution
    ///
    /// Do not call in-place operations that may reallocate storage
    /// (e.g., `resize_`) on the source tensor while a
    /// [`MutBorrowedDLTensor`] derived from it is alive.
    impl<'a> TryFrom<&'a tch::Tensor> for MutBorrowedDLTensor<'a> {
        type Error = DLPackError;

        fn try_from(tensor: &'a tch::Tensor) -> Result<Self, Self::Error> {
            let (shape, strides, managed) = tensor_to_parts(tensor)?;
            Ok(MutBorrowedDLTensor {
                shape,
                strides,
                managed,
                _marker: PhantomData,
            })
        }
    }
}

#[cfg(all(test, feature = "torch"))]
mod torch_tests {
    use super::*;

    #[test]
    fn torch_f32_shape_device_and_dtype() {
        let tensor = tch::Tensor::zeros([100, 128], (tch::Kind::Float, tch::Device::Cpu));
        let dl = BorrowedDLTensor::try_from(&tensor).unwrap();

        assert_eq!(dl.shape.len(), 2);
        assert_eq!(dl.shape[..], [100, 128]);
        assert!(dl.strides.is_none());
        assert_eq!(dl.managed_ref().dl_tensor.device.device_type, ffi::DLDeviceType::kDLCPU);
        assert_eq!(dl.managed_ref().dl_tensor.device.device_id, 0);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.code, ffi::DLDataTypeCode::kDLFloat as u8);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.bits, 32);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.lanes, 1);
    }

    #[test]
    fn torch_transposed_cpu_tensor_has_strides() {
        let tensor = tch::Tensor::zeros([10, 20], (tch::Kind::Float, tch::Device::Cpu));
        let transposed = tensor.transpose(0, 1);
        let dl = BorrowedDLTensor::try_from(&transposed).unwrap();

        assert_eq!(dl.shape[..], [20, 10]);
        let strides = dl.strides.as_ref().unwrap();
        assert_eq!(strides[..], [1, 20]);
    }

    #[test]
    fn torch_bool_dtype_maps_to_dl_bool() {
        let tensor = tch::Tensor::zeros([2, 2], (tch::Kind::Bool, tch::Device::Cpu));
        let dl = BorrowedDLTensor::try_from(&tensor).unwrap();

        assert_eq!(dl.managed_ref().dl_tensor.dtype.code, ffi::DLDataTypeCode::kDLBool as u8);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.bits, 8);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.lanes, 1);
    }

    #[test]
    fn torch_as_ptr_produces_valid_cpu_tensor() {
        let tensor = tch::Tensor::zeros([10, 20], (tch::Kind::Float, tch::Device::Cpu));
        let dl = BorrowedDLTensor::try_from(&tensor).unwrap();
        let ptr = dl.as_ptr();

        let managed = unsafe { &*ptr };
        assert_eq!(managed.dl_tensor.ndim, 2);
        assert!(!managed.dl_tensor.data.is_null());
        assert!(!managed.dl_tensor.shape.is_null());
        assert!(managed.dl_tensor.strides.is_null());
        assert_eq!(unsafe { *managed.dl_tensor.shape }, 10);
        assert_eq!(unsafe { *managed.dl_tensor.shape.add(1) }, 20);
        assert_eq!(managed.dl_tensor.byte_offset, 0);
        assert!(managed.manager_ctx.is_null());
        assert!(managed.deleter.is_none());
    }

    #[test]
    fn torch_mut_borrowed_from_tensor() {
        let tensor = tch::Tensor::zeros([8, 16], (tch::Kind::Float, tch::Device::Cpu));
        let dl = MutBorrowedDLTensor::try_from(&tensor).unwrap();

        assert_eq!(dl.shape[..], [8, 16]);
        assert!(dl.strides.is_none());

        let ptr = dl.as_ptr();
        let managed = unsafe { &*ptr };
        assert_eq!(managed.dl_tensor.ndim, 2);
        assert!(!managed.dl_tensor.data.is_null());
        assert!(!managed.dl_tensor.shape.is_null());
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn ndarray_f32_shape_and_dtype() {
        let arr = Array2::<f32>::zeros((100, 128));
        let dl = BorrowedDLTensor::from(&*arr);

        assert_eq!(dl.shape.len(), 2);
        assert_eq!(dl.shape[..], [100, 128]);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.code, ffi::DLDataTypeCode::kDLFloat as u8);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.bits, 32);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.lanes, 1);
    }

    #[test]
    fn ndarray_contiguous_has_no_strides() {
        let arr = Array2::<f32>::zeros((10, 20));
        let dl = BorrowedDLTensor::from(&*arr);
        assert!(dl.strides.is_none());
    }

    #[test]
    fn ndarray_transposed_has_strides() {
        let arr = Array2::<f32>::zeros((10, 20));
        let transposed = arr.t();
        let dl = BorrowedDLTensor::from(&*transposed);

        assert_eq!(dl.shape[..], [20, 10]);
        let strides = dl.strides.as_ref().unwrap();
        assert_eq!(strides[..], [1, 20]);
    }

    #[test]
    fn ndarray_data_ptr_is_non_null() {
        let arr = Array2::<f64>::zeros((4, 4));
        let dl = BorrowedDLTensor::from(&*arr);
        assert!(!dl.managed_ref().dl_tensor.data.is_null());
    }

    #[test]
    fn ndarray_device_is_cpu() {
        let arr = Array2::<f32>::zeros((2, 2));
        let dl = BorrowedDLTensor::from(&*arr);
        assert_eq!(dl.managed_ref().dl_tensor.device.device_type, ffi::DLDeviceType::kDLCPU);
        assert_eq!(dl.managed_ref().dl_tensor.device.device_id, 0);
    }

    #[test]
    fn ndarray_byte_offset_is_zero() {
        let arr = Array2::<f32>::zeros((2, 2));
        let dl = BorrowedDLTensor::from(&*arr);
        assert_eq!(dl.managed_ref().dl_tensor.byte_offset, 0);
    }

    #[test]
    fn as_ptr_produces_valid_tensor() {
        let arr = Array2::<f32>::zeros((10, 20));
        let dl = BorrowedDLTensor::from(&*arr);
        let ptr = dl.as_ptr();

        let managed = unsafe { &*ptr };
        assert_eq!(managed.dl_tensor.ndim, 2);
        assert!(!managed.dl_tensor.shape.is_null());
        assert!(managed.dl_tensor.strides.is_null());
        assert_eq!(unsafe { *managed.dl_tensor.shape }, 10);
        assert_eq!(unsafe { *managed.dl_tensor.shape.add(1) }, 20);
        assert_eq!(managed.dl_tensor.dtype.bits, 32);
        assert!(managed.manager_ctx.is_null());
        assert!(managed.deleter.is_none());
    }

    #[test]
    fn ndarray_mut_borrowed_requires_mut_ref() {
        let mut arr = Array2::<f32>::zeros((10, 20));
        let dl = MutBorrowedDLTensor::from(&mut *arr);

        assert_eq!(dl.shape[..], [10, 20]);
        assert!(dl.strides.is_none());

        let ptr = dl.as_ptr();
        let managed = unsafe { &*ptr };
        assert_eq!(managed.dl_tensor.ndim, 2);
        assert!(!managed.dl_tensor.data.is_null());
        assert!(!managed.dl_tensor.shape.is_null());
    }
}
