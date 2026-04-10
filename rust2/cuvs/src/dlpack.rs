/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! DLPack tensor view types.
//!
//! Three tensor view types are provided:
//!
//! * [`DLTensorView`] — a read-only view created from tensor-like inputs.
//!   Use for C API parameters that only *read* data (datasets, queries).
//!
//! * [`DLTensorViewMut`] — a writable view created from mutable tensor handles.
//!   Use for C API parameters that *write* results (neighbors, distances).
//!
//! * [`ReturnedDLTensor`] — a non-owning view returned by the cuVS C API that
//!   owns the returned DLPack metadata and runs its deleter on drop.
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
/// Returned by [`DLTensorView::to_c`] / [`DLTensorViewMut::to_c`].  The
/// lifetime parameter ensures the view (which owns the shape and strides
/// arrays that the C struct points into) outlives this handle.
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
// Shared view implementation
// ---------------------------------------------------------------------------

/// Generates a DLPack tensor view struct with the shared constructor and
/// metadata-to-C conversion method.
///
/// Both [`DLTensorView`] and [`DLTensorViewMut`] have identical layouts and
/// core operations; only their `PhantomData` marker (and therefore the safety
/// contract on the referenced data) differs.  Type-specific impls live outside
/// this macro.
macro_rules! dl_tensor_view {
    (
        $(#[$meta:meta])*
        pub struct $name:ident<'a>($marker:ty);
    ) => {
        $(#[$meta])*
        pub struct $name<'a> {
            pub(crate) data: *mut std::ffi::c_void,
            pub(crate) device: ffi::DLDevice,
            pub(crate) dtype: ffi::DLDataType,
            pub(crate) shape: TensorDims,
            pub(crate) strides: Option<TensorDims>,
            _marker: PhantomData<$marker>,
        }

        impl<'a> $name<'a> {
            /// Construct a DLPack view from raw tensor metadata.
            ///
            /// # Safety
            ///
            /// The caller must guarantee that:
            /// - `data` points to initialized tensor storage for the full extent
            ///   described by `shape`, `strides`, and `dtype`
            /// - `device`, `shape`, `strides`, and `dtype` accurately describe
            ///   that storage
            /// - the underlying storage remains valid for the lifetime `'a`:
            ///   immutable for [`DLTensorView`], exclusively writable for
            ///   [`DLTensorViewMut`]
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
                        strides: self.strides.as_ref()
                            .map_or(std::ptr::null_mut(), |s| s.as_ptr() as *mut _),
                        byte_offset: 0,
                    },
                    manager_ctx: std::ptr::null_mut(),
                    deleter: None,
                },
                _borrow: PhantomData,
                }
            }
        }

        impl fmt::Debug for $name<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("shape", &self.shape.as_slice())
                    .field("strides", &self.strides.as_deref())
                    .finish()
            }
        }
    };
}

// ---------------------------------------------------------------------------
// DLTensorView — read-only view for C API inputs
// ---------------------------------------------------------------------------

dl_tensor_view! {
    /// A non-owning, read-only DLPack tensor view.
    ///
    /// Suitable for C API parameters that only *read* the data
    /// (e.g. datasets, queries).
    #[must_use]
    pub struct DLTensorView<'a>(&'a ());
}

// ---------------------------------------------------------------------------
// DLTensorViewMut — writable view for C API outputs
// ---------------------------------------------------------------------------

dl_tensor_view! {
    /// A non-owning, writable DLPack tensor view.
    ///
    /// Constructed from a mutable reference (`&mut ndarray::ArrayRef` or
    /// `&mut tch::Tensor`). Suitable for C API parameters that *write* results
    /// (e.g. neighbors, distances).
    #[must_use]
    pub struct DLTensorViewMut<'a>(&'a mut ());
}

// ---------------------------------------------------------------------------
// Conversions between view types
// ---------------------------------------------------------------------------

impl<'a> From<DLTensorViewMut<'a>> for DLTensorView<'a> {
    fn from(view: DLTensorViewMut<'a>) -> Self {
        Self {
            data: view.data,
            device: view.device,
            dtype: view.dtype,
            shape: view.shape,
            strides: view.strides,
            _marker: PhantomData,
        }
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

impl<'a> IntoDlTensor<'a> for &'a ReturnedDLTensor<'a> {
    fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError> {
        let managed = &self.managed;
        let ndim = usize::try_from(managed.dl_tensor.ndim)
            .map_err(|_| DLPackError::InvalidMetadata("negative ndim"))?;
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
#[must_use]
pub struct ReturnedDLTensor<'a> {
    managed: ffi::DLManagedTensor,
    _owner: PhantomData<&'a ()>,
}

impl<'a> ReturnedDLTensor<'a> {
    /// # Safety
    ///
    /// The `init` closure must fully initialize the `DLManagedTensor` when
    /// it returns `CUVS_SUCCESS`.  In particular, `ndim`, `shape`, `strides`,
    /// `device`, and `dtype` must describe valid, accessible memory.
    pub(crate) unsafe fn from_ffi(
        init: impl FnOnce(*mut ffi::DLManagedTensor) -> ffi::cuvsError_t,
    ) -> Result<Self, LibraryError> {
        let mut managed = MaybeUninit::<ffi::DLManagedTensor>::zeroed();
        check_cuvs(init(managed.as_mut_ptr()))?;
        Ok(Self {
            // SAFETY: the caller's contract guarantees the closure fully
            // initialized the struct on the success path.
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

    fn array_view<'a, A, D>(
        arr: &'a ndarray::ArrayRef<A, D>,
    ) -> Result<DLTensorView<'a>, DLPackError>
    where
        A: DType,
        D: ndarray::Dimension,
    {
        // TensorDims keeps ≤3 dims on the stack (covers the common case).
        let shape: TensorDims = arr.shape().iter().map(|&d| d as i64).collect();
        let strides: Option<TensorDims> = if arr.is_standard_layout() {
            None
        } else {
            Some(arr.strides().iter().map(|&s| s as i64).collect())
        };
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
        let shape: TensorDims = arr.shape().iter().map(|&d| d as i64).collect();
        let strides: Option<TensorDims> = if arr.is_standard_layout() {
            None
        } else {
            Some(arr.strides().iter().map(|&s| s as i64).collect())
        };
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

    fn tensor_view<'a>(tensor: &'a tch::Tensor) -> Result<DLTensorView<'a>, DLPackError> {
        let shape: TensorDims = tensor.size().into_iter().collect();
        let strides: Option<TensorDims> = if tensor.is_contiguous() {
            None
        } else {
            Some(tensor.stride().into_iter().collect())
        };
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
        let shape: TensorDims = tensor.size().into_iter().collect();
        let strides: Option<TensorDims> = if tensor.is_contiguous() {
            None
        } else {
            Some(tensor.stride().into_iter().collect())
        };
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
    fn returned_tensor_from_ffi_zeroes_unwritten_fields() {
        // SAFETY: the closure fully initializes every DLManagedTensor field.
        let returned = unsafe {
            ReturnedDLTensor::from_ffi(|ptr| {
                (*ptr).dl_tensor.data = std::ptr::null_mut();
                (*ptr).dl_tensor.device = ffi::DLDevice {
                    device_type: ffi::DLDeviceType::kDLCPU,
                    device_id: 0,
                };
                (*ptr).dl_tensor.ndim = 0;
                (*ptr).dl_tensor.dtype = ffi::DLDataType {
                    code: ffi::DLDataTypeCode::kDLFloat as u8,
                    bits: 32,
                    lanes: 1,
                };
                (*ptr).dl_tensor.shape = std::ptr::null_mut();
                (*ptr).dl_tensor.strides = std::ptr::null_mut();
                (*ptr).deleter = None;
                ffi::cuvsError_t::CUVS_SUCCESS
            })
        }
        .unwrap();

        assert_eq!(returned.managed.dl_tensor.byte_offset, 0);
        assert!(returned.managed.manager_ctx.is_null());
    }
}
