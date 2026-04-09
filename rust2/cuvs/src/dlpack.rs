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
//! * [`DLTensorViewMut`] — a writable view created from mutable or
//!   interior-mutable tensor handles. Use for C API parameters that *write*
//!   results (neighbors, distances).
//!
//! * [`ReturnedDLTensor`] — a non-owning view returned by the cuVS C API that
//!   owns the returned DLPack metadata and runs its deleter on drop.
//!
//! The traits [`IntoDlTensor`] and [`IntoDlTensorMut`] are the public entry
//! point for adapting external tensor types into these views. Custom backends
//! can implement those traits directly, typically by calling the unsafe
//! [`DLTensorView::from_raw_parts`] or [`DLTensorViewMut::from_raw_parts`]
//! constructors from a small, well-audited conversion layer.

use std::cell::UnsafeCell;
use std::convert::Infallible;
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
/// Most cuVS bindings use 1-D/2-D tensors, with IVF-PQ precomputed codebooks
/// requiring a 3-D tensor view.
const MAX_DIMS: usize = 3;

pub(crate) type TensorDims = ArrayVec<[i64; MAX_DIMS]>;

/// A public conversion trait for read-only tensor inputs.
pub trait IntoDlTensor<'a> {
    fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError>;
}

/// A public conversion trait for writable tensor inputs.
pub trait IntoDlTensorMut<'a> {
    fn into_dl_tensor_mut(self) -> Result<DLTensorViewMut<'a>, DLPackError>;
}

impl<'a, T> IntoDlTensor<'a> for T
where
    DLTensorView<'a>: TryFrom<T>,
    DLPackError: From<<DLTensorView<'a> as TryFrom<T>>::Error>,
{
    fn into_dl_tensor(self) -> Result<DLTensorView<'a>, DLPackError> {
        DLTensorView::try_from(self).map_err(Into::into)
    }
}

impl<'a, T> IntoDlTensorMut<'a> for T
where
    DLTensorViewMut<'a>: TryFrom<T>,
    DLPackError: From<<DLTensorViewMut<'a> as TryFrom<T>>::Error>,
{
    fn into_dl_tensor_mut(self) -> Result<DLTensorViewMut<'a>, DLPackError> {
        DLTensorViewMut::try_from(self).map_err(Into::into)
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
    /// The tensor rank exceeds what cuVS currently supports.
    #[error("unsupported tensor rank: {0}")]
    UnsupportedRank(usize),
    /// A strides slice did not match the tensor rank.
    #[error("strides length {strides} does not match tensor rank {ndim}")]
    StridesLenMismatch { ndim: usize, strides: usize },
    /// The source tensor reported invalid DLPack metadata.
    #[error("invalid DLPack metadata: {0}")]
    InvalidMetadata(&'static str),
}

// Infallible `From` conversions auto-derive `TryFrom<_, Error = Infallible>`;
// this lets the blanket `IntoDlTensor` impl treat them uniformly.
impl From<Infallible> for DLPackError {
    fn from(x: Infallible) -> Self {
        match x {}
    }
}

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

fn view_parts(
    data: *mut std::ffi::c_void,
    device: ffi::DLDevice,
    shape: &[i64],
    strides: Option<&[i64]>,
    dtype: ffi::DLDataType,
) -> Result<
    (
        TensorDims,
        Option<TensorDims>,
        UnsafeCell<ffi::DLManagedTensor>,
    ),
    DLPackError,
> {
    if shape.len() > MAX_DIMS {
        return Err(DLPackError::UnsupportedRank(shape.len()));
    }
    let shape: TensorDims = shape.iter().copied().collect();

    let strides = match strides {
        Some(values) if values.len() != shape.len() => Err(DLPackError::StridesLenMismatch {
            ndim: shape.len(),
            strides: values.len(),
        }),
        Some(values) => Ok(Some(values.iter().copied().collect())),
        None => Ok(None),
    }?;

    let managed = new_managed_tensor(data, device, shape.len() as i32, dtype);
    Ok((shape, strides, managed))
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
    // SAFETY: UnsafeCell permits interior mutation. Only the shape/strides
    // pointer fields are written; the arrays they point into are owned by the
    // enclosing struct and are not modified.
    unsafe {
        (*ptr).dl_tensor.shape = shape.as_ptr() as *mut _;
        (*ptr).dl_tensor.strides = match strides {
            Some(s) => s.as_ptr() as *mut _,
            None => std::ptr::null_mut(),
        };
    }
    ptr
}

fn new_dl_tensor_view<'a>(
    shape: TensorDims,
    strides: Option<TensorDims>,
    managed: UnsafeCell<ffi::DLManagedTensor>,
) -> DLTensorView<'a> {
    DLTensorView {
        shape,
        strides,
        managed,
        _marker: PhantomData,
    }
}

fn new_dl_tensor_view_mut<'a>(
    shape: TensorDims,
    strides: Option<TensorDims>,
    managed: UnsafeCell<ffi::DLManagedTensor>,
) -> DLTensorViewMut<'a> {
    DLTensorViewMut {
        shape,
        strides,
        managed,
        _marker: PhantomData,
    }
}

// ---------------------------------------------------------------------------
// DLTensorView — read-only view for C API inputs
// ---------------------------------------------------------------------------

/// A non-owning, read-only DLPack tensor view.
///
/// Suitable for C API parameters that only *read* the data
/// (e.g. datasets, queries).
pub struct DLTensorView<'a> {
    shape: TensorDims,
    strides: Option<TensorDims>,
    managed: UnsafeCell<ffi::DLManagedTensor>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> DLTensorView<'a> {
    /// Construct a read-only DLPack view from raw tensor metadata.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that:
    /// - `data` points to initialized tensor storage for the full extent
    ///   described by `shape`, `strides`, and `dtype`
    /// - `device`, `shape`, `strides`, and `dtype` accurately describe that
    ///   storage
    /// - the underlying storage remains valid and immutable for the lifetime
    ///   associated with the returned view
    pub unsafe fn from_raw_parts(
        data: *mut std::ffi::c_void,
        device: ffi::DLDevice,
        shape: &[i64],
        strides: Option<&[i64]>,
        dtype: ffi::DLDataType,
    ) -> Result<Self, DLPackError> {
        let (shape, strides, managed) = view_parts(data, device, shape, strides, dtype)?;
        Ok(new_dl_tensor_view(shape, strides, managed))
    }

    fn from_managed_tensor(managed: &ffi::DLManagedTensor) -> Result<Self, DLPackError> {
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
            Self::from_raw_parts(
                managed.dl_tensor.data,
                managed.dl_tensor.device,
                shape,
                strides,
                managed.dl_tensor.dtype,
            )
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut ffi::DLManagedTensor {
        bind_dl_managed_ptr(&self.managed, &self.shape, &self.strides)
    }

    pub(crate) fn dl_tensor(&self) -> &ffi::DLTensor {
        unsafe { &(*self.as_ptr()).dl_tensor }
    }

    #[cfg(test)]
    fn managed_ref(&self) -> &ffi::DLManagedTensor {
        unsafe { &*self.managed.get() }
    }
}

impl<'a> From<&'a DLTensorView<'a>> for DLTensorView<'a> {
    fn from(view: &'a DLTensorView<'a>) -> Self {
        let dl = view.dl_tensor();
        let managed = new_managed_tensor(dl.data, dl.device, dl.ndim, dl.dtype);
        new_dl_tensor_view(view.shape, view.strides, managed)
    }
}

impl<'a> From<DLTensorViewMut<'a>> for DLTensorView<'a> {
    fn from(view: DLTensorViewMut<'a>) -> Self {
        new_dl_tensor_view(view.shape, view.strides, view.managed)
    }
}

impl<'a> From<&'a DLTensorViewMut<'a>> for DLTensorView<'a> {
    fn from(view: &'a DLTensorViewMut<'a>) -> Self {
        let dl = view.dl_tensor();
        let managed = new_managed_tensor(dl.data, dl.device, dl.ndim, dl.dtype);
        new_dl_tensor_view(view.shape, view.strides, managed)
    }
}

impl<'a> TryFrom<&'a ReturnedDLTensor<'a>> for DLTensorView<'a> {
    type Error = DLPackError;

    fn try_from(value: &'a ReturnedDLTensor<'a>) -> Result<Self, Self::Error> {
        Self::from_managed_tensor(&value.managed)
    }
}

impl fmt::Debug for DLTensorView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DLTensorView")
            .field("shape", &self.shape.as_slice())
            .field("strides", &self.strides.as_ref().map(|s| s.as_slice()))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DLTensorViewMut — writable view for C API outputs
// ---------------------------------------------------------------------------

/// A non-owning, writable DLPack tensor view.
///
/// Constructed from a mutable reference (`&mut ndarray::ArrayBase`) or from a
/// shared `&tch::Tensor` (PyTorch tensors have interior mutability). Suitable
/// for C API parameters that *write* results (e.g. neighbors, distances).
pub struct DLTensorViewMut<'a> {
    shape: TensorDims,
    strides: Option<TensorDims>,
    managed: UnsafeCell<ffi::DLManagedTensor>,
    _marker: PhantomData<&'a mut ()>,
}

impl<'a> DLTensorViewMut<'a> {
    /// Construct a writable DLPack view from raw tensor metadata.
    ///
    /// # Safety
    ///
    /// In addition to the [`DLTensorView::from_raw_parts`] invariants, the data
    /// region described by the tensor must be exclusively writable for the
    /// lifetime associated with the returned view.
    pub unsafe fn from_raw_parts(
        data: *mut std::ffi::c_void,
        device: ffi::DLDevice,
        shape: &[i64],
        strides: Option<&[i64]>,
        dtype: ffi::DLDataType,
    ) -> Result<Self, DLPackError> {
        let (shape, strides, managed) = view_parts(data, device, shape, strides, dtype)?;
        Ok(new_dl_tensor_view_mut(shape, strides, managed))
    }

    pub(crate) fn as_ptr(&self) -> *mut ffi::DLManagedTensor {
        bind_dl_managed_ptr(&self.managed, &self.shape, &self.strides)
    }

    pub(crate) fn dl_tensor(&self) -> &ffi::DLTensor {
        unsafe { &(*self.as_ptr()).dl_tensor }
    }
}

impl fmt::Debug for DLTensorViewMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DLTensorViewMut")
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

    impl<'a, A, S, D> From<&'a ndarray::ArrayBase<S, D>> for DLTensorView<'a>
    where
        A: DType,
        S: ndarray::Data<Elem = A>,
        D: ndarray::Dimension,
    {
        fn from(arr: &'a ndarray::ArrayBase<S, D>) -> Self {
            let shape: Vec<i64> = arr.shape().iter().map(|&d| d as i64).collect();
            let strides: Option<Vec<i64>> = if arr.is_standard_layout() {
                None
            } else {
                Some(arr.strides().iter().map(|&s| s as i64).collect())
            };
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
                .expect("ndarray shape and strides must be valid DLPack metadata")
            }
        }
    }

    impl<'a, A, S, D> From<&'a mut ndarray::ArrayBase<S, D>> for DLTensorViewMut<'a>
    where
        A: DType,
        S: ndarray::DataMut<Elem = A>,
        D: ndarray::Dimension,
    {
        fn from(arr: &'a mut ndarray::ArrayBase<S, D>) -> Self {
            let shape: Vec<i64> = arr.shape().iter().map(|&d| d as i64).collect();
            let strides: Option<Vec<i64>> = if arr.is_standard_layout() {
                None
            } else {
                Some(arr.strides().iter().map(|&s| s as i64).collect())
            };
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
                .expect("ndarray shape and strides must be valid DLPack metadata")
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

    /// # Caution
    ///
    /// `tch::Tensor` has interior mutability. Do not call in-place operations
    /// that may reallocate storage (e.g., `resize_`) on the source tensor while
    /// a [`DLTensorView`] derived from it is alive.
    impl<'a> TryFrom<&'a tch::Tensor> for DLTensorView<'a> {
        type Error = DLPackError;

        fn try_from(tensor: &'a tch::Tensor) -> Result<Self, Self::Error> {
            let shape = tensor.size();
            let strides = if tensor.is_contiguous() {
                None
            } else {
                Some(tensor.stride())
            };
            unsafe {
                DLTensorView::from_raw_parts(
                    tensor.data_ptr() as *mut _,
                    device_to_dl(tensor.device())?,
                    &shape,
                    strides.as_deref(),
                    kind_to_dl_dtype(tensor.kind()),
                )
            }
        }
    }

    /// PyTorch tensors use refcounted C++ storage with interior mutability, so
    /// a shared `&tch::Tensor` is sufficient for writable views.
    ///
    /// # Caution
    ///
    /// Do not call in-place operations that may reallocate storage
    /// (e.g., `resize_`) on the source tensor while a [`DLTensorViewMut`]
    /// derived from it is alive.
    impl<'a> TryFrom<&'a tch::Tensor> for DLTensorViewMut<'a> {
        type Error = DLPackError;

        fn try_from(tensor: &'a tch::Tensor) -> Result<Self, Self::Error> {
            let shape = tensor.size();
            let strides = if tensor.is_contiguous() {
                None
            } else {
                Some(tensor.stride())
            };
            unsafe {
                DLTensorViewMut::from_raw_parts(
                    tensor.data_ptr() as *mut _,
                    device_to_dl(tensor.device())?,
                    &shape,
                    strides.as_deref(),
                    kind_to_dl_dtype(tensor.kind()),
                )
            }
        }
    }
}

#[cfg(all(test, feature = "torch"))]
mod torch_tests {
    use super::*;

    #[test]
    fn torch_f32_shape_device_and_dtype() {
        let tensor = tch::Tensor::zeros([100, 128], (tch::Kind::Float, tch::Device::Cpu));
        let dl = DLTensorView::try_from(&tensor).unwrap();

        assert_eq!(dl.shape.len(), 2);
        assert_eq!(dl.shape[..], [100, 128]);
        assert!(dl.strides.is_none());
        assert_eq!(
            dl.managed_ref().dl_tensor.device.device_type,
            ffi::DLDeviceType::kDLCPU
        );
        assert_eq!(dl.managed_ref().dl_tensor.device.device_id, 0);
        assert_eq!(
            dl.managed_ref().dl_tensor.dtype.code,
            ffi::DLDataTypeCode::kDLFloat as u8
        );
        assert_eq!(dl.managed_ref().dl_tensor.dtype.bits, 32);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.lanes, 1);
    }

    #[test]
    fn torch_transposed_cpu_tensor_has_strides() {
        let tensor = tch::Tensor::zeros([10, 20], (tch::Kind::Float, tch::Device::Cpu));
        let transposed = tensor.transpose(0, 1);
        let dl = DLTensorView::try_from(&transposed).unwrap();

        assert_eq!(dl.shape[..], [20, 10]);
        let strides = dl.strides.as_ref().unwrap();
        assert_eq!(strides[..], [1, 20]);
    }

    #[test]
    fn torch_bool_dtype_maps_to_dl_bool() {
        let tensor = tch::Tensor::zeros([2, 2], (tch::Kind::Bool, tch::Device::Cpu));
        let dl = DLTensorView::try_from(&tensor).unwrap();

        assert_eq!(
            dl.managed_ref().dl_tensor.dtype.code,
            ffi::DLDataTypeCode::kDLBool as u8
        );
        assert_eq!(dl.managed_ref().dl_tensor.dtype.bits, 8);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.lanes, 1);
    }

    #[test]
    fn torch_as_ptr_produces_valid_cpu_tensor() {
        let tensor = tch::Tensor::zeros([10, 20], (tch::Kind::Float, tch::Device::Cpu));
        let dl = DLTensorView::try_from(&tensor).unwrap();
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
    fn torch_mut_view_from_tensor() {
        let tensor = tch::Tensor::zeros([8, 16], (tch::Kind::Float, tch::Device::Cpu));
        let dl = DLTensorViewMut::try_from(&tensor).unwrap();

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
        let dl = DLTensorView::from(&arr);

        assert_eq!(dl.shape.len(), 2);
        assert_eq!(dl.shape[..], [100, 128]);
        assert_eq!(
            dl.managed_ref().dl_tensor.dtype.code,
            ffi::DLDataTypeCode::kDLFloat as u8
        );
        assert_eq!(dl.managed_ref().dl_tensor.dtype.bits, 32);
        assert_eq!(dl.managed_ref().dl_tensor.dtype.lanes, 1);
    }

    #[test]
    fn ndarray_contiguous_has_no_strides() {
        let arr = Array2::<f32>::zeros((10, 20));
        let dl = DLTensorView::from(&arr);
        assert!(dl.strides.is_none());
    }

    #[test]
    fn ndarray_transposed_has_strides() {
        let arr = Array2::<f32>::zeros((10, 20));
        let transposed = arr.t();
        let dl = DLTensorView::from(&transposed);

        assert_eq!(dl.shape[..], [20, 10]);
        let strides = dl.strides.as_ref().unwrap();
        assert_eq!(strides[..], [1, 20]);
    }

    #[test]
    fn ndarray_data_ptr_is_non_null() {
        let arr = Array2::<f64>::zeros((4, 4));
        let dl = DLTensorView::from(&arr);
        assert!(!dl.managed_ref().dl_tensor.data.is_null());
    }

    #[test]
    fn ndarray_device_is_cpu() {
        let arr = Array2::<f32>::zeros((2, 2));
        let dl = DLTensorView::from(&arr);
        assert_eq!(
            dl.managed_ref().dl_tensor.device.device_type,
            ffi::DLDeviceType::kDLCPU
        );
        assert_eq!(dl.managed_ref().dl_tensor.device.device_id, 0);
    }

    #[test]
    fn ndarray_byte_offset_is_zero() {
        let arr = Array2::<f32>::zeros((2, 2));
        let dl = DLTensorView::from(&arr);
        assert_eq!(dl.managed_ref().dl_tensor.byte_offset, 0);
    }

    #[test]
    fn as_ptr_produces_valid_tensor() {
        let arr = Array2::<f32>::zeros((10, 20));
        let dl = DLTensorView::from(&arr);
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
    fn ndarray_mut_view_requires_mut_ref() {
        let mut arr = Array2::<f32>::zeros((10, 20));
        let dl = DLTensorViewMut::from(&mut arr);

        assert_eq!(dl.shape[..], [10, 20]);
        assert!(dl.strides.is_none());

        let ptr = dl.as_ptr();
        let managed = unsafe { &*ptr };
        assert_eq!(managed.dl_tensor.ndim, 2);
        assert!(!managed.dl_tensor.data.is_null());
        assert!(!managed.dl_tensor.shape.is_null());
    }
}
