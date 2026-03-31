/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! DLPack tensor view trait and implementations.

use std::marker::PhantomData;

use tinyvec::ArrayVec;

use crate::ffi;

/// Maximum tensor dimensions for stack-allocated shape/strides buffers.
///
/// The cuVS C API only uses 1-D vectors and 2-D matrices.
const MAX_DIMS: usize = 2;

pub(crate) type TensorDims = ArrayVec<[i64; MAX_DIMS]>;

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

pub struct BorrowedDLTensor<'a> {
    shape: TensorDims,
    strides: Option<TensorDims>,
    managed: ffi::DLManagedTensor,
    _marker: PhantomData<&'a ()>,
}

impl<'a> BorrowedDLTensor<'a> {
    /// Return a pointer to the inner `DLManagedTensor`, binding the
    /// shape/strides pointers to their current addresses.
    pub fn as_mut_ptr(&mut self) -> *mut ffi::DLManagedTensor {
        self.managed.dl_tensor.shape = self.shape.as_mut_ptr();
        self.managed.dl_tensor.strides = match &mut self.strides {
            Some(s) => s.as_mut_ptr(),
            None => std::ptr::null_mut(),
        };
        &mut self.managed
    }
}

// ---------------------------------------------------------------------------
// ndarray implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::*;

    impl<'a, A, D> From<&'a ndarray::ArrayRef<A, D>> for BorrowedDLTensor<'a>
    where
        A: DType,
        D: ndarray::Dimension,
    {
        fn from(arr: &'a ndarray::ArrayRef<A, D>) -> Self {
            let shape: TensorDims = arr.shape().iter().map(|&d| d as i64).collect();
            let strides: Option<TensorDims> = if arr.is_standard_layout() {
                None
            } else {
                Some(arr.strides().iter().map(|&s| s as i64).collect())
            };
            let ndim = shape.len() as i32;
            BorrowedDLTensor {
                shape,
                strides,
                managed: ffi::DLManagedTensor {
                    dl_tensor: ffi::DLTensor {
                        data: arr.as_ptr() as *mut _,
                        device: ffi::DLDevice {
                            device_type: ffi::DLDeviceType::kDLCPU,
                            device_id: 0,
                        },
                        ndim,
                        dtype: A::dl_dtype(),
                        // Null until as_mut_ptr() binds them.
                        shape: std::ptr::null_mut(),
                        strides: std::ptr::null_mut(),
                        byte_offset: 0,
                    },
                    manager_ctx: std::ptr::null_mut(),
                    deleter: None,
                },
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

    fn device_to_dl(device: tch::Device) -> ffi::DLDevice {
        match device {
            tch::Device::Cpu => ffi::DLDevice {
                device_type: ffi::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            tch::Device::Cuda(id) => ffi::DLDevice {
                device_type: ffi::DLDeviceType::kDLCUDA,
                device_id: id as i32,
            },
            _ => unimplemented!("unsupported tch device: {:?}", device),
        }
    }

    impl<'a> From<&'a tch::Tensor> for BorrowedDLTensor<'a> {
        fn from(tensor: &'a tch::Tensor) -> Self {
            let shape: TensorDims = tensor.size().into_iter().collect();
            let strides: Option<TensorDims> = if tensor.is_contiguous() {
                None
            } else {
                Some(tensor.stride().into_iter().collect())
            };
            let ndim = shape.len() as i32;
            BorrowedDLTensor {
                shape,
                strides,
                managed: ffi::DLManagedTensor {
                    dl_tensor: ffi::DLTensor {
                        data: tensor.data_ptr() as *mut _,
                        device: device_to_dl(tensor.device()),
                        ndim,
                        dtype: kind_to_dl_dtype(tensor.kind()),
                        shape: std::ptr::null_mut(),
                        strides: std::ptr::null_mut(),
                        byte_offset: 0,
                    },
                    manager_ctx: std::ptr::null_mut(),
                    deleter: None,
                },
                _marker: PhantomData,
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
        let dl = BorrowedDLTensor::from(&tensor);

        assert_eq!(dl.shape.len(), 2);
        assert_eq!(dl.shape[..], [100, 128]);
        assert!(dl.strides.is_none());
        assert_eq!(dl.managed.dl_tensor.device.device_type, ffi::DLDeviceType::kDLCPU);
        assert_eq!(dl.managed.dl_tensor.device.device_id, 0);
        assert_eq!(dl.managed.dl_tensor.dtype.code, ffi::DLDataTypeCode::kDLFloat as u8);
        assert_eq!(dl.managed.dl_tensor.dtype.bits, 32);
        assert_eq!(dl.managed.dl_tensor.dtype.lanes, 1);
    }

    #[test]
    fn torch_transposed_cpu_tensor_has_strides() {
        let tensor = tch::Tensor::zeros([10, 20], (tch::Kind::Float, tch::Device::Cpu));
        let transposed = tensor.transpose(0, 1);
        let dl = BorrowedDLTensor::from(&transposed);

        assert_eq!(dl.shape[..], [20, 10]);
        let strides = dl.strides.as_ref().unwrap();
        assert_eq!(strides[..], [1, 20]);
    }

    #[test]
    fn torch_bool_dtype_maps_to_dl_bool() {
        let tensor = tch::Tensor::zeros([2, 2], (tch::Kind::Bool, tch::Device::Cpu));
        let dl = BorrowedDLTensor::from(&tensor);

        assert_eq!(dl.managed.dl_tensor.dtype.code, ffi::DLDataTypeCode::kDLBool as u8);
        assert_eq!(dl.managed.dl_tensor.dtype.bits, 8);
        assert_eq!(dl.managed.dl_tensor.dtype.lanes, 1);
    }

    #[test]
    fn torch_as_mut_ptr_produces_valid_cpu_tensor() {
        let tensor = tch::Tensor::zeros([10, 20], (tch::Kind::Float, tch::Device::Cpu));
        let mut dl = BorrowedDLTensor::from(&tensor);
        let ptr = dl.as_mut_ptr();

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
        assert_eq!(dl.managed.dl_tensor.dtype.code, ffi::DLDataTypeCode::kDLFloat as u8);
        assert_eq!(dl.managed.dl_tensor.dtype.bits, 32);
        assert_eq!(dl.managed.dl_tensor.dtype.lanes, 1);
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
        assert!(!dl.managed.dl_tensor.data.is_null());
    }

    #[test]
    fn ndarray_device_is_cpu() {
        let arr = Array2::<f32>::zeros((2, 2));
        let dl = BorrowedDLTensor::from(&*arr);
        assert_eq!(dl.managed.dl_tensor.device.device_type, ffi::DLDeviceType::kDLCPU);
        assert_eq!(dl.managed.dl_tensor.device.device_id, 0);
    }

    #[test]
    fn ndarray_byte_offset_is_zero() {
        let arr = Array2::<f32>::zeros((2, 2));
        let dl = BorrowedDLTensor::from(&*arr);
        assert_eq!(dl.managed.dl_tensor.byte_offset, 0);
    }

    #[test]
    fn as_mut_ptr_produces_valid_tensor() {
        let arr = Array2::<f32>::zeros((10, 20));
        let mut dl = BorrowedDLTensor::from(&*arr);
        let ptr = dl.as_mut_ptr();

        let managed = unsafe { &*ptr };
        assert_eq!(managed.dl_tensor.ndim, 2);
        assert!(!managed.dl_tensor.shape.is_null());
        assert!(managed.dl_tensor.strides.is_null()); // contiguous
        assert_eq!(unsafe { *managed.dl_tensor.shape }, 10);
        assert_eq!(unsafe { *managed.dl_tensor.shape.add(1) }, 20);
        assert_eq!(managed.dl_tensor.dtype.bits, 32);
        assert!(managed.manager_ctx.is_null());
        assert!(managed.deleter.is_none());
    }
}
