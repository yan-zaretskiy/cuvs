/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared filter payloads for nearest-neighbor search APIs.

use std::marker::PhantomData;

use crate::dlpack::{DLPackError, DLTensorView, IntoDlTensor};
use crate::ffi;

/// Error returned when constructing an invalid filter payload.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FilterError {
    /// The filter tensor must be a 1-D vector of packed 32-bit words.
    #[error("filter must be a 1-D tensor")]
    InvalidRank,
    /// The bitset tensor must be in device-accessible memory.
    #[error("filter must use device-accessible memory")]
    InvalidDevice,
    /// The bitset tensor must use scalar (non-vectorized) 32-bit words.
    #[error("filter must use scalar 32-bit words (`u32` or reinterpretable `i32`)")]
    InvalidDType,
    /// The filter tensor must be contiguous (no strides).
    #[error("filter must be contiguous")]
    NonContiguous,
    /// The source tensor could not be converted to a DLPack view.
    #[error(transparent)]
    Conversion(#[from] DLPackError),
}

impl From<std::convert::Infallible> for FilterError {
    fn from(x: std::convert::Infallible) -> Self {
        match x {}
    }
}

/// Marker for a row-level bitset filter.
pub enum Bitset {}

/// Marker for a per-query bitmap filter.
pub enum Bitmap {}

/// Shared search filter options for nearest-neighbor search APIs.
///
/// Support varies by algorithm:
/// - brute force: `Bitset` and `Bitmap`
/// - CAGRA: `Bitset`
/// - IVF-Flat: `Bitset`
pub enum SearchFilter<'a> {
    /// Reuse one row-level bitset for every query.
    Bitset(Filter<'a, Bitset>),
    /// Use a per-query bitmap of allowed `(query, row)` pairs.
    Bitmap(Filter<'a, Bitmap>),
}

/// Type-level mapping from a Rust filter marker to the C `cuvsFilterType`.
pub trait FilterKind {
    const FILTER_TYPE: ffi::cuvsFilterType;
}

impl FilterKind for Bitset {
    const FILTER_TYPE: ffi::cuvsFilterType = ffi::cuvsFilterType::BITSET;
}

impl FilterKind for Bitmap {
    const FILTER_TYPE: ffi::cuvsFilterType = ffi::cuvsFilterType::BITMAP;
}

/// Packed filter words used to include or exclude dataset rows during search.
///
/// The kind parameter determines whether the packed words are interpreted as:
/// - [`Filter<Bitset>`]: one bit per dataset row
/// - [`Filter<Bitmap>`]: one bit per `(query, dataset_row)` pair
///
/// The current CAGRA C bridge expects a device-accessible 1-D vector of
/// `uint32` words. For ergonomic torch integration, `tch` `int32` tensors are
/// also accepted and re-tagged as `uint32` in the stored view metadata.
pub struct Filter<'a, K: FilterKind> {
    tensor: DLTensorView<'a>,
    _kind: PhantomData<K>,
}

impl<'a, K: FilterKind> Filter<'a, K> {
    /// Create a packed filter from a tensor-like input supported by
    /// [`IntoDlTensor`].
    pub fn new<T>(filter_words: T) -> Result<Self, FilterError>
    where
        T: IntoDlTensor<'a>,
    {
        let mut tensor = filter_words.into_dl_tensor()?;

        if tensor.ndim() != 1 {
            return Err(FilterError::InvalidRank);
        }

        if !matches!(
            tensor.device().device_type,
            ffi::DLDeviceType::kDLCUDA
                | ffi::DLDeviceType::kDLCUDAHost
                | ffi::DLDeviceType::kDLCUDAManaged
        ) {
            return Err(FilterError::InvalidDevice);
        }

        if tensor.strides().is_some() {
            return Err(FilterError::NonContiguous);
        }

        let dtype = tensor.dtype();
        let is_32_bit = dtype.bits == 32;
        let is_scalar = dtype.lanes == 1;
        let is_int = (dtype.code == ffi::DLDataTypeCode::kDLInt as u8)
            || (dtype.code == ffi::DLDataTypeCode::kDLUInt as u8);

        if !is_32_bit || !is_scalar || !is_int {
            return Err(FilterError::InvalidDType);
        }

        // Retag i32 metadata as u32 so all later to_c() calls produce the
        // same bitset representation expected by the C API.
        if dtype.code == ffi::DLDataTypeCode::kDLInt as u8 {
            tensor.set_dtype(ffi::DLDataType {
                code: ffi::DLDataTypeCode::kDLUInt as u8,
                bits: 32,
                lanes: 1,
            });
        }

        Ok(Self {
            tensor,
            _kind: PhantomData,
        })
    }
}

impl SearchFilter<'_> {
    /// Build a stack-local [`DLManagedTensor`](ffi::DLManagedTensor) for the
    /// filter payload. The returned [`ManagedTensorRef`] borrows the filter's
    /// view, so the compiler ensures the filter outlives the C struct.
    pub(crate) fn to_c(&self) -> crate::dlpack::ManagedTensorRef<'_> {
        match self {
            Self::Bitset(f) => f.tensor.to_c(),
            Self::Bitmap(f) => f.tensor.to_c(),
        }
    }

    pub(crate) fn filter_type(&self) -> ffi::cuvsFilterType {
        match self {
            Self::Bitset(_) => ffi::cuvsFilterType::BITSET,
            Self::Bitmap(_) => ffi::cuvsFilterType::BITMAP,
        }
    }

    pub(crate) fn uses_bitmap(&self) -> bool {
        matches!(self, Self::Bitmap(_))
    }
}

/// Build a [`cuvsFilter`](ffi::cuvsFilter) from an optional search filter.
///
/// When a filter is present, the caller must pass a `managed_out` slot that
/// will hold the stack-local `DLManagedTensor`; the returned `cuvsFilter`
/// points into it.
pub(crate) fn prepare_filter<'a>(
    filter: Option<&'a SearchFilter<'_>>,
    managed_out: &mut Option<crate::dlpack::ManagedTensorRef<'a>>,
) -> ffi::cuvsFilter {
    match filter {
        Some(f) => {
            let managed = managed_out.insert(f.to_c());
            ffi::cuvsFilter {
                addr: managed.as_mut_ptr() as usize,
                type_: f.filter_type(),
            }
        }
        None => ffi::cuvsFilter {
            addr: 0,
            type_: ffi::cuvsFilterType::NO_FILTER,
        },
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    #[test]
    fn filter_retags_i32_metadata_at_construction() {
        let bitset = tch::Tensor::from_slice(&[0b0110i32]).to(tch::Device::Cuda(0));
        let filter = Filter::<Bitset>::new(&bitset).unwrap();

        let dtype = filter.tensor.dtype();
        assert_eq!(dtype.code, ffi::DLDataTypeCode::kDLUInt as u8);
        assert_eq!(dtype.bits, 32);
        assert_eq!(dtype.lanes, 1);
    }

    #[test]
    fn search_filter_to_c_produces_retagged_managed_tensor() {
        let bitset = tch::Tensor::from_slice(&[0b0110i32]).to(tch::Device::Cuda(0));
        let filter = SearchFilter::Bitset(Filter::<Bitset>::new(&bitset).unwrap());
        let managed = filter.to_c();

        assert_eq!(
            managed.inner.dl_tensor.dtype.code,
            ffi::DLDataTypeCode::kDLUInt as u8
        );
        assert_eq!(filter.filter_type(), ffi::cuvsFilterType::BITSET);
    }
}
