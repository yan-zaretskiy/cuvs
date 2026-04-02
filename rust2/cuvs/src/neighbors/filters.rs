/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared filter payloads for nearest-neighbor search APIs.

use std::marker::PhantomData;

use crate::dlpack::BorrowedDLTensor;
use crate::ffi;

/// Error returned when constructing an invalid filter payload.
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    /// The filter tensor must be a 1-D vector of packed 32-bit words.
    #[error("filter must be a 1-D tensor")]
    InvalidRank,
    /// The bitset tensor must be in device-accessible memory.
    #[error("filter must use device-accessible memory")]
    InvalidDevice,
    /// The bitset tensor must use 32-bit words.
    #[error("filter must use 32-bit words (`u32` or reinterpretable `i32`)")]
    InvalidDType,
}

/// Marker for a row-level bitset filter.
pub enum Bitset {}

/// Marker for a per-query bitmap filter.
pub enum Bitmap {}

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
/// also accepted and re-tagged as `uint32` in the temporary DLPack metadata
/// passed to the C API.
pub struct Filter<'a, K: FilterKind> {
    tensor: BorrowedDLTensor<'a>,
    _kind: PhantomData<K>,
}

impl<'a, K: FilterKind> Filter<'a, K> {
    /// Create a packed filter from a tensor-like input supported by
    /// [`BorrowedDLTensor`].
    pub fn new<T>(filter_words: &'a T) -> Result<Self, FilterError>
    where
        BorrowedDLTensor<'a>: From<&'a T>,
    {
        let tensor = BorrowedDLTensor::from(filter_words);
        let ptr = tensor.as_ptr();
        let dl = unsafe { &(*ptr).dl_tensor };

        if dl.ndim != 1 {
            return Err(FilterError::InvalidRank);
        }

        if !matches!(
            dl.device.device_type,
            ffi::DLDeviceType::kDLCUDA
                | ffi::DLDeviceType::kDLCUDAHost
                | ffi::DLDeviceType::kDLCUDAManaged
        ) {
            return Err(FilterError::InvalidDevice);
        }

        let is_32_bit = dl.dtype.bits == 32;
        let is_int = (dl.dtype.code == ffi::DLDataTypeCode::kDLInt as u8)
            || (dl.dtype.code == ffi::DLDataTypeCode::kDLUInt as u8);

        if !is_32_bit || !is_int {
            return Err(FilterError::InvalidDType);
        }

        Ok(Self {
            tensor,
            _kind: PhantomData,
        })
    }

    pub(crate) fn as_cuvs_filter(&self) -> ffi::cuvsFilter {
        let ptr = self.tensor.as_ptr();
        let dl = unsafe { &mut (*ptr).dl_tensor };

        if dl.dtype.code == ffi::DLDataTypeCode::kDLInt as u8 && dl.dtype.bits == 32 {
            dl.dtype = ffi::DLDataType {
                code: ffi::DLDataTypeCode::kDLUInt as u8,
                bits: 32,
                lanes: 1,
            };
        }

        ffi::cuvsFilter {
            addr: ptr as usize,
            type_: K::FILTER_TYPE,
        }
    }
}
