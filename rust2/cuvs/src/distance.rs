/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Distance metrics for nearest neighbor search.

use crate::ffi;

/// Distance metric used for building and searching nearest neighbor indices.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum DistanceType {
    /// L2 (squared Euclidean) distance.
    L2Expanded,
    /// L2 distance with square root.
    L2SqrtExpanded,
    /// Cosine distance.
    CosineExpanded,
    /// L1 (Manhattan) distance.
    L1,
    /// L2 distance (unexpanded form).
    L2Unexpanded,
    /// L2 distance with square root (unexpanded form).
    L2SqrtUnexpanded,
    /// Inner product.
    InnerProduct,
    /// Chebyshev (L-infinity) distance.
    Linf,
    /// Canberra distance.
    Canberra,
    /// Generalized Minkowski (Lp) distance.
    LpUnexpanded,
    /// Correlation distance.
    CorrelationExpanded,
    /// Jaccard distance.
    JaccardExpanded,
    /// Hellinger distance.
    HellingerExpanded,
    /// Haversine (great-circle) distance.
    Haversine,
    /// Bray-Curtis distance.
    BrayCurtis,
    /// Jensen-Shannon divergence.
    JensenShannon,
    /// Hamming distance.
    HammingUnexpanded,
    /// Kullback-Leibler divergence.
    KLDivergence,
    /// Russell-Rao distance.
    RusselRaoExpanded,
    /// Dice-Sorensen distance.
    DiceExpanded,
    /// Bitwise Hamming distance.
    BitwiseHamming,
    /// Precomputed distance matrix.
    Precomputed,
}

impl From<DistanceType> for ffi::cuvsDistanceType {
    fn from(v: DistanceType) -> Self {
        use DistanceType::*;
        match v {
            L2Expanded => Self::L2Expanded,
            L2SqrtExpanded => Self::L2SqrtExpanded,
            CosineExpanded => Self::CosineExpanded,
            L1 => Self::L1,
            L2Unexpanded => Self::L2Unexpanded,
            L2SqrtUnexpanded => Self::L2SqrtUnexpanded,
            InnerProduct => Self::InnerProduct,
            Linf => Self::Linf,
            Canberra => Self::Canberra,
            LpUnexpanded => Self::LpUnexpanded,
            CorrelationExpanded => Self::CorrelationExpanded,
            JaccardExpanded => Self::JaccardExpanded,
            HellingerExpanded => Self::HellingerExpanded,
            Haversine => Self::Haversine,
            BrayCurtis => Self::BrayCurtis,
            JensenShannon => Self::JensenShannon,
            HammingUnexpanded => Self::HammingUnexpanded,
            KLDivergence => Self::KLDivergence,
            RusselRaoExpanded => Self::RusselRaoExpanded,
            DiceExpanded => Self::DiceExpanded,
            BitwiseHamming => Self::BitwiseHamming,
            Precomputed => Self::Precomputed,
        }
    }
}

impl From<ffi::cuvsDistanceType> for DistanceType {
    fn from(v: ffi::cuvsDistanceType) -> Self {
        use ffi::cuvsDistanceType::*;
        match v {
            L2Expanded => Self::L2Expanded,
            L2SqrtExpanded => Self::L2SqrtExpanded,
            CosineExpanded => Self::CosineExpanded,
            L1 => Self::L1,
            L2Unexpanded => Self::L2Unexpanded,
            L2SqrtUnexpanded => Self::L2SqrtUnexpanded,
            InnerProduct => Self::InnerProduct,
            Linf => Self::Linf,
            Canberra => Self::Canberra,
            LpUnexpanded => Self::LpUnexpanded,
            CorrelationExpanded => Self::CorrelationExpanded,
            JaccardExpanded => Self::JaccardExpanded,
            HellingerExpanded => Self::HellingerExpanded,
            Haversine => Self::Haversine,
            BrayCurtis => Self::BrayCurtis,
            JensenShannon => Self::JensenShannon,
            HammingUnexpanded => Self::HammingUnexpanded,
            KLDivergence => Self::KLDivergence,
            RusselRaoExpanded => Self::RusselRaoExpanded,
            DiceExpanded => Self::DiceExpanded,
            BitwiseHamming => Self::BitwiseHamming,
            Precomputed => Self::Precomputed,
        }
    }
}
