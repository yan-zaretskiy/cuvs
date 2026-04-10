/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Distance metrics for nearest neighbor search and pairwise distance.

use ordered_float::OrderedFloat;

use crate::dlpack::{DLPackError, IntoDlTensor, IntoDlTensorMut};
use crate::error::{LibraryError, check_cuvs};
use crate::ffi;
use crate::resources::Resources;

const DEFAULT_METRIC_ARG: f32 = 2.0;

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
    /// Generalized Minkowski (Lp) distance with exponent `p`.
    LpUnexpanded(OrderedFloat<f32>),
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

impl DistanceType {
    pub(crate) fn metric_arg(self) -> f32 {
        match self {
            Self::LpUnexpanded(p) => p.into_inner(),
            _ => DEFAULT_METRIC_ARG,
        }
    }
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
            LpUnexpanded(_) => Self::LpUnexpanded,
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
            LpUnexpanded => Self::LpUnexpanded(OrderedFloat(DEFAULT_METRIC_ARG)),
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

/// Error type for pairwise distance operations.
#[derive(Debug, thiserror::Error)]
pub enum DistanceError {
    /// The C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
}

/// Fills `distances` with pairwise distances between rows of `x` and rows of `y`.
///
/// The C API expects `x` as `[n, k]`, `y` as `[m, k]`, and `distances` as a
/// pre-allocated `[n, m]` matrix (row-major layout per DLPack). Device placement
/// and dtypes must match what cuVS supports for the chosen [`DistanceType`].
/// Use [`DistanceType::LpUnexpanded`] to provide the Minkowski exponent `p`;
/// all other metrics use the C API default argument of `2.0`.
pub fn pairwise_distance<'x, 'y, 'd, X, Y, Dist>(
    res: &Resources,
    x: X,
    y: Y,
    distances: Dist,
    metric: DistanceType,
) -> Result<(), DistanceError>
where
    X: IntoDlTensor<'x>,
    Y: IntoDlTensor<'y>,
    Dist: IntoDlTensorMut<'d>,
{
    let x = x.into_dl_tensor()?;
    let y = y.into_dl_tensor()?;
    let distances = distances.into_dl_tensor_mut()?;
    let status = unsafe {
        ffi::cuvsPairwiseDistance(
            res.handle(),
            x.as_ptr(),
            y.as_ptr(),
            distances.as_ptr(),
            metric.into(),
            metric.metric_arg(),
        )
    };
    check_cuvs(status)?;
    Ok(())
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;
    use crate::resources::Resources;

    #[test]
    fn pairwise_distance_l2_sqrt_matches_torch_cdist() {
        let n = 16i64;
        let m = 24i64;
        let k = 8i64;
        let x = tch::Tensor::randn([n, k], (tch::Kind::Float, tch::Device::Cuda(0)));
        let y = tch::Tensor::randn([m, k], (tch::Kind::Float, tch::Device::Cuda(0)));
        let mut distances = tch::Tensor::zeros([n, m], (tch::Kind::Float, tch::Device::Cuda(0)));

        let res = Resources::new().unwrap();
        pairwise_distance(&res, &x, &y, &mut distances, DistanceType::L2SqrtUnexpanded).unwrap();

        let expected = tch::Tensor::cdist(&x, &y, 2.0, None::<i64>);
        let got: Vec<Vec<f32>> = Vec::try_from(&distances).unwrap();
        let exp: Vec<Vec<f32>> = Vec::try_from(&expected).unwrap();
        for (row_idx, (got_row, exp_row)) in got.iter().zip(&exp).enumerate() {
            for (col_idx, (got_val, exp_val)) in got_row.iter().zip(exp_row).enumerate() {
                let i = row_idx * exp_row.len() + col_idx;
                assert!(
                    (*got_val - *exp_val).abs() < 1e-3,
                    "row mismatch at {i}: got {} expected {}",
                    got_val,
                    exp_val
                );
            }
        }
    }

    #[test]
    fn pairwise_distance_lp_matches_torch_cdist() {
        let n = 10i64;
        let m = 12i64;
        let k = 6i64;
        let p = 3.0f64;
        let x = tch::Tensor::randn([n, k], (tch::Kind::Float, tch::Device::Cuda(0)));
        let y = tch::Tensor::randn([m, k], (tch::Kind::Float, tch::Device::Cuda(0)));
        let mut distances = tch::Tensor::zeros([n, m], (tch::Kind::Float, tch::Device::Cuda(0)));

        let res = Resources::new().unwrap();
        pairwise_distance(
            &res,
            &x,
            &y,
            &mut distances,
            DistanceType::LpUnexpanded(OrderedFloat(p as f32)),
        )
        .unwrap();

        let expected = tch::Tensor::cdist(&x, &y, p, None::<i64>);
        let got: Vec<Vec<f32>> = Vec::try_from(&distances).unwrap();
        let exp: Vec<Vec<f32>> = Vec::try_from(&expected).unwrap();
        for (row_idx, (got_row, exp_row)) in got.iter().zip(&exp).enumerate() {
            for (col_idx, (got_val, exp_val)) in got_row.iter().zip(exp_row).enumerate() {
                let i = row_idx * exp_row.len() + col_idx;
                assert!(
                    (*got_val - *exp_val).abs() < 1e-3,
                    "row mismatch at {i}: got {} expected {}",
                    got_val,
                    exp_val
                );
            }
        }
    }

    #[test]
    fn non_lp_metrics_use_default_metric_arg() {
        assert_eq!(DistanceType::L2Expanded.metric_arg(), DEFAULT_METRIC_ARG);
        assert_eq!(
            DistanceType::LpUnexpanded(OrderedFloat(3.5)).metric_arg(),
            3.5
        );
        assert_eq!(
            DistanceType::from(ffi::cuvsDistanceType::LpUnexpanded),
            DistanceType::LpUnexpanded(OrderedFloat(DEFAULT_METRIC_ARG))
        );
    }
}
