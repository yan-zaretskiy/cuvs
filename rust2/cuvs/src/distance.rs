/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Distance metrics for nearest neighbor search and pairwise distance.

use ordered_float::OrderedFloat;

use crate::dlpack::{AsDLTensor, AsMutDLTensor, DLTensorFfi};
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

/// Fills `distances` with pairwise distances between rows of `x` and rows of `y`.
///
/// The C API expects `x` as `[n, k]`, `y` as `[m, k]`, and `distances` as a
/// pre-allocated `[n, m]` matrix (row-major layout per DLPack). Device placement
/// and dtypes must match what cuVS supports for the chosen [`DistanceType`].
/// Use [`DistanceType::LpUnexpanded`] to provide the Minkowski exponent `p`;
/// all other metrics use the C API default argument of `2.0`.
pub fn pairwise_distance(
    res: &Resources,
    x: &impl AsDLTensor,
    y: &impl AsDLTensor,
    distances: &impl AsMutDLTensor,
    metric: DistanceType,
) -> Result<(), LibraryError> {
    let status = unsafe {
        ffi::cuvsPairwiseDistance(
            res.handle(),
            x.ffi_ptr(),
            y.ffi_ptr(),
            distances.ffi_ptr(),
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
    use crate::dlpack::{BorrowedDLTensor, MutBorrowedDLTensor};
    use crate::resources::Resources;

    #[test]
    fn pairwise_distance_l2_sqrt_matches_torch_cdist() {
        let n = 16i64;
        let m = 24i64;
        let k = 8i64;
        let x = tch::Tensor::randn([n, k], (tch::Kind::Float, tch::Device::Cuda(0)));
        let y = tch::Tensor::randn([m, k], (tch::Kind::Float, tch::Device::Cuda(0)));
        let distances =
            tch::Tensor::zeros([n, m], (tch::Kind::Float, tch::Device::Cuda(0)));

        let res = Resources::new().unwrap();
        let x_dl = BorrowedDLTensor::try_from(&x).unwrap();
        let y_dl = BorrowedDLTensor::try_from(&y).unwrap();
        let dist_dl = MutBorrowedDLTensor::try_from(&distances).unwrap();
        pairwise_distance(
            &res,
            &x_dl,
            &y_dl,
            &dist_dl,
            DistanceType::L2SqrtUnexpanded,
        )
        .unwrap();

        let expected = tch::Tensor::cdist(&x, &y, 2.0, None::<i64>);
        let n_el = (n * m) as usize;
        let mut got = vec![0f32; n_el];
        let mut exp = vec![0f32; n_el];
        distances.copy_data(&mut got, n_el);
        expected.copy_data(&mut exp, n_el);
        for i in 0..n_el {
            assert!(
                (got[i] - exp[i]).abs() < 1e-3,
                "row mismatch at {i}: got {} expected {}",
                got[i],
                exp[i]
            );
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
        let distances = tch::Tensor::zeros([n, m], (tch::Kind::Float, tch::Device::Cuda(0)));

        let res = Resources::new().unwrap();
        let x_dl = BorrowedDLTensor::try_from(&x).unwrap();
        let y_dl = BorrowedDLTensor::try_from(&y).unwrap();
        let dist_dl = MutBorrowedDLTensor::try_from(&distances).unwrap();
        pairwise_distance(
            &res,
            &x_dl,
            &y_dl,
            &dist_dl,
            DistanceType::LpUnexpanded(OrderedFloat(p as f32)),
        )
        .unwrap();

        let expected = tch::Tensor::cdist(&x, &y, p, None::<i64>);
        let n_el = (n * m) as usize;
        let mut got = vec![0f32; n_el];
        let mut exp = vec![0f32; n_el];
        distances.copy_data(&mut got, n_el);
        expected.copy_data(&mut exp, n_el);
        for i in 0..n_el {
            assert!(
                (got[i] - exp[i]).abs() < 1e-3,
                "row mismatch at {i}: got {} expected {}",
                got[i],
                exp[i]
            );
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
