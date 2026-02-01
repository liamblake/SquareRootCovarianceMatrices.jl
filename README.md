# SquareRootCovarianceMatrices

[![Build Status](https://github.com/liamblake/SquareRootCovarianceMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/liamblake/SquareRootCovarianceMatrices.jl/actions/workflows/CI.yml?query=branch%3Amain)

For any matrix $S \in \mathbb{R}^{n \times m}$, the matrix $P = S S^T$ is positive semidefinite (PSD). This esoteric package implements a positive semidefinite matrix type by storing the square root matrix $S$ instead of the full matrix $P$.

*Copilot was used to help write the test suite for this package.*

## Operations
The following operations are implemented for `PSDMatrix` objects, which is entirely informed by my selfish needs and use-cases:

- Downdate of a PSD matrix by a low-rank update (i.e. computing $A - \alpha u u^T$): `X_A_Xt!(A::PSDMatrix, u::AbstractVector, α::Real = 1.0)`.


