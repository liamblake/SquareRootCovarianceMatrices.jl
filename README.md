# SquareRootCovarianceMatrices

[![Build Status](https://github.com/liamblake/SquareRootCovarianceMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/liamblake/SquareRootCovarianceMatrices.jl/actions/workflows/CI.yml?query=branch%3Amain)

For any matrix $S \in \mathbb{R}^{n \times m}$, the matrix $P = S S^T$ is positive semidefinite (PSD). This esoteric package implements a positive semidefinite matrix type by storing the square root matrix $S$ instead of the full matrix $P$.

*Copilot was used to help write the test suite for this package.*

## Operations
The following in-place operations are implemented for `PSDMatrix` objects, which is entirely informed by my selfish needs and use-cases:
- Multiplication by a scalar ($a X$): `LinearAlgebra.lmul!(a::Real, A::PSDMatrix)`
- Congruence transformation ($X P X^T$ for an arbitrary matrix $X$): `X_A_Xt!(A::PSDMatrix, X::AbstractMatrix)`.
- Downdate of a PSD matrix by a low-rank update (i.e. computing $A - a u u^T$): `downdate!(A::PSDMatrix, u::AbstractVector, a::Real = 1.0)`.
- Calculation of the $k$-largest eigenvalues and eigenvectors: `eigs!(A::PSDMatrix; k::Int = 1)`
- Calculation of the $k$-largest right singular values and vectors of $S^T S$: `rsvd!(A::PSDMatrix; k::Int = 1)`.
