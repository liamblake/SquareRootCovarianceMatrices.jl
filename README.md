# SquareRootCovarianceMatrices

[![Build Status](https://github.com/liamblake/SquareRootCovarianceMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/liamblake/SquareRootCovarianceMatrices.jl/actions/workflows/CI.yml?query=branch%3Amain)

For any matrix $S \in \mathbb{R}^{n \times m}$, the matrix $P = S S^T$ is positive semidefinite (PSD). This esoteric package implements a positive semidefinite matrix type by storing the square root matrix $S$ instead of the full matrix $P$.



*Copilot was used to help write the test suite for this package.*


