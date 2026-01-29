using Test
using LinearAlgebra
using SquareRootCovarianceMatrices

@testset "PSDMatrix Operations" begin
    # Construct two positive definite matrices
    A_mat = [2.0 1.0; 1.0 2.0]
    B_mat = [3.0 0.5; 0.5 1.5]
    A = PSDMatrix(A_mat)
    B = PSDMatrix(B_mat)

    # Test addition
    C = A + B
    @test Matrix(C) ≈ A_mat + B_mat

    # Test scalar multiplication
    α = 2.0
    D = α * A
    @test Matrix(D) ≈ α * A_mat

    # Test lmul!
    A2 = PSDMatrix(A_mat)
    lmul!(α, A2)
    @test Matrix(A2) ≈ α * A_mat

    # Test X_A_Xt!
    X = [1.0 2.0; 0.0 1.0]
    A3 = PSDMatrix(A_mat)
    X_A_Xt!(A3, X)
    @test Matrix(A3) ≈ X * A_mat * X'

    # Test trace
    @test isapprox(tr(A), tr(A_mat))
    @test isapprox(tr(A), tr(A_mat))
end
