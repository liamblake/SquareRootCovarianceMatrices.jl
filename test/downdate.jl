using Test
using LinearAlgebra
using SquareRootCovarianceMatrices

@testset "PSDMatrix downdate!" begin
    # Start with a positive definite matrix
    A_mat = [5.0 2.0; 2.0 2.0]
    A = PSDMatrix(A_mat)

    # Downdate by a rank-1 vector
    u = [1.0, 1.0]
    a = 1.0
    # Compute reference downdated matrix
    A_ref = A_mat - a * (u * u')
    # Only proceed if A_ref is still positive semidefinite
    @test all(eigvals(A_ref) .>= -1e-10)

    # Apply downdate!
    downdate!(A, copy(u), a)
    @test Matrix(A)≈A_ref atol=1e-8

    # Test with a different scaling
    A2 = PSDMatrix(A_mat)
    a2 = 0.5
    A_ref2 = A_mat - a2 * (u * u')
    @test all(eigvals(A_ref2) .>= -1e-10)
    downdate!(A2, copy(u), a2)
    @test Matrix(A2)≈A_ref2 atol=1e-8
end
