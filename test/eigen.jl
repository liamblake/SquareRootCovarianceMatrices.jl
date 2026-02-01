using Test
using LinearAlgebra
using SquareRootCovarianceMatrices

@testset "PSDMatrix Eigen and SVD" begin
    @testset "Small 2x2 Matrix" begin
        # Construct a positive definite matrix
        A_mat = [4.0 2.0; 2.0 3.0]
        A = PSDMatrix(A_mat)

        # Test eigen decomposition (all eigenvalues/vectors)
        eig = eigen(A)
        eig_ref = eigen(A_mat)
        @test isapprox(sort(eig.values; rev = true), sort(eig_ref.values; rev = true); atol = 1e-8)

        # Test that eigenvalues are sorted in descending order
        @test issorted(eig.values, rev = true)

        # Test that eigenvectors reconstruct the matrix
        A_reconstructed = eig.vectors * Diagonal(eig.values) * eig.vectors'
        @test isapprox(A_reconstructed, A_mat; atol = 1e-8)

        # Test that eigenvectors are orthonormal
        @test isapprox(eig.vectors' * eig.vectors, I; atol = 1e-8)

        # Test eigen decomposition (top k=1 eigenvalue/vector)
        eig1 = eigen(A; k = 1)
        @test eig1.values[1] ≈ maximum(eig_ref.values)
        @test length(eig1.values) == 1
        @test size(eig1.vectors, 2) == 1

        # Test that top eigenvector is normalized
        @test isapprox(norm(eig1.vectors[:, 1]), 1.0; atol = 1e-8)

        # Test right_svd (should match svdvals of sqrt)
        svd_A = right_svd(A)
        svd_ref = svd(A.sqrt)
        @test isapprox(svd_A.values, sort(svd_ref.S; rev = true); atol = 1e-8)

        # Test that right singular vectors are orthonormal
        @test isapprox(svd_A.vectors' * svd_A.vectors, I; atol = 1e-8)

        # Test right_svd with k=1
        svd_A1 = right_svd(A; k = 1)
        @test svd_A1.values[1] ≈ maximum(svd_ref.S)
        @test length(svd_A1.values) == 1
    end

    @testset "Larger 5x5 Matrix" begin
        # Create a larger random PSD matrix
        n = 5
        B_sqrt = randn(n, n)
        B_mat = B_sqrt * B_sqrt'
        B = PSDMatrix(B_mat)

        # Test all eigenvalues
        eig = eigen(B)
        eig_ref = eigen(B_mat)
        @test isapprox(sort(eig.values; rev = true), sort(eig_ref.values; rev = true); atol = 1e-6)
        @test issorted(eig.values, rev = true)
        @test length(eig.values) == n

        # Test top k=3 eigenvalues
        eig3 = eigen(B; k = 3)
        @test length(eig3.values) == 3
        @test issorted(eig3.values, rev = true)
        # Top 3 should match the largest 3 from full decomposition
        @test isapprox(eig3.values, eig.values[1:3]; atol = 1e-6)

        # Test that eigenvectors are orthonormal
        @test isapprox(eig3.vectors' * eig3.vectors, I; atol = 1e-6)
    end

    @testset "Diagonal Matrix" begin
        # Test with a diagonal matrix (special case)
        D_mat = Diagonal([5.0, 3.0, 1.0])
        D = PSDMatrix(Matrix(D_mat))

        eig = eigen(D)
        @test isapprox(sort(eig.values; rev = true), [5.0, 3.0, 1.0]; atol = 1e-8)

        # Top 2 eigenvalues
        eig2 = eigen(D; k = 2)
        @test isapprox(eig2.values, [5.0, 3.0]; atol = 1e-8)
    end

    @testset "Rank Deficient Matrix" begin
        # Create a rank-2 matrix in 3D space using "pass" mode
        U = [1.0 0.0; 0.0 1.0; 0.0 0.0]
        S = Diagonal([2.0, sqrt(2.0)])  # Use square roots for the sqrt matrix
        R_sqrt = U * S
        R = PSDMatrix(R_sqrt; sqrt_mode = "pass", check = false)

        eig = eigen(R; k = 2)  # Only request 2 eigenvalues since rank is 2
        # Should have 2 eigenvalues
        @test length(eig.values) == 2
        @test isapprox(sort(eig.values; rev = true), [4.0, 2.0]; atol = 1e-6)
    end

    @testset "Identity Matrix" begin
        I_mat = Matrix{Float64}(I, 3, 3)
        I_psd = PSDMatrix(I_mat)

        eig = eigen(I_psd)
        @test all(isapprox.(eig.values, 1.0; atol = 1e-8))

        # Test k=1 for identity
        eig1 = eigen(I_psd; k = 1)
        @test isapprox(eig1.values[1], 1.0; atol = 1e-8)
    end
end

@testset "PSDMatrix eigmax!" begin
    @testset "Small 2x2 Matrix" begin
        # Construct a positive definite matrix
        A_mat = [4.0 2.0; 2.0 3.0]
        A = PSDMatrix(A_mat)

        # Compute reference largest eigenpair
        eig = eigen(A_mat)
        idx = argmax(eig.values)
        λ_ref = eig.values[idx]
        v_ref = eig.vectors[:, idx]

        # Test eigmax!
        v = zeros(eltype(A_mat), size(A, 1))
        λ = eigmax!(v, A)
        # Largest eigenvalue
        @test isapprox(λ, λ_ref; atol = 1e-8)
        # Eigenvector (up to sign)
        @test isapprox(abs(dot(v, v_ref)), 1.0; atol = 1e-8)
        # Eigenvector should be normalized
        @test isapprox(norm(v), 1.0; atol = 1e-8)
    end

    @testset "Larger Matrix" begin
        # Test with a larger matrix
        n = 10
        B_sqrt = randn(n, n)
        B_mat = B_sqrt * B_sqrt'
        B = PSDMatrix(B_mat)

        eig_ref = eigen(B_mat)
        λ_max_ref = maximum(eig_ref.values)
        idx = argmax(eig_ref.values)
        v_max_ref = eig_ref.vectors[:, idx]

        v = zeros(n)
        λ_max = eigmax!(v, B)

        @test isapprox(λ_max, λ_max_ref; atol = 1e-6)
        @test isapprox(abs(dot(v, v_max_ref)), 1.0; atol = 1e-6)
        @test isapprox(norm(v), 1.0; atol = 1e-8)
    end

    @testset "Diagonal Matrix" begin
        # Test with diagonal matrix
        D_mat = Diagonal([10.0, 5.0, 3.0, 1.0])
        D = PSDMatrix(Matrix(D_mat))

        v = zeros(4)
        λ = eigmax!(v, D)

        @test isapprox(λ, 10.0; atol = 1e-8)
        # For diagonal matrix, eigenvector should align with standard basis
        @test maximum(abs.(v)) ≈ 1.0
    end

    @testset "Error Handling" begin
        # Test dimension mismatch error
        A_mat = [4.0 2.0; 2.0 3.0]
        A = PSDMatrix(A_mat)
        v_wrong = zeros(3)  # Wrong size

        @test_throws ArgumentError eigmax!(v_wrong, A)
    end

    @testset "Verify Eigenpair Property" begin
        # Verify that the returned eigenpair satisfies Av = λv
        A_mat = [4.0 2.0; 2.0 3.0]
        A = PSDMatrix(A_mat)

        v = zeros(2)
        λ = eigmax!(v, A)

        # Compute Av
        Av = A_mat * v
        λv = λ * v

        @test isapprox(Av, λv; atol = 1e-8)
    end
end
