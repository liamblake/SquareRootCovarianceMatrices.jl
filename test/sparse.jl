using Test
using LinearAlgebra
using SparseArrays
using SquareRootCovarianceMatrices

@testset "Sparse PSDMatrix" begin
    sqrt_a = sparse([2.0 0.0 0.0;
                     1.0 3.0 0.0;
                     0.5 1.0 4.0])
    sqrt_b = sparse([1.5 0.0 0.0;
                     0.25 2.0 0.0;
                     0.75 0.5 1.0])

    a = PSDMatrix(sqrt_a; sqrt_mode = "pass", check = false)
    b = PSDMatrix(sqrt_b; sqrt_mode = "pass", check = false)

    a_ref = Matrix(sqrt_a * sqrt_a')
    b_ref = Matrix(sqrt_b * sqrt_b')

    @testset "Constructor types" begin
        # Test that types have been recorded correctly
        @test isa(a, PSDMatrix{Float64, SparseMatrixCSC{Float64, Int}})
    end

    @testset "Operations" begin
        @test a.sqrt isa SparseMatrixCSC
        @test size(a) == (3, 3)
        @test a[1, 2] ≈ a_ref[1, 2]
        @test a[3, 3] ≈ a_ref[3, 3]
        @test Matrix(a) ≈ a_ref
        @test ishermitian(a)
        @test isposdef(a)

        c = a + b
        @test Matrix(c) ≈ a_ref + b_ref
        @test c.sqrt isa SparseMatrixCSC

        α = 2.0
        d = α * a
        @test Matrix(d) ≈ α * a_ref

        a_scaled = copy(a)
        lmul!(α, a_scaled)
        @test Matrix(a_scaled) ≈ α * a_ref

        x = sparse([1.0 0.0 0.0;
                    0.5 1.0 0.0;
                    0.0 0.0 1.0])

        xax_t = X_A_Xt(a, x)
        x_ref = Matrix(x * a_ref * x')
        @test Matrix(xax_t) ≈ x_ref

        a_inplace = copy(a)
        X_A_Xt!(a_inplace, x)
        @test Matrix(a_inplace) ≈ x_ref

        @test tr(a) ≈ tr(a_ref)
        @test det(a) ≈ det(a_ref)
        @test a == copy(a)
        @test isapprox(a, a_ref)

        a_copy = copy(a)
        @test a_copy.sqrt isa SparseMatrixCSC
        @test Matrix(a_copy) ≈ a_ref

        copied = PSDMatrix(sqrt_b; sqrt_mode = "pass", check = false)
        copyto!(copied, a)
        @test Matrix(copied) ≈ a_ref
        @test copied.sqrt isa SparseMatrixCSC

        @test_throws "The inverse of a sparse matrix can often be dense" inv(a)
    end

    @testset "Eigen and SVD" begin
        eig_ref = eigen(a_ref)

        eig2 = eigen(a; k = 2)
        @test length(eig2.values) == 2
        @test isapprox(eig2.values, reverse(eig_ref.values)[1:2]; atol = 1e-8)

        svd_a = right_svd(a; k = 2)
        svd_ref = svd(Matrix(a.sqrt))
        @test isapprox(svd_a.values, sort(svd_ref.S; rev = true)[1:2]; atol = 1e-8)

        @test isapprox(svd_a.vectors' * svd_a.vectors, I; atol = 1e-8)

        v = zeros(eltype(a_ref), size(a, 1))
        λ = eigmax!(v, a)
        idx = 3
        v_ref = eig_ref.vectors[:, idx]
        @test isapprox(λ, eig_ref.values[idx]; atol = 1e-8)
        @test isapprox(abs(dot(v, v_ref)), 1.0; atol = 1e-6)
        @test isapprox(norm(v), 1.0; atol = 1e-8)
    end
end