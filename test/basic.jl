using LinearAlgebra

@testset "PSDMatrix Constructor" begin
    # Test basic constructor with Cholesky decomposition
    A = [4.0 2.0; 2.0 2.0]
    psd = PSDMatrix(A)
    @test psd isa PSDMatrix{Float64}
    @test size(psd) == (2, 2)

    # Test constructor with "pass" mode
    sqrt_mat = [2.0 0.0; 1.0 1.0]  # Valid square root
    psd_pass = PSDMatrix(sqrt_mat * sqrt_mat'; sqrt_mode = "pass", check = false)
    @test psd_pass isa PSDMatrix{Float64}

    # Test error for invalid sqrt_mode
    @test_throws ArgumentError PSDMatrix(A, sqrt_mode = "invalid")
end

@testset "AbstractMatrix Interface" begin
    A = [4.0 2.0; 2.0 3.0]
    psd = PSDMatrix(A)

    # Test size
    @test size(psd) == (2, 2)
    @test size(psd, 1) == 2
    @test size(psd, 2) == 2

    # Test getindex
    @test psd[1, 1]≈4.0 atol=1e-12
    @test psd[1, 2]≈2.0 atol=1e-12
    @test psd[2, 1]≈2.0 atol=1e-12
    @test psd[2, 2]≈3.0 atol=1e-12

    # Test that setindex! throws error
    @test_throws ArgumentError psd[1, 1]=5.0

    # Test Matrix conversion
    reconstructed = Matrix(psd)
    @test reconstructed≈A atol=1e-12
    @test reconstructed isa Matrix{Float64}
end

@testset "Matrix Properties" begin
    A = [9.0 3.0; 3.0 2.0]
    psd = PSDMatrix(A)

    # Test symmetry/hermitian property
    @test ishermitian(psd)

    # Test positive definiteness
    @test isposdef(psd)

    # Test with rank-deficient matrix (not full rank)
    # Create a matrix that is PSD but not positive definite (has zero eigenvalue)
    rank_def_sqrt = [1.0 0.0; 1.0 0.0]  # This creates a rank-1 matrix
    psd_singular = PSDMatrix(rank_def_sqrt * rank_def_sqrt'; sqrt_mode = "pass", check = false)
    @test !isposdef(psd_singular)  # Should not be positive definite due to rank deficiency
end

@testset "Matrix Operations" begin
    A = [4.0 1.0; 1.0 2.0]
    psd = PSDMatrix(A)

    # Test matrix-vector multiplication
    v = [1.0, -1.0]
    result = psd * v
    expected = A * v
    @test result≈expected atol=1e-12

    # Test element-wise operations
    @test sum(psd)≈sum(A) atol=1e-12

    # Test iteration
    psd_elements = [psd[i, j] for i in 1:2, j in 1:2]
    @test psd_elements≈A atol=1e-12
end

@testset "Different Data Types" begin
    # Test with Float32
    A32 = Float32[4.0 2.0; 2.0 3.0]
    psd32 = PSDMatrix(A32)
    @test psd32 isa PSDMatrix{Float32}
    @test eltype(psd32) == Float32
    @test Matrix(psd32)≈A32 atol=1e-5

    # Test with integers (should be converted to Float64)
    A_int = [4 2; 2 3]
    psd_int = PSDMatrix(float(A_int))
    @test psd_int isa PSDMatrix{Float64}
    @test Matrix(psd_int)≈float(A_int) atol=1e-12
end

@testset "Edge Cases" begin
    # Test 1x1 matrix
    A1 = reshape([4.0], 1, 1)
    psd1 = PSDMatrix(A1)
    @test size(psd1) == (1, 1)
    @test psd1[1, 1]≈4.0 atol=1e-12

    # Test larger matrix
    n = 5
    A_large = rand(n, n)
    A_large = A_large' * A_large + I  # Make it PSD
    psd_large = PSDMatrix(A_large)
    @test size(psd_large) == (n, n)
    @test Matrix(psd_large)≈A_large atol=1e-10
    @test ishermitian(psd_large)
end

@testset "Display and Printing" begin
    A = [4.0 2.0; 2.0 3.0]
    psd = PSDMatrix(A)

    # Test that show doesn't throw error
    @test nothing == @test_nowarn show(IOBuffer(), psd)

    # Test with larger matrix (should show truncated version)
    A_large = rand(6, 6)
    A_large = A_large' * A_large + I
    psd_large = PSDMatrix(A_large)
    @test nothing == @test_nowarn show(IOBuffer(), psd_large)
end