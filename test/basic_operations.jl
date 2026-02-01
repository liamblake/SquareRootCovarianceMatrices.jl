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

@testset "PSDMatrix Copy Operations" begin
    @testset "Basic copy" begin
        # Create a test matrix
        A_mat = [4.0 2.0; 2.0 3.0]
        A = PSDMatrix(A_mat)
        
        # Test copy creates a new independent object
        B = copy(A)
        @test Matrix(B) ≈ Matrix(A)
        @test B.sqrt ≈ A.sqrt
        
        # Verify independence: modifying copy doesn't affect original
        B.sqrt[1, 1] = 999.0
        @test A.sqrt[1, 1] ≈ 2.0  # Original unchanged
        @test B.sqrt[1, 1] ≈ 999.0  # Copy modified
    end
    
    @testset "copy with different sizes" begin
        # Test 3x3 matrix
        C_mat = [5.0 1.0 0.5; 1.0 4.0 1.0; 0.5 1.0 3.0]
        C = PSDMatrix(C_mat)
        C_copy = copy(C)
        
        @test size(C_copy) == size(C)
        @test Matrix(C_copy) ≈ Matrix(C)
        @test C_copy.sqrt ≈ C.sqrt
        
        # Test 1x1 matrix
        D_mat = [2.0;;]
        D = PSDMatrix(D_mat)
        D_copy = copy(D)
        
        @test Matrix(D_copy) ≈ Matrix(D)
    end
    
    @testset "copyto!" begin
        # Create source and destination matrices
        A_mat = [4.0 2.0; 2.0 3.0]
        B_mat = [9.0 3.0; 3.0 5.0]
        A = PSDMatrix(A_mat)
        B = PSDMatrix(B_mat)
        
        # Store original B values for verification
        B_orig_sqrt = copy(B.sqrt)
        
        # Copy A into B
        result = copyto!(B, A)
        
        # Test that B now equals A
        @test Matrix(B) ≈ Matrix(A)
        @test B.sqrt ≈ A.sqrt
        
        # Test that B's sqrt was actually modified
        @test !(B.sqrt ≈ B_orig_sqrt)
        
        # Test that result is B (returns destination)
        @test result === B
        
        # Test that A is unchanged
        @test Matrix(A) ≈ A_mat
    end
    
    @testset "copyto! with same object" begin
        # Test copying to itself (should work without issues)
        A_mat = [4.0 2.0; 2.0 3.0]
        A = PSDMatrix(A_mat)
        A_sqrt_orig = copy(A.sqrt)
        
        copyto!(A, A)
        
        @test A.sqrt ≈ A_sqrt_orig
        @test Matrix(A) ≈ A_mat
    end
    
    @testset "copyto! independence" begin
        # Verify that modifying destination after copyto! doesn't affect source
        A_mat = [4.0 2.0; 2.0 3.0]
        B_mat = [9.0 3.0; 3.0 5.0]
        A = PSDMatrix(A_mat)
        B = PSDMatrix(B_mat)
        
        A_sqrt_orig = copy(A.sqrt)
        
        copyto!(B, A)
        
        # Modify B
        B.sqrt[1, 1] = 777.0
        
        # A should be unchanged
        @test A.sqrt ≈ A_sqrt_orig
        @test A.sqrt[1, 1] ≠ 777.0
    end
    
    @testset "copy preserves sqrt_mode behavior" begin
        # Test that copied matrix works with "pass" mode
        A_mat = [4.0 2.0; 2.0 3.0]
        A = PSDMatrix(A_mat)
        
        B = copy(A)
        
        # Both should have same underlying sqrt representation
        @test B.sqrt ≈ A.sqrt
        
        # Verify both reconstruct to same matrix
        @test Matrix(B) ≈ Matrix(A)
        @test Matrix(B) ≈ A_mat
    end
    
    @testset "copyto! with larger matrices" begin
        # Test with 5x5 matrix
        n = 5
        A_sqrt = randn(n, n)
        A_mat = A_sqrt * A_sqrt'
        B_sqrt = randn(n, n)
        B_mat = B_sqrt * B_sqrt'
        
        A = PSDMatrix(A_mat)
        B = PSDMatrix(B_mat)
        
        copyto!(B, A)
        
        @test Matrix(B) ≈ Matrix(A) atol=1e-10
        @test B.sqrt ≈ A.sqrt
    end
    
    @testset "Type stability" begin
        # Test that copy preserves type
        A_f64 = PSDMatrix([4.0 2.0; 2.0 3.0])
        B_f64 = copy(A_f64)
        @test eltype(B_f64.sqrt) === Float64
        
        A_f32 = PSDMatrix([4.0f0 2.0f0; 2.0f0 3.0f0])
        B_f32 = copy(A_f32)
        @test eltype(B_f32.sqrt) === Float32
        
        # Test copyto! type matching
        C_f64 = PSDMatrix([9.0 3.0; 3.0 5.0])
        copyto!(C_f64, A_f64)
        @test eltype(C_f64.sqrt) === Float64
    end
end
