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

    A4 = X_A_Xt(A3, X)
    @test Matrix(A4) ≈ X * Matrix(A3) * X'

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

        @test Matrix(B)≈Matrix(A) atol=1e-10
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

@testset "PSDMatrix Inverse" begin
    @testset "Basic inverse" begin
        # Create a test positive definite matrix
        A_mat = [4.0 2.0; 2.0 3.0]
        A = PSDMatrix(A_mat)

        # Compute inverse using the PSDMatrix method
        A_inv = inv(A)

        # Test that A * A_inv ≈ I
        @test Matrix(A) * A_inv ≈ I
        @test A_inv * Matrix(A) ≈ I

        # Test that result matches standard matrix inverse
        @test A_inv ≈ inv(A_mat)
    end

    @testset "Inverse with different sizes" begin
        # Test 3x3 matrix
        B_mat = [5.0 1.0 0.5; 1.0 4.0 1.0; 0.5 1.0 3.0]
        B = PSDMatrix(B_mat)
        B_inv = inv(B)

        @test Matrix(B) * B_inv ≈ I
        @test B_inv ≈ inv(B_mat)

        # Test 1x1 matrix
        C_mat = [2.0;;]
        C = PSDMatrix(C_mat)
        C_inv = inv(C)

        @test Matrix(C) * C_inv ≈ I
        @test C_inv ≈ inv(C_mat)
    end

    @testset "Inverse with diagonal matrix" begin
        # Test with diagonal matrix (simpler case)
        D_mat = Diagonal([2.0, 3.0, 5.0])
        D = PSDMatrix(Matrix(D_mat))
        D_inv = inv(D)

        @test Matrix(D) * D_inv ≈ I
        @test D_inv ≈ inv(D_mat)
    end

    @testset "Inverse type stability" begin
        # Test Float64
        A_f64 = PSDMatrix([4.0 2.0; 2.0 3.0])
        A_inv_f64 = inv(A_f64)
        @test eltype(A_inv_f64) === Float64

        # Test Float32
        A_f32 = PSDMatrix([4.0f0 2.0f0; 2.0f0 3.0f0])
        A_inv_f32 = inv(A_f32)
        @test eltype(A_inv_f32) === Float32
    end

    @testset "Inverse accuracy with random matrices" begin
        # Test with larger random positive definite matrix
        n = 5
        A_sqrt = randn(n, n)
        A_mat = A_sqrt * A_sqrt' + I  # Add I to ensure positive definiteness
        A = PSDMatrix(A_mat)
        A_inv = inv(A)

        @test Matrix(A) * A_inv≈I atol=1e-10
        @test A_inv * Matrix(A)≈I atol=1e-10
        @test A_inv≈inv(A_mat) atol=1e-10
    end
end

@testset "PSDMatrix Determinant" begin
    @testset "Determinant with square sqrt matrix" begin
        # Create a test positive definite matrix with square sqrt representation
        A_mat = [4.0 2.0; 2.0 3.0]
        A = PSDMatrix(A_mat)
        
        # Ensure the sqrt is square
        @test size(A.sqrt, 1) == size(A.sqrt, 2)
        
        # Test determinant matches the standard matrix determinant
        @test det(A) ≈ det(A_mat)
        
        # Test that det(A) = det(sqrt)^2 for square sqrt
        @test det(A) ≈ det(A.sqrt)^2
    end
    
    @testset "Determinant with rectangular sqrt matrix" begin
        # Create a PSDMatrix with rectangular sqrt representation
        # For this, we need to manually construct with a non-square sqrt
        sqrt_rect = [1.0 0.0; 0.0 2.0; 1.0 1.0]  # 3x2 matrix
        A = PSDMatrix(sqrt_rect; sqrt_mode="pass", check=false)
        
        # Ensure the sqrt is rectangular
        @test size(A.sqrt, 1) != size(A.sqrt, 2)
        
        # Compute the full matrix
        A_mat = sqrt_rect' * sqrt_rect
        
        # Test determinant matches the full matrix determinant
        @test det(A) ≈ det(A_mat)
        
        # Test that det(A) = det(sqrt' * sqrt) for rectangular sqrt
        @test det(A) ≈ det(A.sqrt' * A.sqrt)
    end
    
    @testset "Determinant with different sizes" begin
        # Test 3x3 matrix
        B_mat = [5.0 1.0 0.5; 1.0 4.0 1.0; 0.5 1.0 3.0]
        B = PSDMatrix(B_mat)
        @test det(B) ≈ det(B_mat)
        
        # Test 1x1 matrix
        C_mat = [2.0;;]
        C = PSDMatrix(C_mat)
        @test det(C) ≈ det(C_mat)
        @test det(C) ≈ 2.0
        
        # Test 4x4 matrix
        D_mat = [6.0 2.0 1.0 0.5; 2.0 5.0 1.0 0.5; 1.0 1.0 4.0 0.5; 0.5 0.5 0.5 3.0]
        D = PSDMatrix(D_mat)
        @test det(D) ≈ det(D_mat)
    end
    
    @testset "Determinant with diagonal matrix" begin
        # Test with diagonal matrix (product of diagonal elements)
        D_mat = Diagonal([2.0, 3.0, 5.0])
        D = PSDMatrix(Matrix(D_mat))
        
        @test det(D) ≈ det(D_mat)
        @test det(D) ≈ 2.0 * 3.0 * 5.0
        @test det(D) ≈ 30.0
    end
    
    @testset "Determinant type stability" begin
        # Test Float64
        A_f64 = PSDMatrix([4.0 2.0; 2.0 3.0])
        det_f64 = det(A_f64)
        @test typeof(det_f64) === Float64
        
        # Test Float32
        A_f32 = PSDMatrix([4.0f0 2.0f0; 2.0f0 3.0f0])
        det_f32 = det(A_f32)
        @test typeof(det_f32) === Float32
    end
    
    @testset "Determinant with scaled matrices" begin
        # Test that det(α * A) = α^n * det(A) for n×n matrix
        A_mat = [4.0 2.0; 2.0 3.0]
        A = PSDMatrix(A_mat)
        α = 2.0
        
        A_scaled = α * A
        
        # For 2x2 matrix: det(α * A) = α^2 * det(A)
        @test det(A_scaled) ≈ α^2 * det(A)
        @test det(A_scaled) ≈ det(α * A_mat)
    end
    
    @testset "Determinant accuracy with random matrices" begin
        # Test with larger random positive definite matrix
        n = 5
        A_sqrt = randn(n, n)
        A_mat = A_sqrt * A_sqrt' + I  # Add I to ensure positive definiteness
        A = PSDMatrix(A_mat)
        
        @test det(A)≈det(A_mat) rtol=1e-9
        @test det(A) > 0  # PSD matrices have non-negative determinants
    end
    
    @testset "Determinant with rectangular sqrt (overdetermined)" begin
        # Create another rectangular case with more rows than columns
        sqrt_rect = [1.0 0.0; 2.0 1.0; 0.0 3.0; 1.0 1.0]  # 4x2 matrix
        A = PSDMatrix(sqrt_rect; sqrt_mode="pass", check=false)
        
        # Ensure the sqrt is rectangular
        @test size(A.sqrt, 1) > size(A.sqrt, 2)
        
        # Compute the full matrix
        A_mat = sqrt_rect' * sqrt_rect
        
        # Test determinant
        @test det(A) ≈ det(A_mat)
        @test det(A) > 0
    end
end
