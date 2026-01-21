using LinearAlgebra

mutable struct PSDMatrix{T <: Real} <: AbstractMatrix{T}
    sqrt::Matrix{T}
end

function PSDMatrix(mat::Matrix{T}; sqrt_mode::String = "chol", check::Bool = true) where {T <: Real}
    if sqrt_mode == "pass"
        sqrt = mat
        if check
            if any(eigvals(sqrt * transpose(sqrt)) .<= 0)
                throw(ArgumentError("The provided matrix is not a valid square root of a positive definite matrix."))
            end
        end
    elseif sqrt_mode == "chol"
        sqrt = cholesky(mat).L
    else
        throw(ArgumentError("Unsupported sqrt_mode: $sqrt_mode"))
    end
    return PSDMatrix{T}(sqrt)
end

# Basic AbstractMatrix interface implementation

"""
    size(A::PSDMatrix)

Return the size of the positive semidefinite matrix A.
Since A = sqrt * sqrt', the size is determined by the square root matrix dimensions.
"""
function Base.size(A::PSDMatrix)
    n = size(A.sqrt, 1)
    return (n, n)
end

"""
    getindex(A::PSDMatrix, i::Int, j::Int)

Get the element at position (i, j) of the positive semidefinite matrix A.
The actual matrix is computed as A = sqrt * sqrt'.
"""
function Base.getindex(A::PSDMatrix{T}, i::Int, j::Int) where {T}
    # Compute the (i,j) element of sqrt * sqrt'
    return sum(A.sqrt[i, k] * A.sqrt[j, k] for k in 1:size(A.sqrt, 2))
end

"""
    setindex!(A::PSDMatrix, v, i::Int, j::Int)

Setting individual elements is not supported for PSDMatrix as it would break
the positive semidefinite constraint. Use reconstruction methods instead.
"""
function Base.setindex!(A::PSDMatrix, v, i::Int, j::Int)
    throw(ArgumentError("Direct element assignment not supported for PSDMatrix. " *
                        "Modifying individual elements would break positive semidefinite constraint."))
end

# Additional useful methods for better performance and functionality

"""
    Matrix(A::PSDMatrix)

Convert the PSDMatrix to a regular Matrix by computing sqrt * sqrt'.
"""
function Base.Matrix(A::PSDMatrix{T}) where {T}
    return A.sqrt * A.sqrt'
end

"""
    show(io::IO, A::PSDMatrix)

Display method for PSDMatrix.
"""
function Base.show(io::IO, A::PSDMatrix{T}) where {T}
    n = size(A, 1)
    print(io, "$(n)×$(n) PSDMatrix{$T}:")
    if n <= 4
        print(io, "\n", Matrix(A))
    else
        print(io, " (use Matrix(A) to see full matrix)")
    end
end

"""
    ishermitian(A::PSDMatrix)

PSDMatrix is always Hermitian (symmetric for real matrices).
"""
LinearAlgebra.ishermitian(A::PSDMatrix) = true

"""
    isposdef(A::PSDMatrix)

Check if the matrix is positive definite.
"""
function LinearAlgebra.isposdef(A::PSDMatrix)
    # A PSD matrix is positive definite if all eigenvalues are positive
    # For square root representation, this means sqrt has full rank
    return rank(A.sqrt) == size(A.sqrt, 1)
end
