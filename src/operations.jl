"""
    X_A_Xt!(A::PSDMatrix, X::AbstractMatrix)

Update the PSDMatrix A in place to represent X * A * X'.
"""
function X_A_Xt!(A::PSDMatrix{T}, X::AbstractMatrix{T}) where {T}
    A.sqrt .= X * A.sqrt
end

"""
    Base.:+(A::PSDMatrix, B::PSDMatrix)

Return a new PSDMatrix representing the sum of two PSDMatrix objects.
"""
function Base.:+(A::PSDMatrix{T}, B::PSDMatrix{T}) where {T}
    return PSDMatrix(A.sqrt * A.sqrt' + B.sqrt * B.sqrt') # Defaults to chol
end

"""
    Base.:*(a::Number, A::PSDMatrix)

Return a new PSDMatrix representing the scalar multiplication of a PSDMatrix.
"""
function Base.:*(a::Number, A::PSDMatrix{T}) where {T}
    if a < 0
        throw(ArgumentError("Cannot scale PSDMatrix by a negative scalar."))
    end
    return PSDMatrix(sqrt(a) * A.sqrt; sqrt_mode = "pass")
end

function LinearAlgebra.lmul!(a::Number, A::PSDMatrix{T}) where {T}
    if a < 0
        throw(ArgumentError("Cannot scale PSDMatrix by a negative scalar."))
    end
    lmul!(sqrt(a), A.sqrt)
    return A
end

LinearAlgebra.tr(A::PSDMatrix) = sum(norm(A.sqrt[i, :])^2 for i = 1:size(A.sqrt, 1))

function Base.copy(A::PSDMatrix{T}) where {T}
    return PSDMatrix(copy(A.sqrt); sqrt_mode = "pass")
end

function Base.copyto!(dest::PSDMatrix{T}, src::PSDMatrix{T}) where {T}
    copyto!(dest.sqrt, src.sqrt)
    return dest
end

function Base.:(==)(A::PSDMatrix, B::PSDMatrix)
    return Matrix(A) == Matrix(B)
end

function Base.:(==)(A::PSDMatrix, B::AbstractMatrix)
    return Matrix(A) == B
end

function Base.:(==)(A::AbstractMatrix, B::PSDMatrix)
    return A == Matrix(B)
end

function Base.isapprox(A::PSDMatrix, B::PSDMatrix; kwargs...)
    return isapprox(Matrix(A), Matrix(B); kwargs...)
end

function Base.isapprox(A::PSDMatrix, B::AbstractMatrix; kwargs...)
    return isapprox(Matrix(A), B; kwargs...)
end

function Base.isapprox(A::AbstractMatrix, B::PSDMatrix; kwargs...)
    return isapprox(A, Matrix(B); kwargs...)
end