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

LinearAlgebra.tr(A::PSDMatrix) = sum(norm(A.sqrt[i, :])^2 for i in 1:size(A.sqrt, 1))
