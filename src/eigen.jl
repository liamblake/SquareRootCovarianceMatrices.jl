using Arpack

"""
    LinearAlgebra.eigen(A::PSDMatrix; k::Int = size(A, 1))

Overload of the LinearAlgebra.eigen function to compute eigenvalues and eigenvectors of a PSDMatrix.
Only the top k eigenvalues/vectors are computed (defaulting to almost all of them) and the eigenvalues are sorted in descending order. This uses ARPACK internally, so only up to min(n,m) - 1 eigenvalues can be computed efficiently. If k = n then the full matrix is computed and the standard LinearAlgebra.eigen function is called. The purpose of this implementation is to be efficient for a small number of eigenvalues of very large matrices.
"""
function LinearAlgebra.eigen(A::PSDMatrix{T}; k::Int = size(A, 1)) where {T <: Real}
    # Compute SVD of the square root matrix
    # For A = sqrt * sqrt', eigenvalues of A are the squared singular values of sqrt
    # and eigenvectors of A are the left singular vectors of sqrt
    svd_call_min = min(size(A.sqrt)...)
    if k < 1 || k > size(A, 1)
        throw(ArgumentError("k must be between 1 and the size of the matrix (n = $n)."))
    end

    if k >= svd_call_min
        E = LinearAlgebra.eigen(Symmetric(A); sortby = λ -> -real(λ))  # Fall back to full eigen decomposition for all eigenvalues
        return LinearAlgebra.Eigen(E.values[1:k], E.vectors[:, 1:k])
    else
        Z, _, _, _, _ = svds(A.sqrt; nsv = k)
        return LinearAlgebra.Eigen(Z.S .^ 2, Z.U)
    end
end

"""
    right_svd(A::PSDMatrix; k::Int = size(A, 1))

Compute the top k right singular values/vectors of A.sqrt, i.e. the eigenvalues/vectors of A'A.
"""
function right_svd(A::PSDMatrix{T}; k::Int = size(A, 1)) where {T <: Real}
    svd_call_min = min(size(A.sqrt)...)
    if k < 1 || k > size(A, 1)
        throw(ArgumentError("k must be between 1 and the size of the matrix (n = $n)."))
    end

    if k >= svd_call_min
        E = LinearAlgebra.eigen(Symmetric(A.sqrt' * A.sqrt); sortby = λ -> -real(λ))  # Fall back to full eigen decomposition for all eigenvalues
        return LinearAlgebra.Eigen(sqrt.(E.values[1:k]), E.vectors[:, 1:k])
    else
        Z, _, _, _, _ = svds(A.sqrt; nsv = k)
        return LinearAlgebra.Eigen(Z.S, Z.V)
    end
    # end
end

"""
    eigmax!(v::AbstractVector{T}, A::PSDMatrix)

Compute the maximum eigenpair of the PSDMatrix A using the power method and store the vector in v.
"""
function eigmax!(v::AbstractVector{T}, A::PSDMatrix{T}) where {T <: Real}
    n = size(A, 1)
    if length(v) != n
        throw(ArgumentError("Length of vector v must match the size of PSDMatrix A."))
    end
    # TODO: use power method to find the largest eigenvalue/vector without allocations
    max_eig = LinearAlgebra.eigen(A; k = 1)
    v .= max_eig.vectors[:, 1]
    return max_eig.values[1]
end