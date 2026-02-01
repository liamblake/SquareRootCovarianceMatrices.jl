using TSVD

"""
    LinearAlgebra.eigen(A::PSDMatrix; k::Int = size(A, 1))

Overload of the LinearAlgebra.eigen function to compute eigenvalues and eigenvectors of a PSDMatrix.
Only the top k eigenvalues/vectors are computed (defaulting to all of them) and the eigenvalues are sorted in descending order.
"""
function LinearAlgebra.eigen(A::PSDMatrix{T}; k::Int = size(A, 1)) where {T <: Real}
    # Compute SVD of the square root matrix
    # For A = sqrt * sqrt', eigenvalues of A are the squared singular values of sqrt
    # and eigenvectors of A are the left singular vectors of sqrt

    n = size(A.sqrt, 1)

    # Use standard SVD for full decomposition or small matrices
    # tsvd has issues when k == n
    if k >= n || n <= 10
        S = svd(A.sqrt)
        # Sort by descending eigenvalues and take top k
        eigenvalues = S.S .^ 2
        perm = sortperm(eigenvalues; rev = true)[1:k]
        return LinearAlgebra.Eigen(eigenvalues[perm], S.U[:, perm])
    else
        # Use truncated SVD for large matrices with k < n
        U, s, _ = tsvd(A.sqrt, k)
        return LinearAlgebra.Eigen(s .^ 2, U)
    end
end

"""
    right_svd(A::PSDMatrix; k::Int = size(A, 1))

Compute the top k right singular values/vectors of A.sqrt, i.e. the eigenvalues/vectors of A'A.
"""
function right_svd(A::PSDMatrix{T}; k::Int = size(A, 1)) where {T <: Real}
    n = size(A.sqrt, 1)

    # Use standard SVD for full decomposition or small matrices
    if k >= n || n <= 10
        S = svd(A.sqrt)
        # Sort by descending singular values and take top k
        perm = sortperm(S.S; rev = true)[1:k]
        return LinearAlgebra.Eigen(S.S[perm], S.V[:, perm])
    else
        # Use truncated SVD for large matrices with k < n
        _, s, V = tsvd(A.sqrt, k)
        return LinearAlgebra.Eigen(s, V)
    end
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