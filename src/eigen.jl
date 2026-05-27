using KrylovKit
using Random

"""
    LinearAlgebra.eigen(A::PSDMatrix; k::Int = size(A, 1))

Overload of the LinearAlgebra.eigen function to compute eigenvalues and eigenvectors of a PSDMatrix.
Only the top k eigenvalues/vectors are computed (defaulting to almost all of them) and the eigenvalues are sorted in descending order. This uses KrylovKit.svdsolve internally, which uses a random initialisation vector. An rng instance can be passed to control the random seed used in this initialisation. If the solver fails to converge to the requested number of singular values, an error is thrown.
"""
function LinearAlgebra.eigen(
        A::PSDMatrix{T}; k::Int = size(A, 1), rng = Random.default_rng()) where {T <: Real}
    # Compute SVD of the square root matrix
    # For A = sqrt * sqrt', eigenvalues of A are the squared singular values of sqrt
    # and eigenvectors of A are the left singular vectors of sqrt
    if k < 1 || k > size(A, 1)
        throw(ArgumentError("k must be between 1 and the size of the matrix (n = $n)."))
    end

    if k == size(A, 1)
        E = LinearAlgebra.eigen(Symmetric(A); sortby = λ -> -real(λ))  # Fall back to full eigen decomposition for all eigenvalues
        return LinearAlgebra.Eigen(E.values[1:k], E.vectors[:, 1:k])
    else
        vals, lvecs, _, info = svdsolve(A.sqrt, rand(rng, T, size(A, 1)), k)

        if info.converged < k
            throw(ProcessFailedException("KrylovKit.svdsolve did not converge to the requested number of singular values. Converged: $(info.converged), Requested: $k"))
        end

        return LinearAlgebra.Eigen(vals[1:k] .^ 2, stack(lvecs[1:k]))
    end
end

"""
    right_svd(A::PSDMatrix; k::Int = size(A, 1))

Compute the top k right singular values/vectors of A.sqrt, i.e. the eigenvalues/vectors of A'A. These
are stored in a Eigen object.
"""
function right_svd(
        A::PSDMatrix{T}; k::Int = size(A, 1), rng = Random.default_rng()) where {T <: Real}
    if k < 1 || k > size(A, 1)
        throw(ArgumentError("k must be between 1 and the size of the matrix (n = $n)."))
    end

    vals, _, rvecs, info = svdsolve(A.sqrt, rand(rng, T, size(A, 1)), k)
    if info.converged < k
        throw(ProcessFailedException("KrylovKit.svdsolve did not converge to the requested number of singular values. Converged: $(info.converged), Requested: $k"))
    end
    return LinearAlgebra.Eigen(vals[1:k], stack(rvecs[1:k]))
end

"""
    eigmax!(v::AbstractVector{T}, A::PSDMatrix)

Compute the maximum eigenpair of the PSDMatrix A using the power method and store the vector in v.
"""
function eigmax!(
        v::AbstractVector{T}, A::PSDMatrix{T}; rng = Random.default_rng()) where {T <: Real}
    n = size(A, 1)
    if length(v) != n
        throw(ArgumentError("Length of vector v must match the size of PSDMatrix A."))
    end
    # TODO: use power method to find the largest eigenvalue/vector without allocations
    max_eig = LinearAlgebra.eigen(A; k = 1, rng = rng)
    copyto!(v, max_eig.vectors[:, 1])
    return max_eig.values[1]
end