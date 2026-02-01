"""
Downdate the PSDMatrix A by removing the contribution of the vector u, i.e. A := A - a*u*u'.
The downdate is performed in-place on the square root of A. The vector u is also modified inplace during the process
"""
function downdate!(A::PSDMatrix{T}, u::Vector{T}, a::T = T(1)) where {T <: Real}
    n = size(A, 1)
    @assert length(u)==n "Length of u must match the size of A."
    α = sqrt(a)

    for k in 1:n
        uk = α * u[k]
        r = sqrt(A.sqrt[k, k]^2 - uk^2)
        c = r / A.sqrt[k, k]
        s = uk / A.sqrt[k, k]
        A.sqrt[k, k] = r
        if k < n
            A.sqrt[(k + 1):n, k] = (A.sqrt[(k + 1):n, k] - s * α * u[(k + 1):n]) / c
            u[(k + 1):n] = c * u[(k + 1):n] - s * A.sqrt[(k + 1):n, k] / α
        end
    end
end