module SquareRootCovarianceMatrices
using LinearAlgebra

include("psdmatrix.jl")
include("operations.jl")
include("eigen.jl")
include("downdate.jl")

export PSDMatrix, X_A_Xt!, right_svd, eigmax!, downdate!

end
