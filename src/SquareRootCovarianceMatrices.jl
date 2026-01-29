module SquareRootCovarianceMatrices
using LinearAlgebra

include("psdmatrix.jl")
include("operations.jl")
# include("eigen.jl")

export PSDMatrix, X_A_Xt!

end
