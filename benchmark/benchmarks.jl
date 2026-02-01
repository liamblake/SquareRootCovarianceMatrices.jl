using BenchmarkTools
using SquareRootCovarianceMatrices

include("eigen.jl")

function create_benchmark()
    suite = BenchmarkGroup()

    suite["Eigen comparison"] = create_eigen_benchmark()

    return suite
end

const SUITE = create_benchmark()