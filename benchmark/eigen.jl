using LinearAlgebra

module EigenBenchmark

using SquareRootCovarianceMatrices
using Random

const rng = MersenneTwister(1234)
const test_sqrt = randn(rng, 1000, 1000)
const test_psd = PSDMatrix(test_sqrt; sqrt_mode = "pass", check = false)
const test_left_svd = test_sqrt * test_sqrt'
const test_right_svd = test_sqrt' * test_sqrt

end

function create_eigen_benchmark()
    suite = BenchmarkGroup()

    suite["eigen_full"] = @benchmarkable LinearAlgebra.eigen($EigenBenchmark.test_psd)
    suite["eigen_top10"] = @benchmarkable LinearAlgebra.eigen($EigenBenchmark.test_psd; k = 10)
    suite["right_svd_full"] = @benchmarkable right_svd($EigenBenchmark.test_psd)
    suite["right_svd_top10"] = @benchmarkable right_svd($EigenBenchmark.test_psd; k = 10)

    suite["eigen left native"] = @benchmarkable LinearAlgebra.eigen($EigenBenchmark.test_left_svd)
    suite["eigen right native"] = @benchmarkable LinearAlgebra.eigen($EigenBenchmark.test_right_svd)
    suite["svd native"] = @benchmarkable LinearAlgebra.svd($EigenBenchmark.test_sqrt)

    return suite
end