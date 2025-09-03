**This is currently a WIP and most of the interface reflects my own needs and use cases.**

This stems from my attempts to manipulate covariance matrices while still enforcing positive semi-definiteness and frustrations in using [JuliaStats/PDMats.jl](https://github.com/JuliaStats/PDMats.jl) (which cannot appropriately handle positive *semi*-definite matrices, which are rank deficient) and [invenia/PDMatsExtras.jl](https://github.com/invenia/PDMatsExtras.jl) (unmaintained).

