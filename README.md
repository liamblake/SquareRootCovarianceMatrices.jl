**This is currently a WIP and most of the interface reflects my own needs and use cases.**

This stems from my attempts to manipulate covariance matrices while still enforcing positive semi-definiteness and frustrations in using [JuliaStats/PDMats.jl](https://github.com/JuliaStats/PDMats.jl) (which cannot appropriately handle positive *semi*-definite matrices, which are rank deficient) and [invenia/PDMatsExtras.jl](https://github.com/invenia/PDMatsExtras.jl) (unmaintained).

A real-valued matrix $A$ is positive semi-definite if and only if it can be decomposed as $A = BB^T$ for some other matrix $B$. This decomposition is not unique, but we can use it to characterise $A$. A common choice is to ensure that $B$ is lower triangular (the Cholesky decomposition), but $B$ does not need to be triangular in general. 

The principal behind this package is that we can represent a positive semi-definite matrix $A$ by only storing a decomposition $B$ and calculating $A = BB^T$ on the fly when needed. A subset of operations on $A$ are implemented by changing $B$ accordingly. 

## Where would this be useful?
TODO

