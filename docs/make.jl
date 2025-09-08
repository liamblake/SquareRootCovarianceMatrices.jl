using SquareRootCovarianceMatrices
using Documenter

DocMeta.setdocmeta!(SquareRootCovarianceMatrices, :DocTestSetup, :(using SquareRootCovarianceMatrices); recursive=true)

makedocs(;
    modules=[SquareRootCovarianceMatrices],
    authors="Liam A. A. Blake <liam.blake@adelaide.edu.au>",
    sitename="SquareRootCovarianceMatrices.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
