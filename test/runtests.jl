using BayesianVARs
using Test

@testset "Basics" begin include("basics.jl") end
@testset "Instantiation" begin include("instantiation.jl") end
@testset "Priors and Estimation" begin include("matlab_matching.jl") end