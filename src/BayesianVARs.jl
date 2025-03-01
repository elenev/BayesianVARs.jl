module BayesianVARs

using   Distributions,      # For analytically defined priors and posteriors
        LinearAlgebra,      # For matrix operations
        Random,             # For sampling from posteriors
        Statistics,         # For calculating statistics and extending some
        Format              # For formatting output

# Import functions that will overloaded for the BayesianVARs module
import Base: rand, length, size

# Import functions necessary to define a new custom distribution
import Distributions: logpdf, cdf, quantile, minimum, maximum, insupport, rand!, _logpdf
import StatsBase: mean, var, modes, mode, skewness, kurtosis

# Include files
include("varmodel.jl")
include("priors.jl")
include("parameters.jl")
include("deterministic_distro.jl")
include("simulation.jl")
include("estimation.jl")
include("output.jl")
include("irf.jl")

end
