struct DeterministicUnivariateDistribution{T} <: DiscreteUnivariateDistribution
    x::T
end
rand(::AbstractRNG, d::DeterministicUnivariateDistribution) = d.x
logpdf(d::DeterministicUnivariateDistribution, x) = x == d.x ? 0.0 : -Inf
cdf(d::DeterministicUnivariateDistribution, x) = x < d.x ? 0.0 : 1.0
quantile(d::DeterministicUnivariateDistribution, p) = p < 1.0 ? d.x : Inf
minimum(d::DeterministicUnivariateDistribution) = d.x
maximum(d::DeterministicUnivariateDistribution) = d.x
insupport(d::DeterministicUnivariateDistribution, x) = x == d.x
mean(d::DeterministicUnivariateDistribution) = d.x
var(::DeterministicUnivariateDistribution) = 0.
mode(d::DeterministicUnivariateDistribution) = d.x
modes(d::DeterministicUnivariateDistribution) = [d.x]
skewness(::DeterministicUnivariateDistribution) = 0.
kurtosis(::DeterministicUnivariateDistribution) = 0.

struct DeterministicMultivariateDistribution{T} <: DiscreteMultivariateDistribution
    x::T
end
length(d::DeterministicMultivariateDistribution) = length(d.x)
eltype(d::DeterministicMultivariateDistribution) = eltype(d.x)
rand!(::AbstractRNG, d::DeterministicMultivariateDistribution, x::AbstractArray) = copyto!(x, d.x)
_logpdf(d::DeterministicMultivariateDistribution, x) = x == d.x ? 0.0 : -Inf
mean(d::DeterministicMultivariateDistribution) = d.x
var(d::DeterministicMultivariateDistribution) = fill!(similar(d.x),0.)
cov(d::DeterministicMultivariateDistribution) = fill!(similar(d.x,length(d),length(d)),0.)
mode(d::DeterministicMultivariateDistribution) = d.x
modes(d::DeterministicMultivariateDistribution) = [d.x]

struct DeterministicMatrixDistribution{T} <: ContinuousMatrixDistribution
    x::T
end
size(d::DeterministicMatrixDistribution) = size(d.x)
eltype(d::DeterministicMatrixDistribution) = eltype(d.x)
rand!(::AbstractRNG, d::DeterministicMatrixDistribution, x::AbstractMatrix) = copyto!(x, d.x)
_logpdf(d::DeterministicMatrixDistribution, x) = x == d.x ? 0.0 : -Inf
mean(d::DeterministicMatrixDistribution) = d.x
var(d::DeterministicMatrixDistribution) = fill!(similar(d.x),0.)