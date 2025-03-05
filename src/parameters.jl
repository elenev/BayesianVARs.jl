abstract type AbstractVARParameters end

struct VARParameters{T0,T1,T2}<:AbstractVARParameters
    meta::T0
    Φ::T1
    Σ::T2
    function VARParameters(meta::T0, Φ::T1, Σ::T2) where {T0<:AbstractVARMeta,T1,T2}
        new{T0,T1,T2}(meta,Φ,Σ)
    end
end

metadata(p::VARParameters) = p.meta
coefficients(p::VARParameters) = p.Φ
covariance(p::VARParameters) = p.Σ
length(p::VARParameters) = length(p.Φ)
nseries(p::VARParameters) = length(p.meta)
nlags(p::VARParameters) = nlags(p.meta)

function _split_coefficients(M, P, Φ)
    ar = ntuple( p -> Φ[(p-1)*M .+ (1:M), :]', P)
    constant = Φ[end,:][:]

    return constant, ar
end

function _reshape_coefficients(M, vecΦ::AbstractVector)
    return reshape(vecΦ, :, M)
end

function _reshape_coefficients(M, Φ::AbstractMatrix)
    return Φ
end

# Model constructor from parameters
function VARModel(p::AbstractVARParameters)
    meta = metadata(p)
    M = length(meta)
    P = nlags(meta)
    Φ = _reshape_coefficients(M, coefficients(p))
    C, A = _split_coefficients(M, P, Φ)
    Σ = covariance(p)
    return VARModel(meta, C, A, Σ)
end

# Parameters constructor from model
_coefficients_to_matrix(lags, constant) = hcat(lags..., constant)'

function VARParameters(mdl::AbstractVARModel)
    Φ = _coefficients_to_matrix(lags(mdl), constant(mdl))
    Σ = covariance(mdl)
    return VARParameters(metadata(mdl), Φ, Σ)
end


## Instantiate model from distribution
function _is_instantiated(params::AbstractVARParameters)
    return coefficients(params) isa AbstractVecOrMat{<:Number} && 
            covariance(params) isa AbstractMatrix{<:Number}
end

function instantiate(f::Function, distros::VARParameters)
    if _is_instantiated(distros)
        f = _scalar_version(f)
    elseif is_priors(distros)
        return distros = initialize(distros)
    end

    vecΦ = f(coefficients(distros))
    Σ = f(covariance(distros))
    
    return VARParameters(metadata(distros), vecΦ, Σ)
end

mean(d::AbstractVARParameters) = instantiate(mean,d)
serror(d::AbstractVARParameters) = instantiate(serror,d)
rand(rng::AbstractRNG, d::AbstractVARParameters) = instantiate(x -> rand(rng,x),d)
rand(d::AbstractVARParameters) = rand(Random.default_rng(), d)


# serror definitions for distributions
serror(x::AbstractVecOrMat{<:Number}) = zeros(eltype(x), size(x))
serror(x::AbstractVector{<:AbstractVecOrMat}) = std(x)
serror(x::ContinuousUnivariateDistribution) = std(x)
serror(x::ContinuousDistribution) = sqrt.(var(x))

_scalar_version(f::Function) = x -> x
_scalar_version(f::typeof(serror)) = x -> zeros(eltype(x), size(x))