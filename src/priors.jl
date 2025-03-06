abstract type Prior end
abstract type CoefficientPrior <: Prior end
abstract type CovariancePrior <: Prior end

# vec(Φ) ~ N(μ, V)
struct UnconditionalNormalPrior{T1,T2} <: CoefficientPrior
    μ::T1
    V::T2
end

# vec(Φ) | Σ ~ N(μ, Σ ⊗ V)
struct ConditionalNormalPrior{T1,T2} <: CoefficientPrior 
    μ::T1
    V::T2
end

# vec(Φ)ᵢ ~ F
struct VectorOfPriors{T<:UnivariateDistribution} <: CoefficientPrior
    priors::Vector{T}
end
Base.getindex(p::VectorOfPriors, i) = p.priors[i]
Base.size(p::VectorOfPriors) = size(p.priors)
Base.length(p::VectorOfPriors) = length(p.priors)
Base.lastindex(p::VectorOfPriors) = lastindex(p.priors)

# Σ ~ IW(ν, Ψ)
struct InverseWishartPrior{T1,T2} <: CovariancePrior
   ν::T1
   Ψ::T2 
end

# Σ = Ψ
struct DeterministicPrior{T} <: CovariancePrior
    Ψ::T
end


## Traits for VARParameters containing priors
VARPriors = VARParameters{T0,T1,T2} where {T0,T1<:CoefficientPrior,T2<:CovariancePrior}
is_priors(p::AbstractVARParameters) = isa(coefficients(p), CoefficientPrior) && isa(covariance(p), CovariancePrior)

abstract type AbstractPriorTraits end
struct IsNormal<:AbstractPriorTraits end
struct IsConjugate<:AbstractPriorTraits end
struct IsSemiConjugate<:AbstractPriorTraits end
struct IsIntractable<:AbstractPriorTraits end

function _prior_traits(priors::AbstractVARParameters)
    Φ_prior = coefficients(priors)
    Σ_prior = covariance(priors)
    if Σ_prior isa DeterministicPrior
        return IsNormal()
    elseif Φ_prior isa ConditionalNormalPrior && Σ_prior isa InverseWishartPrior
        return IsConjugate()
    elseif Φ_prior isa UnconditionalNormalPrior && Σ_prior isa InverseWishartPrior
        return IsSemiConjugate()
    else
        return IsIntractable()
    end
end


## Minnesota prior
function minnesota(meta; ρ=zeros(length(meta)),
    v₀=100., d=2., νₓ=nothing, σ²=ones(length(meta)), vₘ=100., 
    ν::Integer=length(meta)+10,
    prior=UnconditionalNormalPrior, Σ=nothing )

    if νₓ === nothing && prior != ConditionalNormalPrior
        νₓ = 1.
    end

    # If covariance matrix is known, use it for normalization
    #if Σ !== nothing && prior == UnconditionalNormalPrior
    #    σ² = diag(Σ)
    #end
    M = length(meta)
    P = nlags(meta)

    # Prior on Φ
    # Means
    # First lag
    expand_ρ(ρ::Number) = ρ*ones(M)
    expand_ρ(ρ::AbstractVector) = ρ
    μ₁ = diagm(0 => expand_ρ(ρ))

    # Subsequent lags and constant
    μ = hcat(μ₁, zeros(M,M*(P-1) + 1))
    vecμ = vec(μ')

    # Variances
    if prior == UnconditionalNormalPrior || prior == VectorOfPriors
        V11 = I.*(v₀ - νₓ) + νₓ*σ².*σ²'
        V = vcat([V11./p^d for p in 1:P]...)
        V = vcat(V, vₘ*ones(1,M))
        vecV = vec(V)

        if prior == UnconditionalNormalPrior
            V = diagm(0 => V)
            Φ = UnconditionalNormalPrior(vecμ, vecV)
        else
            Φ = VectorOfPriors([Normal(μᵢ, vᵢ) for (μᵢ, vᵢ) in zip(vecμ, vecV)])
        end
    elseif prior == ConditionalNormalPrior
        if νₓ !== nothing && νₓ != v₀
            @warn "νₓ is set and ≠ v₀, but νₓ is unused in ConditionalNormalPrior."
        end
        V11 = v₀*ones(M)
        V = vcat([V11./p^d for p in 1:P]...)
        append!(V, vₘ)
        V = diagm(0 => vec(V'))
        Φ = ConditionalNormalPrior(vecμ, V)
    elseif Prior == VectorOfPriors

    else
        error("prior type must be either UnconditionalNormalPrior or ConditionalNormalPrior")
    end

    # Prior on Σ
    if Σ === nothing
        Ψ = diagm(0 => σ²)
        Σ = InverseWishartPrior(ν, Ψ)
    else
        Σ = DeterministicPrior(Σ)
    end

    return VARParameters(meta, Φ, Σ)
end