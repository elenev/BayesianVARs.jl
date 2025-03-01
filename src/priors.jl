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
ispriors(p::AbstractVARParameters) = isa(coefficients(p), CoefficientPrior) && isa(covariance(p), CovariancePrior)

abstract type AbstractPriorTraits end
struct IsNormal<:AbstractPriorTraits end
struct IsConjugate<:AbstractPriorTraits end
struct IsSemiConjugate<:AbstractPriorTraits end

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
        return nothing
    end
end


## Minnesota prior
function minnesota(M::Integer, P::Integer; ρ=zeros(M),
    v₀=100., d=2., νₓ=nothing, σ²=ones(M), vₘ=100., ν::Integer=M+10,
    prior=UnconditionalNormalPrior, Σ=nothing )

    if νₓ === nothing && prior == UnconditionalNormalPrior
        νₓ = 1.
    end

    # If covariance matrix is known, use it for normalization
    #if Σ !== nothing && prior == UnconditionalNormalPrior
    #    σ² = diag(Σ)
    #end

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
    if prior == UnconditionalNormalPrior
        V11 = I.*(v₀ - νₓ) + νₓ*σ².*σ²'
        V = vcat([V11./p^d for p in 1:P]...)
        V = vcat(V, vₘ*ones(1,M))
        V = diagm(0 => vec(V))

        Φ = UnconditionalNormalPrior(vecμ, V)
    elseif prior == ConditionalNormalPrior
        if νₓ !== nothing && νₓ != v₀
            @warn "νₓ is set and ≠ v₀, but νₓ is unused in ConditionalNormalPrior."
        end
        V11 = v₀*ones(M)
        V = vcat([V11./p^d for p in 1:P]...)
        append!(V, vₘ)
        V = diagm(0 => vec(V'))
        Φ = ConditionalNormalPrior(vecμ, V)
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

    return VARParameters(M, P, Φ, Σ)
end