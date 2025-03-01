## Estimation
# Collapse data 
function block_diagonalize!(X, Yslice)
    P, M = size(Yslice)
    N = M*P + 1
    for p=1:P
        rel_idx = (p-1)*M .+ (1:M)
        Yp = @view Yslice[end-p+1,:]
        for m=1:M
            X[m,N*(m-1) .+ rel_idx] .= Yp
        end
    end
end

function collapse_data(data, P, Ω = nothing)
    T, M = size(data)
    N = M*(M*P+1)
    Sxx = zeros(N, N)
    Sxy = zeros(N)
    Syy = zeros(M, M)

    if T>0
        #X = kron( sparse(I, M, M), ones(1, P*M+1) )
        #Ω = Ω === nothing ? sparse(I,M,M) : Ω
        X = kron(I(M), ones(1, P*M+1) )
        Ω = Ω === nothing ? I(M) : Ω

        Syy .+= data'*data

        XΩ = zeros(N,M)

        for t=P+1:T
            Yt = @view data[t,:]
            Yslice = @view data[t-P:t-1,:]
            block_diagonalize!(X, Yslice)
            mul!(XΩ, X', Ω)
            mul!(Sxx, XΩ, X, 1.0, 1.0)
            mul!(Sxy, XΩ, Yt, 1.0, 1.0)
        end
    end

    return Sxx, Syy, Sxy, T
end

function collapse_data_matrix(data, P)
    T, M = size(data)
    N = M*P+1
    Sxx = zeros(N, N)
    Sxy = zeros(N, M)
    Syy = zeros(M, M)

    if T>P
        X = ones(1,N)

        Y = @view data[P+1:end,:]
        Syy .+= Y'*Y

        for t=P+1:T
            Yt = @view data[t,:]
            Yslice = @view data[t-1:-1:t-P,:]
            X[1:end-1] .= view(Yslice',:)
            mul!(Sxx, X', X, 1.0, 1.0)
            mul!(Sxy, X', Yt', 1.0, 1.0)
        end
    end

    return Sxx, Syy, Sxy, T-P
end

function collapse_errors(vecΦ, data, P)
    T, M = size(data)
    See = zeros(M, M)
    X = kron(I(M), ones(1, P*M+1) )
    E = zeros(M)

    if T>P
        for t=P+1:T
            Yt = @view data[t,:]
            Yslice = @view data[t-P:t-1,:]
            block_diagonalize!(X, Yslice)
            E .= Yt
            mul!(E,X,vecΦ,-1.0,1.0)
            mul!(See, E, E', 1.0, 1.0)
        end 
    end

    return See
end

# Estimate
function estimate(priors::AbstractVARParameters, data::AbstractMatrix{<:Number})
    ispriors(priors) || error("The first argument must contain priors.")
    return _estimate(_prior_traits(priors), priors, data)
end

# Estimate Normal Model
function _estimate(::IsNormal, priors, data)

    # Unpack
    Φ_prior = coefficients(priors)
    Σ_prior = covariance(priors)
    (; μ, V) = Φ_prior
    Σ = Σ_prior.Ψ

    # Collapse data
    Sxx, Syy, Sxy, T = collapse_data(data, nlags(priors), inv(Σ))

    # Get the Φ prior variance (could be different from priormdl.Φ.V) if it's a ConditionalNormalPrior
    V = Φ_prior isa ConditionalNormalPrior ? kron(Σ, V) : V
    V⁻ = inv(V)

    # Define posterior
    V_posterior = inv(Sxx + V⁻)
    μ_posterior = V_posterior * (Sxy + V⁻*μ)
    V_posterior = (V_posterior + V_posterior')/2
    return  VARParameters(priors.M, priors.P, MultivariateNormal(μ_posterior, V_posterior), 
                             DeterministicMatrixDistribution(Σ))
end

# Estimate Conjugate (Dependent Prior) Model
function _estimate(::IsConjugate, priors, data)

    # Unpack
    Φ_prior = coefficients(priors)
    Σ_prior = covariance(priors)
    (; μ, V) = Φ_prior
    (; Ψ, ν) = Σ_prior

    M = size(Ψ,1)
    P = nlags(priors)
    μ_matrix = reshape(μ, M*P+1, M)

    # Collapse data
    Sxx, Syy, Sxy, T = collapse_data_matrix(data, nlags(priors))

    # Invert the across-coefficient variance
    V⁻ = inv(V)

    # Define posterior
    V⁻_posterior = Sxx + V⁻
    V_posterior = inv(V⁻_posterior)
    μmatrix_posterior = V_posterior * (Sxy + V⁻*μ_matrix)
    V_posterior = (V_posterior + V_posterior')/2

    # Define posterior for Σ
    ν_posterior = ν + T
    Ψ_posterior = Ψ + Syy + μ_matrix'*V⁻*μ_matrix -
                     μmatrix_posterior'*V⁻_posterior*μmatrix_posterior
    Ψ_posterior = (Ψ_posterior + Ψ_posterior')/2

    μ_posterior = vec(μmatrix_posterior)

    Σ_posterior = InverseWishart(ν_posterior, Ψ_posterior)
    V_posterior = kron(mean(Σ_posterior), V_posterior)
    Φ_posterior = MvTDist(ν_posterior, μ_posterior, V_posterior)
    #Φ_posterior = MultivariateNormal(μ_posterior, V_posterior)

    return  VARParameters(priors.M, priors.P, Φ_posterior, Σ_posterior )
end

function _initialize_distributions(Φ_prior::UnconditionalNormalPrior, Σ_prior::InverseWishartPrior)
    # Distributions are independent so just construct them
    Φ_priordist = MvNormal(Φ_prior.μ, Φ_prior.V)
    Σ_priordist = InverseWishart(Σ_prior.ν, Σ_prior.Ψ)
    return Φ_priordist, Σ_priordist
end

function _initialize_distributions(Φ_prior::CoefficientPrior, Σ_prior::DeterministicPrior)
    # Initialize covariance distribution first
    Σ_priordist = DeterministicMatricDistribution(Σ_prior.Ψ)
    # If Φ is a ConditionalNormalPrior, then incorporate the mean of the covariance prior distribution into the prior covariance
    # Otherwise, just use V
    V = priors.Φ isa ConditionalNormalPrior ? kron(mean(Σ_priordist), Φ_prior.V) : Φ_prior.V
    Φ_priordist = MvNormal(priors.Φ.μ, V)
    return Φ_priordist, Σ_priordist
end

function initialize(priors::AbstractVARParameters)
    #ispriors(priors) || error("The first argument must contain priors.")
    Φ_distro, Σ_distro = _initialize_distributions(coefficients(priors), covariance(priors))
    return VARParameters(metadata(priors), Φ_distro, Σ_distro)
end

# Estimate Semi-conjugate (Independent Prior) Model
function _estimate(::IsSemiConjugate, priors, data; 
                rng = Random.default_rng(), N=50_000, BURN=1_000) 

    # Initialize
    init_mdl = initialize(priors)
    init_sample = rand(rng, init_mdl)
    ν_posterior = priors.Σ.ν + size(data,1) - nlags(priors)

    meta = metadata(priors)
    Φ_prior = coefficients(priors)
    Σ_prior = covariance(priors)

    # Gibbs sampling
    chains = Vector{typeof(init_sample)}(undef, N+BURN)
    chains[1] = init_sample
    for n=2:N+BURN
        # Condition Φ distribution on previous Σ sample
        Σ_fixed = DeterministicPrior(chains[n-1].Σ)
        d = estimate(VARParameters(meta, Φ_prior, Σ_fixed), data)
        vecΦ_sample = rand(rng, d.Φ)

        # Condition Σ distribution on Φ sample
        Ψ_posterior = Σ_prior.Ψ + collapse_errors(vecΦ_sample, data, nlags(priors))
        Σ_posterior = InverseWishart(ν_posterior, Ψ_posterior)
        Σ_sample = rand(rng, Σ_posterior)

        chains[n] = VARParameters(meta, vecΦ_sample, Σ_sample)
    end

    chains = chains[BURN+1:end]
    return VARParameters(meta, 
                        [coefficients(chain) for chain in chains], 
                        [covariance(chain) for chain in chains]
                        )
end
