abstract type IRFMethod end
struct Cholesky <: IRFMethod end
struct Generalized <: IRFMethod end

function _recursive_add!(Ωₜ, Ωvec, Φvec)
    for (Ωᵢ,Φᵢ) in zip(Ωvec,Φvec)
        mul!(Ωₜ, Φᵢ, Ωᵢ, 1.0, 1.0)
    end
end

function irf(mdl::AbstractVARModel, T::Integer; method::IRFMethod=Generalized())
    @assert isstable(mdl) "Model must be stable to compute IRFs"
    M = length(mdl)
    P = nlags(mdl)
    A = lags(mdl)
    Σ = covariance(mdl)

    if method isa Cholesky
        Σᵃ = cholesky(Σ).U
        scale = ones(M)
    elseif method isa Generalized
        Σᵃ = Σ
        scale = sqrt.(diag(Σ)).^(-1)
    end

    # 1st Dimension: Innovation/Shock
    # 2nd Dimension: Response
    # 3rd Dimension: Time
    IRF = zeros(T,M,M)
    Ω = ntuple(i -> zeros(M,M), T+1)
    Ω[1] .= I(M)
    Cₜ = zeros(M,M)
    evec = [x for x in eachcol(I(M))]
    for t=1:T
        # Period
        Astart = 1
        Afinish = min(t,P)
        Ωstart = t-Astart
        Ωfinish = t-Afinish
        #println("t = $t  | Ai ∈ {$Astart,$Afinish}, Ωi ∈ {$Ωstart, $Ωfinish} ")
        _recursive_add!(Ω[t+1], Ω[1+Ωstart:-1:1+Ωfinish], A[Astart:Afinish])

        mul!(Cₜ, Ω[t+1], Σᵃ)
        for i=1:M
            # Innovation to y_i
            e = evec[i]
            e[i] = 1.
            mul!(view(IRF,t,i,:), Cₜ, e)
            IRF[t,i,:] .*= scale[i]
        end
    end

    return IRF
end

function irf(d::AbstractVARParameters, T::Integer; N=10_000, rng=Random.default_rng(), kwargs...)
    out = stack(i -> irf(VARModel(rand(rng, d)), T; kwargs...), 1:N)
    return out
end