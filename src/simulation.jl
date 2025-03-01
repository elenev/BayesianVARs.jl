## Simulation
function _default_initial_conditions(mdl::AbstractVARModel)
    _check_eltype(mdl)
    if isstable(mdl)
        return steadystate(mdl)
    else
        return zeros(eltype(mdl), length(mdl))
    end
end

function _repeat_initial_conditions(mdl::AbstractVARModel, Y0::AbstractVector)
    return repeat(Y0', nlags(mdl), 1)
end

function _repeat_initial_conditions(mdl::AbstractVARModel, Y0::AbstractMatrix)
    return Y0
end

function simulate(mdl::AbstractVARModel, T::Integer; 
   Yinit=_default_initial_conditions(mdl),
   rng::AbstractRNG=Random.default_rng())
   
   _check_eltype(mdl)

   Yinit = _repeat_initial_conditions(mdl, Yinit)
   

   M = length(mdl)
   P = nlags(mdl)
   AR = lags(mdl)
   Σ = covariance(mdl)

   # Initialize
   Y = zeros(eltype(Yinit), T, M)
   Y[1:P,:] .= Yinit

   # Innovations distribution
   D = MvNormal(zeros(M), Σ)

   # Simulate
   for t=P+1:T
       Yt_mean = mdl.Constant + mapreduce(*,+,AR,eachrow(@view Y[t-1:-1:t-P,:]))
       Y[t,:] .= Yt_mean + rand(rng, D)
   end

   return Y
end