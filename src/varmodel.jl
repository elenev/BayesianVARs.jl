## Model metadata
abstract type AbstractVARMeta end

# Default implementation
Base.@kwdef struct VARMeta{M,S} <: AbstractVARMeta
    m::Int = 1                                          # Number of variables
    p::Int = 1                                          # Number of lags
    names::NTuple{M,S} = ntuple(i -> Symbol("Y$i"), m)  # Names of the variables
    function VARMeta(m::Int, p::Int, names::NTuple{M,S}) where {M,S}
        m > 0 || error("The number of variables must be positive")
        p > 0 || error("The number of lags must be positive")
        m == M || error("The number of series must be equal to the number of variables.")
        new{M,S}(m,p,names)
    end
end

nlags(meta::AbstractVARMeta) = meta.p
length(meta::AbstractVARMeta) = meta.m
series_names(meta::AbstractVARMeta) = meta.names

# Abstract supertype
abstract type AbstractVARModel end

# Default implementation
struct VARModel{M,N,T1,T2,T3} <: AbstractVARModel
    meta::M
    constant::T1
    ar::NTuple{N,T2}
    covariance::T3
    function VARModel(meta::M1, constant::T1, ar::NTuple{N,T2}, covariance::T3) where
        {M1<:AbstractVARMeta, T1<:AbstractVector, T2<:AbstractMatrix, T3<:AbstractMatrix, N}
        M = length(meta)
        P = nlags(meta)
        M == length(constant) ||
            error("The length of constant must M, where M = length(meta)")
        N == P ||
            error("The length of AR must be equal to P, where P = nlags(meta)")
        any(size(Φᵢ)==(M,M) for Φᵢ in ar) || 
            error("Each Element of AR must be an M x M matrix, where M = length(meta)")
        size(covariance)==(M,M) ||
            error("Covariance must be an MxM matrix, where M = length(meta)")
        eltype(constant) == eltype(covariance) && all(eltype(Φᵢ) == eltype(constant) for Φᵢ in ar) ||
            error("The element types of constant, AR, and covariance must be the same")
        new{M1,N,T1,T2,T3}(meta,constant,ar,covariance)
    end
end

# Convenience constructors
_make_ar_tuple(AR::AbstractMatrix) = (AR,)
_make_ar_tuple(AR::NTuple) = AR
_determine_metadata(ar,covariance) = VARMeta(p = length(ar), m = size(covariance,1))
_default_constant(meta, covariance) = zeros(eltype(covariance), length(meta))

const ARType = Union{AbstractMatrix, NTuple{N,<:AbstractMatrix} where N}

# Omitting metadata
function VARModel(constant::AbstractVector, ar::ARType, covariance::AbstractMatrix)
    ar = _make_ar_tuple(ar)
    meta = _determine_metadata(ar, covariance)
    return VARModel(meta, constant, ar, covariance)
end

# Omitting constant
function VARModel(meta::AbstractVARMeta, ar::ARType, covariance::AbstractMatrix)
    ar = _make_ar_tuple(ar)
    constant = _default_constant(meta, covariance)
    return VARModel(meta, constant, ar, covariance)
end

# Omitting metadata and constant
function VARModel(ar::ARType, covariance::AbstractMatrix)
    ar = _make_ar_tuple(ar)
    meta = _determine_metadata(ar, covariance)
    return VARModel(meta, ar, covariance)
end

# Convenience constructor for VAR(1)
function VARModel(Constant::AbstractVector, 
    AR::AbstractMatrix, Covariance::AbstractMatrix)
    return VARModel(Constant, (AR,), Covariance)
end

# Pass-through accessors
length(mdl::VARModel) = length(mdl.meta)
nlags(mdl::VARModel) = nlags(mdl.meta)
series_names(mdl::VARModel) = series_names(mdl.meta)
eltype(mdl::VARModel) = eltype(mdl.constant)

# Accessors
metadata(mdl::VARModel) = mdl.meta
constant(mdl::VARModel) = mdl.constant
lags(mdl::VARModel) = mdl.ar
covariance(mdl::VARModel) = mdl.covariance

## VAR Model Properties
# These only make sense if eltype is a number
_check_eltype(mdl::AbstractVARModel) = eltype(mdl) <: Number || error("The element type of the VARModel must be a number")

function steadystate(mdl::AbstractVARModel)
    _check_eltype(mdl)
    return (I -  reduce(+,lags(mdl))) \ constant(mdl)
end

function companion_form(mdl::AbstractVARModel)
    M = length(mdl)
    P = nlags(mdl)
    A = zeros(M*P,M*P)
    for p=1:P
        A[1:M,(p-1)*M.+(1:M)] .= lags(mdl)[p]
    end
    A .+= diagm(-M => ones(M*(P-1)))
    return A
end

function is_stable(mdl::AbstractVARModel)
    A = companion_form(mdl)
    return all(abs.(eigvals(A)) .< 1)
end



