using Test
using BayesianVARs
using Distributions, LinearAlgebra

M = 3
P = 2
meta = VARMeta(p=P, m=M)

Φdist = MvNormal(zeros(M*(M*P+1)), I)
Σdist = InverseWishart(34, Matrix{Float64}(I, M, M))
@testset "VARParameters from distro" begin
    mdlparams = VARParameters(meta, Φdist, Σdist)

    mean_instance = instantiate(mean, mdlparams)
    @test coefficients(mean_instance) == mean(Φdist)
    @test covariance(mean_instance) == mean(Σdist)

    serror_instance = instantiate(serror, mdlparams)
    tmp_var = var(Σdist)
    @test coefficients(serror_instance) == ones(M*(M*P+1))
    @test covariance(serror_instance) == sqrt.(tmp_var)
end

N = 1_000
Φvec = [Φ for Φ in eachcol(rand(Φdist, N))]
Σvec = rand(Σdist, N)
@testset "VARParamaters from vector" begin
    mdlparams = VARParameters(meta, Φvec, Σvec)
    mean_instance = instantiate(mean, mdlparams)
    @test coefficients(mean_instance) == mean(Φvec)
    @test covariance(mean_instance) == mean(Σvec)

    serror_instance = instantiate(serror, mdlparams)
    tmp_var = var(Σdist)
    @test coefficients(serror_instance) == std(Φvec)
    @test covariance(serror_instance) == std(Σvec)
end

Φinst = Φvec[1]
Σinst = Σvec[1]
@testset "VARParamaters from scalar" begin
    mdlparams = VARParameters(meta, Φinst, Σinst)
    mean_instance = instantiate(mean, mdlparams)
    @test coefficients(mean_instance) == Φinst
    @test covariance(mean_instance) == Σinst

    serror_instance = instantiate(serror, mdlparams)
    @test coefficients(serror_instance) == zeros(size(Φinst))
    @test covariance(serror_instance) == zeros(size(Σinst))
end

# From priors