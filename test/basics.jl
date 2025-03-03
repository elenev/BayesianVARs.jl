using Test
using BayesianVARs

# Test sets for constructors, accessors, and convenience constructors

## VARMeta
@testset "VARMeta" begin
    @testset "VARMeta(p=2,m=3)" begin
        meta = VARMeta(p=2,m=3)
        @test length(meta) == 3
        @test nlags(meta) == 2
        @test series_names(meta) == (:Y1, :Y2, :Y3)
    end

    @testset "VARMeta(p=1,m=1,names=(\"Y\",))" begin
        meta = VARMeta(p=1,m=1,names=("Y",))
        @test length(meta) == 1
        @test nlags(meta) == 1
        @test series_names(meta) == ("Y",)
    end
end

# VARMdl
c  = [   0.9298,   0.2431,   0.3530];
Φ₁ = [   0.2102   -0.7794    0.6663
        -0.0095    1.4123    0.0257
        -0.0231   -1.2146    0.7187];
Φ₂ = [   0.4418    0.6629   -0.4762
         0.0205   -0.4800   -0.0048
         0.1568    1.1809    0.1610];
Σ  = [   4.9224   -0.0639    0.6477
        -0.0639    0.0978   -0.1626
         0.6477   -0.1626    1.2883];
meta = VARMeta(p=2,m=3)

@testset "VARModel" begin
    @testset "VARModel(meta, c, (Φ₁,Φ₂), Σ)" begin
        mdl = VARModel(meta, c, (Φ₁,Φ₂), Σ)
        @test length(mdl) == 3
        @test nlags(mdl) == 2
        @test series_names(mdl) == (:Y1, :Y2, :Y3)
        @test lags(mdl) == (Φ₁,Φ₂)
        @test constant(mdl) == c
        @test covariance(mdl) == Σ
        @test metadata(mdl) === meta
    end

    @testset "VARModel(c, (Φ₁,Φ₂), Σ)" begin
        mdl = VARModel(c, (Φ₁,Φ₂), Σ)
        @test length(mdl) == 3
        @test nlags(mdl) == 2
        @test series_names(mdl) == (:Y1, :Y2, :Y3)
        @test lags(mdl) == (Φ₁,Φ₂)
        @test constant(mdl) == c
        @test covariance(mdl) == Σ
    end

    @testset "VARModel(c, Φ₁, Σ)" begin
        mdl = VARModel(c, Φ₁, Σ)
        @test nlags(mdl) == 1
        @test lags(mdl) == (Φ₁,)
    end

    @testset "VARModel(Φ₁, Σ)" begin
        mdl = VARModel(Φ₁, Σ)
        @test nlags(mdl) == 1
        @test lags(mdl) == (Φ₁,)
        @test constant(mdl) == zeros(length(mdl))
    end
end

# VARParameters
mdl = VARModel(meta, c, (Φ₁,Φ₂), Σ)
mdlparams = VARParameters(mdl)
@testset "VARParameters" begin
    @testset "VARParameters(mdl)" begin
        @test length(mdlparams) == 3*(3*2+1)
        @test nlags(mdlparams) == 2
        @test nseries(mdlparams) == 3
        @test metadata(mdlparams) == meta
        @test coefficients(mdlparams) == [Φ₁ Φ₂ c]'
        @test covariance(mdlparams) == Σ
    end

    @testset "VARModel(mdlparams)" begin
        mdl2 = VARModel(mdlparams)
        @test length(mdl2) == 3
        @test nlags(mdl2) == 2
        @test lags(mdl2) == (Φ₁,Φ₂)
        @test constant(mdl2) == c
        @test covariance(mdl2) == Σ
    end
end