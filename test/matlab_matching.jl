using Test
using BayesianVARs
using MAT, CSV
import Distributions: MvNormal, InverseWishart

function recursive_equals(a, b)
    try
        if eltype(a) <: Integer && eltype(b) <: Integer
            return a == b
        else
            return a ≈ b
        end
    catch
        return all(
            recursive_equals(getfield(a,f),getfield(b,f)) for f in propertynames(a)
        )
    end
end

function get_matlab_priors(meta, path)
    matlab_results = matread(path)

    normal_prior = matlab_results["Normal"]["Prior"]
    matlab_normal = VARParameters(
        meta,
        UnconditionalNormalPrior(
            normal_prior["Phi"]["Mu"][:],
            normal_prior["Phi"]["V"]
        ),
        DeterministicPrior(
            normal_prior["Sigma"]["Psi"]
        )
    )

    mniw_prior = matlab_results["MNIW"]["Prior"]
    matlab_conjugate = VARParameters(
        meta,
        ConditionalNormalPrior(
            mniw_prior["Phi"]["Mu"][:],
            mniw_prior["Phi"]["V"]
        ),
        InverseWishartPrior(
            mniw_prior["Sigma"]["nu"],
            mniw_prior["Sigma"]["Psi"]
        )
    )

    gibbs_prior = matlab_results["Gibbs"]["Prior"]
    matlab_semiconjugate = VARParameters(
        meta,
        UnconditionalNormalPrior(
            gibbs_prior["Phi"]["Mu"][:],
            gibbs_prior["Phi"]["V"]
        ),
        InverseWishartPrior(
            gibbs_prior["Sigma"]["nu"],
            gibbs_prior["Sigma"]["Psi"]
        )
    )

    return matlab_normal, matlab_conjugate, matlab_semiconjugate
end

function get_matlab_posteriors(meta, path)
    matlab_results = matread(path)

    normal_posterior = matlab_results["Normal"]["Post"]
    matlab_normal = VARParameters(
        meta,
        MvNormal(normal_posterior["Phi"]["Mu"][:], normal_posterior["Phi"]["V"]),
        BayesianVARs.DeterministicMatrixDistribution(normal_posterior["Sigma"]["Psi"])
        )

    mniw_posterior = matlab_results["MNIW"]["Post"]
    Sigma_posterior = InverseWishart(mniw_posterior["Sigma"]["nu"], mniw_posterior["Sigma"]["Psi"])
    V = kron(mean(Sigma_posterior), mniw_posterior["Phi"]["V"])
    V = (V + V')/2
    Phi_posterior = MvNormal(mniw_posterior["Phi"]["Mu"][:], V)
    matlab_conjugate = VARParameters(
        meta,
        Phi_posterior,
        Sigma_posterior
        )

    gibbs_posterior = matlab_results["Gibbs"]["Post"]
    matlab_semiconjugate = VARParameters(
        meta,
        [Φ for Φ in eachcol(gibbs_posterior["Phi"])],
        [Σ for Σ in eachslice(gibbs_posterior["Sigma"],dims=3)]
        )

    return matlab_normal, matlab_conjugate, matlab_semiconjugate
end

function get_matlab_minnesota_config(path)
    matlab_results = matread(path)

    config = matlab_results["minnesota"]
    return (ρ = config["Center"],
            v₀ = config["SelfLag"],
            νₓ = config["CrossLag"],
            d = config["Decay"],
            vₘ = config["VarianceX"])
end

# Load data
data_path = "simdata.csv"
csv_data = CSV.File(data_path)  # Read CSV file
data = Float64.(stack(collect(x) for x in csv_data))'

# Load MATLAB results
matlab_results_path = "matlab_output.mat"
meta = VARMeta(p=2, m=3)

mn = get_matlab_minnesota_config(matlab_results_path)
m_normal_prior, m_conjugate_prior, m_semiconjugate_prior = get_matlab_priors(meta, matlab_results_path)
m_normal_posterior, m_conjugate_posterior, m_semiconjugate_posterior = get_matlab_posteriors(meta, matlab_results_path)

# Test priors
normal_prior = minnesota(meta; mn..., prior=UnconditionalNormalPrior, Σ=m_normal_prior.Σ.Ψ)
conjugate_prior = minnesota(meta; mn..., prior=ConditionalNormalPrior)
semiconjugate_prior = minnesota(meta; mn..., prior=UnconditionalNormalPrior)
@testset "Priors" begin
    @test recursive_equals(normal_prior, m_normal_prior)
    @test recursive_equals(conjugate_prior, m_conjugate_prior)
    @test recursive_equals(semiconjugate_prior, m_semiconjugate_prior)
end

# Test estimation
estimate = BayesianVARs.estimate
normal_posterior = estimate(normal_prior, data)
conjugate_posterior = estimate(conjugate_prior, data, use_normal=true)
semiconjugate_posterior = estimate(semiconjugate_prior, data,N=100_000)

function compare_simulated_posteriors(mat,jul;tol=0.01,moments=(mean,serror))
    for f in moments
        matcoefs = f(mat)
        julcoefs = f(jul)
        @test all(abs(m-j) < tol for (m,j) in zip(
            coefficients(matcoefs),coefficients(julcoefs)))
        @test all(abs(m-j) < tol for (m,j) in zip(
            covariance(matcoefs),covariance(julcoefs)))
    end
end

@testset "Posteriors" begin
    @test recursive_equals(normal_posterior, m_normal_posterior)
    @test recursive_equals(conjugate_posterior, m_conjugate_posterior)
    compare_simulated_posteriors(semiconjugate_posterior, m_semiconjugate_posterior)
end