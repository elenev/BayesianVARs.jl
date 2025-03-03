using Test
using BayesianVARs
using MAT, Distributions

function Base.:(==)(a::VARParameters, b::VARParameters)
    return a.Φ == b.Φ && a.Σ == b.Σ
end

matlab_results_path = "test/matlab_output.mat"
meta = VARMeta(p=2, m=3)

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
            normal_prior["Sigma"]
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
        matlab_results["Normal"]["Sigma"]
        )

    mniw_posterior = matlab_results["MNIW"]["Post"]
    Sigma_posterior = InverseWishart(mniw_posterior["Sigma"]["nu"], mniw_posterior["Sigma"]["Psi"])
    V = kron(mean(Sigma_posterior), inv(matlab_results["MNIW"]["Phi"]["V"]))
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

matlab_normal, matlab_conjugate, matlab_semiconjugate = get_matlab_results(meta, matlab_results_path)