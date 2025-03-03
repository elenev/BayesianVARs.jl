using Test
using BayesianVARs
using MAT, Distributions

function Base.:(==)(a::VARParameters, b::VARParameters)
    return a.Φ == b.Φ && a.Σ == b.Σ
end

matlab_results_path = "test/matlab_output.mat"
meta = VARMeta(p=2, m=3)

function get_matlab_results(meta, path)
    matlab_results = matread(path)
    matlab_normal = VARParameters(
        meta,
        MvNormal(matlab_results["Normal"]["Phi"]["Mu"][:], matlab_results["Normal"]["Phi"]["V"]),
        matlab_results["Normal"]["Sigma"]
        )

    Sigma_posterior = InverseWishart(matlab_results["MNIW"]["Sigma"]["nu"], matlab_results["MNIW"]["Sigma"]["Psi"])
    V = kron(mean(Sigma_posterior), inv(matlab_results["MNIW"]["Phi"]["V"]))
    V = (V + V')/2
    Phi_posterior = MvNormal(matlab_results["MNIW"]["Phi"]["Mu"][:], V)
    matlab_conjugate = VARParameters(
        meta,
        Phi_posterior,
        Sigma_posterior
        )

    matlab_semiconjugate = VARParameters(
        meta,
        [Φ for Φ in eachcol(matlab_results["Gibbs"]["Phi"])],
        [Σ for Σ in eachslice(matlab_results["Gibbs"]["Sigma"],dims=3)]
        )

    return matlab_normal, matlab_conjugate, matlab_semiconjugate
end

matlab_normal, matlab_conjugate, matlab_semiconjugate = get_matlab_results(meta, matlab_results_path)