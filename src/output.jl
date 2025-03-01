## Summarize
function summarize(d::VARParameters)
    M = length(d)
    N_params = ncoefs(d)
    if !(d.Σ isa DeterministicMatrixDistribution)
        N_params += M^2
        if d.Σ isa InverseWishart
            desc = "conjugate priors"
        else
            desc = "analytically intractable priors"
        end
    else
        desc = "normal priors and fixed Σ"
    end
    println("""
        Bayesian VAR under $desc
        Number of equations:            $M
        Number of estimated Parameters: $N_params""");

    mean_model = VARModel(mean(d))
    se_model = VARModel(serror(d))
    println("""
                      |   Mean     Std  
         --------------------------------""")

    fe = FormatExpr(" Constant[{}] | {:7.4f} | {:7.4f}\n")
    for (i,(m,s)) in enumerate(zip(mean_model.Constant, se_model.Constant))
        printfmt(fe, i, m, s)
    end

    fe = FormatExpr(" AR[{}][{},{}]  | {:7.4f} | {:7.4f}\n")
    indices = CartesianIndices((M,M))
    for (p,Φᵢ) in enumerate(mean_model.AR)
        for (i,(m,s)) in enumerate(zip(Φᵢ, se_model.AR[p]))
            r = indices[i][1]
            c = indices[i][2]
            printfmt(fe, p, r, c, m, s)
        end
    end

    println("     Innovation Covariance Matrix")
    print("          ")
    for i=1:M
        printfmt("|     Y$i    ")
    end
    print("\n----------")
    for i=1:M
        print("|-----------")
    end
    for i=1:M
        print("\n     Y$i   |")
        for j=1:M
            printfmt(" {:9.4f} |", mean_model.Covariance[i,j])
        end
        print("\n          |")
        for j=1:M
            printfmt(" ({:7.4f}) |", se_model.Covariance[i,j])
        end
    end
    println()
end