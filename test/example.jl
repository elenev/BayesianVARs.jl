using BayesianVARs
using CSV

# Define a 3D VAR(2)
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

truemdl = VARModel(c, (Φ₁,Φ₂), Σ);

T = 100;
data = simulate(truemdl, T);
CSV.write("simdata.csv", (; (Symbol(k) => v for (k,v) in zip(["INFL","UNRATE","FEDFUNDS"], eachcol(data)))...))

normal_priors = minnesota(3,2,ρ=0.9, 
            prior=UnconditionalNormalPrior, Σ=truemdl.Covariance);

normal_posterior = estimate(normal_priors, data);
summarize(normal_posterior)

conjugate_priors = minnesota(3,2,ρ=0.9,
            prior=ConditionalNormalPrior);

conjugate_posterior = estimate(conjugate_priors, data);
summarize(conjugate_posterior)

semiconjugate_priors = minnesota(3,2,ρ=0.9,
            prior=UnconditionalNormalPrior);

semiconjugate_posterior = estimate(semiconjugate_priors, data);
summarize(semiconjugate_posterior)
