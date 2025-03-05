%% Load julia-simulated data
simdata = readtable("simdata.csv");

%% True Covariance
S  = [   4.9224   -0.0639    0.6477
        -0.0639    0.0978   -0.1626
         0.6477   -0.1626    1.2883];

%% Minnesota prior configuration
minnesota = struct;
minnesota.Center = 0.9;
minnesota.SelfLag = 100;
minnesota.CrossLag = 1;
minnesota.Decay = 2;
minnesota.VarianceX = 100;

%% Normal bayesian model
NormMdl = bayesvarm(3,2,'SeriesNames',simdata.Properties.VariableNames, ...
        minnesota, ...
        'ModelType','normal', 'Sigma',S);
NormEstMdl = estimate(NormMdl, simdata{:,:});

NormStruct = struct;
NormStruct.Prior.Phi.Mu = NormMdl.Mu;
NormStruct.Prior.Phi.V = NormMdl.V;
NormStruct.Prior.Sigma.Psi = NormMdl.Covariance;
NormStruct.Post.Phi.Mu = NormEstMdl.Mu;
NormStruct.Post.Phi.V = NormEstMdl.V;
NormStruct.Post.Sigma.Psi = NormEstMdl.Covariance;

%% Conjugate bayesian model
MNIWMdl = bayesvarm(3,2,'SeriesNames',simdata.Properties.VariableNames, ...
        minnesota, 'ModelType','conjugate');
MNIWEstMdl = estimate(MNIWMdl, simdata{:,:});

MNIWStruct = struct;
MNIWStruct.Prior.Phi.Mu = MNIWMdl.Mu;
MNIWStruct.Prior.Phi.V = MNIWMdl.V;
MNIWStruct.Prior.Sigma.nu = MNIWMdl.DoF;
MNIWStruct.Prior.Sigma.Psi = MNIWMdl.Omega;
MNIWStruct.Post.Phi.Mu = MNIWEstMdl.Mu;
MNIWStruct.Post.Phi.V = MNIWEstMdl.V;
MNIWStruct.Post.Sigma.nu = MNIWEstMdl.DoF;
MNIWStruct.Post.Sigma.Psi = MNIWEstMdl.Omega;

%% Semi-conjugate bayesian model
GibbsMdl = bayesvarm(3,2,'SeriesNames',simdata.Properties.VariableNames, ...
        minnesota, 'ModelType','semiconjugate');
GibbsEstMdl = estimate(GibbsMdl, simdata{:,:},'NumDraws',1e5);

GibbsStruct = struct;
GibbsStruct.Prior.Phi.Mu = GibbsMdl.Mu;
GibbsStruct.Prior.Phi.V = GibbsMdl.V;
GibbsStruct.Prior.Sigma.nu = GibbsMdl.DoF;
GibbsStruct.Prior.Sigma.Psi = GibbsMdl.Omega;
GibbsStruct.Post.Phi = GibbsEstMdl.CoeffDraws;
GibbsStruct.Post.Sigma = GibbsEstMdl.SigmaDraws;

%% Save results to struct
out = struct('Normal',NormStruct, ...
        'MNIW',MNIWStruct, ...
        'Gibbs',GibbsStruct, ...
        'minnesota', minnesota);
save('matlab_output.mat', '-struct', 'out');