# BayesianVARs.jl

BayesianVARs.jl is a Julia package for estimating Bayesian Vector Autoregressions (VARs). This package provides tools for defining, estimating, and simulating VAR models with Bayesian priors.

## Installation

To install BayesianVARs.jl, you can use the Julia package manager. Open the Julia REPL and run:

```julia
using Pkg
Pkg.add("BayesianVARs")
```

## Usage

### Defining a VAR Model

You can define a VAR model using the `VARModel` and `VARMeta` types. For example:

```julia
using BayesianVARs

meta = VARMeta(3, 2, ["GDP", "Inflation", "Interest Rate"])  # 3 variables, 2 lags
constant = [0.1, 0.2, 0.3]
ar = ([0.5 0.1 0.0; 0.0 0.5 0.1; 0.1 0.0 0.5],)
covariance = [0.1 0.0 0.0; 0.0 0.1 0.0; 0.0 0.0 0.1]

model = VARModel(meta, constant, ar, covariance)
```

You can omit the constant (default is a vector of zeros) and/or omit the `meta` argument and have the package infer the metadata from the shapes of `ar` and `covariance`. The default variable names are `:Y1`, `:Y2`, etc.

### Setting Priors

TBD

### Estimating a VAR Model

You can estimate a VAR model using the `estimate` function. For example:

```julia
data = randn(100, 3)  # 100 observations of 3 variables
priors = minnesota(3, 2)

estimated_model = estimate(priors, data)
```

### Simulating a VAR Model

You can simulate data from a VAR model using the `simulate` function. For example:

```julia
simulated_data = simulate(model, 100)
```

### Impulse Response Functions

You can compute impulse response functions (IRFs) using the `irf` function. For example:

```julia
irfs = irf(model, 10)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.