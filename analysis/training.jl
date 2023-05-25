#=
    training.jl

    TRAIN MODEL (AT THE POPULATION LEVEL)

    1. Estimate noise model
    2. Define prior distributions
=#

using DataFrames, DataFramesMeta, Plots, StatsPlots, CSV
using DifferentialEquations
using Distributions
using Optim
using LinearAlgebra
using AdaptiveMCMC
using Random
using .Threads


################################################
## LOAD DATA AND MODELS
################################################

include("../analysis/data.jl")
include("../analysis/models.jl")

################################################
## ESTIMATE NOISE MODEL
################################################

# Loop through patients in training data set and fit using least squares
fitval = Float64[]
resids = Float64[]
for id = training_id

    # Get data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Fit model (least squares)
    lp₀ = log.([0.5,1.0,0.5,0.5,0.5,0.01])
    model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(timeᵢ),output=:total)
    resid = lp -> norm(model(lp).(timeᵢ) - volsᵢ)
    optim = optimize(resid,lp₀)
    
    # Calculate and store residuals as a function of yfitted
    y = model(optim.minimizer).(timeᵢ)
    ε = volsᵢ - y

    # Store
    fitval = [fitval; y]
    resids = [resids; ε]
    
end

# Store results (supplementary figure)
CSV.write("analysis/residuals.csv",DataFrame(fitval=fitval,resids=resids))

# Fit noise model (maximum likelihood)
#  ε ~ N(0.0, α₁ + α₂ * y)
loglike = α -> logpdf(MvNormal(α[1] .+ α[2] * fitval),resids)
α̂ = optimize(α -> -loglike(α),[0.1,0.1]).minimizer

# Store noise model
CSV.write("analysis/noise_model.csv",DataFrame(α=α̂))

################################################
## POPULATION-LEVEL POSTERIOR
################################################

# Load noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α

# Store posteriors and likelihood densities
D = Array{Vector}(undef,length(training_id))
X = Array{Matrix}(undef,length(training_id))

# Load prior
include("prior1.jl")

# Loop through patients
@time begin
    @threads for i = eachindex(training_id)

        # Patient id
        id = training_id[i]

        # Get data
        timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

        # Model
        lp₀ = log.([0.5,1.0,0.5,0.5,0.5,0.01])
        model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(timeᵢ),output=:total,verbose=false)
        loglike = lp -> begin
            m = model(lp).(timeᵢ)
            σ = α[1] .+ α[2] * m
            loglikelihood(Normal(),(volsᵢ - m) ./ σ)
        end
        logpost = lp -> insupport(prior1,lp) ? loglike(lp) + logpdf(prior1,lp) : -Inf

        # MCMC
        @time res = adaptive_rwm(lp₀,logpost,110000,algorithm=:aswam,thin=10,b=10000)

        # Store posteriors
        D[i] = res.D[1]
        X[i] = res.X

    end
end


################################################
## STORE PRIOR2
################################################

df = DataFrame(
    [vcat([fill(training_id[i],length(D[i][1:10:end])) for i = eachindex(training_id)]...) vcat([d[1:10:end] for d in D]...) vcat([d[1:10:end] for d in D]...) hcat([x[:,1:10:end] for x in X]...)'],
    ["id";"D";"class";"log_" .* twocompartment_params]
)

################################################
## CLASSIFY SAMPLES
################################################

# Include classifier
include("classifier.jl")

# Classify each posterior sample
df.class = [classify(lp) for lp in eachrow(Matrix(df[!,4:end]))]

################################################
## SAVE POSTERIORS
################################################

CSV.write("analysis/prior2.csv",df)

################################################
## BREAKDOWN OF CLASSES
################################################

P = [count(df.class .== i) / nrow(df) for i = 1:4]
df2 = DataFrame(class=1:4,proportion=P)
CSV.write("analysis/class_proportions.csv",df2)