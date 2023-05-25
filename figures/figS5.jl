#=
    Figures S5

    Actual patients (remain)
=#

using CSV, DataFrames, DataFramesMeta
using Plots, StatsPlots
using DifferentialEquations
using Distributions
using .Threads
using StatsBase
using KernelDensity
using LinearAlgebra

include("defaults.jl")
include("../analysis/models.jl")
include("../analysis/data.jl")
include("../analysis/classifier.jl")

################################################
## LOAD PRIOR SAMPLES
################################################

include("../analysis/prior1.jl")
df = CSV.read("analysis/prior2.csv",DataFrame)
prior2 = Matrix(df)[:,4:end]
prior2c = [eachrow(prior2)...]

# Bandwidth (Silverman's rule)
n,d = size(prior2)
Σ = Diagonal((4 / (d + 2))^(1 / (d + 4)) * n^(-1 / (d + 4)) * std.(eachcol(prior2)))^2
kd = MvNormal(Σ)

# Expansion factor
β = 2.0

# Sample from prior
prior = zeros(40000,d) .+ Inf
for i = axes(prior,1)
    while !insupport(prior1,prior[i,:])
        prior[i,:] = rand(prior2c) + β * rand(kd)
    end
end
priorc = [eachrow(prior)...]

# Load noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α


################################################
## FUNCTION TO CALCULATE WEIGHTS
################################################

function get_weights(timeᵢ,doseᵢ,vols)
    # Observation times
    time = timeᵢ[eachindex(vols)]
    # Model
    model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(time),output=:total)
    # Calculate weights
    w = zeros(size(prior,1))
    @threads for i = eachindex(w)
        # Simulate model
        y = model(prior[i,:]).(time)
        # Compute likelihood
        w[i] = pdf(MvNormal(y,α[1] .+ α[2] * y),vols)
    end
    return w / sum(w)
end

################################################
## GET REMAINING PATIENTS
################################################

remaining_id = setdiff(unique(data.ID),[patient_id; training_id])

################################################
## PERFORM INFERENCE USING PARTIAL DATA
################################################

# Up to the following measurement times
upto = [2,3,4,5,6,7,8]

# Initialise storage for weights and synthetic patient data
w = [Array{Vector}(undef,length(upto)) for _ = 1:length(remaining_id)]

# Loop through each patient
for (i,id) in enumerate(remaining_id)

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Loop through observation times (to get weights)
    for j = eachindex(upto)
        max_idx = min(length(timeᵢ),upto[j])
        w[i][j] = get_weights(timeᵢ,doseᵢ,volsᵢ[1:max_idx])
    end

end

################################################
## Figure S5
################################################

plts = [Array{Any}(undef,length(upto)) for _ = 1:length(remaining_id)]

# Loop through plots
for (i,id) = enumerate(remaining_id), j = eachindex(upto)

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Fix for some patients with fewer observations
    max_idx = min(length(timeᵢ),upto[j])

    # Patient specific model
    model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(timeᵢ),output=:total)

    # Make predictions
    lps = sample(priorc,Weights(w[i][j]),200)
    plts[i][j] = plot([model(lp) for lp in lps],label="",c=:black,α=0.05,xlim=(0.0,timeᵢ[max_idx]))
    plot!(plts[i][j],[model(lp) for lp in lps],label="",c=col_posterior,α=0.1,xlim=(timeᵢ[max_idx],maximum(timeᵢ)))

    # Plot observed data at present
    scatter!(plts[i][j],timeᵢ[1:max_idx],volsᵢ[1:max_idx],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[1:max_idx],volsᵢ[1:max_idx],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Plot future observed data
    scatter!(plts[i][j],timeᵢ[max_idx+1:end],volsᵢ[max_idx+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[max_idx+1:end],volsᵢ[max_idx+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

    # Plot present time
    vline!(plts[i][j],[timeᵢ[max_idx]],c=:black,lw=2.0,ls=:dash,label="",widen=true)
    
end

plts_patient = [plot(plts[i][1:2:end]...,layout=grid(1,4),link=:all,ylim=(0.0,2.0),xticks=0:14:56,xlim=(-4.0,60.0)) for i in eachindex(remaining_id)]

# Figure S5
figS5 = plot(plts_patient...,layout=grid(7,1),size=(800,1200),xlabel="Time [d]",link=:all)
savefig(figS5,"$(@__DIR__)/figS5.svg")
