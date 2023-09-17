#=
    Figure 1

    Clinical data, motivation figure and verification of model.

=#

using DataFrames, DataFramesMeta, Plots, StatsPlots, CSV
using DifferentialEquations
using Distributions
using Optim
using LinearAlgebra
using AdaptiveMCMC
using .Threads

include("defaults.jl")

################################################
## LOAD DATA AND MODELS, AND PRIOR
################################################

include("../analysis/data.jl")
include("../analysis/models.jl")
include("../analysis/inference.jl")
include("../analysis/prior1.jl")

################################################
## LOAD POSTERIOR1 (ALL PATIENTS)
################################################

posterior1_allpatients = CSV.read("analysis/posterior1_allpatients.csv",DataFrame)

################################################
## PRODUCE PLOTS
################################################

# Setup
nsim = 1000

# Store plots
plts = [plot() for _ = eachindex(patient_id)]

# Loop through patients
for i = eachindex(patient_id)
    
    # Patient id
    id = patient_id[i]

    # Load patient data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Plotting vector
    tplt = range(0.0,56.0,200)
    
    # Posterior samples for this patient
    df = @subset(posterior1_allpatients,:id .== id)

    # Posterior sample with highest density
    lp = Vector(df[findmax(df.D)[2],:][5:end])

    # 95% prediction interval
    vsim = hcat([model(Vector(df[idx,5:end]),tplt,doseᵢ) for idx = rand(1:nrow(df),nsim)]...)
    l,u = [[quantile(v,q) for v in eachrow(vsim)] for q = [0.025,0.975]]

    # Plot model fit
    plot!(plts[i],tplt,model(lp),frange=(l,u),c=col_prior2,lw=2.0,label="",fα=0.3)

    # Plot data
    scatter!(plts[i],timeᵢ,volsᵢ,ylim=(0.0,:auto),widen=true,msc=:black,ms=8.0,msw=2.0,mc=:white,label="")
    scatter!(plts[i],timeᵢ,volsᵢ,ylim=(0.0,:auto),widen=true,c=:black,label="")

    # Formatting
    plot!(xticks=0:7:56.0,box=:on)

end

# Figure 1
fig1 = plot(plts...,ylim=(0.0,1.6),size=(700,450),xlim=(0.0,56.0),xlabel="Time [d]",ylabel="Volume [FC]")
add_plot_labels!(fig1)
savefig(fig1,"$(@__DIR__)/fig1.svg")
fig1