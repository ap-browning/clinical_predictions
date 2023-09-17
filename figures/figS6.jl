#=
    Figures S6

    Actual patients (remain)
=#

using CSV, DataFrames, DataFramesMeta
using Plots, StatsPlots
using StatsBase
using LinearAlgebra
using .Threads

include("defaults.jl")
include("../analysis/models.jl")
include("../analysis/data.jl")
include("../analysis/classifier.jl")
include("../analysis/inference.jl")

################################################
## LOAD PRIOR SAMPLES
################################################

prior2 = CSV.read("analysis/prior2.csv",DataFrame)
prior2.row = 1:nrow(prior2)
X = Matrix(prior2[:,2:end-1])
Xc = [eachrow(X)...]

# Load noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α

################################################
## GET REMAINING PATIENTS
################################################

remaining_id = setdiff(unique(data.ID),[patient_id; training_id])

################################################
## PERFORM INFERENCE USING PARTIAL DATA
################################################

W = Array{Any}(undef,length(remaining_id))
M = Array{Any}(undef,length(remaining_id))
M̄ = Array{Any}(undef,length(remaining_id))

# Loop through each patient
@threads for i = eachindex(remaining_id)

    # Patient id
    id = remaining_id[i]

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Calculate weights
    W[i],M[i] = get_weights(X,α,timeᵢ,doseᵢ,volsᵢ)

    # Calculate full model traces for plotting
    M̄[i] = hcat([model(lp,timeᵢ,doseᵢ;ret=:function).(tplt) for lp = Xc]...)'

end

################################################
## Figure S6
################################################
 
# Which time indices to plot upto?
upto = [2,4,6,8]

plts = [[plot() for _ = eachindex(upto)] for _ = eachindex(remaining_id)]

# Loop through plots
for (i,id) = enumerate(remaining_id), j = eachindex(upto)

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Plotting vector
    tplt = range(0.0,maximum(timeᵢ),200)

    # Adjust for patients with only 7 patients
    if j == length(upto)
        upto[end] = length(timeᵢ)
    end

    # Confidence bands
    l1,l2,u2,u1 = [[quantile(v,Weights(W[i][:,upto[j]]),q) for v in eachcol(M̄[i])] for q = [0.025,0.25,0.75,0.975]]

    # Mean
    μ = [mean(v,Weights(W[i][:,upto[j]])) for v in eachcol(M̄[i])]

    display([i,j])

    # Index of current upto time
    idx_max = tplt[end] == timeᵢ[upto[j]] ? length(tplt) : findfirst(tplt .> timeᵢ[upto[j]])


    # Plot confidence bands
    plts[i][j] = plot(tplt[1:idx_max],u1[1:idx_max],frange=l1[1:idx_max],c=:black,lw=0.0,label="",fα=0.4)
    plot!(plts[i][j],tplt[1:idx_max],u2[1:idx_max],frange=l2[1:idx_max],c=:black,lw=0.0,label="",fα=0.4)
    plot!(plts[i][j],tplt[idx_max:end],u1[idx_max:end],frange=l1[idx_max:end],c=col_posterior,lw=0.0,label="",fα=0.4)
    plot!(plts[i][j],tplt[idx_max:end],u2[idx_max:end],frange=l2[idx_max:end],c=col_posterior,lw=0.0,label="",fα=0.4)

    # Plot mean
    plot!(plts[i][j],tplt[1:idx_max],μ[1:idx_max],c=:black,lw=2.0,label="",fα=0.4)
    plot!(plts[i][j],tplt[idx_max:end],μ[idx_max:end],c=col_posterior,lw=2.0,label="",fα=0.4)

    # Plot observed data at present
    scatter!(plts[i][j],timeᵢ[1:upto[j]],volsᵢ[1:upto[j]],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[1:upto[j]],volsᵢ[1:upto[j]],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Plot future observed data
    scatter!(plts[i][j],timeᵢ[upto[j]+1:end],volsᵢ[upto[j]+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[upto[j]+1:end],volsᵢ[upto[j]+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

    # Plot present time
    vline!(plts[i][j],[timeᵢ[upto[j]]],c=:black,lw=2.0,ls=:dash,label="",widen=true)
    
    # Calculate and plot R² value
    R² = median([Bayesian_R²(M[i][idx,:],volsᵢ) for idx = sample(1:nrow(prior2),Weights(W[i][:,upto[j]]),1000)])
    R² = round(R²,sigdigits=3)
    plot!(plts[i][j],title="R² = $R²",titlefontsize=10)

end

plts_patient = [plot(plts[i]...,layout=grid(1,4),link=:all,ylim=(0.0,2.0),xticks=0:14:56,xlim=(-4.0,60.0)) for i in eachindex(remaining_id)]

# Figure S6
figS6 = plot(plts_patient...,layout=grid(length(remaining_id),1),size=(800,1200),xlabel="Time [d]",link=:all)
savefig(figS6,"$(@__DIR__)/figS6.svg")
figS6