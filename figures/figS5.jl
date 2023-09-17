#=
    Figure S5

    Additional synthetic patients
=#

using CSV, DataFrames, DataFramesMeta
using Plots, StatsPlots
using StatsBase
using LinearAlgebra
using Random 
using .Threads

include("defaults.jl")
include("../analysis/models.jl")
include("../analysis/data.jl")
include("../analysis/inference.jl")

################################################
## PLOT PARAMETERS
################################################

tmax = 56.0
tplt = range(0.0,tmax,200)

################################################
## LOAD PRIOR SAMPLES
################################################

prior2 = CSV.read("analysis/prior2.csv",DataFrame)
prior2.row = 1:nrow(prior2)
X = Matrix(prior2[:,2:end-1])
Xc = [eachrow(X)...]

################################################
## CREATE THREE SYNTHETIC PATIENTS (from within posterior1)
################################################

# Load posterior1
posterior1 = CSV.read("analysis/posterior1.csv",DataFrame)
posterior1.row = 1:nrow(posterior1)

# Load noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α

# Treatment regime
timeᵢ = [0;2:8] * 7.0
doseᵢ = (2.0*7.0):(8.0*7.0)
doseᵢ = doseᵢ[mod.(doseᵢ,7) .< 5]

# Model
model(lp) = model(lp,timeᵢ,doseᵢ;ret=:function)

# Synthetic patient parameters (now sampled from full posterior)
rng = MersenneTwister(4)
patient_rows = [rand(rng,@subset(posterior1,:class .== i).row) for i = [1,1,2,2,3,3,4,4]]
lp = [Vector(posterior1[row,5:end]) for row in patient_rows]

# Store noisy volume measurements
vols = Array{Vector}(undef,length(patient_rows))

# Loop through each patient, add noise
for i in eachindex(patient_rows)

    # Generate synthetic data
    vols[i] = model(lp[i]).(timeᵢ)
    vols[i][2:end] += rand(rng,MvNormal(α[1] .+ α[2] * vols[i][2:end]))

end

################################################
## PERFORM INFERENCE USING PARTIAL DATA
################################################

W = Array{Any}(undef,length(patient_rows))
M = Array{Any}(undef,length(patient_rows))
M̄ = Array{Any}(undef,length(patient_rows))

# Loop through each patient
@threads for i = eachindex(patient_rows)
    
    # Calculate weights
    W[i],M[i] = get_weights(X,α,timeᵢ,doseᵢ,vols[i])

    # Calculate full model traces for plotting
    M̄[i] = hcat([model(lp,timeᵢ,doseᵢ;ret=:function).(tplt) for lp = Xc]...)'

end

################################################
## Figure S5
################################################

# Which time indices to plot upto?
upto = [2,4,6,8]

plts = [Array{Any}(undef,length(upto)) for _ = 1:length(patient_rows)]

# Loop through plots
for i = eachindex(patient_rows), j = eachindex(upto)

    # Confidence bands
    l1,l2,u2,u1 = [[quantile(v,Weights(W[i][:,upto[j]]),q) for v in eachcol(M̄[i])] for q = [0.025,0.25,0.75,0.975]]

    # Mean
    μ = [mean(v,Weights(W[i][:,upto[j]])) for v in eachcol(M̄[i])]

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
    scatter!(plts[i][j],timeᵢ[1:upto[j]],vols[i][1:upto[j]],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[1:upto[j]],vols[i][1:upto[j]],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Plot future observed data
    scatter!(plts[i][j],timeᵢ[upto[j]+1:end],vols[i][upto[j]+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[upto[j]+1:end],vols[i][upto[j]+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

    # Plot present time
    vline!(plts[i][j],[timeᵢ[upto[j]]],c=:black,lw=2.0,ls=:dash,label="",widen=true)

    # Calculate and plot R² value
    R² = median([Bayesian_R²(M[i][idx,:],vols[i]) for idx = sample(1:nrow(prior2),Weights(W[i][:,upto[j]]),1000)])
    R² = round(R²,sigdigits=3)
    plot!(plts[i][j],title="R² = $R²",titlefontsize=10)

end

plts_patient = [plot(plts[i]...,layout=grid(1,4),link=:all,ylim=(0.0,1.3),xticks=0:14:maximum(timeᵢ),xlim=(-4.0,maximum(timeᵢ) + 4.0)) for i in eachindex(patient_rows)]
plot!(plts_patient[5],ylim=(0.0,2.2))
plot!(plts_patient[6],ylim=(0.0,2.2))
plot!(plts_patient[7],ylim=(0.0,9.2),yticks=(0:3:9,["0.0","3.0","6.0","9.0"]))
plot!(plts_patient[8],ylim=(0.0,3.2),yticks=(0:3,["0.0","1.0","2.0","3.0"]))


# Figure 5
figS5 = plot(plts_patient...,layout=grid(8,1),size=(800,1200),xlabel="Time [d]",link=:all)
savefig(figS5,"$(@__DIR__)/figS5.svg")
figS5