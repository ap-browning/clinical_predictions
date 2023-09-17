#=
    Figures S8

    Reproduce Fig. 7 using uninformative prior
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
## PLOT PARAMETERS
################################################

tmax = 56.0
tplt = range(0.0,tmax,200)

################################################
## LOAD PRIOR
################################################

include("../analysis/prior1.jl")

################################################
## GUESS
################################################

lp₀ = log.([0.5,1.0,0.5,0.5,0.5,0.01])

################################################
## INFERENCE
################################################

## Load noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α

## Chain settings
iters   = 150000    # Total iterations
burnin  = 50000     # Iterations discarded as burnin
nchains = 4         # Number of independent chains
thin    = 100       # Thinning (samples to discard)
L       = 4         # Number of temperature levels

## Which time indices to produce predictions up to
upto = [2,4,6,8]

## Store results (model fits)
M = Array{Any}(undef,length(patient_id))
M̄ = Array{Any}(undef,length(patient_id))

## Loop through patients and upto
@threads for i = eachindex(patient_id)

    M̄[i] = Array{Matrix}(undef,length(upto))
    M[i] = Array{Matrix}(undef,length(upto))

    for j = eachindex(upto)

        # Patient id
        id = patient_id[i]

        # Get patient specific data
        timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

        # Construct log density
        llike = lp -> loglike(lp,α,timeᵢ[1:upto[j]],volsᵢ[1:upto[j]],doseᵢ)
        lpost = lp -> insupport(prior1,lp) ? llike(lp) + logpdf(prior1,lp) : -Inf

        # Perform MCMC
        c = mcmc(lpost,lp₀,iters;L,burnin,thin,nchains,param_names=twocompartment_params)

        # Obtain parameter values
        lp = vcat([c.value.data[:,:,i] for i = 1:nchains]...)

        # Solve model
        M[i][j] = hcat([model(lpᵢ,timeᵢ,doseᵢ) for lpᵢ in eachrow(lp)]...)
        M̄[i][j] = hcat([model(lpᵢ,tplt,doseᵢ) for lpᵢ in eachrow(lp)]...)

    end

end

################################################
## Figure S7
################################################
 
plts = [[plot() for _ = eachindex(upto)] for _ = eachindex(patient_id)]

# Loop through plots
for (i,id) = enumerate(patient_id), j = eachindex(upto)

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Confidence bands
    l1,l2,u2,u1 = [[quantile(v,q) for v in eachrow(M̄[i][j])] for q = [0.025,0.25,0.75,0.975]]

    # Mean
    μ = [mean(v) for v in eachrow(M̄[i][j])]

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
    R² = median([Bayesian_R²(m,volsᵢ) for m = eachcol(M[i][j])])
    R² = round(R²,sigdigits=3)
    plot!(plts[i][j],title="R² = $R²",titlefontsize=10)

end

plts_patient = [plot(plts[i]...,layout=grid(1,4),link=:all,ylim=(0.0,2.0),xticks=0:14:56,xlim=(-4.0,60.0)) for i in eachindex(patient_id)]

# Figure S8
figS8 = plot(plts_patient...,layout=grid(4,1),size=(800,700),xlabel="Time [d]",link=:all)
savefig(figS8,"$(@__DIR__)/figS8.svg")
figS8
