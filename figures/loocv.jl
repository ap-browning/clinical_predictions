#=
    loocv.jl

    Perform leave-out-one-cross-validation (LOOCV)
=#

using CSV, DataFrames, DataFramesMeta
using Plots, StatsPlots
using Distributions
using LinearAlgebra
using StatsBase
using .Threads

include("defaults.jl")
include("../analysis/models.jl")
include("../analysis/data.jl")
include("../analysis/inference.jl")

################################################
## LOAD PRIOR AND POSTERIOR (ALL PATIENTS)
################################################

include("../analysis/prior1.jl")
posterior1_allpatients = CSV.read("analysis/posterior1_allpatients.csv",DataFrame)

################################################
## PRIOR SAMPLES FCN
################################################

# Sample `m` samples from a prior constructed that excludes patient `id`
function sample_prior2(id,m=100000)

    # Samples from all patients except the patient of interest
    X = Matrix(@subset(posterior1_allpatients,:id .!= id)[:,5:end])
    Xc = [eachrow(X)...]

    # Bandwidth (Silverman's rule)
    n,d = size(X)
    Σ = Diagonal((4 / (d + 2))^(1 / (d + 4)) * n^(-1 / (d + 4)) * std.(eachcol(X)))^2
    kd = MvNormal(Σ)

    # Expansion factor
    β = 2

    # Create samples
    Y = zeros(m,d) .+ Inf
    for i = axes(Y,1)
        while !insupport(prior1,Y[i,:])
            Y[i,:] = rand(Xc) + β * rand(kd)
        end
    end

    return Y
end



################################################
## CALCULATE WEIGHTS
################################################

# Store prior samples, weights, and model predictions
P = Array{Any}(undef,length(unique(data.ID)))
W = Array{Any}(undef,length(unique(data.ID)))
M = Array{Any}(undef,length(unique(data.ID)))

# Number of samples in prior (will increase this....)
n = 100000

# Loop through all patients
@time @threads for i = eachindex(unique(data.ID))

    # ID
    id = unique(data.ID)[i]

    # Get patient characteristics
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Sample from prior (using all patients except the patient of interest)
    P[i] = sample_prior2(id,n)

    # Matrix of weights
    W[i],M[i] = get_weights(P[i],timeᵢ,doseᵢ,volsᵢ)

end


################################################
## PRODUCE DIAGNOSTIC PLOTS
################################################

# Store R²
R² = zeros(length(unique(data.ID)),10) ./ 0.0

# Store times
T = copy(R²)

# Calculate R²
for i = eachindex(unique(data.ID))

    # ID
    id = unique(data.ID)[i]

    # Get patient characteristics
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Calculate R² (including predictions)
    for j = eachindex(timeᵢ)
        T[i,j] = timeᵢ[j]
        j == 0 && continue
        R²[i,j] = median([Bayesian_R²(M[i][idx,:],volsᵢ) for idx in sample(1:size(P[i],1),Weights(W[i][:,j]),1000)])
    end
end

figSX = scatter(T[:,2:end][:],R²[:,2:end][:],c=:black,label="",α=0.5,xticks=0:7:70,
            xlabel="Time [d]",ylabel="Median R²",xlim=(0.0,63.0),widen=true)

savefig(figSX,"$(@__DIR__)/figSX.svg")


################################################
## PRODUCE PLOTS OF PREDICTIONS (NON-SMOOTH, DIANOSTIC ONLY)
################################################

# Default plot styling
default()

# Which intermediate time point?
pt = 4

# Loop through all patients
for i = eachindex(unique(data.ID))

    # ID
    id = unique(data.ID)[i]

    # Get patient characteristics
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Indices to plot predictions from
    pts = [2,pt,length(timeᵢ)]

    # Create plots
    plts = [plot() for i = 1:4]

    # Loop through pts
    for (j,upto) = enumerate(pts)

        # Plot predictions
        idx = sample(1:size(P[i],1),Weights(W[i][:,upto]),200)
        plot!(plts[j],timeᵢ[1:upto],M[i][idx,1:upto]',label="",c=:black,α=0.05)
        plot!(plts[j],timeᵢ[upto:end],M[i][idx,upto:end]',label="",c=col_posterior,α=0.05)

        # Plot observed data at present
        scatter!(plts[j],timeᵢ[1:upto],volsᵢ[1:upto],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
        scatter!(plts[j],timeᵢ[1:upto],volsᵢ[1:upto],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

        # Plot future observed data
        scatter!(plts[j],timeᵢ[upto+1:end],volsᵢ[upto+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
        scatter!(plts[j],timeᵢ[upto+1:end],volsᵢ[upto+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

        # Plot present time
        vline!(plts[j],[timeᵢ[upto]],c=:black,lw=2.0,ls=:dash,label="",widen=true)
    
        # Format
        plot!(plts[j],xticks=0:14:Int(ceil(maximum(timeᵢ)/7)*7),xlim=(-4.0,maximum(timeᵢ) + 4.0),ylim=(0.0,2.0),ylabel="GTV [FC]",xlabel="Time [d]")

    end
    
    # Plot R² diagnostic
    plot!(plts[end],T[i,:],R²[i,:],c=:black,lw=2.0,xticks=0:14:Int(ceil(maximum(timeᵢ)/7)*7),xlim=(-4.0,maximum(timeᵢ) + 4.0),label="",ylim=(0.0,1.0),ylabel="R²", xlabel="Time [d]")

    # Plot and save
    fig = plot(plts...,layout=grid(1,4),size=(1000,200),left_margin=5Plots.mm,bottom_margin=7Plots.mm)
    savefig(fig,"analysis/diagnostic/loocv/patient_id_$id.png")

end