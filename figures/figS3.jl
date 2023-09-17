#=
    Figure S3

    Show individual fits for patients in the training set
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
## LOAD POSTERIOR
################################################

posterior1 = CSV.read("analysis/posterior1.csv",DataFrame)

################################################
## PLOT FOR EACH PATIENT IN THE TRAINING SET
################################################

plts = [plot() for _ = training_id]

for (i,id) = enumerate(sort(training_id))

    # Samples
    X = Matrix(@subset(posterior1,:id .== id)[:,5:end])

    # Model fits
    M = hcat([model(lp,id) for lp in eachrow(X)]...)

    # Confidence bands and mean
    l1,l2,u2,u1 = [[quantile(v,q) for v in eachrow(M)] for q = [0.025,0.25,0.75,0.975]]
    μ = mean(M,dims=2)

    # Data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Plot confidence bands
    plot!(plts[i],timeᵢ,u1,frange=l1,c=:black,lw=0.0,label="",fα=0.3)
    plot!(plts[i],timeᵢ,u2,frange=l2,c=:black,lw=0.0,label="",fα=0.3)

    # Plot data
    #scatter!(plts[i],timeᵢ,volsᵢ,ylim=(0.0,:auto),widen=true,msc=:black,ms=8.0,msw=2.0,mc=:white,label="")
    scatter!(plts[i],timeᵢ,volsᵢ,ylim=(0.0,:auto),widen=true,c=:black,label="")

    # Style
    plot!(plts[i],xlim=(0.0,63.0),xticks=0:14:63,ylim=(0.0,2.5),yticks=0.0:1.0:2.0,xlabel="Time [d]",ylabel="GTV [FC]")
    plot!(plts[i],title="Patient ID = $id")

end

################################################
## PRODUCE FIGURE
################################################

figS3 = plot(plts...,layout=grid(8,5),size=(1500,2000),right_margin=15Plots.mm)
savefig(figS3,"$(@__DIR__)/figS3.svg")