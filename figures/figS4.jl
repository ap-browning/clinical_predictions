#=
    Figure S4

    Demonstrate classification algorithm

=#

using Plots, StatsPlots
using DifferentialEquations
using Distributions
using KernelDensity
using LinearAlgebra

include("defaults.jl")
include("../analysis/models.jl")
include("../analysis/data.jl")
include("../analysis/classifier.jl")

################################################
## LOAD PRIOR SAMPLES
################################################

prior2 = CSV.read("analysis/prior2.csv",DataFrame)
prior2.row = 1:nrow(prior2)

X = Matrix(prior2[:,2:end-1])
Xc = [eachrow(X)...]

################################################
## SAMPLE PARAMETER VALUES UNDER EACH CLASSIFICATION
################################################

# Number in each class
neach = 8

# Rows to look at
idx = [sample(@subset(prior2,:class .== i).row,8,replace=false) for i = 1:4]

# Parameter values to look at
lp = [Xc[i] for i in idx]

################################################
## TREATMENT REGIME AND PATIENT SIMULATOR
################################################

# Treatment regime
timeᵢ = [0;2:8] * 7.0
doseᵢ = (2.0*7.0):(8.0*7.0)
doseᵢ = doseᵢ[mod.(doseᵢ,7) .< 5]

# Model
model(lp) = model(lp,timeᵢ,doseᵢ;ret=:function)

################################################
## SIMULATE PATIENTS OF EACH CLASSIFICATION
################################################

# Simulate
f = hcat([[model(lpᵢ) for lpᵢ in lp[c]] for c = 1:4]...);

# Plot
plts = plot.(f,xlim=(0.0,maximum(timeᵢ)),lw=2.0,c=:black)

# Change y limit and ticks
for p in plts
    ymax = max(2.0,ceil(ylims(p)[2]))
    plot!(p,ylim=(0.0,max(2.0,ceil(ymax))),yticks=[0.0,ymax])
    hline!(p,[1.0],α=0.3,lw=2.0,c=:black,ls=:dash)
    vline!(p,[14.0],α=0.3,lw=2.0,c=:black,ls=:dash)
end

figS4 = plot(permutedims(plts,[2,1])...,layout=grid(neach,4),size=(1000,1400),right_margin=10Plots.mm,legend=:none,
    xticks=0:14:56,xlabel="Time [d]",ylabel="Volume [FC]")
savefig("$(@__DIR__)/figS4.svg")