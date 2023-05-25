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

# Sample m samples from each classification
m = 10
lp = [zeros(m,d) .+ Inf for _ = 1:4]
for c = 1:4, i = 1:m
    class = Inf
    while !insupport(prior1,lp[c][i,:]) || class != c
        lp[c][i,:] = rand(prior2c) + β * rand(kd)
        class = classify(lp[c][i,:])
    end
end

################################################
## TREATMENT REGIME AND PATIENT SIMULATOR
################################################

# Treatment regime
timeᵢ = [0;2:8] * 7.0
doseᵢ = (2.0*7.0):(8.0*7.0)
doseᵢ = doseᵢ[mod.(doseᵢ,7) .< 5]

# Model
model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(timeᵢ),output=:total)

################################################
## SIMULATE PATIENTS OF EACH CLASSIFICATION
################################################

# Simulate
f = hcat([[model(lpᵢ) for lpᵢ in eachrow(lp[c])] for c = 1:4]...);

# Plot
plts = plot.(f,xlim=(0.0,maximum(timeᵢ)),lw=2.0,c=:black)

# Change y limit and ticks
for p in plts
    ymax = max(2.0,ceil(ylims(p)[2]))
    plot!(p,ylim=(0.0,max(2.0,ceil(ymax))),yticks=[0.0,ymax])
    hline!(p,[1.0],α=0.3,lw=2.0,c=:black,ls=:dash)
    vline!(p,[14.0],α=0.3,lw=2.0,c=:black,ls=:dash)
end

plot(permutedims(plts,[2,1])...,layout=grid(m,4),size=(700,1500),legend=:none,xticks=0:14:56,xlabel="Time [d]",ylabel="Volume [FC]")
savefig("$(@__DIR__)/figS4.svg")