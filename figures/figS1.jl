#=
    Figure S1

    Parameter sweep
=#

using Plots, StatsPlots
using DifferentialEquations

include("defaults.jl")
include("../analysis/models.jl")
include("../analysis/data.jl")

# Treatment regime
timeᵢ = [0;2:8] * 7.0
doseᵢ = (2.0*7.0):(8.0*7.0)
doseᵢ = doseᵢ[mod.(doseᵢ,7) .< 5]

# Fixed parameters
λ  = 1.0
K  = 5.0
γ  = 0.3
ϕ₀ = 0.2

# Parameters to sweep
ζ = [0.1,0.5,1.0,1.5,2.0]
η = [0.1,0.25,0.5,0.75,1.0]

# Produce plots
plts = [plot() for _ = ζ, _ = η]
for i = eachindex(ζ), j = eachindex(η)
    p = λ,K,γ,ζ[i],η[j],ϕ₀ 
    sol = solve_model_twocompartment(p;tdose=doseᵢ)

    plot!(plts[i,j],t -> sum(sol(t)),xlim=(0.0,56.0),c=:black,label="")
    plot!(plts[i,j],t -> sol(t)[2],xlim=(0.0,56.0),c=:red,label="")
end

# Fig S2
figS1 = plot(plts...,size=(1000,800),lw=2.0,xticks=0:14:56,widen=true)

# Adjust y limits
[plot!(figS1,subplot=i,ylim=(0.0,15.0),yticks=0:5:15) for i = [1,6,11]]
[plot!(figS1,subplot=i,ylim=(0.0,6.0),yticks=0:2:6) for i = [2:5; 7:10; 12:15]]
[plot!(figS1,subplot=i,ylim=(0.0,9.0),yticks=0:3:9) for i = [16]]
[plot!(figS1,subplot=i,ylim=(0.0,3.0),yticks=0:1:3) for i = 17:20]
[plot!(figS1,subplot=i,ylim=(0.0,1.5),yticks=0:0.5:1.5) for i = 22:25]
[plot!(figS1,subplot=i,ylim=(0.0,3.3),yticks=0:1:3) for i = [21]]
figS1

# Save
savefig("$(@__DIR__)/figS1.svg")