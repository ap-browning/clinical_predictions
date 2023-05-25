#=
    Figure S2

    Parameter sweep (classification, with and without noise)
=#

using Plots, StatsPlots
using DifferentialEquations
using CSV, DataFrames

include("defaults.jl")
include("../analysis/models.jl")
include("../analysis/data.jl")
include("../analysis/classifier.jl")

# Noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α

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
ζ = range(0.01,2.0,40)
η = range(0.01,1.0,50)

# Store classifications
C = zeros(length(ζ),length(η))
Ĉ = zeros(length(ζ),length(η))

# Produce plots
plts = [plot() for _ = ζ, _ = η]
for i = eachindex(ζ), j = eachindex(η)
   
    lp = log.([λ,K,γ,ζ[i],η[j],ϕ₀])
    C[i,j] = classify(lp)
    Ĉ[i,j] = classify(lp;α₁=α[1],α₂=α[2])

end

# Colormap
v = [
    RGB(65/255,171/255,93/255),
    RGB(240/255,105/255,19/255),
    RGB(54/255,144/255,192/255),
    RGB(221/255,52/255,150/255)
]

# Plots
figS2a = heatmap(ζ,η,C',c=cgrad(v),clim=(1,4),xlabel="ζ",ylabel="η",legend=:none)
figS2b = heatmap(ζ,η,Ĉ',c=cgrad(v),clim=(1,4),xlabel="ζ",ylabel="η",legend=:none)

# Fig S2
figS2 = plot(figS2a,figS2b,size=(700,300),lw=2.0)

# Save
savefig(figS2,"$(@__DIR__)/figS2.svg")