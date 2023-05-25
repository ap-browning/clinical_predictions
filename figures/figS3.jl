#=
    Figure S3

    Residuals and noise model from least squares fit.
=#

using CSV, DataFrames, DataFramesMeta
using Distributions
using Plots, StatsPlots

include("defaults.jl")

# Load noise model parameters
α = CSV.read("analysis/noise_model.csv",DataFrame).α
 
# Load fitted values and residuals
df = CSV.read("analysis/residuals.csv",DataFrame)

# Plot residuals
figS3 = scatter(fitval,resids,xlabel="Fitted Values",ylabel="Residual",label="",
            c=:black,α=0.4,xlim=(0.0,2.0),ylim=(-0.35,0.35))

# Std and confidence interval
y = range(0.0,2.0,200)              # Fitted values
μ = fill(0.0,length(y))             # Mean
σ = α̂[1] .+ α̂[2] * y                # Std
c = quantile(Normal(),0.975) * σ    # 95% CI

plot!(figS3,y,μ,ribbon=σ,c=:red,ls=:dash,lw=0.0,fα=0.2,label="",z_order=:back)
plot!(figS3,y,μ,ribbon=c,c=:red,ls=:dash,lw=2.0,fα=0.2,label="",z_order=:back)

# Save figure
plot!(figS3,size=(350,250))
savefig("$(@__DIR__)/figS3.svg")