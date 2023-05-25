# Generate synthetic data in same format as clinical data (clinical data available upon request)

using CSV, DataFrames, DataFramesMeta
using Plots, StatsPlots
using DifferentialEquations
using Distributions
using .Threads
using StatsBase
using KernelDensity
using LinearAlgebra

include("../analysis/models.jl")
include("../analysis/data.jl")

# Sample n = 51 synthetic patients (i.e., exact same format as real patients)
df = CSV.read("analysis/prior2.csv",DataFrame)
prior2 = Matrix(df)[:,4:end]
prior2c = [eachrow(prior2)...]

# Treatment regime
timeᵢ = [0;2:8] * 7.0
doseᵢ = (2.0*7.0):(8.0*7.0)
doseᵢ = doseᵢ[mod.(doseᵢ,7) .< 5]

# Model
model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(timeᵢ),output=:total)

# Create data files to match the clinical data files
data_dosetimes = DataFrame(ID=Int[],DoseTimes=Int[])
data_volumes = DataFrame(ID=Int[],Time=Int[],Volume=Float64[])

# Loop through patients
for i = 1:51

    lp = rand(prior2c)

    pt_dosetimes = DataFrame(
        ID=fill(i,length(doseᵢ)),
        DoseTimes=Int.(doseᵢ))
    pt_volumes = DataFrame(
        ID=fill(i,length(timeᵢ)),
        Time=timeᵢ,
        Volume=round.(model(lp).(timeᵢ),digits=4))

    data_dosetimes = [data_dosetimes; pt_dosetimes]
    data_volumes = [data_volumes; pt_volumes]

end

CSV.write("data/data_synthetic_dosetimes.csv",data_dosetimes)
CSV.write("data/data_synthetic_volumes.csv",data_volumes)