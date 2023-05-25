#=
    initial_volumes.jl

    Initial volumes for each class based on prior 2
=#

using CSV, DataFrames, DataFramesMeta

include("../analysis/models.jl")
include("../analysis/data.jl")

################################################
## LOAD PRIOR SAMPLES
################################################

df = CSV.read("analysis/prior2.csv",DataFrame)

# Create array containing initial tumour volumes
init_vols = Array{Any}(undef,maximum(unique(data.ID)))
for id in data.ID

    # Get "raw" data
    timeᵢ = @subset(data, :ID .== id).Time
    volsᵢ = @subset(data, :ID .== id).Volume
    doseᵢ = @subset(dose, :ID .== id).DoseTimes

    # Remove extra pre-measurements for consistency
    idx = findfirst(timeᵢ .≥ doseᵢ[1])
    timeᵢ = timeᵢ[idx-1:end]
    volsᵢ = volsᵢ[idx-1:end]

    init_vols[id] = volsᵢ[1]

end

# Add column to df
df.init_vol = [init_vols[i] for i = Int.(df.id)]

# Mean and std volume by class
μ = [mean(@subset(df,:class .== i).init_vol) for i = 1:4]
σ = [ std(@subset(df,:class .== i).init_vol) for i = 1:4]

μ₁ = mean(vcat([@subset(df,:class .== i).init_vol for i = [1,3,4]]...))
σ₁ =  std(vcat([@subset(df,:class .== i).init_vol for i = [1,3,4]]...))