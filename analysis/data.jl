#=
    data.jl

    Load and normalise data, subset training set
=#

using CSV, DataFrames, DataFramesMeta
using Random, Distributions

# Load data
data = CSV.read("data/data_volumes.csv",DataFrame)
dose = CSV.read("data/data_dosetimes.csv",DataFrame)

# Patient data (for plotting/verification)
patient_id = [14,6,34,5]

# Subset to get training data
training_id = sample(MersenneTwister(1),setdiff(unique(data.ID),patient_id),40,replace=false)

# Create array containing normalised data
normalised_data = Array{Any}(undef,maximum(unique(data.ID)))
for id in data.ID

    # Get "raw" data
    timeᵢ = @subset(data, :ID .== id).Time
    volsᵢ = @subset(data, :ID .== id).Volume
    doseᵢ = @subset(dose, :ID .== id).DoseTimes

    # Remove extra pre-measurements for consistency
    idx = findfirst(timeᵢ .≥ doseᵢ[1])
    timeᵢ = timeᵢ[idx-1:end]
    volsᵢ = volsᵢ[idx-1:end]
    doseᵢ .-= timeᵢ[1]
    timeᵢ .-= timeᵢ[1]

    # Normalise volume
    volsᵢ = volsᵢ / volsᵢ[1]

    normalised_data[id] = [timeᵢ,volsᵢ,doseᵢ]

end