#=
    classifier.jl

    Setup standard patient course of treatment and classification algorithm
=#

include("models.jl")
include("inference.jl")

# Setup "standard" treatment regime
timeᵢ = [0;2:8] * 7.0
doseᵢ = (2.0*7.0):(8.0*7.0)
doseᵢ = doseᵢ[mod.(doseᵢ,7) .< 5]

# Classifier
function classify(lp;α₁=0.0,α₂=0.0,time=timeᵢ,dose=doseᵢ)
    # 1. Fast responder
    # 2. Poor responder
    #       - All above 0.85 of volume at start of treatment
    # 3. Plateaued response
    #       - Last measurement is greater than 20% of initial AND
    #       - Final rate of change is less than 10% of the maximum
    # 4. Pseudo-progression
    #       - Not a poor responder AND 
    #       - Second measurement is greater than 102% of the first

    # Simulate standard patient
    volsᵢ = model(lp,time,dose)[2:end]

    # Add noise (if required)
    if (α₁ > 0.0) || (α₂ > 0.0)
        σ = α₁ .+ α₂ * volsᵢ
        volsᵢ += σ .* randn(length(volsᵢ))
    end

    # Check if (2) Poor Responder
    if all(volsᵢ .> 0.85*volsᵢ[1])
        class = 2
    else
        # Check if (4) Pseudo-progressor
        if volsᵢ[2] > volsᵢ[1]*1.02
            class = 4
        else
            # Check if (3) Plateaued Response
            rel_diffs = abs.(diff(volsᵢ) ./ volsᵢ[1:end-1])
            if (volsᵢ[end] > 0.2) & (rel_diffs[end] < 0.1*maximum(rel_diffs))
                class = 3
            # Else (1) Fast Responder
            else
                class = 1
            end
        end
    end

end