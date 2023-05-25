#=
    Figures 7, 8, and 9

    Actual patients
=#

using CSV, DataFrames, DataFramesMeta
using Plots, StatsPlots
using DifferentialEquations
using Distributions
using .Threads
using StatsBase
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

# Sample from prior
prior = zeros(40000,d) .+ Inf
for i = axes(prior,1)
    while !insupport(prior1,prior[i,:])
        prior[i,:] = rand(prior2c) + β * rand(kd)
    end
end
priorc = [eachrow(prior)...]

# Load noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α


################################################
## FUNCTION TO CALCULATE WEIGHTS
################################################

function get_weights(timeᵢ,doseᵢ,vols)
    # Observation times
    time = timeᵢ[eachindex(vols)]
    # Model
    model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(time),output=:total)
    # Calculate weights
    w = zeros(size(prior,1))
    @threads for i = eachindex(w)
        # Simulate model
        y = model(prior[i,:]).(time)
        # Compute likelihood
        w[i] = pdf(MvNormal(y,α[1] .+ α[2] * y),vols)
    end
    return w / sum(w)
end

################################################
## PERFORM INFERENCE USING PARTIAL DATA
################################################

# Up to the following measurement times
upto = [2,3,4,5,6,7,8]

# Initialise storage for weights and synthetic patient data
w = [Array{Vector}(undef,length(upto)) for _ = 1:length(patient_id)]

# Loop through each patient
for (i,id) in enumerate(patient_id)

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Loop through observation times (to get weights)
    for j = eachindex(upto)
        w[i][j] = get_weights(timeᵢ,doseᵢ,volsᵢ[1:upto[j]])
    end

end

################################################
## Figure 7
################################################

plts = [Array{Any}(undef,length(upto)) for _ = 1:length(patient_id)]

# Loop through plots
for (i,id) = enumerate(patient_id), j = eachindex(upto)

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Patient specific model
    model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(timeᵢ),output=:total)

    # Make predictions
    lps = sample(priorc,Weights(w[i][j]),200)
    plts[i][j] = plot([model(lp) for lp in lps],label="",c=:black,α=0.05,xlim=(0.0,timeᵢ[upto[j]]))
    plot!(plts[i][j],[model(lp) for lp in lps],label="",c=col_posterior,α=0.1,xlim=(timeᵢ[upto[j]],maximum(timeᵢ)))

    # Plot observed data at present
    scatter!(plts[i][j],timeᵢ[1:upto[j]],volsᵢ[1:upto[j]],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[1:upto[j]],volsᵢ[1:upto[j]],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Plot future observed data
    scatter!(plts[i][j],timeᵢ[upto[j]+1:end],volsᵢ[upto[j]+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[upto[j]+1:end],volsᵢ[upto[j]+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

    # Plot present time
    vline!(plts[i][j],[timeᵢ[upto[j]]],c=:black,lw=2.0,ls=:dash,label="",widen=true)
    
end

plts_patient = [plot(plts[i][1:2:end]...,layout=grid(1,4),link=:all,ylim=(0.0,2.0),xticks=0:14:56,xlim=(-4.0,60.0)) for i in eachindex(patient_id)]

# Figure 7
fig7 = plot(plts_patient...,layout=grid(4,1),size=(800,500),xlabel="Time [d]",link=:all)
savefig(fig7,"$(@__DIR__)/fig7.svg")

################################################
## Figure 8
################################################

    ############################################
    ## Row 1: Posterior for γ

    # Color palette
    cols = palette(:RdPu,12)[end-7:end]

    # Setup
    j = 3
    lγ = prior[:,j]
    xv = range(-10.0,0.0,200)

    # Loop through patients
    plts1 = [plot() for _ = patient_id]
    for (i,id) = enumerate(patient_id)

        # Get patient specific data
        timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

        # Plot population posterior (prior)
        f = pdf(kde(lγ),xv)
        plot!(plts1[i],fill(0.0,length(xv)+2),[xv[1];xv;xv[end]],[0.0;f;0.0],c=cols[1],lw=2.0,frange=0.0,label="")

        # Plot successive posteriors
        for j = eachindex(upto)
            f = pdf(kde(lγ,weights=w[i][j]),xv)
            plot!(plts1[i],fill(timeᵢ[upto[j]],length(xv)+2),[xv[1]; xv; xv[end]],[0.0; f; 0.0],lw=2.0,c=cols[j+1],label="")
        end
        
        # Styling
        plot!(
            plts1[i],
            widen=false, zticks=[], xticks=(7.0:7.0:maximum(timeᵢ),[0;14:7:Int(maximum(timeᵢ))]),
            axis=:y,framestyle=:axis,grid=:y,camera=(50,25)
        )
    end

    row1 = plot(plts1...,layout=grid(1,4),size=(1000,300),xlabel="Time [d]",ylabel="log(γ)")

    ############################################
    ## Row 2: Confidence interval for final tumour size

    # Loop through patients
    plts2 = [plot() for _ = patient_id]
    for (i,id) = enumerate(patient_id)

        # Get patient specific data
        timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

        # Patient specific model
        model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(timeᵢ),output=:total)

        # Calculate predicted final tumour volume for each prior sample
        F = [model(lp).(timeᵢ[end]) for lp in priorc]

        # Mean and confidence intervals
        μ = [mean(F,Weights(wᵢⱼ)) for wᵢⱼ in w[i]]
        l = [quantile(F,Weights(wᵢⱼ),0.025) for wᵢⱼ in w[i]]
        u = [quantile(F,Weights(wᵢⱼ),0.975) for wᵢⱼ in w[i]]

        plts2[i] = scatter(timeᵢ[upto],μ,yerror=(μ-l,u-μ),msw=2.0,mc=:black,label="")
    end

    row2 = plot(plts2...,layout=grid(1,4),xlabel="Time [d]", ylabel="Final FC Volume",
        link=:x,xlim=(-4.0,60.0),widen=true,ylim=(0.0,2.0),xticks=0:14:56)


    # Figure 8
    fig8 = plot(row1,row2,layout=grid(2,1),size=(800,450))
    add_plot_labels!(fig8)
    savefig(fig8,"$(@__DIR__)/fig8.svg")

################################################
## Figure 9
################################################

# Class for each prior sample (should this be with the patient specific therapy?)
class = [classify(lp) for lp in priorc]

# Probability of responding to treatment (i.e., not being in class 2)
p = [[sum(wᵢⱼ[class .!= 2]) for wᵢⱼ in w[i]] for i = 1:4]

# Color for each class
cols = [palette(symb)[5] for symb = [:Greens_7,:Oranges_7,:PuBu_7,:RdPu_7]]

## Fig 8
tmax = 70.0
plts = [plot() for _ = eachindex(patient_id)]
for (i,id) in enumerate(patient_id)

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Initialise overall probability response curve
    xc = Float64[]
    pc = Float64[]

    # Loop through observation times
    for j = 1:length(w[i])+1

        # Times for which prediction applies to
        x = j != length(timeᵢ) ? [timeᵢ[j],timeᵢ[j+1]] : [timeᵢ[j],tmax]

        # Weights to use
        if j == 1
            wᵢⱼ = ones(length(class)) / length(class)
        else
            wᵢⱼ = w[i][j-1]
        end

        # Probability of being in each class
        probs = [0.0;cumsum([sum(wᵢⱼ[class .== c]) for c = 1:4][[1,3,4,2]])] # Perturbed (class 2 last)
        colsp = cols[[1,3,4,2]] # Perturbed (class 2 last)

        # Plot
        for k = 1:4
            y₁ = probs[k] * ones(2)
            y₂ = probs[k+1] * ones(2)
            plot!(plts[i],x,y₂,c=colsp[k],frange=y₁,lw=0.0,α=1.0,label="")
        end
        
        # Push back overall response curve
        xc = [xc; x]
        pc = [pc;[probs[end-1],probs[end-1]]]

        # Plot vertical line (if not the first)
        if j > 1
            plot!(plts[i],[x[1],x[1]],[0.0,1.0],lw=1.0,α=0.3,c=:black,label="")
        end

    end

    # Plot overall ℙ(responder) curve
    plot!(plts[i],xc,pc,lw=2.0,c=:black,ls=:dash,label="")

end

fig9 = plot(plts...,xticks=[collect(0:14:tmax);tmax],widen=false,xlabel="Time [d]",ylabel="",size=(500,400))
add_plot_labels!(fig9)
savefig(fig9,"$(@__DIR__)/fig9.svg")