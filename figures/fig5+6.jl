#=
    Figure 5 and 6

    Synthetic patients
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
## CREATE THREE SYNTHETIC PATIENTS (from within prior2)
################################################

# Treatment regime
timeᵢ = [0;2:8] * 7.0
doseᵢ = (2.0*7.0):(8.0*7.0)
doseᵢ = doseᵢ[mod.(doseᵢ,7) .< 5]

# Model
model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(timeᵢ),output=:total)

################################################
## GENERATE SYNTHETIC PATIENT DATA
################################################

# Seeds to generate patients
patient_rows = [30566,1155,12179]
lp = [Vector(df[row,4:end]) for row in patient_rows]

# Store noisy volume measurements
vols = Array{Vector}(undef,length(patient_rows))

# Loop through each patient
for i in eachindex(patient_rows)

    # Generate synthetic data
    vols[i] = model(lp[i]).(timeᵢ)
    vols[i][2:end] += rand(MvNormal(α[1] .+ α[2] * vols[i][2:end]))

end

################################################
## FUNCTION TO CALCULATE WEIGHTS
################################################

function get_weights(vols)
    time = timeᵢ[eachindex(vols)]
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
w = [Array{Vector}(undef,length(upto)) for _ = 1:length(patient_rows)]

# Loop through each patient
for i in eachindex(patient_rows)

    # Loop through observation times (to get weights)
    for j = eachindex(upto)
        w[i][j] = get_weights(vols[i][1:upto[j]])
    end

end

################################################
## Figure 5
################################################

plts = [Array{Any}(undef,length(upto)) for _ = 1:length(patient_rows)]

# Loop through plots
for i = eachindex(patient_rows), j = eachindex(upto)

    # Make predictions
    lps = sample(priorc,Weights(w[i][j]),200)
    plts[i][j] = plot([model(lp) for lp in lps],label="",c=:black,α=0.05,xlim=(0.0,timeᵢ[upto[j]]))
    plot!(plts[i][j],[model(lp) for lp in lps],label="",c=col_posterior,α=0.1,xlim=(timeᵢ[upto[j]],maximum(timeᵢ)))

    # Plot observed data at present
    scatter!(plts[i][j],timeᵢ[1:upto[j]],vols[i][1:upto[j]],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[1:upto[j]],vols[i][1:upto[j]],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Plot future observed data
    scatter!(plts[i][j],timeᵢ[upto[j]+1:end],vols[i][upto[j]+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[upto[j]+1:end],vols[i][upto[j]+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

    # Plot present time
    vline!(plts[i][j],[timeᵢ[upto[j]]],c=:black,lw=2.0,ls=:dash,label="",widen=true)
    
end

plts_patient = [plot(plts[i][1:2:end]...,layout=grid(1,4),link=:all,ylim=(0.0,2.0),xticks=0:14:maximum(timeᵢ),xlim=(-4.0,maximum(timeᵢ) + 4.0)) for i in eachindex(patient_rows)]

# Figure 5
fig5 = plot(plts_patient...,layout=grid(3,1),size=(800,500),xlabel="Time [d]",link=:all)
savefig(fig5,"$(@__DIR__)/fig5.svg")

################################################
## Figure 6
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
    plts1 = [plot() for _ = patient_rows]
    for i = eachindex(patient_rows)

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

    row1 = plot(plts1...,layout=grid(1,3),size=(1000,300),xlabel="Time [d]",ylabel="log(γ)")

    ############################################
    ## Row 2: Confidence interval for final tumour size

    # Calculate predicted final tumour volume for each prior sample
    F = [model(lp).(timeᵢ[end]) for lp in priorc]
    Factual = [model(lpᵢ).(timeᵢ[end]) for lpᵢ in lp]

    # Loop through patients
    plts2 = [plot() for _ = patient_rows]
    for i = eachindex(patient_rows)

        μ = [mean(F,Weights(wᵢⱼ)) for wᵢⱼ in w[i]]
        l = [quantile(F,Weights(wᵢⱼ),0.025) for wᵢⱼ in w[i]]
        u = [quantile(F,Weights(wᵢⱼ),0.975) for wᵢⱼ in w[i]]

        plts2[i] = scatter(timeᵢ[upto],μ,yerror=(μ-l,u-μ),msw=2.0,mc=:black,label="")
        hline!(plts2[i],[Factual[i]],c=col_posterior,lw=2.0,ls=:dash,label="",xticks=0:7:Int(maximum(timeᵢ)))
    end

    row2 = plot(plts2...,layout=grid(1,3),xlabel="Time [d]", ylabel="Final FC Volume",
        link=:x,xlim=(-4.0,60.0),widen=true,ylim=(0.0,2.0))


    # Figure 6
    fig6 = plot(row1,row2,layout=grid(2,1),size=(800,450))
    add_plot_labels!(fig6)
    savefig(fig6,"$(@__DIR__)/fig6.svg")

    fig6