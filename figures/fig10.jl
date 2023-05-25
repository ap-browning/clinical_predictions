#=
    Figure 10

    Predictions using tumour breakdown measurements
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
prior = zeros(100000,d) .+ Inf
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
model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(timeᵢ),output=:both)

################################################
## GENERATE SYNTHETIC PATIENT DATA
################################################

# Seeds to generate patients
patient_row = 30566
lp = Vector(df[patient_row,4:end])

# Generate synthetic data
vols,necs = eachcol(vcat(model(lp).(timeᵢ)...))

# Add noise
vols[2:end] += rand(MvNormal(α[1] .+ α[2] * vols[2:end]))
necs += rand(MvNormal(α[1] .+ α[2] * necs))


################################################
## FUNCTION TO CALCULATE WEIGHTS
################################################

function get_weights(vols,necs)
    time = timeᵢ[eachindex(vols)]
    w = zeros(size(prior,1))
    w2 = zeros(size(prior,1))
    @threads for i = eachindex(w)
        # Simulate model
        y = vcat(model(prior[i,:]).(time)...)
        # Compute likelihood
        w[i] = pdf(MvNormal(y[:,1],α[1] .+ α[2] * y[:,1]),vols) * 
               pdf(MvNormal(y[:,2],α[1] .+ α[2] * y[:,2]),necs) 
        # Compute likelihood without using necrotic data
        w2[i] = pdf(MvNormal(y[:,1],α[1] .+ α[2] * y[:,1]),vols)
    end
    return w / sum(w), w2 / sum(w2)
end


################################################
## PERFORM INFERENCE USING PARTIAL DATA
################################################

# Up to the following measurement times
upto = [2,3,4,5,6,7,8]

# Initialise storage for weights and synthetic patient data
w = Array{Vector}(undef,length(upto))
w2 = Array{Vector}(undef,length(upto))

# Loop through observation times (to get weights)
for j = eachindex(upto)
    w[j],w2[j] = get_weights(vols[1:upto[j]],necs[1:upto[j]])
end

################################################
## FIGURE 10, ROW 1 AND 2
################################################

idx = 3 # Index of "upto" for which we will plot predictions

fig10a = plot()
fig10b = plot()

# Make predictions
lps = sample(priorc,Weights(w[idx]),200)
models = [model(lp) for lp in lps]

# == TOTAL VOLUME ==
    # Plot predictions
    plot!(fig10a,[t -> m(t)[1] for m in models],label="",c=:black,α=0.03,xlim=(0.0,timeᵢ[upto[idx]]))
    plot!(fig10a,[t -> m(t)[1] for m in models],label="",c=col_posterior,α=0.05,xlim=(timeᵢ[upto[idx]],maximum(timeᵢ)))

    # Plot observed data at present
    scatter!(fig10a,timeᵢ[1:upto[idx]],vols[1:upto[idx]],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(fig10a,timeᵢ[1:upto[idx]],vols[1:upto[idx]],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Plot future observed data
    scatter!(fig10a,timeᵢ[upto[idx]+1:end],vols[upto[idx]+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(fig10a,timeᵢ[upto[idx]+1:end],vols[upto[idx]+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

    # Plot present time
    vline!(fig10a,[timeᵢ[upto[idx]]],c=:black,lw=2.0,ls=:dash,label="",widen=true)

# == NECROTIC VOLUME
    # Plot predictions
    plot!(fig10b,[t -> m(t)[2] for m in models],label="",c=:black,α=0.03,xlim=(0.0,timeᵢ[upto[idx]]))
    plot!(fig10b,[t -> m(t)[2] for m in models],label="",c=col_posterior,α=0.05,xlim=(timeᵢ[upto[idx]],maximum(timeᵢ)))

    # Plot observed data at present
    scatter!(fig10b,timeᵢ[1:upto[idx]],necs[1:upto[idx]],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(fig10b,timeᵢ[1:upto[idx]],necs[1:upto[idx]],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Plot future observed data
    scatter!(fig10b,timeᵢ[upto[idx]+1:end],necs[upto[idx]+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(fig10b,timeᵢ[upto[idx]+1:end],necs[upto[idx]+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

    # Plot present time
    vline!(fig10b,[timeᵢ[upto[idx]]],c=:black,lw=2.0,ls=:dash,label="",widen=true)


################################################
## FIGURE 10, ROW 3
################################################

# Calculate predicted final tumour volume for each prior sample
F = [model(lp).(timeᵢ[end])[1] for lp in priorc]
Factual = model(lp).(timeᵢ[end])[1]

# Loop through patients
fig10c = plot()

# Plot predictions using necrotic data
μ = [mean(F,Weights(wᵢⱼ)) for wᵢⱼ in w]
l = [quantile(F,Weights(wᵢⱼ),0.025) for wᵢⱼ in w]
u = [quantile(F,Weights(wᵢⱼ),0.975) for wᵢⱼ in w]

scatter!(fig10c,timeᵢ[upto],μ,yerror=(μ-l,u-μ),msw=2.0,mc=:black,label="")

# Plot predictions using only total volume data
μ = [mean(F,Weights(wᵢⱼ)) for wᵢⱼ in w2]
l = [quantile(F,Weights(wᵢⱼ),0.025) for wᵢⱼ in w2]
u = [quantile(F,Weights(wᵢⱼ),0.975) for wᵢⱼ in w2]

scatter!(fig10c,timeᵢ[upto] .- 3.0,μ,yerror=(μ-l,u-μ),msw=2.0,α=0.5,mc=:black,label="")

# Plot true volume
hline!(fig10c,[Factual[1]],c=col_posterior,lw=2.0,ls=:dash,label="",xticks=0:7:Int(maximum(timeᵢ)))


################################################
## FIGURE 10
################################################

# Format Fig 10c (inset)
fig10cd = plot(fig10c,fig10c,layout=grid(2,1))
plot!(fig10cd,subplot=2,ylim=(0.0,0.5),widen=true)

# Construct figure
fig10 = plot(fig10a,fig10b,fig10cd,layout=grid(1,3),xticks=0:14:maximum(timeᵢ),xlim=(-4.0,maximum(timeᵢ) + 4.0),widen=true,size=(850,250))
plot!(fig10,subplot=3,xticks=(0:14:maximum(timeᵢ),[]),bottom_margin=-3Plots.mm)

# Add plot titles
add_plot_labels!(fig10)
plot!(fig10,subplot=4,title="")

# Add axis labels
plot!(fig10,subplot=1,ylabel="Volume [FC]")
plot!(fig10,subplot=2,ylabel="Necrotic volume")
plot!(fig10,xlabel="Time [d]")
plot!(fig10,subplot=3,xlabel="")

savefig(fig10,"$(@__DIR__)/fig10.svg")
fig10