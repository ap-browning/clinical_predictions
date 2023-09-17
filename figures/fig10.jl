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
include("../analysis/inference.jl")

################################################
## PLOT PARAMETERS
################################################

tmax = 56.0
tplt = range(0.0,tmax,200)

################################################
## LOAD PRIOR SAMPLES
################################################

prior2 = CSV.read("analysis/prior2.csv",DataFrame)
prior2.row = 1:nrow(prior2)
X = Matrix(prior2[:,2:end-1])
Xc = [eachrow(X)...]

################################################
## CREATE THREE SYNTHETIC PATIENTS (from within posterior1)
################################################

# Load posterior1
posterior1 = CSV.read("analysis/posterior1.csv",DataFrame)
posterior1.row = 1:nrow(posterior1)

# Load noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α

# Treatment regime
timeᵢ = [0;2:8] * 7.0
doseᵢ = (2.0*7.0):(8.0*7.0)
doseᵢ = doseᵢ[mod.(doseᵢ,7) .< 5]

# Model
model(lp) = model(lp,timeᵢ,doseᵢ;ret=:function)

# Synthetic patient parameters (now sampled from full posterior)
rng = MersenneTwister(1)
patient_rows = [rand(rng,@subset(posterior1,:class .== i).row) for i = 1:4]

# Resample to get reasonable pseudo-progressor on new treatment plan to match original manuscript
patient_rows = [rand(rng,@subset(posterior1,:class .== i).row) for i = 1:4]
patient_rows = [rand(rng,@subset(posterior1,:class .== i).row) for i = 1:4]
lp = [Vector(posterior1[row,5:end]) for row in patient_rows]

# Store noisy volume measurements
vols = Array{Vector}(undef,length(patient_rows))

# Loop through each patient, add noise
for i in eachindex(patient_rows)

    # Generate synthetic data
    vols[i] = model(lp[i]).(timeᵢ)
    vols[i][2:end] += rand(rng,MvNormal(α[1] .+ α[2] * vols[i][2:end]))

end

# Fast-responder only
lp = lp[1]
vols = vols[1]
necs = hcat(model(lp,timeᵢ,doseᵢ;output=:all)...)[2,:]
#necs += rand(rng,MvNormal(α[1] .+ α[2] * necs))


################################################
## FUNCTION TO CALCULATE WEIGHTS
################################################

function get_weights(prior,α,time,dose,vols,necs;final_only=false)
    # Model prediction for each sampled parameter
    m = hcat([hcat(model(lp,time,dose;output=:all)...)'[:] for lp in eachrow(prior)]...)'
    # Matrix of [GTV necrotic]
    m[:,1:length(time)] += m[:,length(time)+1:end]
    # Corresponding std based on α
    σ = α[1] .+ m * α[2]
    # loglikelihood contribution from each time point
    ll = hcat([logpdf.(d,[vols;necs]) for d in eachrow(Normal.(m,σ))]...)'
    # loglikelihood for each time point (both volume and necrotic, considered to be independent)
    ll = ll[:,1:length(time)] + ll[:,length(time)+1:end]
    # Calculate matrix of weights
    like = exp.(cumsum(ll,dims=2))
    w = like ./ sum(like,dims=1)
    if final_only
        w = w[:,end]
    end
    return w,m
end

################################################
## PERFORM INFERENCE USING PARTIAL DATA
################################################

# Using both necrotic and volume data
W,M = get_weights(X,α,timeᵢ,doseᵢ,vols,necs)

# Model predictions (both)
M̄ = hcat([hcat(model(lp,tplt,doseᵢ;output=:all)...)'[:] for lp in Xc]...)'

# Model predictions (GTV and N(t))
V = M̄[:,1:length(tplt)] + M̄[:,length(tplt)+1:end]
N = M̄[:,length(tplt)+1:end]

# Using only volume data
W2, = get_weights(X,α,timeᵢ,doseᵢ,vols)

################################################
## FIGURE 10, PARTS 1 AND 2
################################################

idx = 4 # Predictions are shown at timeᵢ[idx]
idx_max = findfirst(tplt .> timeᵢ[idx])
idx_max = idx_max === nothing ? length(tplt) : idx_max

fig10a = plot()
fig10b = plot()

# == TOTAL VOLUME ==
    l1,l2,u2,u1 = [[quantile(v,Weights(W[:,idx]),q) for v in eachcol(V)] for q = [0.025,0.25,0.75,0.975]]
    μ = [mean(v,Weights(W[:,idx])) for v in eachcol(V)]

    # Plot confidence bands
    plot!(fig10a,tplt[1:idx_max],u1[1:idx_max],frange=l1[1:idx_max],c=:black,lw=0.0,label="",fα=0.4)
    plot!(fig10a,tplt[1:idx_max],u2[1:idx_max],frange=l2[1:idx_max],c=:black,lw=0.0,label="",fα=0.4)
    plot!(fig10a,tplt[idx_max:end],u1[idx_max:end],frange=l1[idx_max:end],c=col_posterior,lw=0.0,label="",fα=0.4)
    plot!(fig10a,tplt[idx_max:end],u2[idx_max:end],frange=l2[idx_max:end],c=col_posterior,lw=0.0,label="",fα=0.4)

    # Plot mean
    plot!(fig10a,tplt[1:idx_max],μ[1:idx_max],c=:black,lw=2.0,label="",fα=0.4)
    plot!(fig10a,tplt[idx_max:end],μ[idx_max:end],c=col_posterior,lw=2.0,label="",fα=0.4)

    # Plot observed data at present
    scatter!(fig10a,timeᵢ[1:idx],vols[1:idx],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(fig10a,timeᵢ[1:idx],vols[1:idx],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Plot future observed data
    scatter!(fig10a,timeᵢ[idx+1:end],vols[idx+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(fig10a,timeᵢ[idx+1:end],vols[idx+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

    # Plot present time
    vline!(fig10a,[timeᵢ[idx]],c=:black,lw=2.0,ls=:dash,label="",widen=true)

# == NECROTIC VOLUME
    l1,l2,u2,u1 = [[quantile(v,Weights(W[:,idx]),q) for v in eachcol(N)] for q = [0.025,0.25,0.75,0.975]]
    μ = [mean(v,Weights(W[:,idx])) for v in eachcol(N)]

    # Plot confidence bands
    plot!(fig10b,tplt[1:idx_max],u1[1:idx_max],frange=l1[1:idx_max],c=:black,lw=0.0,label="",fα=0.4)
    plot!(fig10b,tplt[1:idx_max],u2[1:idx_max],frange=l2[1:idx_max],c=:black,lw=0.0,label="",fα=0.4)
    plot!(fig10b,tplt[idx_max:end],u1[idx_max:end],frange=l1[idx_max:end],c=col_posterior,lw=0.0,label="",fα=0.4)
    plot!(fig10b,tplt[idx_max:end],u2[idx_max:end],frange=l2[idx_max:end],c=col_posterior,lw=0.0,label="",fα=0.4)

    # Plot mean
    plot!(fig10b,tplt[1:idx_max],μ[1:idx_max],c=:black,lw=2.0,label="",fα=0.4)
    plot!(fig10b,tplt[idx_max:end],μ[idx_max:end],c=col_posterior,lw=2.0,label="",fα=0.4)

    # Plot observed data at present
    scatter!(fig10b,timeᵢ[1:idx],necs[1:idx],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(fig10b,timeᵢ[1:idx],necs[1:idx],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Plot future observed data
    scatter!(fig10b,timeᵢ[idx+1:end],necs[idx+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(fig10b,timeᵢ[idx+1:end],necs[idx+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

    # Plot present time
    vline!(fig10b,[timeᵢ[idx]],c=:black,lw=2.0,ls=:dash,label="",widen=true)

plot(fig10a,fig10b)

################################################
## FIGURE 10, PART 3
################################################

# Calculate predicted final tumour volume for each prior sample
F = M[:,length(timeᵢ)]
Factual = model(lp).(timeᵢ[end])[1]

# Loop through patients
fig10c = plot()

# Plot predictions using necrotic data
μ = [mean(F,Weights(wᵢⱼ)) for wᵢⱼ in eachcol(W)]
l1,l2,u2,u1 = [[quantile(F,Weights(wᵢⱼ),q) for wᵢⱼ in eachcol(W)] for q = [0.025,0.25,0.75,0.975]]
for j = eachindex(timeᵢ)[2:end]
    plot!(fig10c,timeᵢ[j]*[1.0,1.0],[l1[j],u1[j]],lw=8.0,c=:blue,α=0.2,label="")
    plot!(fig10c,timeᵢ[j]*[1.0,1.0],[l2[j],u2[j]],lw=8.0,c=:blue,α=0.3,label="")
end
scatter!(fig10c,timeᵢ[2:end],μ[2:end],c=:blue,ms=4.0,label="")

# Plot predictions using only total volume data
μ = [mean(F,Weights(wᵢⱼ)) for wᵢⱼ in eachcol(W2)]
l1,l2,u2,u1 = [[quantile(F,Weights(wᵢⱼ),q) for wᵢⱼ in eachcol(W2)] for q = [0.025,0.25,0.75,0.975]]
for j = eachindex(timeᵢ)[2:end]
    plot!(fig10c,timeᵢ[j]*[1.0,1.0] .- 2.0,[l1[j],u1[j]],lw=8.0,c=:black,α=0.3,label="")
    plot!(fig10c,timeᵢ[j]*[1.0,1.0] .- 2.0,[l2[j],u2[j]],lw=8.0,c=:black,α=0.3,label="")
end
scatter!(fig10c,timeᵢ[2:end] .- 2.0,μ[2:end],c=:black,ms=4.0,label="")

# Plot true volume
hline!(fig10c,[Factual[1]],c=col_posterior,lw=2.0,ls=:dash,label="",xticks=0:7:Int(maximum(timeᵢ)))


################################################
## FIGURE 10
################################################

# Format Fig 10c (inset)
fig10cd = plot(fig10c,fig10c,layout=grid(2,1))
plot!(fig10cd,subplot=1,ylim=(0.0,2.5),widen=true)
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