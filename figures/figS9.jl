#=
    Figure S9

    Training procedure using a standard hierarchical method.

=#

using Turing
using .Threads

include("../figures/defaults.jl")

################################################
## LOAD DATA AND MODELS
################################################

include("../analysis/data.jl")
include("../analysis/models.jl")
include("../analysis/inference.jl")
include("../analysis/classifier.jl")

################################################
## SETUP PRIOR (INCL. α)
################################################

# Model parameter prior
include("../analysis/prior1.jl")

################################################
## SETUP INITIAL GUESS (USING SAMPLES FROM INDIVIDUAL POSTERIORS)
################################################

# Work with a subset (computationally easier)
ids = sample(MersenneTwister(1),unique(posterior1.id),10,replace=false)
rows = [id ∈ ids for id = posterior1.id]

# Load posterior from individual-level analysis
posterior1 = CSV.read("analysis/posterior1.csv",DataFrame)[rows,:]

# Sample from posterior to form initial guess
function sample_initial_guess()

    # Create initial guess for four chains
    lp = hcat([Vector(rand(eachrow(@subset(posterior1,:id .== id)))[5:end]) for id in ids]...)

    # Population level mean and variance
    μlp = mean(lp,dims=2)
    lσlp = log.(std(lp,dims=2))

    # Noise model
    lα = log.(CSV.read("analysis/noise_model.csv",DataFrame).α)

    # Construct initial guess
    [μlp; lσlp; lα; lp'[:]]

end

################################################
## FULL MODEL
################################################

# Bounds on parameters
pmin,pmax = extrema(prior1)

# Define hierarchical model
@model function mdl1(y)
    
    # Priors for the means
    μlλ  ~ Uniform(-10.0,0.0)
    μlK  ~ Uniform(0.0,5.0)
    μlγ  ~ Uniform(-10.0,0.0)
    μlζ  ~ Uniform(-10.0,3.0)
    μlη  ~ Uniform(-10.0,3.0)
    μlϕ₀ ~ Uniform(-5.0,0.0)

    # Priors for the standard deviations
    lσlλ  ~ Uniform(-10.0,3.0)
    lσlK  ~ Uniform(-10.0,3.0)
    lσlγ  ~ Uniform(-10.0,3.0)
    lσlζ  ~ Uniform(-10.0,3.0)
    lσlη  ~ Uniform(-10.0,3.0)
    lσlϕ₀ ~ Uniform(-10.0,3.0)

    # Standard deviations
    lσ₁ ~ Uniform(-10.0,0.0)
    lσ₂ ~ Uniform(-10.0,0.0)

    # Individual-level parameters
    lλ  ~ filldist(TruncatedNormal(μlλ,exp(lσlλ),pmin[1],pmax[1]),length(y))
    lK  ~ filldist(TruncatedNormal(μlK,exp(lσlK),pmin[2],pmax[2]),length(y))
    lγ  ~ filldist(TruncatedNormal(μlγ,exp(lσlγ),pmin[3],pmax[3]),length(y))
    lζ  ~ filldist(TruncatedNormal(μlζ,exp(lσlζ),pmin[4],pmax[4]),length(y))
    lη  ~ filldist(TruncatedNormal(μlη,exp(lσlη),pmin[5],pmax[5]),length(y))
    lϕ₀ ~ filldist(TruncatedNormal(μlϕ₀,exp(lσlϕ₀),pmin[6],pmax[6]),length(y))

    for i in eachindex(y)
        # Patient-specific prediction
        μ = model([lλ[i],lK[i],lγ[i],lζ[i],lη[i],lϕ₀[i]],ids[i])
        # Observation vector
        y[i] ~ MvNormal(μ,exp(lσ₁) .+ max.(0.0,μ) * exp(lσ₂))
    end

end

################################################
## RUN HMC, FOUR CHAINS
################################################

# Data
y = [normalised_data[id][2] for id in ids]

# Chains
C = Array{Any}(undef,4)
@threads for i = 1:4
     C[i] = sample(mdl1(y),HMC(0.01,5),50000,init_params=sample_initial_guess())
end

################################################
## EXTRACT RESULTS
################################################

# Indices from each chain to keep (discard first 10,000)
idx = 10000:10:50000

# Per-patient parameters
X = [vcat([hcat([c.value.data[idx,14+i+npt*j,1] for j = 0:5]...) for c in C]...) for i = 1:npt]
X̂ = vcat(X...)

# Population-level parameters
μ  = vcat([c[idx,1:6,1].value.data for c in C]...)[:,:,1]
lσ = vcat([c[idx,7:12,1].value.data for c in C]...)[:,:,1]

################################################
## (1) DENSITY PLOT EXAMPLE
################################################

i = 3; j = 1; # Parameters on y and x axes
cmaps = [:Greens_7,:Oranges_7,:PuBu_7,:RdPu_7]
lims = [
    [-5.0,0.0], # λ
    [0.0,5.0],  # K
    [-3.0,0.0], # γ
    [-6.0,3.0], # ζ
    [-10.0,3.0],# η,
    [-5.0,0.0]  # ϕ₀
]
plt1_a = plot()
for class = [1,2,3,4]
    @df @subset(posterior1,:class .== class) density2d!(plt1_a,cols(4 + j),cols(4 + i),
        levels=10,fill=true,c=cmaps[class],lw=0.0,z_clip=0.5,α=0.4,xlim=lims[j],ylim=lims[i])
    plot!(plt1_a,xlabel=names(df)[4 + j],ylabel=names(df)[4 + i])
end
plt1_a

idx = sample(1:size(μ,1),20000)
xv = [rand(TruncatedNormal(μ[k,j],exp(lσ[k,j]),pmin[j],pmax[j])) for k in idx]
yv = [rand(TruncatedNormal(μ[k,i],exp(lσ[k,i]),pmin[i],pmax[i])) for k in idx]

plt1_b = density2d(xv,yv,xlim=lims[j],ylim=lims[i],fill=true,z_clip=0.5,levels=10,c=:bone,lw=0.0)
plot!(plt1_b,xlabel=names(df)[4 + j],ylabel=names(df)[4 + i])

plt1 = plot(plt1_a,plt1_b)

################################################
## (2) PLOT PATIENT TRAJECTORIES
################################################

# Setup "standard" treatment regime
timeᵢ = [0;2:8] * 7.0
doseᵢ = (2.0*7.0):(8.0*7.0)
doseᵢ = doseᵢ[mod.(doseᵢ,7) .< 5]
tplt = range(extrema(timeᵢ)...,200)

# Plot new patient from our approach
plt2_a = plot()
for i = 1:200
    lp = Vector(rand(eachrow(posterior1))[5:end])
    mplt = model(lp,tplt,doseᵢ)
    plot!(plt2_a,tplt,mplt,c=:blue,α=0.1,label="")
end
plt2_a

# Plot new patient trajectories from hierarchical model
plt2_b = plot()
for i = 1:200
    ik = rand(1:size(μ,1))
    lp = rand.(TruncatedNormal.(μ[ik,:],exp.(lσ[ik,:]),pmin,pmax))
    mplt = model(lp,tplt,doseᵢ)
    plot!(plt2_b,tplt,mplt,c=:blue,α=0.1,label="")
end
plt2_b

# Plot all data from the training set...
for plt = [plt2_a,plt2_b], i = 1:10
    timeᵢ,volsᵢ,doseᵢ = normalised_data[ids[i]]
    scatter!(plt,timeᵢ,volsᵢ,c=:black,label="")
end

plot!(plt2_b,ylim=[0.0,5.0])

plt2 = plot(plt2_a,plt2_b,xtick=0:14:56,ylabel="GTV",xlabel="Time [d]")

################################################
## FIGURE S9
################################################

figS9 = plot(plt1,plt2,layout=@layout([a{0.6h}; b]),size=(700,600))
add_plot_labels!(figS9)
savefig(figS9,"$(@__DIR__)/figS9.svg")