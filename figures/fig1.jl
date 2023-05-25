#=
    Figure 1

    Clinical data, motivation figure and verification of model.

=#

using DataFrames, DataFramesMeta, Plots, StatsPlots, CSV
using DifferentialEquations
using Distributions
using Optim
using LinearAlgebra
using AdaptiveMCMC
using .Threads

include("defaults.jl")

################################################
## LOAD DATA AND MODELS
################################################

include("../analysis/data.jl")
include("../analysis/models.jl")

################################################
## INFERENCE USING STANDARD PRIOR
################################################

# Noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α

# Prior
include("../analysis/prior1.jl")

# Store posterior samples and likelihood values
X = Array{Any}(undef,4)
D = Array{Any}(undef,4)

# Loop through patients
@threads for i = eachindex(patient_id)

    # Patient id
    id = patient_id[i]

    # Load patient data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Initial guess (for i == 4, found through random starts)
    lp₀ = i == 4 ? [-1.3, 0.1, -1.1, -3.4, -4.5, -4.0] :
        log.([0.5,1.0,0.5,0.5,0.5,0.01])

    # Model
    model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=maximum(timeᵢ),output=:total)
    loglike = lp -> begin
        m = model(lp).(timeᵢ)
        σ = α[1] .+ α[2] * m
        loglikelihood(Normal(),(volsᵢ - m) ./ σ)
    end
    logpost = lp -> insupport(prior1,lp) ? loglike(lp) + logpdf(prior1,lp) : -Inf

    # MCMC
    #@time res = adaptive_rwm(lp₀,logpost,110000,algorithm=:aswam,thin=100,b=10000)
    @time res = adaptive_rwm(lp₀,logpost,10000,algorithm=:aswam,thin=100,b=5000)

    # Store
    D[i] = res.D[1]
    X[i] = res.X

end


################################################
 # # PRODUCE PLOTS
################################################

# Setup
tmax = 56.0
tplt = range(0.0,tmax,200)
nsim = 1000

# Store plots
plts = [plot() for _ = eachindex(patient_id)]

# Loop through patients
for i = eachindex(patient_id)

    # Patient id
    id = patient_id[i]

    # Load patient data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Model
    model = lp -> solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=56.0,output=:total)

    # 95% prediction interval
    vsim = hcat([solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=56.0,output=:total).(tplt)
                for lp = rand([eachcol(X[i])...],nsim)]...)
    l,u = [[quantile(v,q) for v in eachrow(vsim)] for q = [0.025,0.975]]

    # Best fit
    lp = X[i][:,findmax(D[i])[2]]
    m  = solve_model_twocompartment(exp.(lp);tdose=doseᵢ,tend=56.0,output=:total).(tplt)

    # Plot model fit
    plot!(plts[i],tplt,m,frange=(l,u),c=col_prior2,lw=2.0,label="",fα=0.3)

    # Plot data
    scatter!(plts[i],timeᵢ,volsᵢ,ylim=(0.0,:auto),widen=true,msc=:black,ms=8.0,msw=2.0,mc=:white,label="")
    scatter!(plts[i],timeᵢ,volsᵢ,ylim=(0.0,:auto),widen=true,c=:black,label="")
    
    plot!(xticks=0:7:56.0,box=:on)

end

# Figure 1
fig1 = plot(plts...,ylim=(0.0,1.6),size=(700,450),xlabel="Time [d]",ylabel="Volume [FC]")
add_plot_labels!(fig1)
savefig(fig1,"$(@__DIR__)/fig1.svg")