#=
    inference.jl

Useful tools for performing inference with the two-compartment model.
    
=#

using AdaptiveMCMC
using Distributions
using MCMCChains
using MCMCDiagnosticTools
using .Threads

include("data.jl")

################################################
## SOLVE THE MODEL
################################################

function model(lp,time,dose;output=:total,ret=:vector)
    mdl = solve_model_twocompartment(exp.(lp);tdose=dose,tend=maximum(time),output,verbose=false)
    if ret == :function
        return mdl
    else
        return mdl.(time)
    end
end
model(lp,id;kwargs...) = model(lp,normalised_data[id][[1,3]]...;kwargs...)

################################################
## LIKELIHOOD FUNCTIONS
################################################

# For a patient with given time, vols, dose
function loglike(lp,α,time,vols,dose)
    # Model prediction
    m = model(lp,time,dose)
    # Standard deviation
    σ = α[1] .+ α[2] * m
    # Calculate loglikelihood
    loglikelihood(MvNormal(m,σ),vols)
end

# For a specific patient in the data set
loglike(lp,α,id) = loglike(lp,α,normalised_data[id]...)

################################################
## MCMC
################################################

function mcmc(logpost,x₀,iters;algorithm=:aswam,nchains=1,L=1,burnin=1,thin=1,param_names=nothing,parallel=false)

    # Perform MCMC
    res = Array{Any}(undef,nchains)
    if parallel
        @threads for i = 1:nchains
            res[i] = adaptive_rwm(x₀,logpost,iters;L,algorithm,thin,b=burnin)
        end
    else
        for i = 1:nchains
            res[i] = adaptive_rwm(x₀,logpost,iters;L,algorithm,thin,b=burnin)
        end
    end

    # Create MCMCChains object
    Chains(
        cat([r.X' for r in res]...,dims=3),
        param_names === nothing ? ["p$i" for i = 1:size(res[1].X,1)] : param_names;
        evidence = cat([r.D' for r in res]...,dims=3),
        start=burnin+1,
        thin=thin,
        iterations=burnin:thin:iters
    )
end


################################################
## FUNCTION TO CALCULATE WEIGHTS
################################################

function get_weights(prior,α,time,dose,vols;final_only=false)
    # Model prediction for each sampled parameter
    m = hcat([model(lp,time,dose) for lp in eachrow(prior)]...)'
    # Corresponding std based on α
    σ = α[1] .+ m .* α[2]
    # loglikelihood contribution from each time point
    ll = hcat([logpdf.(d,vols) for d in eachrow(Normal.(m,σ))]...)'
    # Calculate matrix of weights
    like = exp.(cumsum(ll,dims=2))
    w = like ./ sum(like,dims=1)
    if final_only
        w = w[:,end]
    end
    return w,m
end

################################################
## R² statistic
################################################

function Bayesian_R²(fit,obs)
    var(fit) / (var(fit) + var(fit - obs))
end

