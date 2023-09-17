#=
    training.jl

    Estimate α₁ and α₂, parameters relating to the noise process.

=#

using AdaptiveMCMC
using .Threads
using LinearAlgebra
using KernelDensity
using MCMCChains
using MCMCDiagnosticTools
using Optim

include("../figures/defaults.jl")

################################################
## LOAD DATA AND MODELS
################################################

include("data.jl")
include("models.jl")
include("inference.jl")

################################################
## SETUP PRIOR (INCL. α)
################################################

# Model parameter prior
include("../analysis/prior1.jl")

################################################
## GUESS
################################################

lp₀ = log.([0.5,2.0,0.5,0.5,0.5,0.01])
lθ₀ = vcat(lp₀,log.([0.1,0.1])) # Including noise

################################################
## ESTIMATE NOISE BY PERFORMING MCMC ON NOISE PARAMS SIMULTANEOUSLY
################################################

## Chain settings
iters = 1000000
burnin = 500000
thin = 100

## Store chains
C = Array{Chains}(undef,length(training_id))

## Perform MCMC
@threads for i = eachindex(training_id)

    # Patient ID
    id = training_id[i]

    # Construct log density
    llike = lθ -> loglike(lθ[1:end-2],exp.(lθ[end-1:end]),id)
    lpost = lθ -> insupport(prior1,lθ) ? llike(lθ) + logpdf(prior1α,lθ) : -Inf

    # Perform MCMC and store the result
    C[i] = mcmc(lpost,lθ₀,iters;burnin,thin,param_names=twocompartment_params_expanded);

end

## Convert to matrix of samples for α
X = vcat([c.value.data[:,:,1] for c in C]...)[:,end-1:end]

## Find MAP using kernel density estimation
func = InterpKDE(kde(X))
α = exp.(optimize(x -> -pdf(func,x...),mean(X,dims=1)[:]).minimizer)

## Save noise model
CSV.write("analysis/noise_model.csv",DataFrame(α=α))


################################################
## TRAIN TO EACH PATIENT INDIVIDUALLY (ALL PATIENTS)
################################################

## Load noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α

## Chain settings
iters   = 150000    # Total iterations
burnin  = 50000     # Iterations discarded as burnin
nchains = 4         # Number of independent chains
thin    = 100       # Thinning (samples to discard)
L       = 4         # Number of temperature levels

## Store chains
C = Array{Chains}(undef,length(unique(data.ID)))

r = falses(length(C))

## Perform MCMC
@threads for i = eachindex(unique(data.ID))
    
    # Patient ID
    id = unique(data.ID)[i]

    # Construct log density
    llike = lp -> loglike(lp,α,id)
    lpost = lp -> insupport(prior1,lp) ? llike(lp) + logpdf(prior1,lp) : -Inf

    # Perform MCMC and store the result
    C[i] = mcmc(lpost,lp₀,iters;L,burnin,thin,nchains,param_names=twocompartment_params);

    # Display
    r[i] = true
    ndone = count(r)
    ntotal = length(r)
    display("Completed $ndone / $ntotal")

end

## Create and save figures
default(); gr();
for (i,id) = enumerate(unique(data.ID))
    # Chain figure
    fig = plot(C[i],size=(1000,1600),bottom_margin=10Plots.mm,right_margin=20Plots.mm)
    savefig(fig,"analysis/diagnostic/individual_mcmc/patient_id_$id.png")
end

## Create MCMC diagnostic table
R̂ = hcat([ess_rhat(c).nt.rhat for c in C]...)'
df_diag = hcat(
    DataFrame(ID=unique(data.ID)),
    DataFrame(in_training=in_training = [i ∈ training_id for i in unique(data.ID)]),
    DataFrame(round.(R̂,digits=3),twocompartment_params)
)
CSV.write("analysis/diagnostic/mcmc_diagnostics.csv",df_diag)


################################################
## CREATE DATAFRAME
################################################

# Number of values to keep for each (chain)
neach = div(2000,nchains)

# Index of values to keep in prior (keep 1000 equally spaced)
idx = Int.(round.(range(1,size(C[1].value.data,1),neach)))

# Array of matrices to store (samples)
X = [vcat([c.value.data[idx,:,i] for i = 1:nchains]...) for c in C]
D = [vcat([c.logevidence[1,1,i][idx] for i = 1:nchains]...) for c in C]

# Get initial volumes for storage
init_vols = [@subset(data, :ID .== id, :Time .== 0.0).Volume[1] for id in unique(data.ID)]

# Create and store a data frame
df = DataFrame(
    [vcat([fill(unique(data.ID)[i],length(D[i])) for i = eachindex(unique(data.ID))]...) vcat(D...) vcat(D...) vcat([fill(init_vols[i],length(D[i])) for i = eachindex(unique(data.ID))]...)  vcat(X...)],
    ["id";"D";"class";"init_volume";"log_" .* twocompartment_params]
)

df.id = Int.(df.id)

################################################
## CLASSIFY SAMPLES
################################################

# Include classifier
include("classifier.jl")

# Classify each posterior sample (based on the standard patient)
df.class = [classify(Vector(r[5:end])) for r in eachrow(df)]

# Fix column types
df.id = Int.(df.id)
df.class = Int.(df.class)

# Classifications of 10,000 prior samples
prior1_classes = [classify(rand(prior1)) for _ = 1:10000]
prior1_classpoor = count(prior1_classes .== 2) / length(prior1_classes)

################################################
## SAVE RESULTS FOR ALL PATIENTS (FOR LOOCV)
################################################

CSV.write("analysis/posterior1_allpatients.csv",df)

################################################
## CREATE AND SAVE POSTERIOR1 (TRAINING SET ONLY)
################################################

# Create posterior1 using only patients in the training set
idx = [id ∈ training_id for id in df.id]
posterior1 = df[idx,:]

# Save posterior1
CSV.write("analysis/posterior1.csv",posterior1)

# Save plots
for (i,id) = enumerate(training_id)

    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Plot posterior samples
    fig = plot()
    for r = eachrow(@subset(posterior1,:id .== id))
        plot!(fig,timeᵢ,model(Vector(r[5:end]),id),c=:black,α=0.01,label="")
    end

    # Plot data
    scatter!(fig,timeᵢ,volsᵢ,ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(fig,timeᵢ,volsᵢ,ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Label
    plot!(fig,xlabel="Time [d]",ylabel="GTV [FC]",xticks=0:7:100)

    # Save
    savefig(fig,"analysis/diagnostic/training_posterior1_fits/patient_id_$id.png")
end

################################################
## CREATE SAMPLES FROM PRIOR2 (EXPANDED POSTERIOR 1)
################################################

# Number of samples
m = 100000

# Posterior samples to a matrix
X = Matrix(posterior1[:,5:end])
Xc = [eachrow(X)...]

# Bandwidth (Silverman's rule)
n,d = size(X)
Σ = Diagonal((4 / (d + 2))^(1 / (d + 4)) * n^(-1 / (d + 4)) * std.(eachcol(X)))^2
kd = MvNormal(Σ)

# Expansion factor
β = 2

# Create samples
Y = zeros(m,d) .+ Inf
for i = axes(Y,1)
    while !insupport(prior1,Y[i,:])
        # Sample dynamical parameters from prior independently of noise parameters
        Y[i,:] = [rand(Xc)[1:end-2]; rand(Xc)[end-1:end]] + β * rand(kd)
    end
end

# Classify samples (standard patient)
prior2_classes = [classify(lp) for lp in eachrow(Y[:,1:end])]

# Create and save data frame
prior2 = hcat(
    DataFrame(class=prior2_classes),
    DataFrame(Y,names(posterior1)[5:end])
)
CSV.write("analysis/prior2.csv",prior2)