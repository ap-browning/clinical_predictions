#=
    Figures 7, 8, and 9

    Actual patients
=#

using CSV, DataFrames, DataFramesMeta
using Plots, StatsPlots
using StatsBase
using LinearAlgebra
using .Threads

include("defaults.jl")
include("../analysis/models.jl")
include("../analysis/data.jl")
include("../analysis/classifier.jl")
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

# Load noise model
α = CSV.read("analysis/noise_model.csv",DataFrame).α

################################################
## PERFORM INFERENCE USING PARTIAL DATA
################################################

W = Array{Any}(undef,length(patient_id))
M = Array{Any}(undef,length(patient_id))
M̄ = Array{Any}(undef,length(patient_id))

# Loop through each patient
@threads for i = eachindex(patient_id)

    # Patient id
    id = patient_id[i]

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Calculate weights
    W[i],M[i] = get_weights(X,α,timeᵢ,doseᵢ,volsᵢ)

    # Calculate full model traces for plotting
    M̄[i] = hcat([model(lp,timeᵢ,doseᵢ;ret=:function).(tplt) for lp = Xc]...)'

end

################################################
## Figure 7
################################################
 
# Which time indices to plot upto?
upto = [2,4,6,8]

plts = [[plot() for _ = eachindex(upto)] for _ = eachindex(patient_id)]

# Loop through plots
for (i,id) = enumerate(patient_id), j = eachindex(upto)

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Confidence bands
    l1,l2,u2,u1 = [[quantile(v,Weights(W[i][:,upto[j]]),q) for v in eachcol(M̄[i])] for q = [0.025,0.25,0.75,0.975]]

    # Mean
    μ = [mean(v,Weights(W[i][:,upto[j]])) for v in eachcol(M̄[i])]


    # Index of current upto time
    idx_max = tplt[end] == timeᵢ[upto[j]] ? length(tplt) : findfirst(tplt .> timeᵢ[upto[j]])

    # Plot confidence bands
    plts[i][j] = plot(tplt[1:idx_max],u1[1:idx_max],frange=l1[1:idx_max],c=:black,lw=0.0,label="",fα=0.4)
    plot!(plts[i][j],tplt[1:idx_max],u2[1:idx_max],frange=l2[1:idx_max],c=:black,lw=0.0,label="",fα=0.4)
    plot!(plts[i][j],tplt[idx_max:end],u1[idx_max:end],frange=l1[idx_max:end],c=col_posterior,lw=0.0,label="",fα=0.4)
    plot!(plts[i][j],tplt[idx_max:end],u2[idx_max:end],frange=l2[idx_max:end],c=col_posterior,lw=0.0,label="",fα=0.4)

    # Plot mean
    plot!(plts[i][j],tplt[1:idx_max],μ[1:idx_max],c=:black,lw=2.0,label="",fα=0.4)
    plot!(plts[i][j],tplt[idx_max:end],μ[idx_max:end],c=col_posterior,lw=2.0,label="",fα=0.4)

    # Plot observed data at present
    scatter!(plts[i][j],timeᵢ[1:upto[j]],volsᵢ[1:upto[j]],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[1:upto[j]],volsᵢ[1:upto[j]],ylim=(0.0,:auto),ms=3,widen=true,c=:black,label="")

    # Plot future observed data
    scatter!(plts[i][j],timeᵢ[upto[j]+1:end],volsᵢ[upto[j]+1:end],ylim=(0.0,:auto),widen=true,msc=:black,ms=6.0,α=0.5,msw=2.0,mc=:white,label="",xlim=extrema(timeᵢ))
    scatter!(plts[i][j],timeᵢ[upto[j]+1:end],volsᵢ[upto[j]+1:end],ylim=(0.0,:auto),ms=3,widen=true,c=:black,α=0.5,label="")

    # Plot present time
    vline!(plts[i][j],[timeᵢ[upto[j]]],c=:black,lw=2.0,ls=:dash,label="",widen=true)
    
    # Calculate and plot R² value
    R² = median([Bayesian_R²(M[i][idx,:],volsᵢ) for idx = sample(1:nrow(prior2),Weights(W[i][:,upto[j]]),1000)])
    R² = round(R²,sigdigits=3)
    plot!(plts[i][j],title="R² = $R²",titlefontsize=10)

end

plts_patient = [plot(plts[i]...,layout=grid(1,4),link=:all,ylim=(0.0,2.0),xticks=0:14:56,xlim=(-4.0,60.0)) for i in eachindex(patient_id)]

# Figure 7
fig7 = plot(plts_patient...,layout=grid(4,1),size=(800,700),xlabel="Time [d]",link=:all)
savefig(fig7,"$(@__DIR__)/fig7.svg")
fig7

################################################
## Figure 8
################################################

    ############################################
    ## Row 1: Posterior for γ

    # Color palette
    cols = palette(:RdPu,12)[end-8:end]

    # Setup
    lγ = prior2[:,4]
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
        for j = eachindex(timeᵢ)
            f = pdf(kde(lγ,weights=W[i][:,j]),xv)
            plot!(plts1[i],fill(timeᵢ[j],length(xv)+2),[xv[1]; xv; xv[end]],[0.0; f; 0.0],lw=2.0,c=cols[j+1],label="")
        end
        
        # Styling
        plot!(
            plts1[i],
            widen=false, zticks=[], xticks=0.0:7.0:56.0,
            axis=:xy,framestyle=:axis,grid=:y,camera=(50,25)
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

        # Calculate predicted final tumour volume for each prior sample
        F = M[i][:,end]

        # Calculate statistics
        μ = [mean(F,Weights(Wᵢⱼ)) for Wᵢⱼ in eachcol(W[i])]
        l1 = [quantile(F,Weights(Wᵢⱼ),0.025) for Wᵢⱼ in eachcol(W[i])]
        u1 = [quantile(F,Weights(Wᵢⱼ),0.975) for Wᵢⱼ in eachcol(W[i])]
        l2 = [quantile(F,Weights(Wᵢⱼ),0.25) for Wᵢⱼ in eachcol(W[i])]
        u2 = [quantile(F,Weights(Wᵢⱼ),0.75) for Wᵢⱼ in eachcol(W[i])]

        # Plot
        for j = eachindex(timeᵢ)[2:end]
            plot!(plts2[i],timeᵢ[j]*[1.0,1.0],[l1[j],u1[j]],lw=8.0,c=:black,α=0.3,label="")
            plot!(plts2[i],timeᵢ[j]*[1.0,1.0],[l2[j],u2[j]],lw=8.0,c=:black,α=0.3,label="")
        end
        scatter!(plts2[i],timeᵢ[2:end],μ[2:end],c=:black,ms=4.0,label="")

        # Add observed final volume
        hline!(plts2[i],[volsᵢ[end]],lw=2.0,ls=:dashdot,c=:red,label="")

    end

    row2 = plot(plts2...,layout=grid(1,4),xlabel="Time [d]", ylabel="Final FC Volume",
        link=:x,xlim=(-4.0,60.0),widen=true,ylim=(0.01,10.0),yaxis=:log10,yticks=10.0 .^ (-2.0:2.0),xticks=0:7:56)


    # Figure 8
    fig8 = plot(row1,row2,layout=@layout([a; b]),size=(800,450))
    add_plot_labels!(fig8)
    savefig(fig8,"$(@__DIR__)/fig8.svg")

################################################
## Figure 9
################################################

# Class for each prior sample (should this be with the patient specific therapy?)
class = prior2.class

# Probability of responding to treatment (i.e., not being in class 2)
p = [[sum(Wᵢⱼ[prior2.class .!= 2]) for Wᵢⱼ in eachcol(W[i])] for i = 1:4]

# Color for each class
cols = [palette(symb)[5] for symb = [:Greens_7,:Oranges_7,:PuBu_7,:RdPu_7]]

# Produce figure
tmax = 70.0
plts = [plot() for _ = eachindex(patient_id)]
for (i,id) in enumerate(patient_id)

    # Get patient specific data
    timeᵢ,volsᵢ,doseᵢ = normalised_data[id]

    # Initialise overall probability response curve
    xc = Float64[]
    pc = Float64[]

    # Loop through observation times
    for j = eachindex(timeᵢ)

        # Times for which prediction applies to
        x = j != length(timeᵢ) ? [timeᵢ[j],timeᵢ[j+1]] : [timeᵢ[j],tmax]

        # Weights to use
        Wᵢⱼ = W[i][:,j]

        # Probability of being in each class
        probs = [0.0;cumsum([sum(Wᵢⱼ[class .== c]) for c = 1:4][[1,3,4,2]])] # Perturbed (class 2 last)
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