#=
    Figure 4

    Classify posterior samples (according to patient types)
    
=#

using CSV, DataFrames, DataFramesMeta
using Plots, StatsPlots

include("defaults.jl")

################################################
## LOAD PRIOR AND POSTERIOR SAMPLES
################################################

include("../analysis/prior1.jl")
df = CSV.read("analysis/prior2.csv",DataFrame)

################################################
## PLOTS
################################################

# Start from λ (ignore V₀)
offset = 3

# Settings
cmaps = [:Greens_7,:Oranges_7,:PuBu_7,:RdPu_7]
lims = [
    [-5.0,0.0], # λ
    [0.0,5.0],  # K
    [-3.0,0.0], # γ
    [-6.0,3.0], # ζ
    [-10.0,3.0],# η,
    [-5.0,0.0]  # ϕ₀
]

# Produce plots
plts = [plot() for i = 1:6, j = 1:6]
for i = 1:6, j = 1:6
    if i == j
        for class = [1,2,3,4]
            @df @subset(df,:class .== class) density!(plts[i,i],cols(offset + i),
                frange=0.0,fα=0.2,lw=2.0,c=palette(cmaps[class])[5],xlim=lims[i],trim=true,label="")
            plot!(plts[i,j],xlabel=names(df)[offset + j])
        end
    elseif j < i
        # Produce 2D density plots
        for class = [1,2,3,4]
            @df @subset(df,:class .== class) density2d!(plts[i,j],cols(offset + j),cols(offset + i),
                levels=10,fill=true,c=cmaps[class],lw=0.0,z_clip=0.4,α=0.4,xlim=lims[j],ylim=lims[i])
            plot!(plts[i,j],xlabel=names(df)[offset + j],ylabel=names(df)[offset + i])
        end
    else
        # Turn off axes
        plot!(plts[i,j],[],[],box=:none,axis=false,label="")
    end
end

# Produce legend
for class = [1,2,3,4]
    plot!(plts[1,end],[],[],frange=0.0,c=palette(cmaps[class])[5],lw=0.0,α=0.8,label="$class",legend=:topright)
end

# Fig 4
fig4 = plot(permutedims(plts)...,size=(1200,1150))

# Save
savefig(fig4,"$(@__DIR__)/fig4.svg")
