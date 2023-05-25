#=
    Figure 3

    Posteriors from training data.
    
=#

using CSV, DataFrames, DataFramesMeta
using Plots, StatsPlots

include("defaults.jl")

################################################
## LOAD PRIOR AND POSTERIOR SAMPLES
################################################

include("../analysis/prior1.jl")
X = Matrix(CSV.read("analysis/prior2.csv",DataFrame))[:,4:end]

################################################
## PLOT
################################################

xrange = [extrema(p) .+ abs(-(extrema(p)...)) .* (-0.05,0.05) for p in prior1.v]

plts = [plot() for _ in prior1.v]
for i = eachindex(plts)

    # Plot prior
    plot!(plts[i],x -> pdf(prior1.v[i],x),xlim=xrange[i],c=col_prior1,lw=2.0,frange=0.0,fα=0.2,label="")

    # Plot posterior
    density!(plts[i],X[:,i],xlim=xrange[i],c=col_prior2,lw=2.0,frange=0.0,fα=0.3,label="",trim=true)

    # Add decorations
    plot!(plts[i],xlabel="log $(twocompartment_params[i])")

    # Ticks
    plot!(plts[i],widen=false)
    ymax = ylims(plts[i])[2]
    plot!(plts[i],yticks=(range(0,ymax,5),[]),ylim=(0.0,ymax*1.05))

end

# Empty plot (for legend)
plt2 = plot([],[],frange=0.0,fα=0.3,c=col_prior1,label="Prior")
plot!(plt2,[],[],frange=0.0,fα=0.3,c=col_prior2,label="Posterior")
plot!(plt2,axis=:off,box=:off,grid=:none,xticks=[],yticks=[])

# Figure 3
fig3 = plot(plts...,plt2,layout=grid(3,3),size=(800,500))
add_plot_labels!(fig3)
savefig(fig3,"$(@__DIR__)/fig3.svg")