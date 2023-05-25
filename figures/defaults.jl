gr()
default()
default(
    fontfamily="Helvetica",
    tick_direction=:out,
    guidefontsize=9,
    annotationfontfamily="Helvetica",
    annotationfontsize=10,
    annotationhalign=:left,
    box=:on,
    msw=0.0,
    lw=1.5
)

alphabet = "abcdefghijklmnopqrstuvwxyz"

function add_plot_labels!(plt;offset=0)
    n = length(plt.subplots)
    for i = 1:n
        plot!(plt,subplot=i,title="($(alphabet[i+offset]))")
    end
    plot!(
        titlelocation = :left,
        titlefontsize = 10,
        titlefontfamily = "Arial"
    )
end

# Colours
col_prior1 = "#2d79ff"
col_prior2 = "#af52de"
col_posterior = "#f63a30"


using KernelDensity
using Plots, StatsPlots

"""
    density2d(x,y,...)

Create 2D kernel density plot.
"""
@userplot density2d
@recipe function f(kc::density2d; levels=10, clip=((-3.0, 3.0), (-3.0, 3.0)), z_clip = nothing)
    x,y = kc.args

    x = vec(x)
    y = vec(y)

    k = KernelDensity.kde((x, y))
    z = k.density / maximum(k.density)
    if !isnothing(z_clip)
        z[z .< z_clip * maximum(z)] .= NaN
    end

    legend --> false

    @series begin
        seriestype := contourf
        colorbar := false
        (collect(k.x), collect(k.y), z')
    end

end