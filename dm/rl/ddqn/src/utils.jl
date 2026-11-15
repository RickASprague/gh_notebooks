# Plotting / reporting helpers, no RL logic.

using UnicodePlots                       # learning curve in the terminal
using CairoMakie                         # learning curve as a PNG (resources/<env>.png)

# Where a learning curve goes. Bool still works for train!(...; plots=true) (= UnicodePlot).
@enum PlotStyle NoPlot UnicodePlot MakiePlot BothPlots

Base.convert(::Type{PlotStyle}, b::Bool) = b ? UnicodePlot : NoPlot

# Trailing moving average: avgᵢ = mean(returns[max(1, i−w+1):i]).
moving_avg(x, w) = [sum(x[max(1, i - w + 1):i]) / min(i, w) for i ∈ eachindex(x)]

# Plot y-range: at least ylim, widened to the data.
yrange(returns, ylim) = (min(ylim[1], minimum(returns)), max(ylim[2], maximum(returns)))

# Episode returns (dots) with a 50-episode moving average (line) and the "solved" level.
function learning_curve(returns; window=50, solved=200, ylim=(-400, 320), title="Learning curve")
    n = length(returns)
    avg = moving_avg(returns, window)
    plt = scatterplot(1:n, returns; name="return", marker=:dot, color=:normal,
                      xlabel="episode", ylabel="return", title,
                      width=70, height=18, ylim=yrange(returns, ylim))
    lineplot!(plt, 1:n, avg; name="avg ($window)", color=:cyan)
    hline!(plt, solved; name="solved", color=:green)
    return plt
end

# Same curve with Makie. Calls are qualified: UnicodePlots and Makie export overlapping names.
function learning_curve_figure(returns; window=50, solved=200, ylim=(-400, 320),
                               title="LunarLander-v3 DQN learning curve")
    n = length(returns)
    fig = CairoMakie.Figure(size=(1000, 550), fontsize=16)
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="episode", ylabel="return G", title)
    CairoMakie.scatter!(ax, 1:n, returns; color=(:gray40, 0.35), markersize=4, label="return")
    CairoMakie.lines!(ax, 1:n, moving_avg(returns, window); color=:dodgerblue, linewidth=2.5,
                      label="avg ($window)")
    CairoMakie.hlines!(ax, solved; color=:seagreen, linestyle=:dash, linewidth=2, label="solved ($solved)")
    CairoMakie.ylims!(ax, yrange(returns, ylim)...)
    CairoMakie.axislegend(ax; position=:rb)
    return fig
end

function save_learning_curve(returns, path; kw...)
    mkpath(dirname(path))
    CairoMakie.save(path, learning_curve_figure(returns; kw...); px_per_unit=2)
    return path
end

# Dispatch on the PlotStyle: terminal, PNG, both, or nothing. kw (solved, ylim, title) → both plots.
function show_curve(returns, style; path, kw...)
    style = convert(PlotStyle, style)
    isempty(returns) && return nothing
    style ∈ (UnicodePlot, BothPlots) && println(learning_curve(returns; kw...))
    style ∈ (MakiePlot, BothPlots) && println("learning curve → ", save_learning_curve(returns, path; kw...))
    return nothing
end
