# Integrate a test particle past a fixed point charge at the origin.
#   dx/dt = v,   dv/dt = α x / r³,   r = ‖x‖,   α = kqQ/m
#
#   julia --project src/trajectory.jl

ENV["GKSwstype"] = "100"   # headless GR
using OrdinaryDiffEq, LinearAlgebra, Plots

const α = -1.0   # α < 0: attractive (opposite charges);  α > 0: repulsive

function coulomb!(du, u, p, t)
    x, v = u[1:2], u[3:4]
    du[1:2] = v
    du[3:4] = α * x / norm(x)^3
end

x0 = [3.0, 1.0]    # initial position
v0 = [-1.0, 0.0]   # initial velocity (moving left)
tspan = (0.0, 4.0)

sol = solve(ODEProblem(coulomb!, [x0; v0], tspan), Tsit5(); saveat = range(tspan..., length = 60))
X = reduce(hcat, sol.u)

plt = plot(X[1, :], X[2, :]; label = "trajectory", lw = 2, marker = :circle, ms = 2,
    xlabel = "x", ylabel = "y", title = "Charged Particle Trajectory", aspect_ratio = :equal)
scatter!(plt, [0], [0]; label = "point charge", ms = 10, color = :red)
scatter!(plt, [x0[1]], [x0[2]]; label = "t₀", ms = 10, color = :green, marker = :star5)

out = joinpath(@__DIR__, "..", "resources", "coulomb_trajectory.png")
mkpath(dirname(out))
savefig(plt, out)
println("saved $out")
