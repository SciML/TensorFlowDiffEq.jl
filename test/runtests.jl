using TensorFlowDiffEq
using Base.Test

using Plots; plotly()
using DiffEqBase

f = (t,u) -> -u/5 + exp(-t/5)*cos(t)
(::typeof(f))(::Type{Val{:analytic}},t,u0) =  exp(-t/5)*(u0 + sin(t))
prob = ODEProblem(f,0.0,(0.0,2.0))

sol = solve(prob,odetf(),dt=0.02)
plot(sol,plot_analytic=true)
