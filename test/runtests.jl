using TensorFlowDiffEq
using Base.Test

using Plots; plotly()
using DiffEqBase

# Toy problem 1

f = (t,u) -> -u/5 + exp(-t/5).*cos(t)
(::typeof(f))(::Type{Val{:analytic}},t,u0) =  exp(-t/5)*(u0 + sin(t))
prob = ODEProblem(f,0.0,(0.0,2.0))
sol = solve(prob,odetf(),dt=0.02)

#plot(sol,plot_analytic=true)


# The real issue

function lorenz(t,u)
    du1 = 10.0(u[2]-u[1])
    du2 = u[1].*(28.0-u[3]) - u[2]
    du3 = u[1].*u[2] - (8/3)*u[3]

    [du1 du2 du3]
end

prob = ODEProblem(lorenz,[1.0,1.0,0.0],(0.0,2.0))
sol = solve(prob,odetf(),dt=0.02)
println("Test complete")
