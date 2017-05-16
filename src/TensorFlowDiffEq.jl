module TensorFlowDiffEq

using DiffEqBase, TensorFlow, Compat
import DiffEqBase: solve

# Abstract Types
@compat abstract type TensorFlowAlgorithm <: AbstractODEAlgorithm end

# DAE Algorithms
immutable odetf <: TensorFlowAlgorithm end

export odetf

## Solve for DAEs uses raw_solver

function solve(
    prob::AbstractODEProblem,
    alg::TensorFlowAlgorithm,
    timeseries = [], ts = [], ks = [];
    verbose=true, dt = nothing,
    callback = nothing, abstol = 1/10^6, reltol = 1/10^3,
    saveat = Float64[], adaptive = true, maxiters = 1000,
    timeseries_errors = true, save_everystep = isempty(saveat),
    dense = save_everystep,
    save_start = true, save_timeseries = nothing,
    userdata = nothing,
    kwargs...)

    u0 = prob.u0
    f = prob.f
    tspan = prob.tspan

    if dt == nothing
      error("dt must be set.")
    end

    sess=Session(Graph())
    hl_width=64


    @tf begin #Automatically name nodes based on RHS
        t = placeholder(Float32; shape=[-1])
        tt = expand_dims(t, 2) #make it a matrix

        w1 = get_variable([1,hl_width], Float32)
        b1 = get_variable([hl_width], Float32)
        w2 = get_variable([hl_width,1], Float32)

        u = tt.*nn.sigmoid(tt*w1 + b1)*w2


        du_dt = gradients(u, tt)
        deq_rhs = f(tt,u) # - u/5 + exp(-tt/5).*cos(tt) # Should be f.(tt,u)


        loss = reduce_sum((du_dt - deq_rhs).^2)
        opt = train.minimize(train.AdamOptimizer(), loss)
    end

    t_obs = collect(tspan[1]:dt:tspan[2])

    run(sess, global_variables_initializer())
    for ii in 1:maxiters
        _, loss_o = run(sess, [opt, loss], Dict(t=>t_obs))
        if verbose && ii%50 == 1
            println(loss_o)
        end
    end


    u_net = run(sess, u, Dict(t=>t_obs))

    build_solution(prob,alg,t_obs,u_net,
               dense = dense,
               timeseries_errors = timeseries_errors,
               retcode = :Success)

  end

end # module
