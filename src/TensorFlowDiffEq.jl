module TensorFlowDiffEq

using DiffEqBase, TensorFlow, Compat
import DiffEqBase: solve

# Abstract Types
@compat abstract type TensorFlowAlgorithm <: AbstractODEAlgorithm end

# DAE Algorithms
immutable odetf <: TensorFlowAlgorithm
  hl_width::Int
end
odetf(;hl_width=64) = odetf(hl_width)

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
    tspan = prob.tspan

    if dt == nothing
      error("dt must be set.")
    end

    sess=Session(Graph())
    hl_width=alg.hl_width
    f = prob.f

    t_obs = collect(tspan[1]:dt:tspan[2])

    @tf begin #Automatically name nodes based on RHS
        t = constant(t_obs) #for now, just code this in as a constant
        tt = expand_dims(t, 2) #make it a matrix

        w1 = get_variable([1,hl_width], Float64)
        b1 = get_variable([hl_width], Float64)
        w2 = get_variable([hl_width,length(u0)], Float64)

        u = u0 + tt.*nn.sigmoid(tt*w1 + b1)*w2


        du_dt = gradients(u, tt)
        # This whole section is a static graph generation
        # bound to the length of t_obs
        # todo: rewrite to by dynamic

        deq_rhs_steps = Vector{Tensor}()
        for t_ii in 1:get_shape(t,1)
            val =  f(t[t_ii], u[t_ii,:])
            push!(deq_rhs_steps, val)
        end
        deq_rhs = vcat(deq_rhs_steps...)

        loss = reduce_sum((du_dt - deq_rhs).^2)
        opt = train.minimize(train.AdamOptimizer(), loss)
    end


    run(sess, global_variables_initializer())
    for ii in 1:maxiters
        _, loss_o = run(sess, [opt, loss])
        if verbose && ii%50 == 1
            println(loss_o)
        end
    end


    u_net = run(sess, u)

    if typeof(u0) <: AbstractArray
        timeseries = Vector{typeof(u0)}(0)
        for i=1:size(u_net, 1)
            push!(timeseries, reshape(view(u_net, i, :)', size(u0)))
        end
    else
            timeseries = vec(u_net)
    end

    build_solution(prob,alg,t_obs,timeseries,
               dense = dense,
               timeseries_errors = timeseries_errors,
               retcode = :Success)

  end

end # module
