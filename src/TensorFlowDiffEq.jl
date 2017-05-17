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
    dense = save_everystep, progress_steps = 50,
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

    uElType = eltype(u0)
    tType = typeof(tspan[1])
    outdim = length(u0)
    t_obs = collect(tspan[1]:tType(dt):tspan[2])


    @tf begin #Automatically name nodes based on RHS
        t = constant(t_obs)
        tt = expand_dims(t, 2) #make it a matrix

        # u_trial trail network definition
        w1 = get_variable([1,hl_width], uElType)
        b1 = get_variable([hl_width], uElType)
        w2 = get_variable([hl_width,outdim], uElType)

        u = u0 + tt.*nn.sigmoid(tt*w1 + b1)*w2

        du_dt = hcat(map(1:outdim) do u_ii
                gradients(u[:,u_ii], tt)
        end...)
        deq_rhs = f(tt,u) # - u/5 + exp(-tt/5).*cos(tt) # Should be f.(tt,u)


        loss = reduce_mean((du_dt - deq_rhs).^2)
        opt = train.minimize(train.AdamOptimizer(), loss)
    end

    run(sess, global_variables_initializer())

    if (verbose)
      @show run(sess, size(u))
      @show run(sess, size(du_dt))
      @show run(sess, size(deq_rhs))
    end

    for ii in 1:maxiters
        _, loss_o = run(sess, [opt, loss])
        if verbose && ii%progress_steps == 1
            @show run(sess, size(du_dt))
            println(loss_o)
        end
    end

    verbose && println("get final estimates with u_net(t)")
    u_net, du_dt_net = run(sess, [u, du_dt])
    verbose && println("preparing results")

    if typeof(u0) <: AbstractArray
        timeseries = Vector{typeof(u0)}(0)
        if dense
            du = Vector{typeof(u0)}(0)
        end
    for i=1:size(u_net, 1)
        push!(timeseries, reshape(view(u_net, i, :)', size(u0)))
        if dense
            push!(du, reshape(view(du_dt_net, i, :)', size(u0)))
        end
    end
    else
        timeseries = vec(u_net)
        if dense
          du = vec(du_dt_net)
        end
    end
  
    if !dense
      du = []
    end

    build_solution(prob,alg,t_obs,timeseries,
               dense = dense, du = du,
               timeseries_errors = timeseries_errors,
               retcode = :Success)

  end

end # module
