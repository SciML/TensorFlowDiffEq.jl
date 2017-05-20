module TensorFlowDiffEq

using DiffEqBase, TensorFlow, Compat
import DiffEqBase: solve

# Abstract Types
@compat abstract type TensorFlowAlgorithm <: AbstractODEAlgorithm end

# DAE Algorithms
immutable odetf{Opt<:train.Optimizer} <: TensorFlowAlgorithm
  hl_widths::Vector{Int}
  optimizer::Opt
end

"""
    odetf
Specify the hyper-parameters used by the TensorFlow solver.
Including the optimisation hyper-paramters

for example
```julia
odetf(
    256,
    # use a neural network with 256 neurons in one-hiddlen layer in trail solution
    optimizer=train.MomentumOptimizer(0.5, 0.9)
    # Fit the network using a momentum based optimizser, learning rate of 0.5, momentum of 0.9
)
```

To make a deep network, pass a `Vector` to `hl_width`, given the width of each hiddden layer
Eg a 3 layer network
```julia
odetf(hl_width = [64, 128, 64])
```
"""
function odetf(;hl_width=64, optimizer=train.AdamOptimizer())
    odetf(hl_width, optimizer)
end

odetf(hl_width::Integer, optimizer) = odetf([hl_width], optimizer)


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
    f = prob.f

    uElType = eltype(u0)
    tType = typeof(tspan[1])
    outdim = length(u0)
    t_obs = collect(tspan[1]:tType(dt):tspan[2])


    @tf begin #Automatically name nodes based on RHS
        t = constant(t_obs)
        tt = expand_dims(t, 2) #make it a matrix

        # u_trial trail network definition
        z = tt
        for (ii, hl_width) in enumerate(alg.hl_widths)
            width_below = get_shape(z, 2)
            w = get_variable("w_$ii", [width_below, hl_width], uElType)
            b = get_variable("b_$ii", [hl_width], uElType)
            z = nn.sigmoid(z*w + b; name="z_$ii")
        end
        width_below = get_shape(z, 2)
        w_out = get_variable([width_below, outdim], uElType)

        u = u0 + tt.*z*w_out

        if outdim>1 #FIXME: Bug in Tensorflow.jl will not concat a single tensor
            du_dt = hcat(map(1:outdim) do u_ii
                gradients(u[:,u_ii], tt)
            end...)
        else
            du_dt = gradients(u,tt)
        end
        
        deq_rhs = f(tt,u) # - u/5 + exp(-tt/5).*cos(tt) # Should be f.(tt,u)


        loss = reduce_mean((du_dt - deq_rhs).^2)
        opt = train.minimize(alg.optimizer, loss)
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
