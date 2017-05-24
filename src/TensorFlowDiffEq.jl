module TensorFlowDiffEq

using DiffEqBase, TensorFlow, Compat
import DiffEqBase: solve, interpolation

# Abstract Types
@compat abstract type TensorFlowAlgorithm <: AbstractODEAlgorithm end

# DAE Algorithms
immutable odetf <: TensorFlowAlgorithm
  hl_width::Int
end
odetf(;hl_width=64) = odetf(hl_width)

export odetf


immutable TensorFlowInterpolation{uElType}
    sess::Session
    u::Tensor{uElType}
end


grad_node(interp, deriv::Type{Val{0}}) = interp.u

function grad_node{N}(interp, deriv::Type{Val{N}})
    N >= 0 || throw(DomainError())
    #GOLDPLATE: this code could be enhanced, a lot. Right now it creates new nodes in the graph every time it is called. Which is a minor waste. The fix makes the code pretty hard to read and so I do not think it is worth it.
    as_default(interp.sess.graph) do
        prev = grad_node(interp, Val{N-1})
        tt = interp.sess.graph["tt"] #load the `tt` node from the graph
        grads(prev, tt)
    end
end

function (id::TensorFlowInterpolation){N}(tvals, idxs, deriv::Type{Val{N}})
    gn = grad_node(id, derviv)
    vals = run(id.sess, gn, Dict(t=>tvals))
    #PREM-OPT: the indexing could be moved inside the network, to avoid even calculating gradients for columns are are not using, by first slicing `interp.u` inside the `grad_node` function.
    vals[:,idxs]'
end

(id::TensorFlowInterpolation)(tval::Number, idxs, deriv) = id([tval], idxs, deriv)

function (id::TensorFlowInterpolation){N}(v, tvals, idxs, deriv::Type{Val{N}})
    # In-place version, noting that truely inplace operations between julia and tensorflow are actually impossible
    v[:] = @view id(tvals, idxs, deriv)[:]
end

"Calculate the gradient per column of us, with regards to ts"
function grads(us, ts)
    outdim = get_shape(us, 2)
    hcat(map(1:outdim) do u_ii
        gradients(us[:,u_ii], ts)
    end...)
end


## Solve for DAEs uses raw_solver

function solve(
    prob::AbstractODEProblem,
    alg::TensorFlowAlgorithm,
    timeseries = [], ts = [], ks = [];
    verbose=true, dt = nothing, maxiters = Int(1e4),
    progress_steps = 100, dense = true,
    timeseries_errors = true,
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
        t = placeholder(tType, shape=[-1])
        tt = expand_dims(t, 2) #make it a matrix

        # u_trial trail network definition
        w1 = get_variable([1,hl_width], uElType)
        b1 = get_variable([hl_width], uElType)
        w2 = get_variable([hl_width,outdim], uElType)

        u = u0 + tt.*nn.sigmoid(tt*w1 + b1)*w2
        du_dt = grads(u, tt)
        deq_rhs = f(tt,u)


        loss = reduce_mean((du_dt - deq_rhs).^2)
        opt = train.minimize(train.AdamOptimizer(), loss)
    end

    run(sess, global_variables_initializer())

    if (verbose)
        @show run(sess, size(u), Dict(t=>t_obs))
        @show run(sess, size(du_dt), Dict(t=>t_obs))
        @show run(sess, size(deq_rhs), Dict(t=>t_obs))
    end

    for ii in 1:maxiters
        _, loss_o = run(sess, [opt, loss], Dict(t=>t_obs))
        if verbose && ii%progress_steps == 1
            println(loss_o)
        end
    end

    verbose && println("get final estimates with u_net(t)")
    u_net, du_dt_net = run(sess, [u, du_dt], Dict(t=>t_obs))
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
               interp = TensorFlowInterpolation(sess, u),
               timeseries_errors = timeseries_errors,
               retcode = :Success)

  end

end # module
