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


immutable TensorFlowInterpolation
    sess::Session
end


grad_node(graph, deriv::Type{Val{0}}) = graph["u"]
grad_node(graph, deriv::Type{Val{1}}) = graph["du_dt"]

function grad_node{N}(graph, deriv::Type{Val{N}})
    N >= 0 || throw(DomainError())
    name = "du$(N)_dt$(N)"
    if haskey(graph, name)
            # Don't create the node if it already exists
        graph["name"]
        else
    end
    as_default(sess.graph) do

        grads(
        v0  = graph["u"]
        v1 = graph["du_dt"

        end

function (id::TensorFlowInterpolation){N}(tvals, idxs, deriv::Type{Val{N}})
end

const tf=TensorFlow #required to use @op to register name automatically with @tf blcoks
"Calculate the gradient per column of us, with regards to ts"
TensorFlow.@op function grads(us, ts; name=nothing)
    outdim = get_shape(us, 2)
    dus_dts = hcat(map(1:outdim) do u_ii
        gradients(us[:,u_ii], ts)
    end...)
    
    identity(dus_dts, name=name) #hack to give it a name. 
    #FIXME: hcat could do with a name field
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
               timeseries_errors = timeseries_errors,
               retcode = :Success)

  end

end # module
