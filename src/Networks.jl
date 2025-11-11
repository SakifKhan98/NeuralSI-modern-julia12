##############################
# Networks.jl (Julia ≥1.12)
##############################

# Keep Flux/DataLoader, drop CUDA/DiffEqFlux/Transformers/Plots/TBLogger
using Flux
# using Flux.Data: DataLoader
using MLUtils: DataLoader
using Random

# CairoMakie for visualizations (consistent with ODEfunctions.jl)
using CairoMakie

# ------------------------------------------------------------------
# Split layer: fanout to multiple subpaths, same as your original
# ------------------------------------------------------------------
struct Split{T}
    paths::T
end
Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

# ------------------------------------------------------------------
# Minimal positional embedding (replaces Transformers.Basic.PositionEmbedding)
# We keep the same API you use in main.jl:
#     pe = PositionEmbedding(1)
#     NN_input = pe(xlmap[cpu(index_input)]')'
# Default behavior = identity (no change in dimensions).
# If you want sinusoidal features later, flip `mode=:sinusoid` below.
# ------------------------------------------------------------------
struct PositionEmbedding
    d::Int
    mode::Symbol
    scale::Float32
end
PositionEmbedding(d::Integer; mode::Symbol = :identity, scale::Real = 1.0) =
    PositionEmbedding(d, mode, Float32(scale))

function loader(datalength, minibatch_size, sample_ratio, rng=123)
    # pick a random subset of indices
    idx = Random.randsubseq(MersenneTwister(rng), 1:datalength, sample_ratio)
    # truncate to a multiple of minibatch_size
    k = length(idx) - rem(length(idx), minibatch_size)
    idx = idx[1:k]
    # make fixed-size batches
    batches = [idx[i:i+minibatch_size-1] for i in 1:minibatch_size:k]
    return batches  # iterate with: for idxbatch in train_loader
end


function (pe::PositionEmbedding)(x::AbstractArray)
    # x is typically (features × batch) OR (batch × features) depending on your pipeline.
    # Your code transposes twice around this call, so returning `x` as-is preserves shapes.
    if pe.mode === :identity
        return x
    elseif pe.mode === :sinusoid
        # Simple sinusoidal along the *last* dimension; keep shape-compatible.
        # Applies sin/cos to the input and concatenates features.
        # Use with care if your NN dims expect the original width.
        s = sin.(x .* pe.scale)
        c = cos.(x .* pe.scale)
        return vcat(x, s, c)
    else
        return x
    end
end

# ------------------------------------------------------------------
# Sampler/DataLoader (unchanged behavior)
# ------------------------------------------------------------------
function loader(datalength, minibatch_size, sample_ratio, rng=123)
    idx = Random.randsubseq(MersenneTwister(rng), 1:datalength, sample_ratio)  # random subset
    # ensure length is multiple of minibatch_size
    k = length(idx) - rem(length(idx), minibatch_size)
    train_index = Flux.cpu(idx[1:k])
    return DataLoader(train_index; batchsize=minibatch_size, shuffle=true)
end

# ------------------------------------------------------------------
# Feedforward blocks (same math as before)
# ------------------------------------------------------------------
function def_FFNN_sigmoid(dim::AbstractVector{<:Integer})
    layer = Dense(dim[1], dim[2], sigmoid)
    for i in 2:length(dim)-1
        layer = cat(layer, Dense(dim[i], dim[i+1], sigmoid); dims=1)
    end
    return Chain(layer...)
end

function def_FFNN_no_digmoid_output_vc(dim::AbstractVector{<:Integer})
    layer = Dense(dim[end-1], dim[end], x -> abs(x))
    for i in 2:length(dim)-1
        layer = cat(Dense(dim[end-i], dim[end-i+1], sigmoid), layer; dims=1)
    end
    return Chain(layer...)
end

function NN_model(NNdims1::AbstractVector{<:Integer}, NNdims2::AbstractVector{<:Integer})
    NN1 = def_FFNN_sigmoid(NNdims1)
    NN1 = Chain(NN1[:]..., x -> 0.6f0 .* x .+ 0.7f0)   # same affine post-scale/shift
    NN2 = def_FFNN_no_digmoid_output_vc(NNdims2)
    model = Chain(Split(NN1, NN2))
    return model
end

# ------------------------------------------------------------------
# Globals expected from main.jl/environment:
#   pe, xlmap, re, NNparam, func!, x0, solver, tspan, tl, y, Nx
#   solveODE(...) is defined in ODEfunctions.jl
# ------------------------------------------------------------------

# lossfunc keeps your original semantics:
# - Reconstructs model via re(NNparam)
# - Computes p = [vp; vc]
# - Predicts displacement via solveODE
# - MAE on selected indices, scaled by 1e4
# NOTE: `loss_value` and `p` are kept as globals because your callback uses them.
function lossfunc(index_input)
    NN_input = pe(xlmap[Flux.cpu(index_input)]')'      # preserve your shapes
    vp, vc = re(NNparam)(NN_input)                     # model forward pass
    global p = vcat(vp, vc)
    prediction = solveODE(func!; p = p, x0 = x0, solver = solver, tspan = tspan, tl = tl)
    loss = 1e4 * mean(abs, (y .- prediction)[Flux.cpu(index_input)])
    global loss_value = loss
    return loss
end

# ------------------------------------------------------------------
# Visual helpers — CairoMakie versions
# ------------------------------------------------------------------
function visual_vp!(ax, v)
    scatter!(ax, xll, p0[1:Nx]; markersize=6, label="True")
    lines!(ax,   xll, v[1:Nx];   label="Predict")
    ax.title  = "Modulus coefficient P"
    ax.xlabel = "x"
    ax.ylabel = "value"
end

# --- REPLACE visual_vc ---
function visual_vc!(ax, v)
    scatter!(ax, xll[2:end-1], p0[Nx+2:end-1]; markersize=6, label="True")
    lines!(ax,   xll[2:end-1], v[Nx+2:end-1];   label="Predict")
    ax.title  = "Damping C"
    ax.xlabel = "x"
    ax.ylabel = "value"
end

function cb_vp_vc()
    fig = Figure(size = (700, 520))  # was resolution=...
    ax1 = Axis(fig[1,1])
    ax2 = Axis(fig[2,1])

    visual_vp!(ax1, p)
    axislegend(ax1; position=:lt)

    visual_vc!(ax2, p)
    axislegend(ax2; position=:rt)

    display(fig)

    println("loss: ", loss_value)
    @info "loss" loss = loss_value
    return false
end

# Optionally, keep a composite "results" figure similar to the original.
# `jetmap` is provided in ODEfunctions.jl (CairoMakie version).
function results(v)
    predict = solveODE(func!; p=v, x0=x0, solver=solver, tspan=tspan, tl=tl)
    diff_u  = abs.(y .- predict)

    fig = Figure(size = (1000, 900))

    axP = Axis(fig[1,1]); visual_vp!(axP, v); axislegend(axP; position=:lt)
    axC = Axis(fig[1,2]); visual_vc!(axC, v); axislegend(axC; position=:rt)

    axT = Axis(fig[2,1]); heatmap!(axT, (Array(y) .* 1e3)';  colormap=:jet);   axT.title = "True"
    axY = Axis(fig[2,2]); heatmap!(axY, (Array(predict) .* 1e3)'; colormap=:jet); axY.title = "Predict"
    axE = Axis(fig[3,1]); heatmap!(axE, (Array(diff_u) .* 1e3)'; colormap=:jet); axE.title = "Error"

    display(fig)
    return fig
end