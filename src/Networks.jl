##############################
# Networks.jl (Julia ≥1.12)
##############################

using Flux
using Functors
using Flux: cpu
using MLUtils: DataLoader
using Random
using CairoMakie

# -----------------------------------------------------------------------------
# Split layer (same behavior as before)
# -----------------------------------------------------------------------------
struct Split{T}
    paths::T
end
Split(paths...) = Split(paths)
Functors.@functor Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

# -----------------------------------------------------------------------------
# Minimal PositionEmbedding — default identity
# -----------------------------------------------------------------------------
struct PositionEmbedding
    d::Int
    mode::Symbol
    scale::Float32
end
PositionEmbedding(d::Integer; mode::Symbol = :identity, scale::Real = 1.0) =
    PositionEmbedding(d, mode, Float32(scale))

function (pe::PositionEmbedding)(x::AbstractArray)
    pe.mode === :identity && return x
    if pe.mode === :sinusoid
        s = sin.(x .* pe.scale)
        c = cos.(x .* pe.scale)
        return vcat(x, s, c)   # NOTE: changes feature width; adjust dims if we use this
    end
    return x
end

# -----------------------------------------------------------------------------
# Data loader — random subset, batched by MLUtils
# -----------------------------------------------------------------------------
function loader(datalength, minibatch_size, sample_ratio, rng=123)
    idx = Random.randsubseq(MersenneTwister(rng), 1:datalength, sample_ratio)
    k   = length(idx) - rem(length(idx), minibatch_size)
    idx = idx[1:k]
    # DataLoader will slice this vector into batches of Int indices
    return DataLoader(idx; batchsize=minibatch_size, shuffle=true)
end

# -----------------------------------------------------------------------------
# FFNN blocks (match original)
# -----------------------------------------------------------------------------
function def_FFNN_sigmoid(dim::AbstractVector{<:Integer})
    layer = Dense(dim[1], dim[2], sigmoid)
    for i in 2:length(dim)-1
        layer = cat(layer, Dense(dim[i], dim[i+1], sigmoid); dims=1)
    end
    return Chain(layer...)
end

# Scaled damping head: map outputs to a realistic 15..25 range
_vc_out(x) = 15f0 .+ 10f0 .* σ.(x)

function def_FFNN_no_digmoid_output_vc(dim::AbstractVector{<:Integer})
    layer = Dense(dim[end-1], dim[end], _vc_out)
    for i in 2:length(dim)-1
        layer = cat(Dense(dim[end-i], dim[end-i+1], sigmoid), layer; dims=1)
    end
    return Chain(layer...)
end

function NN_model(NNdims1::AbstractVector{<:Integer}, NNdims2::AbstractVector{<:Integer})
    NN1 = def_FFNN_sigmoid(NNdims1)
    NN1 = Chain(NN1[:]..., x -> 0.6f0 .* x .+ 0.7f0)   # same affine shift/scale as original
    NN2 = def_FFNN_no_digmoid_output_vc(NNdims2)
    model = Chain(Split(NN1, NN2))
    return model
end

# -----------------------------------------------------------------------------
# Globals expected from main.jl (they are set there with `global ...`)
# -----------------------------------------------------------------------------
# Nx, xll, p0, xlmap, pe, func!, x0, solver, tspan, tl, y
# A1, A2, A3, A4, EI, rhoA, b, forced_time, force_magnitude
# abstol, reltol, prob_template, prob2_template
# NNparam, re
# Also: we write to globals `p` and `loss_value` (used by callbacks/plots).

# -----------------------------------------------------------------------------
# Loss function — uses ODEProblem template + tolerances (fast!)
# -----------------------------------------------------------------------------
function lossfunc(index_input)
    # 1) Build NN input
    NN_input = pe(xlmap[Flux.cpu(index_input)]')'

    # 2) Forward pass
    vp, vc = re(NNparam)(NN_input)

    # 3) Keep parameters in a stable physical band (helps solver)
    vp = clamp.(vp, 0.5f0, 1.6f0)   # modulus P(x)
    vc = clamp.(vc, 10f0, 30f0)     # damping C(x); head already ~15..25

    # 4) Assemble parameter vector
    global p = vcat(vp, vc)

    # 5) Solve ODE robustly during training, in Float64 for stability
    prob64 = remake(prob_template; p = Float64.(p), u0 = Float64.(x0))
    sol = solve(
        prob64, solver;
        saveat   = tl,
        abstol   = abstol,
        reltol   = reltol,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()),
        save_everystep = false,
    )

    # 6) Extract prediction and cast back to Float32 to match data
    prediction = Float32.(Array(sol)[1, :, :])  # (Nx × Nt)

    # 7) Loss (same scale as original)
    raw_mae = mean(abs, (y .- prediction)[Flux.cpu(index_input)])
    loss    = 1e4 * raw_mae
    global loss_value = loss

    # Optional: log MAE without touching the AD tape
    Zygote.ignore() do
        @info "mae_unscaled" mae = raw_mae
    end

    return loss
end

# -----------------------------------------------------------------------------
# Visual helpers (CairoMakie, single-figure layout)
# -----------------------------------------------------------------------------
function visual_vp!(ax, v)
    scatter!(ax, xll, p0[1:Nx]; markersize=6, label="True")
    lines!(ax,   xll, v[1:Nx];   label="Predict", linewidth=3)
    ax.title  = "Modulus coefficient P"
    ax.xlabel = "x"
    ax.ylabel = "value"
end

function visual_vc!(ax, v)
    scatter!(ax, xll[2:end-1], p0[Nx+2:end-1]; markersize=6, label="True")
    lines!(ax,   xll[2:end-1], v[Nx+2:end-1];   label="Predict", linewidth=3)
    ax.title  = "Damping C"
    ax.xlabel = "x"
    ax.ylabel = "value"
end

function cb_vp_vc()
    fig = Figure(size = (720, 520))
    ax1 = Axis(fig[1,1])
    ax2 = Axis(fig[2,1])

    visual_vp!(ax1, p); axislegend(ax1; position=:lt)
    visual_vc!(ax2, p); axislegend(ax2; position=:rt)

    display(fig)

    println("loss: ", loss_value)
    @info "loss_scaled" loss = loss_value
    return false
end

# Composite results figure (optional; not used during training by default)
function results(v)
    # Predict displacement with current p = v
    prob = remake(prob_template; p = v)
    sol  = solve(prob, solver; saveat = tl, abstol = abstol, reltol = reltol)
    predict = Array(sol)[1, :, :]
    diff_u  = abs.(y .- predict)

    fig = Figure(size = (1000, 900))

    axP = Axis(fig[1,1]); visual_vp!(axP, v); axislegend(axP; position=:lt)
    axC = Axis(fig[1,2]); visual_vc!(axC, v); axislegend(axC; position=:rt)

    axT = Axis(fig[2,1]); heatmap!(axT, xll, tl, (y .* 1e3)';       colormap=:viridis); axT.title = "True"
    axY = Axis(fig[2,2]); heatmap!(axY, xll, tl, (predict .* 1e3)'; colormap=:plasma);  axY.title = "Predict"
    axE = Axis(fig[3,1]); heatmap!(axE, xll, tl, (diff_u .* 1e3)';  colormap=:magma);   axE.title = "Error ×1e3"

    display(fig)
    return fig
end
