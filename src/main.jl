# =========================
# main.jl  (Julia ≥ 1.12)
# NeuralSI-Julia12 Modernization
# =========================

using Random, LinearAlgebra, Statistics
using DelimitedFiles
using DifferentialEquations
using SciMLSensitivity
using ModelingToolkit
using Flux
using Optimisers
using Zygote

# Make Makie headless & non-interactive during training (prevents REPL/window issues)
try
    using CairoMakie
    CairoMakie.activate!(type = "png")   # render to PNG surfaces
    Makie.inline!(true)                  # don't try to pop up a window
catch
    # if Makie isn't loaded yet (no plotting path), this is harmless
end

# --- local deps ---
include("ODEfunctions.jl")         # beam_nonlinear!, solveODE, get_BandedMatrix, jetmap
include("Networks.jl")             # loader, PositionEmbedding, NN_model, lossfunc, cb_vp_vc, visual_*
include("FrechetDistance.jl")      # frdist
include("plotting.jl")            # plot_elemental, plot_perf_summary

# Create output folders if missing (assuming you run `julia --project=. src/main.jl` from repo root)
isdir("../data")         || mkpath("../data")
isdir("../results")      || mkpath("../results")
isdir("../results/figs") || mkpath("../results/figs")

# ------------------ Problem setup: space, time, materials ------------------
Random.seed!(1234)

# Space & Time
Nx   = 16
Lx   = 0.4f0
xl   = LinRange(0, Lx, Nx+2)
dx   = xl[2] - xl[1]
xll  = xl[2:end-1]                         # interior points (length Nx)

Nt    = 160
tmax  = 0.045f0
tl    = LinRange(0, tmax, Nt)
dt    = tl[2] - tl[1]
tspan = (0.0f0, tmax)

# Banded operators
A1, A2, A3, A4 = get_BandedMatrix(Nx, dx)

# Geometry & Modulus
thick = 0.005f0
width = 0.05f0
E     = 70f9
rho   = 2700f0
rhoA  = rho * thick * width
I     = width * thick^3 / 12
EI    = E * I

vp0 = Array{Float32}(.25f0 .* sin.(xll ./ Lx .* 2f0*pi) .+ 1.0f0)     # Nx
vc0 = Array{Float32}((0.8f0 .+ 0.3f0 .* xll ./ Lx) .* 20f0)           # Nx

# Force (used by ODE RHS)
forced_time     = 0.02f0
force_magnitude = 1000f0

# Initial condition, solver, tolerances
x0     = zeros(Float32, 2, Nx)                  # states: [disp; vel]
b      = ones(Float32, Nx, 1); b[[1, Nx]] .= 0f0
solver = Tsit5()
abstol = 1e-8; reltol = 1e-8

# ODE RHS alias (must exist before building problems)
func! = beam_nonlinear!

# Parameters and problem templates (build AFTER func!, x0, tspan)
p0 = vcat(vp0, vc0)                              # (2Nx)
tspan2 = (0.0f0, 2*tmax)

prob_template  = ODEProblem(func!, x0, tspan,  p0)
prob2_template = ODEProblem(func!, x0, tspan2, p0)

# ------------------ ground truth (training target) ------------------
@time y = solveODE(func!; p = p0, x0 = x0, solver = solver, tspan = tspan, tl = tl)  # (Nx × Nt)
jetmap(y; title = "True displacement (mm-scaled)")

# ------------------ network & training data ------------------
sample_ratio   = 0.5f0
epochs         = 20
minibatch_size = 16

# Feature map (same shapes as original code path)
xlmap = repeat(xl ./ Lx, 1, Nt)                   # (Nx+2 × Nt)
pe    = PositionEmbedding(1)                      # identity embedding (keeps dims)

# data loader yields index batches (lossfunc expects indices)
train_loader = loader(Nx * Nt, minibatch_size, sample_ratio)  # rng seeded inside

# Toggle training-time plotting:
const ENABLE_TRAIN_PLOTS = false

# Lightweight callback that *doesn't* open figures (safe in VS Code)
function cb_logonly()
    @info "loss_scaled" loss = loss_value
    return false
end
# (Optional) callback printing/plots (uses global `p` & `loss_value` from lossfunc)
cb = ENABLE_TRAIN_PLOTS ? cb_vp_vc : cb_logonly

# Network dims
NNdims1 = [minibatch_size, 32, 32, 16, Nx]
NNdims2 = NNdims1
model   = NN_model(NNdims1, NNdims2)

# Flatten parameters → (NNparam, re) for Flux ≥0.14
NNparam, re = Flux.destructure(model)

# Make variables accessible from Networks.jl lossfunc (which uses globals)
# (These assignments ensure visibility there.)
global Nx, xll, p0, xlmap, pe, func!, x0, solver, tspan, tl, y,
       A1, A2, A3, A4, EI, rhoA, b, forced_time, force_magnitude,
       abstol, reltol, prob_template, prob2_template
# `NNparam` and `re` are captured by lossfunc via closure; we keep them global too.
global NNparam, re
# Initialize p so downstream code has a value even before training finishes
global p = p0

# ---------------------- smoke test: loss & callback ----------------------
@time lossfunc(rand(1:Nx*Nt, minibatch_size))
@info "smoke_mae_scaled" loss = loss_value
# call cb rarely to avoid slow plotting during training
cb()

# ---------------------- training (modern Flux + Optimisers) ----------------------
try
    # ----- Phase 1 -----
    opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.AdamW(0.005f0))
    opt = Flux.setup(opt, NNparam)
    for epoch_idx in 1:10
        for idxbatch in train_loader
            gs = Flux.gradient(NNparam) do _
                lossfunc(idxbatch)
            end
            Flux.update!(opt, NNparam, gs[1])
        end
        (epoch_idx % 5 == 0) ? cb() : @info "loss_scaled" loss = loss_value
    end

    # ----- Phase 2 -----
    opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.AdamW(0.001f0))
    opt = Flux.setup(opt, NNparam)
    for epoch_idx in 1:(epochs - 10)
        for idxbatch in train_loader
            gs = Flux.gradient(NNparam) do _
                lossfunc(idxbatch)
            end
            Flux.update!(opt, NNparam, gs[1])
        end
        (epoch_idx % 5 == 0) ? cb() : @info "loss_scaled" loss = loss_value
    end
catch e
    # Print the true error to the terminal (VS Code often hides it)
    Base.showerror(stderr, e, catch_backtrace())
    flush(stderr)
    rethrow()
end
try
    # ----- Phase 1 -----
    opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.AdamW(0.005f0))
    opt = Flux.setup(opt, NNparam)
    for epoch_idx in 1:10
        for idxbatch in train_loader
            gs = Flux.gradient(NNparam) do _
                lossfunc(idxbatch)
            end
            Flux.update!(opt, NNparam, gs[1])
        end
        (epoch_idx % 5 == 0) ? cb() : @info "loss_scaled" loss = loss_value
    end

    # ----- Phase 2 -----
    opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.AdamW(0.001f0))
    opt = Flux.setup(opt, NNparam)
    for epoch_idx in 1:(epochs - 10)
        for idxbatch in train_loader
            gs = Flux.gradient(NNparam) do _
                lossfunc(idxbatch)
            end
            Flux.update!(opt, NNparam, gs[1])
        end
        (epoch_idx % 5 == 0) ? cb() : @info "loss_scaled" loss = loss_value
    end
catch e
    # Print the true error to the terminal (VS Code often hides it)
    Base.showerror(stderr, e, catch_backtrace())
    flush(stderr)
    rethrow()
end


println("\ntraining done!")

# ---------------------- check performance & MTK solve ----------------------
# Fréchet distance between learned and true parameters
a  = frdist(p[1:Nx],        p0[1:Nx],       xll)
b_ = frdist(p[Nx+2:end-1],  p0[Nx+2:end-1], xll[2:end-1])   # name `b_` to avoid clash with mask `b`
println("Fréchet(P): ", a, "   Fréchet(C): ", b_)

# Interpolation prediction via ModelingToolkit structural_simplify
tl2 = LinRange(0, 2*tmax, 2*Nt)

prob = ODEProblem(func!, x0, tspan, p)
sys  = modelingtoolkitize(prob)
sysS = structural_simplify(sys)
fastprob  = ODEProblem(sysS, x0, tspan,  p)
fastprob2 = ODEProblem(sysS, x0, (0.0f0, 2*tmax), p)

pred1_sol = solve(
    fastprob, ImplicitEulerExtrapolation();
    p=p, saveat=tl,
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()),
    abstol=1e-8, reltol=1e-8
)
pred1 = Array(pred1_sol)[1:2:Nx*2, :]   # take displacement rows

pred2_sol = solve(
    fastprob2, ImplicitEulerExtrapolation();
    p=p, saveat=tl2,
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()),
    abstol=1e-8, reltol=1e-8
)
pred2 = Array(pred2_sol)[1:2:Nx*2, :]

# Ground truth on extended horizon and errors
y2     = solveODE(func!; p = p0, x0 = x0, solver = solver, tspan = (0.0f0, 2*tmax), tl = tl2)
error1 = mean(abs, pred1 .- y)
error2 = mean(abs, pred2 .- y2)
println("Interpolation MAE: ", error1)
println("Extrapolation  MAE: ", error2)

# ---------------------- save outputs ----------------------
writedlm("../data/y.txt",               y)
writedlm("../data/y-extrapolate.txt",   y2)
writedlm("../data/NeuralSI-pred.txt",   pred1)
writedlm("../data/NeuralSI-pred2.txt",  pred2)

println("Saved outputs under ../data and figures under ../results/figs")
