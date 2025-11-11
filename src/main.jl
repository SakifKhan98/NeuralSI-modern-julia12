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
using .Plotting


# ---- stable project paths ----
const PROJ_ROOT   = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR    = joinpath(PROJ_ROOT, "data")
const RESULTS_DIR = joinpath(PROJ_ROOT, "results")
const FIGS_DIR    = joinpath(RESULTS_DIR, "figs")
mkpath(DATA_DIR); mkpath(RESULTS_DIR); mkpath(FIGS_DIR)

# Create output folders if missing (assuming you run `julia --project=. src/main.jl` from repo root)
isdir("../data")         || mkpath("../data")
isdir("../results")      || mkpath("../results")
isdir("../results/figs") || mkpath("../results/figs")

const QUICK_TEST = true

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
solver = TRBDF2()
abstol = 1e-8
reltol = 1e-8

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

if QUICK_TEST
    # Fewer epochs & fewer batches per epoch
    epochs          = 2
    sample_ratio    = 0.10f0      # only use 10% of samples per epoch
    minibatch_size  = 64          # larger batches to reduce steps

    # # Keep training stable but fast
    # global USE_SHORT_WINDOW = true   # (if you added the short-window feature)
    # const ENABLE_TRAIN_PLOTS = false # don’t open plots during training
end

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


const MAX_BATCHES = QUICK_TEST ? 10 : typemax(Int)
# ---------------------- training (modern Flux + Optimisers) ----------------------
try
    # How many batches per epoch to run (throttle when QUICK_TEST=true)
    # ----- Phase 1 -----
    opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.AdamW(0.005f0))
    opt = Flux.setup(opt, NNparam)

    for epoch_idx in 1:min(10, epochs)
        bcount = 0
        for idxbatch in train_loader
            bcount += 1
            loss_val = lossfunc(idxbatch)          # if your lossfunc returns (loss, mae), use its 'loss' part here
            gs = Flux.gradient(NNparam) do _
                loss_val
            end
            Flux.update!(opt, NNparam, gs[1])
            (bcount >= MAX_BATCHES) && break
        end
        (epoch_idx % 5 == 0) ? cb() : @info "training_progress" epoch = epoch_idx loss_scaled = loss_value
    end

    # ----- Phase 2 -----
    if epochs > 10
        opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0f0), Optimisers.AdamW(0.001f0))
        opt = Flux.setup(opt, NNparam)

        for epoch_idx in 11:epochs
            bcount = 0
            for idxbatch in train_loader
                bcount += 1
                loss_val = lossfunc(idxbatch)
                gs = Flux.gradient(NNparam) do _
                    loss_val
                end
                Flux.update!(opt, NNparam, gs[1])
                (bcount >= MAX_BATCHES) && break
            end
            (epoch_idx % 5 == 0) ? cb() : @info "training_progress" epoch = epoch_idx loss_scaled = loss_value
        end
    end
catch e
    # Print the true error to the terminal (VS Code often hides it)
    Base.showerror(stderr, e, catch_backtrace())
    flush(stderr)
    rethrow()
end

writedlm("../results/p_final.txt", p)
writedlm("../results/NNparam_final.txt", NNparam)
println("\ntraining done!")

if QUICK_TEST
    # ---------------------- quick numeric evaluation (no MTK) ----------------------
    # Use Float64 for robust ODE solves; cast back to Float32 for error calc/saving
    x0_eval = Float64.(x0)
    p_eval  = Float64.(p)

    # Interpolation horizon
    prob_eval   = ODEProblem(func!, x0_eval, (0.0, Float64(tmax)), p_eval)
    pred1_full  = Array(solve(prob_eval, TRBDF2(); saveat = Float64.(tl), abstol = 1e-8, reltol = 1e-8))
    pred1       = Float32.(pred1_full[1, :, :])   # (Nx × Nt) displacement

    # Extrapolation horizon (2×)
    tl2_64      = LinRange(0.0, 2*Float64(tmax), 2*Nt)
    prob_eval2  = ODEProblem(func!, x0_eval, (0.0, 2*Float64(tmax)), p_eval)
    pred2_full  = Array(solve(prob_eval2, TRBDF2(); saveat = tl2_64, abstol = 1e-8, reltol = 1e-8))
    pred2       = Float32.(pred2_full[1, :, :])   # (Nx × 2Nt)

    # Ground-truth on the 2× horizon (with p0)
    y2_full     = Array(solve(ODEProblem(func!, x0_eval, (0.0, 2*Float64(tmax)), Float64.(p0)),
                              TRBDF2(); saveat = tl2_64, abstol = 1e-8, reltol = 1e-8))
    y2          = Float32.(y2_full[1, :, :])

    # Errors (note: y is your Nx×Nt Float32 training target)
    error1 = mean(abs, pred1 .- y)
    error2 = mean(abs, pred2 .- y2)
    println("Interpolation MAE: ", error1)
    println("Extrapolation  MAE: ", error2)

    writedlm(joinpath(RESULTS_DIR, "p_final.txt"), p)
    writedlm(joinpath(RESULTS_DIR, "NNparam_final.txt"), NNparam)

    writedlm(joinpath(DATA_DIR, "y.txt"),              y)
    writedlm(joinpath(DATA_DIR, "y-extrapolate.txt"),  y2)
    writedlm(joinpath(DATA_DIR, "NeuralSI-pred.txt"),  pred1)
    @info "saved" file = joinpath(DATA_DIR, "NeuralSI-pred.txt") exists = isfile(joinpath(DATA_DIR, "NeuralSI-pred1.txt"))
    writedlm(joinpath(DATA_DIR, "NeuralSI-pred2.txt"), pred2)
    @info "saved" file = joinpath(DATA_DIR, "NeuralSI-pred2.txt") exists = isfile(joinpath(DATA_DIR, "NeuralSI-pred2.txt"))
else
    # ---------- EVALUATION WITHOUT MTK ----------
    # Save trained params *before* any evaluation, so you don't lose them if something crashes.

    # Use Float64 just for solving to avoid Float32 dt<eps issues
    x0_eval = Float64.(x0)
    p_eval  = Float64.(p)
    tl64    = Float64.(tl)
    tl2     = LinRange(0.0, 2*Float64(tmax), 2*Nt)

    # Interpolation
    prob_eval  = ODEProblem(func!, x0_eval, (0.0, Float64(tmax)), p_eval)
    pred1_full = Array(solve(prob_eval, TRBDF2(); saveat=tl64, abstol=1e-8, reltol=1e-8))
    pred1      = pred1_full[1:2:2Nx, :]             # displacement rows

    # Extrapolation
    prob_eval2  = ODEProblem(func!, x0_eval, (0.0, 2*Float64(tmax)), p_eval)
    pred2_full  = Array(solve(prob_eval2, TRBDF2(); saveat=tl2, abstol=1e-8, reltol=1e-8))
    pred2       = pred2_full[1:2:2Nx, :]

    # Ground truth at double horizon (from p0)
    y2_full = Array(solve(ODEProblem(func!, x0_eval, (0.0, 2*Float64(tmax)), Float64.(p0)),
                          TRBDF2(); saveat=tl2, abstol=1e-8, reltol=1e-8))
    y2      = y2_full[1:2:2Nx, :]

    error1 = mean(abs, pred1 .- Float32.(y))   # y is your original Float32 training target
    error2 = mean(abs, pred2 .- Float32.(y2))
    println("Interpolation MAE: ", error1)
    println("Extrapolation  MAE: ", error2)
    
    writedlm(joinpath(RESULTS_DIR, "p_final.txt"), p)
    writedlm(joinpath(RESULTS_DIR, "NNparam_final.txt"), NNparam)

    writedlm(joinpath(DATA_DIR, "y.txt"),              y)
    writedlm(joinpath(DATA_DIR, "y-extrapolate.txt"),  y2)
    writedlm(joinpath(DATA_DIR, "NeuralSI-pred.txt"),  pred1)
    @info "saved" file = joinpath(DATA_DIR, "NeuralSI-pred.txt") exists = isfile(joinpath(DATA_DIR, "NeuralSI-pred.txt"))
    writedlm(joinpath(DATA_DIR, "NeuralSI-pred2.txt"), pred2)
    @info "saved" file = joinpath(DATA_DIR, "NeuralSI-pred2.txt") exists = isfile(joinpath(DATA_DIR, "NeuralSI-pred2.txt"))
end

# ---- Make & save figures ----
Plotting.generate_all_figures(
    xll = xll,
    tl  = tl,
    tl2 = LinRange(0.0, 2*tmax, 2*Nt),  # same tl2 you used in evaluation
    y   = y,
    y2  = y2,
    pred1 = pred1,
    pred2 = pred2,
    vp0 = vp0,
    vc0 = vc0,
    p   = p,     # final learned (vcat(vp, vc))
    p0  = p0,    # ground truth (vcat(vp0, vc0))
    Nx  = Nx,
    myforce_fn = myforce,                 # from ODEfunctions.jl
    outdir = joinpath(RESULTS_DIR, "figs")  # or just "results/figs"
)
println("Saved figures → ", joinpath(RESULTS_DIR, "figs"))


println("Saved outputs under ../data and figures under ../results/figs")