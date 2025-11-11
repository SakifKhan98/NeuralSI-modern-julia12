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
using BenchmarkTools

# --- local deps ---
include("ODEfunctions.jl")         # beam_nonlinear!, solveODE, get_BandedMatrix, jetmap
include("Networks.jl")             # loader, PositionEmbedding, NN_model, lossfunc, cb_vp_vc, visual_*
include("FrechetDistance.jl")      # frdist

# Create output folders if missing
isdir("../data")        || mkpath("../data")
isdir("../results")     || mkpath("../results")
isdir("../results/figs")|| mkpath("../results/figs")

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

# Force
forced_time     = 0.02f0
force_magnitude = 1000f0

# Initial condition & solver
p0     = vcat(vp0, vc0)                          # (2Nx)
solver = RK4()                                   # (Tsit5() also fine)
x0     = zeros(Float32, 2, Nx)                   # states: [disp; vel]
b      = ones(Float32, Nx, 1); b[[1, Nx]] .= 0f0 # boundary mask

# ------------------ ground truth (training target) ------------------
func! = beam_nonlinear!                           # from ODEfunctions.jl
@time y = solveODE(func!; p = p0, x0 = x0, solver = solver, tspan = tspan, tl = tl)  # (Nx × Nt)
jetmap(y; title = "True displacement (mm-scaled)")

# ------------------ network & training data ------------------
sample_ratio   = 0.5f0
epochs         = 20
minibatch_size = 16

# Feature map (same shapes as original code path)
xlmap = repeat(xl ./ Lx, 1, Nt)                   # (Nx+2 × Nt)
pe    = PositionEmbedding(1)                      # identity embedding (keeps dims)

# data loader yields index batches (your lossfunc expects indices)
train_loader = loader(Nx * Nt, minibatch_size, sample_ratio)  # rng seeded inside

# (Optional) callback printing/plots (uses global `p` & `loss_value` from lossfunc)
cb = cb_vp_vc

# Network dims
NNdims1 = [minibatch_size, 32, 32, 16, Nx]
NNdims2 = NNdims1
model   = NN_model(NNdims1, NNdims2)

# Flatten parameters → (NNparam, re) pairs with Flux ≥0.14
NNparam, re = Flux.destructure(model)

# Make variables accessible from Networks.jl lossfunc (which uses globals)
# (These are already defined above; the assignment here just ensures they’re in global scope.)
global Nx, xll, p0, xlmap, pe, func!, x0, solver, tspan, tl, y, A1, A2, A3, A4, EI, rhoA, b, forced_time, force_magnitude
# `NNparam` and `re` are captured by lossfunc via closure; we keep them in global too.
global NNparam, re

# ---------------------- smoke test: loss & callback ----------------------
@time lossfunc(rand(1:Nx*Nt, minibatch_size))
@time cb()

# ---------------------- training (modern Flux + Optimisers) ----------------------
# Phase 1: lr = 1e-2 for 10 epochs
opt = Flux.setup(Optimisers.AdamW(0.01f0), NNparam)

for epoch_idx in 1:10
    println("epoch $(epoch_idx):")
    @time begin
        for idxbatch in train_loader
            gs = Flux.gradient(NNparam) do _
                lossfunc(idxbatch)                 # must use NNparam via closure + re()
            end
            Flux.update!(opt, NNparam, gs[1])
        end
        cb()
    end
end

# Phase 2: lr = 1e-3 for remaining epochs
opt = Flux.setup(Optimisers.AdamW(0.001f0), NNparam)

for epoch_idx in 1:(epochs - 10)
    println("epoch $(epoch_idx + 10):")
    @time begin
        for idxbatch in train_loader
            gs = Flux.gradient(NNparam) do _
                lossfunc(idxbatch)
            end
            Flux.update!(opt, NNparam, gs[1])
        end
        cb()
    end
end

println("\ntraining done!")

# ---------------------- check performance & MTK solve ----------------------
# Fréchet distance between learned and true parameters
# (lossfunc sets global `p = vcat(vp, vc)` each step; use final `p` here)
a = frdist(p[1:Nx],             p0[1:Nx],             xll)
b_ = frdist(p[Nx+2:end-1],      p0[Nx+2:end-1],       xll[2:end-1])   # name `b_` to avoid clash with boundary mask `b`
println("Fréchet(P): ", a, "   Fréchet(C): ", b_)

# Interpolation prediction via MTK structural_simplify
tl2    = LinRange(0, 2*tmax, 2*Nt)
tspan2 = (0.0f0, 2*tmax)

prob = ODEProblem(func!, x0, tspan, p)
sys  = modelingtoolkitize(prob)
sysS = structural_simplify(sys)
fastprob = ODEProblem(sysS, x0, tspan, p)

@time pred1_sol = solve(
    fastprob, ImplicitEulerExtrapolation();
    p=p, saveat=tl,
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()),
    abstol=1e-8, reltol=1e-8
)
pred1 = Array(pred1_sol)[1:2:Nx*2, :]   # take displacement rows

@benchmark solve(
    fastprob, ImplicitEulerExtrapolation();
    p=p, saveat=tl,
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()),
    abstol=1e-8, reltol=1e-8
)

jetmap(pred1; title = "NeuralSI prediction (interp)")

# Extrapolation
prob2     = ODEProblem(func!, x0, tspan2, p)
fastprob2 = ODEProblem(sysS,  x0, tspan2, p)

@time pred2_sol = solve(
    fastprob2, ImplicitEulerExtrapolation();
    p=p, saveat=tl2,
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()),
    abstol=1e-8, reltol=1e-8
)
pred2 = Array(pred2_sol)[1:2:Nx*2, :]

@benchmark solve(
    fastprob2, ImplicitEulerExtrapolation();
    p=p, saveat=tl2,
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()),
    abstol=1e-8, reltol=1e-8
)

# Ground truth on extended horizon and errors
y2     = solveODE(func!; p = p0, x0 = x0, solver = solver, tspan = tspan2, tl = tl2)
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
