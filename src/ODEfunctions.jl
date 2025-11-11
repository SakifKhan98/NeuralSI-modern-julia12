##############################
# ODEfunctions.jl  (Julia ≥1.12)
##############################

# Core deps (no CUDA, no DiffEqFlux)
using DifferentialEquations
using ModelingToolkit
using LinearAlgebra
using Statistics
using Random
using DelimitedFiles
using BandedMatrices

# NOTE: We do NOT `using Plots` anymore.
# If you still want the old Plots-based heatmap, see the note at the bottom.

# ------------------------------------------------------------------
# External globals expected from main.jl:
#   Nx, EI, rhoA, b, forced_time, force_magnitude
#   x0, solver, tspan, tl
#   A1, A2, A3, A4  (via get_BandedMatrix)
# ------------------------------------------------------------------

@inline function myforce(t)::Float32
    # returns a scalar force scale; broadcasting with `b` happens in beam_nonlinear!
    # (-sign(t - forced_time) + 1) * force_magnitude / 2
    # keep Float32 to match rest of pipeline
    return ((-sign(t - forced_time) + 1.0f0) * (force_magnitude * 0.5f0))::Float32
end

function beam_nonlinear!(du, u, p, t)
    # u, du are size (2, Nx)
    # p = [pK; cdamp] with pK = stiffness (Nx), cdamp = damping (Nx)
    @inbounds begin
        du[1, :] = u[2, :]

        pK  = @view p[1:Nx]
        cdp = @view p[Nx+1:end]          # damping vector (Nx)

        u1  = @view u[1, :]
        u2  = @view u[2, :]

        # Elastic terms: (Nx) each
        term = (A2 * (pK .* (A2 * u1))) .+
               (A1 * (pK .* (2f0 .* (A3 * u1)))) .+
               (pK .* (A4 * u1))

        # Boundary mask as a vector (Nx)
        bvec = vec(b)

        # External force (scalar) broadcasted over mask (Nx)
        fext = (myforce(t) / rhoA) .* bvec

        # Damping: (cdp .* u2) / rhoA, then masked by b
        damp = ((cdp .* u2) ./ rhoA) .* bvec

        # Final acceleration row
        du[2, :] = .-(EI / rhoA) .* term .+ fext .- damp
    end
    return nothing
end


function solveODE(func!; p=p, x0=x0, solver=solver, tspan=tspan, tl=tl)
    # Wrapper used by main.jl to produce ground-truth trajectories.
    # Returns only the first state (displacement) like original `[1,:,:]`.
    prob = ODEProblem(func!, x0, tspan, p)
    sol  = solve(prob, solver; saveat = tl)
    A = Array(sol)                  # size: (2, Nx, length(tl))
    return @view A[1, :, :]         # displacement field (Nx × Nt)
end

function get_BandedMatrix(N::Integer, dx::Real)
    T  = Float32
    A4 = zeros(T, N, N)
    A3 = zeros(T, N, N)
    A2 = zeros(T, N, N)
    A1 = zeros(T, N, N)

    # A4:  [1, -4, 6, -4, 1] on bands with boundary tweaks
    @inbounds for i in 1:N
        if i-2 ≥ 1; A4[i, i-2] =  T(1);  end
        if i-1 ≥ 1; A4[i, i-1] = -T(4);  end
        A4[i, i] = T(6)
        if i+1 ≤ N; A4[i, i+1] = -T(4);  end
        if i+2 ≤ N; A4[i, i+2] =  T(1);  end
    end
    # boundary rows: convert values to T *before* assignment
    vA4 = convert.(T, [1.0, -4/5, 1/5])         # => Float32[1.0, -0.8, 0.2]
    A4[1, 1:3] .= vA4
    A4[end, end-2:end] .= reverse(vA4)

    # A3:  [-1/2, 1, 0, -1, 1/2] with zeroed boundary rows
    @inbounds for i in 1:N
        if i-2 ≥ 1; A3[i, i-2] = -T(1/2); end
        if i-1 ≥ 1; A3[i, i-1] =  T(1);   end
        if i+1 ≤ N; A3[i, i+1] = -T(1);   end
        if i+2 ≤ N; A3[i, i+2] =  T(1/2); end
    end
    A3[1, :]   .= 0
    A3[end, :] .= 0

    # A2:  [1, -2, 1] with zeroed boundary rows
    @inbounds for i in 1:N
        if i-1 ≥ 1; A2[i, i-1] =  T(1);   end
        A2[i, i] = -T(2)
        if i+1 ≤ N; A2[i, i+1] =  T(1);   end
    end
    A2[1, :]   .= 0
    A2[end, :] .= 0

    # A1:  [-1/2, 0, 1/2] with zeroed boundary rows
    @inbounds for i in 1:N
        if i-1 ≥ 1; A1[i, i-1] = -T(1/2); end
        if i+1 ≤ N; A1[i, i+1] =  T(1/2); end
    end
    A1[1, :]   .= 0
    A1[end, :] .= 0

    dxT = T(dx)
    return A1/dxT, A2/(dxT^2), A3/(dxT^3), A4/(dxT^4)
end


# ------------------------------------------------------------------
# Lightweight heatmap utility using CairoMakie (instead of Plots)
# Keeps your previous `jetmap(y)` call working without Plots.jl.
# ------------------------------------------------------------------
function jetmap(data; title::AbstractString = "", tl = tl)
    # data is (Nx × Nt); we plot Nt on x-axis
    @eval using CairoMakie
    fig = Figure(size = (800, 500))                    # was: resolution=(...)
    ax  = Axis(fig[1,1], xlabel = "Time index", ylabel = "Space index", title = title)
    heatmap!(ax, (Array(data) .* 1e3)'; colormap = :jet)
    display(fig)
    return fig, ax
end

# ------------------------------------------------------------------
# Optional: results() visualization kept for convenience.
# It depends on globals `func!`, `y`, etc., just like the original.
# ------------------------------------------------------------------
function results(v; position_p=:bottomleft, position_c=:bottomright)
    # These helpers (visual_vp / visual_vc) were referenced in the original code.
    # If you need them, keep their definitions in your repo; otherwise omit.
    # Here we only compute prediction and show three fields with jetmap.
    predict = solveODE(beam_nonlinear!; p = v, x0 = x0, solver = solver, tspan = tspan, tl = tl)
    diff_u  = abs.(y .- predict)
    u1 = jetmap(y;       title = "True")
    u2 = jetmap(predict; title = "Predict")
    u3 = jetmap(diff_u;  title = "Error")
    return (u1, u2, u3)
end
