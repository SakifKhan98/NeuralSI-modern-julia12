module Plotting

using CairoMakie

# ---------- styling helpers ----------
const _LINEW = 3.0
const _FS    = (900, 600)      # default figure size
const _FS_SQ = (900, 900)

# palette helpers
_col(name) = name
_cmap(name) = name

# make sure output directory exists
_mkdir(dir) = isdir(dir) || mkpath(dir)

# ---------- Fig. 3: Setup (P0, C0, F(t)) ----------
"""
plot_setup_fig3(xll, vp0, vc0, tl, myforce_fn; outdir="results/figs", fname="Fig3_setup")

Generates side-by-side plots of P₀(x), C₀(x), and F(t).
"""

# convert center grid -> bin edges for Makie heatmap
_center_to_edges(v) = range(first(v) - step(v)/2, last(v) + step(v)/2, length = length(v) + 1)


function plot_setup_fig3(xll, vp0, vc0, tl, myforce_fn; outdir="results/figs", fname="Fig3_setup")
    _mkdir(outdir)
    fig = Figure(size=(1000, 320))

    ax1 = Axis(fig[1,1], title="Ground Truth Modulus P₀(x)", xlabel="x (m)", ylabel="P")
    lines!(ax1, xll, vp0; color=_col(:teal), linewidth=_LINEW)

    ax2 = Axis(fig[1,2], title="Ground Truth Damping C₀(x)", xlabel="x (m)", ylabel="C")
    lines!(ax2, xll, vc0; color=_col(:orange), linewidth=_LINEW)

    ax3 = Axis(fig[1,3], title="Applied Force F(t)", xlabel="t (s)", ylabel="Force (N)")
    ft = myforce_fn.(tl)
    lines!(ax3, tl, ft; color=_col(:magenta), linewidth=_LINEW)

    path_png = joinpath(outdir, "$fname.png")
    path_pdf = joinpath(outdir, "$fname.pdf")
    save(path_png, fig); save(path_pdf, fig)
    display(fig)
    return fig
end

# ---------- Fig. 5: Predicted vs True Parameters ----------
"""
plot_fig5_params(xll, p_pred, p_true, Nx; outdir="results/figs", fname="Fig5_P_C_comparison")

Left: P(x) true vs pred. Right: C(x) true vs pred.
"""
function plot_fig5_params(xll, p_pred, p_true, Nx; outdir="results/figs", fname="Fig5_P_C_comparison")
    _mkdir(outdir)
    fig = Figure(size=_FS)

    ax1 = Axis(fig[1,1], title="Predicted vs. True Modulus P(x)", xlabel="x (m)", ylabel="P")
    lines!(ax1, xll, p_true[1:Nx]; color=_col(:black), linewidth=2, label="True")
    lines!(ax1, xll, p_pred[1:Nx]; color=_col(:dodgerblue), linewidth=_LINEW, linestyle=:dash, label="Pred")
    axislegend(ax1)

    ax2 = Axis(fig[1,2], title="Predicted vs. True Damping C(x)", xlabel="x (m)", ylabel="C")
    lines!(ax2, xll, p_true[Nx+1:end]; color=_col(:black), linewidth=2, label="True")
    lines!(ax2, xll, p_pred[Nx+1:end]; color=_col(:crimson), linewidth=_LINEW, linestyle=:dash, label="Pred")
    axislegend(ax2)

    path_png = joinpath(outdir, "$fname.png")
    path_pdf = joinpath(outdir, "$fname.pdf")
    save(path_png, fig); save(path_pdf, fig)
    display(fig)
    return fig
end

# ---------- Fig. 6: Heatmaps (interp/extrap) ----------
"""
response_heatmaps(xll, t, y, yhat; title_prefix, fname, outdir="results/figs")

Three stacked heatmaps: Truth, Prediction, Error (mm scaling shown).
"""
function response_heatmaps(xll, t, y, yhat; title_prefix="Interpolation", fname="Fig6_interp", outdir="results/figs")
    _mkdir(outdir)

    # Ensure plain matrices (avoid SubArray world-age quirks)
    Y    = Matrix(y)      # size: (Nx, Nt)
    Yhat = Matrix(yhat)   # size: (Nx, Nt)
    Err  = abs.(Y .- Yhat)

    # Build bin edges for Makie (length = size + 1)
    xe = _center_to_edges(xll)   # length Nx+1
    te = _center_to_edges(t)     # length Nt+1

    fig = Figure(size=_FS_SQ)

    ax1 = Axis(fig[1,1], title="$title_prefix – Ground Truth", xlabel="x (m)", ylabel="t (s)")
    heatmap!(ax1, xe, te, Y .* 1e3; colormap=_cmap(:viridis))

    ax2 = Axis(fig[2,1], title="$title_prefix – Prediction", xlabel="x (m)", ylabel="t (s)")
    heatmap!(ax2, xe, te, Yhat .* 1e3; colormap=_cmap(:plasma))

    ax3 = Axis(fig[3,1], title="$title_prefix – Error ×1e3", xlabel="x (m)", ylabel="t (s)")
    heatmap!(ax3, xe, te, Err .* 1e3; colormap=_cmap(:magma))

    path_png = joinpath(outdir, "$fname.png")
    path_pdf = joinpath(outdir, "$fname.pdf")
    save(path_png, fig); save(path_pdf, fig)
    display(fig)
    return fig
end

# ---------- Fig. 7: Elemental responses (time traces at positions) ----------
"""
plot_elemental(xidxs, t, y, yhat; outdir="results/figs", fname="Fig7_elemental")

Plots truth vs prediction at indices in xidxs (e.g., midspan and quarter-span).
Displacement is shown in mm.
"""
function plot_elemental(xidxs::AbstractVector{<:Integer}, t, y, yhat; outdir="results/figs", fname="Fig7_elemental")
    _mkdir(outdir)
    fig = Figure(size=_FS)

    ax1 = Axis(fig[1,1], title="Elemental Response — index $(xidxs[1])", xlabel="t (s)", ylabel="u (mm)")
    lines!(ax1, t, y[xidxs[1],:] .* 1e3; color=_col(:black), label="True")
    lines!(ax1, t, yhat[xidxs[1],:] .* 1e3; color=_col(:green), linestyle=:dash, label="Pred")
    axislegend(ax1)

    if length(xidxs) ≥ 2
        ax2 = Axis(fig[1,2], title="Elemental Response — index $(xidxs[2])", xlabel="t (s)", ylabel="u (mm)")
        lines!(ax2, t, y[xidxs[2],:] .* 1e3; color=_col(:black), label="True")
        lines!(ax2, t, yhat[xidxs[2],:] .* 1e3; color=_col(:green), linestyle=:dash, label="Pred")
        axislegend(ax2)
    end

    path_png = joinpath(outdir, "$fname.png")
    path_pdf = joinpath(outdir, "$fname.pdf")
    save(path_png, fig); save(path_pdf, fig)
    display(fig)
    return fig
end

# ---------- Optional: performance summary ----------
"""
plot_perf_summary(error_interp, error_extrap; outdir="results/figs", fname="Fig10_error_summary")

Simple bar chart for MAE summary (log10 scale).
"""
function plot_perf_summary(error_interp, error_extrap; outdir="results/figs", fname="Fig10_error_summary")
    _mkdir(outdir)
    fig = Figure(size=(700, 400))
    ax = Axis(fig[1,1], title="Reproduced NeuralSI Performance", ylabel="Error (log₁₀ scale)")
    xs = [1, 2]
    ys = log10.([error_interp, error_extrap])
    barplot!(ax, xs, ys; color=(:dodgerblue, :orange))
    ax.xticks = (xs, ["Interpolation MAE","Extrapolation MAE"])
    path_png = joinpath(outdir, "$fname.png")
    path_pdf = joinpath(outdir, "$fname.pdf")
    save(path_png, fig); save(path_pdf, fig)
    display(fig)
    return fig
end

# ---------- One-call convenience wrapper ----------
"""
generate_all_figures(; xll, tl, tl2, y, y2, pred1, pred2, vp0, vc0, p, p0, Nx, myforce_fn, outdir="results/figs")

Produces all recommended figures and saves PNG+PDF.
"""
function generate_all_figures(; xll, tl, tl2, y, y2, pred1, pred2, vp0, vc0, p, p0, Nx, myforce_fn, outdir="results/figs")
    _mkdir(outdir)

    # Fig 3
    plot_setup_fig3(xll, vp0, vc0, tl, myforce_fn; outdir=outdir, fname="Fig3_setup")

    # Fig 5 (params)
    plot_fig5_params(xll, p, p0, Nx; outdir=outdir, fname="Fig5_P_C_comparison")

    # Fig 6a (interp)
    response_heatmaps(xll, tl,  y,  pred1; title_prefix="Interpolation", fname="Fig6_interp", outdir=outdir)

    # Fig 6b (extrap)
    response_heatmaps(xll, tl2, y2, pred2; title_prefix="Extrapolation", fname="Fig6_extrap", outdir=outdir)

    # Fig 7 (time traces at mid and quarter)
    mid, quarter = Nx ÷ 2, Nx ÷ 4
    plot_elemental([mid, quarter], tl, y, pred1; outdir=outdir, fname="Fig7_elemental")

    return nothing
end

end # module
