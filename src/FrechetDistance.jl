# FrechetDistance.jl  (Julia ≥1.12)
# Discrete Fréchet distance between two 2D polylines defined on the same x-grid.
# Usage in your code:
#   a = frdist(p[1:Nx],        p0[1:Nx],        xll)
#   b = frdist(p[Nx+2:end-1],  p0[Nx+2:end-1],  xll[2:end-1])

using LinearAlgebra

"""
    frdist(p::AbstractVector, q::AbstractVector, xlist::AbstractVector) -> Real

Compute the discrete Fréchet distance between curves
P = [(x[i], p[i])] and Q = [(x[i], q[i])], using a standard DP algorithm.
All inputs must have the same (non-zero) length.
"""
function frdist(p::AbstractVector, q::AbstractVector, xlist::AbstractVector)
    n = length(p)
    n == length(q)      || throw(ArgumentError("p and q must have the same length"))
    n == length(xlist)  || throw(ArgumentError("xlist length must match p and q"))
    n == 0              && error("Input curves are empty.")

    # Build 2D point sets (rows are points): (n × 2)
    T = promote_type(eltype(p), eltype(q), eltype(xlist))
    x = collect(T.(xlist))
    pp = collect(T.(p))
    qq = collect(T.(q))

    P = hcat(x, pp)
    Q = hcat(x, qq)

    return _dfd(P, Q)
end

# Iterative DP (no recursion). ca[i,j] stores the Fréchet value up to (i,j).
function _dfd(P::AbstractMatrix, Q::AbstractMatrix)
    n = size(P, 1)
    m = size(Q, 1)
    if n == 0 || m == 0
        error("Input curves are empty.")
    end

    T = promote_type(eltype(P), eltype(Q))
    ca = fill(T(-1), n, m)

    # Safe views with @views (no @view inside broadcast)
    @views begin
        dist(i, j) = norm(P[i, :] .- Q[j, :])

        ca[1, 1] = dist(1, 1)
        for i in 2:n
            ca[i, 1] = max(ca[i-1, 1], dist(i, 1))
        end
        for j in 2:m
            ca[1, j] = max(ca[1, j-1], dist(1, j))
        end
        for i in 2:n
            for j in 2:m
                ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]), dist(i, j))
            end
        end
    end

    return ca[n, m]
end
