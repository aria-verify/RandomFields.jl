using SparseArrays
using Oceananigans: Periodic

"""
Determine the sparse matrix representation `(A, b)` of the linear (affine) operator implicitly
defined  by `apply!`.

The sparse matrix `A` and affine offset vector `b` are computed such that
`result_vec .= A * vec(interior(operand)) + b` corresponds to  `apply!(result, operand, args...)`
with `result_vec == vec(interior(result))`. `A` is restricted to acting on interior grid points
and with ordering corresponding to the contiguous layout of `result` and `operand` fields `data`
arrays. For a purely linear operator (homogeneous boundary conditions), `b` is all zeros.

`apply!` must be linear and spatially local with support within `stencil_radius` (in L∞ /
Chebyshev distance). If `verify=true` a second probe is performed with `stencil_radius + 1`
and if the resulting computed `A` differs an error is raised.

Requires `(2*stencil_radius+1)^3` calls to `apply!`, independent of grid size, or
roughly double this if `verify=true`.
"""
function get_sparse_operator(
    result, operand, apply!, args...; stencil_radius::Int=2, verify::Bool=true, atol::Real=0
)
    Nx, Ny, Nz = size(operand)
    @assert size(result) == (Nx, Ny, Nz) "result and operand must share size/location"

    grid = operand.grid
    TX, TY, TZ = topology(grid)
    periodic = (TX <: Periodic, TY <: Periodic, TZ <: Periodic)

    R = stencil_radius
    A, b = probe_operator(result, operand, apply!, args, Nx, Ny, Nz, R, atol, periodic)

    if verify
        A_wide, _ = probe_operator(
            result, operand, apply!, args, Nx, Ny, Nz, R + 1, atol, periodic
        )
        if (A.rowval, A.colptr) != (A_wide.rowval, A_wide.colptr) ||
            !isapprox(nonzeros(A), nonzeros(A_wide); atol=max(atol, 1e-12))
            error(
                "get_sparse_operator: result changed when stencil_radius was " *
                "increased from $R to $(R+1) — increase it and retry.",
            )
        end
    end

    return A, b
end

"""Smallest `P >= 2R+1` dividing `N`, for a periodic axis and `2R+1` otherwise."""
function coloring_period(N::Int, R::Int, is_periodic::Bool)
    P0 = 2R + 1
    !is_periodic && return P0
    P0 > N && error(
        "stencil_radius=$R too large for periodic axis of length $N " *
        "(operator support would wrap onto itself).",
    )
    P = P0
    while N % P != 0
        P += 1
    end
    return P
end

function probe_operator(result, operand, apply!, args, Nx, Ny, Nz, R, atol, periodic)
    Px = coloring_period(Nx, R, periodic[1])
    Py = coloring_period(Ny, R, periodic[2])
    Pz = coloring_period(Nz, R, periodic[3])
    n = Nx * Ny * Nz
    lin = LinearIndices((Nx, Ny, Nz))

    op_int = interior(operand)
    res_int = interior(result)

    zero!(operand)
    fill_halo_regions!(operand)
    zero!(result)
    apply!(result, operand, args...)
    baseline = copy(res_int)
    b = vec(Array(baseline))

    T = eltype(result)

    rows = Int[]
    cols = Int[]
    vals = T[]

    for ck in 0:(Pz - 1), cj in 0:(Py - 1), ci in 0:(Px - 1)
        ri = (ci + 1):Px:Nx
        rj = (cj + 1):Py:Ny
        rk = (ck + 1):Pz:Nz
        (isempty(ri) || isempty(rj) || isempty(rk)) && continue

        zero!(operand)
        @views op_int[ri, rj, rk] .= 1
        fill_halo_regions!(operand)

        zero!(result)
        apply!(result, operand, args...)

        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            v = res_int[i, j, k] - baseline[i, j, k]
            abs(v) <= atol && continue

            xi = nearest_color_index(i, ci, Px, R, Nx, periodic[1])
            xj = nearest_color_index(j, cj, Py, R, Ny, periodic[2])
            xk = nearest_color_index(k, ck, Pz, R, Nz, periodic[3])
            (xi === nothing || xj === nothing || xk === nothing) && continue

            push!(rows, lin[i, j, k])
            push!(cols, lin[xi, xj, xk])
            push!(vals, T(v))
        end
    end

    A = sparse(rows, cols, vals, n, n, +)
    return A, b
end

@inline function nearest_color_index(
    idx::Int, c::Int, P::Int, R::Int, N::Int, is_periodic::Bool
)
    delta = mod(idx - 1 - c, P)
    raw = delta <= R ? idx - delta : idx + (P - delta)
    if is_periodic
        return mod1(raw, N)
    else
        return (1 <= raw <= N) ? raw : nothing
    end
end

"""
Regularize a symmetric stencil operator `A` for which some rows / columns are all zero
due to, for example, presence of inactive immersed cells in the corresponding fields,
by adding an arbitrary value `ε` to the corresponding diagonal entries.
"""
function regularize_operator!(A::SparseMatrixCSC; ε::Real=1.0)
    sum_abs_rows = vec(sum(abs, A; dims=2))
    inactive = findall(iszero, sum_abs_rows)
    for i in inactive
        A[i, i] = ε
    end
    return inactive
end
