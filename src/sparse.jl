using SparseArrays
using Oceananigans: Periodic

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
