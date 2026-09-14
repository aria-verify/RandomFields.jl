abstract type AbstractSolver end
abstract type AbstractIterativeSolver <: AbstractSolver end
abstract type AbstractDirectSolver <: AbstractSolver end

"""
    $(FUNCTIONNAME)(solution, rhs, solver)

Apply the inverse of the linear operator solved for by `solver` to the right-hand side `rhs`
and write the solution in-place to `solution`, where `rhs` and `solution` are both fields.

Solves `A * x = b` where `A` is the linear system solved by `solver`, `b` is the vectorization
of the `rhs` field and `x` is the vectorization of the `solution` field.
"""
function apply_inverse! end

"""
    $(FUNCTIONNAME)(solution, rhs, solver)

Apply the inverse of the square-root of linear operator solved for by `solver` to the right-hand
side `rhs` and write the solution in-place to `solution`, where `rhs` and `solution` are both fields.

Solves `sqrt(A) * x = b` where `A` is the linear system solved by `solver` and `sqrt(A)` is
a matrix such that `sqrt(A) * sqrt(A)' === A`, `b` is the vectorization of the `rhs` field
and `x` is the vectorization of the `solution` field.
"""
function apply_inverse_sqrt! end

"""
    $(FUNCTIONNAME)(solution, rhs, solver)

Apply the inverse of the adjoint of the linear operator solved for by `solver` to the right-hand
side `rhs` and write the solution in-place to `solution`, where `rhs` and `solution` are both fields.

Solves `A' * x = b` where `A'` is the adjoint of the linear system solved by `solver`,
`b` is the vectorization of the `rhs` field and `x` is the vectorization of the `solution` field.
"""
function apply_inverse_adjoint! end

"""
    $(FUNCTIONNAME)(solution, rhs, solver)

Apply the inverse of the adjoint of the square-root of linear operator solved for by `solver`
to the right-hand side `rhs` and write the solution in-place to `solution`, where `rhs` and
`solution` are both fields.

Solves `sqrt(A)' * x = b` where `A` is the linear system solved by `solver` and `sqrt(A)` is
a matrix such that `sqrt(A) * sqrt(A)' === A`, `b` is the vectorization of the `rhs` field
and `x` is the vectorization of the `solution` field.
"""
function apply_inverse_sqrt_adjoint! end

"""Fill `field` with zeros in-place."""
zero!(field::Field) = fill!(field, zero(eltype(field)))

function accumulate_weighted!(accumulator, addend, weight)
    run_kernel!(
        _accumulate_weighted_kernel!,
        accumulator.grid,
        accumulator,
        addend,
        weight;
        active_cells_map=get_active_cells_map(accumulator.grid, Val(:xyz)),
    )
    return nothing
end

"""Linear system solver wrapping Oceananigans iterative conjugate gradients solver."""
struct CGSolver{S,F} <: AbstractIterativeSolver
    "Oceananigans iterative conjugate gradient linear system solver"
    solver::S
    "Field buffer used to store intermediate solutions during computation"
    solution_buffer::F
    "Number of quadrature points to use in integral approximation to square root of linear operator"
    n_sqrt_quadrature_points::Int
end

"""
Construct a conjugate gradient solver with relative tolerance `reltol` and maximum number of
iterations `maxiter`. The half-order (square-root) linear operator inverse is approximated
using quadrature of an integral representation with `n_sqrt_quadrature_points`
quadrature points if relevant.
"""
function CGSolver(
    apply!,
    template_field;
    n_sqrt_quadrature_points=32,
    reltol=1e-7,
    maxiter=prod(size(template_field)),
)
    solver = ConjugateGradientSolver(apply!; template_field, reltol, maxiter)
    return CGSolver(solver, similar(template_field), n_sqrt_quadrature_points)
end

function apply_inverse!(solution, rhs, solver::CGSolver)
    zero!(solution)
    shift_coefficient = 1.0
    solve!(solution, solver.solver, rhs, shift_coefficient)
    fill_halo_regions!(solution)
    mask_immersed_field!(solution)
    return nothing
end

function apply_inverse_sqrt!(solution, rhs, solver::CGSolver)
    zero!(solution)
    # Approximate A^(-1/2) = (2/π) ∫₀^{π/2} (A + tan²θ)⁻¹ sec²θ dθ via midpoint quadrature
    for m in 1:solver.n_sqrt_quadrature_points
        θ = (m - 0.5) * (π / 2) / solver.n_sqrt_quadrature_points
        shift_coefficient = 1.0 + tan(θ)^2
        weight = (1 / solver.n_sqrt_quadrature_points) * sec(θ)^2
        zero!(solver.solution_buffer)
        solve!(solver.solution_buffer, solver.solver, rhs, shift_coefficient)
        accumulate_weighted!(solution, solver.solution_buffer, weight)
    end
    fill_halo_regions!(solution)
    mask_immersed_field!(solution)
    return nothing
end

# Linear system is assumed to be symmetric so inverse is self-adjoint
function apply_inverse_adjoint!(solution, rhs, solver::CGSolver)
    return apply_inverse!(solution, rhs, solver)
end

# Square root of linear system is assumed to be symmetric so inverse is self-adjoint
function apply_inverse_sqrt_adjoint!(solution, rhs, solver::CGSolver)
    return apply_inverse_sqrt!(solution, rhs, solver)
end

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

struct SparseSolver{C,V} <: AbstractDirectSolver
    cholesky_factor::C
    rhs_buffer::V
    solution_buffer::V
end

function SparseSolver(apply!, field_template; stencil_radius::Int=2)
    result = similar(field_template)
    operand = similar(field_template)
    shift_coefficient = 1.0
    A, b = get_sparse_operator(result, operand, apply!, shift_coefficient; stencil_radius)
    @assert iszero(b)
    cholesky_factor = cholesky(Symmetric(A))
    rhs_buffer = similar(vec(interior(field_template)))
    solution_buffer = similar(rhs_buffer)
    return SparseSolver(cholesky_factor, rhs_buffer, solution_buffer)
end

function apply_inverse!(solution, rhs, solver::SparseSolver)
    copyto!(solver.rhs_buffer, rhs)
    ldiv!(solver.solution_buffer, solver.cholesky_factor, solver.rhs_buffer)
    copyto!(solution, solver.solution_buffer)
    fill_halo_regions!(solution)
    mask_immersed_field!(solution)
    return nothing
end

function apply_inverse_sqrt!(solution, rhs, solver::SparseSolver)
    copyto!(solver.rhs_buffer, rhs)
    # \ operation allocates internally in CHOLMOD solve but no ldiv! method currently available
    copyto!(solver.solution_buffer, solver.cholesky_factor.PtL \ solver.rhs_buffer)
    copyto!(solution, solver.solution_buffer)
    fill_halo_regions!(solution)
    mask_immersed_field!(solution)
    return nothing
end

# Linear system is assumed to be symmetric so inverse is self-adjoint
function apply_inverse_adjoint!(solution, rhs, solver::SparseSolver)
    return apply_inverse!(solution, rhs, solver)
end

function apply_inverse_sqrt_adjoint!(solution, rhs, solver::SparseSolver)
    copyto!(solver.rhs_buffer, rhs)
    # \ operation allocates internally in CHOLMOD solve but no ldiv! method currently available
    copyto!(solver.solution_buffer, solver.cholesky_factor.UP \ solver.rhs_buffer)
    copyto!(solution, solver.solution_buffer)
    fill_halo_regions!(solution)
    mask_immersed_field!(solution)
    return nothing
end
