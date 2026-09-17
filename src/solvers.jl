using Oceananigans.Solvers: ConjugateGradientSolver, solve!
using LinearAlgebra

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
    solve!(solution, solver.solver, rhs)
    fill_halo_regions!(solution)
    mask_immersed_field!(solution)
    return nothing
end

function apply_inverse_sqrt!(solution, rhs, solver::CGSolver)
    zero!(solution)
    T = eltype(solution)
    # Approximate A^(-1/2) = (2/π) ∫₀^{π/2} (A + tan²θ)⁻¹ sec²θ dθ via midpoint quadrature
    for m in 1:solver.n_sqrt_quadrature_points
        θ = (m - 0.5) * (π / 2) / solver.n_sqrt_quadrature_points
        shift = T(tan(θ)^2)
        weight = T((1 / solver.n_sqrt_quadrature_points) * sec(θ)^2)
        zero!(solver.solution_buffer)
        solve!(solver.solution_buffer, solver.solver, rhs, shift)
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

struct SparseSolver{S,C,V} <: AbstractDirectSolver
    sparse_operator::S
    cholesky_factor::C
    rhs_buffer::V
    solution_buffer::V
end

function SparseSolver(apply!, field_template; stencil_radius::Int=1, do_checks::Bool=true)
    result = similar(field_template)
    operand = similar(field_template)
    A, b = get_sparse_operator(result, operand, apply!; stencil_radius, verify=do_checks)
    # Operator should be symmetric with respect to Euclidean inner product
    # Use permutedims(A) rather than A' as norm(A - A') computes dense representation
    do_checks && @assert isapprox(A, permutedims(A), rtol=eps(eltype(A)))
    # Operator should be homogeneous and so the affine offset vector b should be all zero
    do_checks && @assert iszero(b)
    if (field_template.grid isa ImmersedBoundaryGrid)
        # For immersed grids sparse operator A will be singular due to apply! having no effect
        # on cell indices corresponding to inactive immersed cells, with corresponding rows /
        # columns with all zero entries. In this case we regularize the operator A by adding
        # an arbitrary value ε=1 to the corresponding diagonal entries. When solving a system
        # in A, because inactive rows/columns are exactly zero, the (non-regularized) A is
        # exactly block diagonal when permuted so that there are contiguous {active, inactive}
        # index sets; adding ε > 0 to inactive diagonal entries only affects the decoupled
        # inactive block and leaves the solution for the active block unchanged, therefore
        # the solution restricted to the active indices is unchanged. Providing the solution
        # is masked to zero inactive indices we will still therefore get a valid solution.
        A, inactive_indices = regularize_operator(A; ε=1.0)
        immersed_indices = get_immersed_indices(field_template)
        do_checks && @assert Set(inactive_indices) == Set(immersed_indices)
    end
    cholesky_factor = cholesky(Symmetric(A))
    rhs_buffer = similar(vec(interior(field_template)))
    solution_buffer = similar(rhs_buffer)
    return SparseSolver(A, cholesky_factor, rhs_buffer, solution_buffer)
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
    # We deliberately do no mask immersed values here as this breaks exchangebility /
    # permutation invariance of output
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
