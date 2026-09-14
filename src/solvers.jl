abstract type AbstractSolver end

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
struct CGSolver{S,F} <: AbstractSolver
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

function apply_half_order_inverse!(solution, rhs, solver::CGSolver)
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
