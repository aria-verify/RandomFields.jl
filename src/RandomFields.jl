module RandomFields

using Oceananigans
using Oceananigans:
    Center,
    Face,
    Flat,
    topology,
    architecture,
    RectilinearGrid,
    LatitudeLongitudeGrid,
    ImmersedBoundaryGrid
using Oceananigans.Fields: location, interior
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: ConjugateGradientSolver, solve!
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using SpecialFunctions: gamma

export generate!, IsotropicMatern, AnisotropicMatern, RandomFieldGenerator

include("parameters.jl")
include("grid_helpers.jl")
include("operators.jl")
include("kernels.jl")
include("workspace.jl")

function matern_spde_alpha(smoothness, dimension)
    two_alpha = round(Int, 2 * smoothness) + dimension
    two_alpha > 0 || throw(ArgumentError("smoothness + dimension/2 must be positive"))
    alpha = two_alpha / 2
    isinteger(alpha) || throw(ArgumentError("smoothness + dimension/2 must be integer"))
    return round(Int, alpha)
end

function generate_white_noise!(field, v, white_noise_scale)
    grid = field.grid
    run_kernel!(_discretize_white_noise_kernel!, grid, field, grid, white_noise_scale, v)
    fill_halo_regions!(field)
    return field
end

function apply_repeated_inverse!(solution_field, source_field, generator::RandomFieldGenerator)
    for _ in 1:generator.k
        zero_field!(solution_field)
        solve!(solution_field, generator.solver, source_field, generator, 1.0)
        fill_halo_regions!(solution_field)
        mask_immersed_values!(solution_field)
        source_field, solution_field = solution_field, source_field
    end
    # Due to name swap in final iteration, source_field corresponds to final solution
    return source_field, solution_field
end

# A^(-1/2) via A^(-1/2) = (2/π) ∫₀^{π/2} (A + tan²θ)⁻¹ sec²θ dθ, midpoint quadrature
function apply_half_order_inverse!(
    field, solution_field, rhs_field, generator::RandomFieldGenerator, quadrature_points
)
    zero_field!(field)
    for m in 1:quadrature_points
        θ = (m - 0.5) * (π / 2) / quadrature_points
        shift_coefficient = 1 + tan(θ)^2
        weight = (1 / quadrature_points) * sec(θ)^2
        zero_field!(solution_field)
        solve!(solution_field, generator.solver, rhs_field, generator, shift_coefficient)
        fill_halo_regions!(solution_field)
        accumulate_weighted!(field, solution_field, weight)
    end
    return field
end

"""
    generate!(field, generator, v; sqrt_quadrature_points = 32)

Overwrite `field` with an (approximate) draw from a mean-zero Gaussian random field with
Matérn covariance, via the SPDE representation `(1 - Σᵢλᵢ²∂ᵢ²)^(ν+d/2) field ∝ white noise`.

- `generator`: a `RandomFieldGenerator` built once via `RandomFieldGenerator(field, parameters)`
   and  reusable across calls, with parameters an instance of `IsotropicMatern` or
   `AnisotropicMatern`.
- `v`: standard normal variates, `size(v) == size(field)`.
- `sqrt_quadrature_points`: quadrature points for the half-order factor, used only when
  `smoothness + dimension/2` is not an integer.
"""
function generate!(field, generator::RandomFieldGenerator, v; sqrt_quadrature_points=32)
    field.grid === generator.grid ||
        throw(ArgumentError("generator was built for a different grid"))
    location(field) === generator.location ||
        throw(ArgumentError("generator was built for a different field location"))

    source_field, solution_field = generator.field_buffer_a, generator.field_buffer_b

    generate_white_noise!(source_field, v, generator.white_noise_scale)

    solution_field, source_field = apply_repeated_inverse!(
        solution_field, source_field, generator
    )

    if generator.half
        apply_half_order_inverse!(
            field, source_field, solution_field, generator, sqrt_quadrature_points
        )
    else
        copy_masked_result!(field, solution_field)
    end

    return nothing
end

end