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
using Oceananigans.ImmersedBoundaries: immersed_cell, mask_immersed_field!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: ConjugateGradientSolver, solve!
using Oceananigans.Utils: launch!, get_active_cells_map
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using SpecialFunctions: gamma

export generate!, IsotropicMatern, AnisotropicMatern, RandomFieldGenerator

include("parameters.jl")
include("grid_helpers.jl")
include("operators.jl")
include("kernels.jl")
include("generator.jl")

"""
    zero!(field)

Fill `field` with zeros in-place.
"""
zero!(field::Field) = fill!(field, zero(eltype(field)))

function generate_white_noise!(field, noise, white_noise_scale)
    grid = field.grid
    active_cells_map = get_active_cells_map(grid, Val(:xyz))
    run_kernel!(
        _discretize_white_noise_kernel!,
        grid,
        field,
        noise,
        white_noise_scale;
        active_cells_map,
    )
    fill_halo_regions!(field)
    isnothing(active_cells_map) && mask_immersed_field!(field)
    return field
end

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

function apply_repeated_inverse!(
    solution_field, source_field, generator::RandomFieldGenerator
)
    for _ in 1:generator.n_inverse_apply
        zero!(solution_field)
        solve!(solution_field, generator.solver, source_field, generator, 1.0)
        fill_halo_regions!(solution_field)
        mask_immersed_field!(solution_field)
        source_field, solution_field = solution_field, source_field
    end
    # Due to name swap in final iteration, source_field corresponds to final solution
    return source_field, solution_field
end

function apply_half_order_inverse!(
    field, solution_field, rhs_field, generator::RandomFieldGenerator
)
    zero!(field)
    # Approximate A^(-1/2) = (2/π) ∫₀^{π/2} (A + tan²θ)⁻¹ sec²θ dθ via midpoint quadrature
    for m in 1:generator.n_sqrt_quadrature_points
        θ = (m - 0.5) * (π / 2) / generator.n_sqrt_quadrature_points
        shift_coefficient = 1 + tan(θ)^2
        weight = (1 / generator.n_sqrt_quadrature_points) * sec(θ)^2
        zero!(solution_field)
        solve!(solution_field, generator.solver, rhs_field, generator, shift_coefficient)
        accumulate_weighted!(field, solution_field, weight)
    end
    fill_halo_regions!(field)
    mask_immersed_field!(solution_field)
    return field
end

"""
    generate!(field, generator, noise)

Overwrite `field` with an (approximate) draw from a mean-zero Gaussian random field with
Matérn covariance, via the SPDE representation `(1 - Σᵢλᵢ²∂ᵢ²)^(ν+d/2) f = τ w` where
`λᵢ` are per-dimension length scale parameters, `ν` a smoothness index, `d` the spatial
dimension, `f` the field being solved for, `τ` an output scaling parameter and `w` a
spatial white noise process. `generator` should be a `RandomFieldGenerator` built once
via `RandomFieldGenerator(field, parameters)` and reusable across calls, with parameters
an instance of `IsotropicMatern` or `AnisotropicMatern` and `noise` is an array of
standard normal variates with `size(noise) == size(field)`.
"""
function generate!(field, generator::RandomFieldGenerator, noise)
    field.grid === generator.grid ||
        throw(ArgumentError("generator was built for a different grid"))
    location(field) === generator.location ||
        throw(ArgumentError("generator was built for a different field location"))

    source_field, solution_field = generator.field_buffer_a, generator.field_buffer_b

    generate_white_noise!(source_field, noise, generator.white_noise_scale)

    solution_field, source_field = apply_repeated_inverse!(
        solution_field, source_field, generator
    )

    if generator.require_half_order
        apply_half_order_inverse!(field, source_field, solution_field, generator)
    else
        copyto!(field, solution_field)
    end

    return nothing
end

end