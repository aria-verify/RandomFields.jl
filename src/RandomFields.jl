"""
Generate Gaussian Markov random fields with Matérn covariance functions on Oceananigans grids.

$(EXPORTS)
"""
module RandomFields

using Oceananigans
using Oceananigans:
    Center,
    Face,
    Flat,
    Periodic,
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
using LinearAlgebra
using SparseArrays
using DocStringExtensions

export generate!
export IsotropicMatern, AnisotropicMatern, RandomFieldGenerator, CGSolver, SparseSolver

@template FUNCTIONS = """
                      $(DOCSTRING)

                      $(TYPEDSIGNATURES)
                      """

@template METHODS = """
                    $(SIGNATURES)

                    $(DOCSTRING)
                    """

@template TYPES = """
                  $(TYPEDEF)

                  $(DOCSTRING)

                  ## Fields

                  $(TYPEDFIELDS)
                  """

include("parameters.jl")
include("grid_helpers.jl")
include("operators.jl")
include("kernels.jl")
include("sparse.jl")
include("solvers.jl")
include("generator.jl")

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
    return nothing
end

"""
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

    source_field, solution_field = generator.field_buffer, field

    generate_white_noise!(source_field, noise, generator.white_noise_scale)

    if generator.require_half_order
        apply_inverse_sqrt!(solution_field, source_field, generator.solver)
        source_field, solution_field = solution_field, source_field
    end

    for _ in 1:generator.n_inverse_apply
        apply_inverse!(solution_field, source_field, generator.solver)
        source_field, solution_field = solution_field, source_field
    end

    # Due to name swap in final iteration, source_field corresponds to final solution.
    # If field does not contain final solution copy from buffer
    field !== source_field && copyto!(field, generator.field_buffer)

    return nothing
end

end
