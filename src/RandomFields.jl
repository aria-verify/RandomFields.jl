"""
Generate Gaussian Markov random fields with Matérn covariance functions on Oceananigans grids.

$(EXPORTS)
"""
module RandomFields

using Oceananigans
using Oceananigans.Fields: location, interior
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, immersed_cell
using Oceananigans.BoundaryConditions: fill_halo_regions!
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

function scale_noise!(result, noise, scale)
    run_kernel!(_scale_noise_kernel!, result.grid, result, noise, scale)
    return nothing
end

"""
Generate white noise and write to `field` using standard normal variate
in `noise` with `size(noise) == size(field)` and scaling by scale factor
and reciprocal of square root of cell volumes cached in `generator`. If
`field` is defined on a `ImmersedBoundaryGrid` and `mask_immersed` is
`true` inactive immersed cells in the output will be masked to zero.
"""
function generate_white_noise!(
    field, noise, generator::RandomFieldGenerator; mask_immersed=false
)
    grid = field.grid
    active_cells_map = mask_immersed ? get_active_cells_map(grid, Val(:xyz)) : nothing
    scale_noise!(field, noise, generator.scale)
    div_by_sqrt_cell_volumes!(field, generator.sqrt_cell_volumes)
    fill_halo_regions!(field)
    mask_immersed && isnothing(active_cells_map) && mask_immersed_field!(field)
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

    scale_noise!(source_field, noise, generator.scale)

    if generator.require_half_order
        solve_sqrt_adjoint!(solution_field, source_field, generator.solver)
        source_field, solution_field = solution_field, source_field
    end

    for _ in 1:generator.n_solve
        solve!(solution_field, source_field, generator.solver)
        source_field, solution_field = solution_field, source_field
    end

    # Due to name swap in final iteration, source_field corresponds to final solution.
    # If field does not contain final solution copy from buffer
    field !== source_field && copyto!(field, generator.field_buffer)

    div_by_sqrt_cell_volumes!(field, generator.sqrt_cell_volumes)

    return nothing
end

end
