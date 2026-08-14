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
using SpecialFunctions: gamma

export generate!, IsotropicMatern, AnisotropicMatern, GRFWorkspace

include("parameters.jl")
include("grid_helpers.jl")
include("directional_operators.jl")
include("kernels.jl")
include("workspace.jl")

function matern_spde_two_alpha(parameters, dimension_count)
    two_nu = 2 * parameters.smoothness
    isinteger(two_nu) || throw(ArgumentError("smoothness must be integer or half-integer"))
    two_alpha = round(Int, two_nu) + dimension_count
    two_alpha > 0 || throw(ArgumentError("smoothness + dimension/2 must be positive"))
    return two_alpha
end

function discretize_white_noise!(ws::GRFWorkspace, v)
    rhs, grid = current_solution(ws), ws.grid
    run_kernel!(
        _discretize_white_noise_kernel!, grid, rhs, grid, ws.scale, ws.inverse_sqrt_volume, v
    )
    fill_halo_regions!(rhs)
    return rhs
end

function apply_repeated_inverse!(ws::GRFWorkspace)
    for _ in 1:ws.k
        zero_field!(next_solution(ws))
        solve!(next_solution(ws), ws.solver, current_solution(ws), ws, 1.)
        fill_halo_regions!(next_solution(ws))
        mask_immersed_values!(next_solution(ws))
        swap_solution_buffers!(ws)
    end
    return ws
end

# A^(-1/2) via A^(-1/2) = (2/π) ∫₀^{π/2} (A + tan²θ)⁻¹ sec²θ dθ, midpoint quadrature
function apply_half_order_inverse!(ws::GRFWorkspace, quadrature_points)
    zero_field!(ws.accumulator)
    for m in 1:quadrature_points
        θ = (m - 0.5) * (π / 2) / quadrature_points
        shift_coefficient = 1 + tan(θ)^2
        weight = (2 / π) * (π / 2 / quadrature_points) * sec(θ)^2
        zero_field!(next_solution(ws))
        solve!(next_solution(ws), ws.solver, current_solution(ws), ws, shift_coefficient)
        fill_halo_regions!(next_solution(ws))
        accumulate_weighted!(ws.accumulator, next_solution(ws), weight)
    end
    return ws.accumulator
end

"""
    generate!(field, workspace, v; sqrt_quadrature_points = 32)

Overwrite `field` with an (approximate) draw from a mean-zero Gaussian random field with
Matérn covariance, via the SPDE representation `(1 - Σᵢλᵢ²∂ᵢ²)^(ν+d/2) field ∝ white noise`.

- `workspace`: a `GRFWorkspace` built once via `GRFWorkspace(field, parameters)` and 
   reusable across calls, with parameters an instance of `IsotropicMatern` or `AnisotropicMatern`.
- `v`: standard normal variates, `size(v) == size(field)`.
- `sqrt_quadrature_points`: quadrature points for the half-order factor, used only when
  `smoothness + dimension/2` is not an integer.
"""
function generate!(
    field,
    workspace::GRFWorkspace,
    v;
    sqrt_quadrature_points=32,
)
    field.grid === workspace.grid ||
        throw(ArgumentError("workspace was built for a different grid"))
    location(field) === workspace.location ||
        throw(ArgumentError("workspace was built for a different field location"))

    workspace.active_is_a = true
    
    discretize_white_noise!(workspace, v)

    apply_repeated_inverse!(workspace)

    result = if workspace.half
        apply_half_order_inverse!(workspace, sqrt_quadrature_points)
    else
        current_solution(workspace)
    end
    copy_masked_result!(field, result)

    return field
end

end