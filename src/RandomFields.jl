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

function configure_weights!(ws, p::IsotropicMatern, active)
    ws.weights = ntuple(_ -> p.length_scale^2, ws.dimension_count)
end

function configure_weights!(ws, p::AnisotropicMatern, active)
    ws.weights = ntuple(n -> p.length_scale[n]^2, ws.dimension_count)
end

function configure_operator!(ws::GRFWorkspace, parameters::MaternParameters, active)
    configure_weights!(ws, parameters, active)
    ws.use_fused_isotropic_path =
        parameters isa IsotropicMatern && !is_immersed_grid(ws.grid)
    return ws
end

function discretize_white_noise!(ws::GRFWorkspace, v, scale)
    rhs, grid = current_solution(ws), ws.grid
    run_kernel!(
        _discretize_white_noise_kernel!, grid, rhs, grid, scale, ws.inverse_sqrt_volume, v
    )
    fill_halo_regions!(rhs)
    return rhs
end

function apply_repeated_inverse!(ws::GRFWorkspace, k)
    for _ in 1:k
        zero_field!(next_solution(ws))
        solve!(next_solution(ws), ws.solver, current_solution(ws), ws)
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
        ws.shift_coefficient = 1 + tan(θ)^2
        weight = (2 / π) * (π / 2 / quadrature_points) * sec(θ)^2
        zero_field!(next_solution(ws))
        solve!(next_solution(ws), ws.solver, current_solution(ws), ws)
        fill_halo_regions!(next_solution(ws))
        accumulate_weighted!(ws.accumulator, next_solution(ws), weight)
    end
    ws.shift_coefficient = 1
    return ws.accumulator
end

"""
    generate!(field, workspace, v, parameters::MaternParameters; sqrt_quadrature_points = 32)

Overwrite `field` with an (approximate) draw from a mean-zero Gaussian random field with
Matérn covariance, via the SPDE representation `(1 - Σᵢλᵢ²∂ᵢ²)^(ν+d/2) field ∝ white noise`.

- `workspace`: a `GRFWorkspace` built once via `GRFWorkspace(field)` and reusable across
  calls (including calls that switch between `IsotropicMatern`/`AnisotropicMatern`).
- `v`: standard normal variates, `size(v) == size(field)`.
- `parameters`: an `IsotropicMatern` or `AnisotropicMatern`.
- `sqrt_quadrature_points`: quadrature points for the half-order factor, used only when
  `smoothness + dimension/2` is not an integer.
"""
function generate!(
    field,
    workspace::GRFWorkspace,
    v,
    parameters::MaternParameters;
    sqrt_quadrature_points=32,
)
    field.grid === workspace.grid ||
        throw(ArgumentError("workspace was built for a different grid"))
    location(field) === workspace.location ||
        throw(ArgumentError("workspace was built for a different field location"))

    active = active_dimensions(workspace.grid)
    check_parameter_dimension(parameters, workspace.dimension_count)

    two_alpha = matern_spde_two_alpha(parameters, workspace.dimension_count)
    k, half = two_alpha ÷ 2, isodd(two_alpha)
    alpha = two_alpha / 2

    configure_operator!(workspace, parameters, active)
    workspace.shift_coefficient = 1
    workspace.active_is_a = true

    scale = variance_matching_constant(parameters, alpha, workspace.dimension_count)
    discretize_white_noise!(workspace, v, scale)

    apply_repeated_inverse!(workspace, k)

    result = if half
        apply_half_order_inverse!(workspace, sqrt_quadrature_points)
    else
        current_solution(workspace)
    end
    copy_masked_result!(field, result)

    return field
end

end