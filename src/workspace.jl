struct GRFWorkspace{G,L,T,F,H,S}
    grid::G
    location::L

    scale::T
    k::Int
    half::Bool
    weights::Tuple{Vararg{T}}

    field_buffer_a::F
    field_buffer_b::F
    inverse_sqrt_volume::F

    modified_helmholtz_operator::H

    solver::S
end

abstract type AbstractModifiedHelmholtzOperator end

struct IsotropicModifiedHelmholtzOperator{L} <: AbstractModifiedHelmholtzOperator
    laplacian_operator::L
end

function kernel(::IsotropicModifiedHelmholtzOperator)
    _isotropic_modified_helmholtz_operator_kernel!
end
differential_operators(op::IsotropicModifiedHelmholtzOperator) = op.laplacian_operator

struct AnisotropicModifiedHelmholtzOperator{D} <: AbstractModifiedHelmholtzOperator
    differential_operators::D
end

function kernel(::AnisotropicModifiedHelmholtzOperator)
    _anisotropic_modified_helmholtz_operator_kernel!
end
function differential_operators(op::AnisotropicModifiedHelmholtzOperator)
    op.differential_operators
end

function apply!(
    result,
    operand,
    operator::AbstractModifiedHelmholtzOperator,
    grid,
    shift_coefficient,
    weights,
)
    run_kernel!(
        kernel(operator),
        grid,
        result,
        operand,
        grid,
        shift_coefficient,
        weights,
        differential_operators(operator),
    )
end

function apply_modified_helmholtz_operator!(
    result, operand, ws::GRFWorkspace, shift_coefficient=1.0
)
    fill_halo_regions!(operand)
    apply!(
        result,
        operand,
        ws.modified_helmholtz_operator,
        ws.grid,
        shift_coefficient,
        ws.weights,
    )
    return result
end

get_weights(p::IsotropicMatern, dimension) = ntuple(_ -> p.length_scale^2, dimension)
get_weights(p::AnisotropicMatern, dimension) = ntuple(n -> p.length_scale[n]^2, dimension)

function GRFWorkspace(
    field, parameters::AbstractMaternParameters; reltol=1e-7, maxiter=prod(size(field))
)
    grid = field.grid
    loc = location(field)
    dimension = dimension_count(grid)

    check_parameter_dimension(parameters, dimension)

    alpha = matern_spde_alpha(parameters.smoothness, dimension)
    k, half = alpha ÷ 2, isodd(alpha)

    scale = variance_matching_constant(parameters, alpha, dimension)

    weights = get_weights(parameters, dimension)

    field_buffer_a, field_buffer_b = similar(field), similar(field)

    inverse_sqrt_volume = similar(field)

    use_isotropic_operator =
        parameters isa IsotropicMatern && !(grid isa ImmersedBoundaryGrid)

    modified_helmholtz_operator = if use_isotropic_operator
        IsotropicModifiedHelmholtzOperator(lookup_operator(:∇², loc...))
    else
        AnisotropicModifiedHelmholtzOperator(directional_operators(grid, loc))
    end

    run_kernel!(
        _inv_sqrt_volume_kernel!,
        grid,
        inverse_sqrt_volume,
        grid,
        lookup_operator(:V⁻¹, loc...),
    )

    solver = ConjugateGradientSolver(
        apply_modified_helmholtz_operator!; template_field=field, reltol, maxiter
    )

    return GRFWorkspace(
        grid,
        loc,
        scale,
        k,
        half,
        weights,
        field_buffer_a,
        field_buffer_b,
        inverse_sqrt_volume,
        modified_helmholtz_operator,
        solver,
    )
end
