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
additional_kernel_args(op::IsotropicModifiedHelmholtzOperator) = (op.laplacian_operator,)

struct SeparableModifiedHelmholtzOperator{D} <: AbstractModifiedHelmholtzOperator
    differential_operators::D
end

function kernel(::SeparableModifiedHelmholtzOperator)
    _separable_modified_helmholtz_operator_kernel!
end
function additional_kernel_args(op::SeparableModifiedHelmholtzOperator)
    (op.differential_operators,)
end

struct NonSeparableModifiedHelmholtzOperator{D,V} <: AbstractModifiedHelmholtzOperator
    differential_operators::D
    volume_reciprocal_operator::V
end

function kernel(::NonSeparableModifiedHelmholtzOperator)
    _nonseparable_modified_helmholtz_operator_kernel!
end
function additional_kernel_args(op::NonSeparableModifiedHelmholtzOperator)
    (op.differential_operators, op.volume_reciprocal_operator)
end

function apply!(
    result,
    operand,
    operator::AbstractModifiedHelmholtzOperator,
    grid,
    location,
    shift_coefficient,
    weights,
)
    run_kernel!(
        kernel(operator),
        grid,
        result,
        operand,
        grid,
        location,
        shift_coefficient,
        weights,
        additional_kernel_args(operator)...,
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
        ws.location,
        shift_coefficient,
        ws.weights,
    )
    return result
end

get_weights(p::IsotropicMatern, dimension) = ntuple(_ -> p.length_scale^2, dimension)
get_weights(p::AnisotropicMatern, dimension) = ntuple(n -> p.length_scale[n]^2, dimension)

function GRFWorkspace(
    field, parameters::MaternParameters; reltol=1e-7, maxiter=prod(size(field))
)
    grid = field.grid
    loc = location(field)
    active = active_dimensions(grid)
    dimension = dimension_count(grid)

    check_parameter_dimension(parameters, dimension)

    alpha = matern_spde_alpha(parameters, dimension)
    k, half = alpha ÷ 2, isodd(alpha)

    scale = variance_matching_constant(parameters, alpha, dimension)

    weights = get_weights(parameters, dimension)

    field_buffer_a, field_buffer_b = similar(field), similar(field)

    volume_reciprocal_operator = lookup_operator(:V⁻¹, loc...)
    inverse_sqrt_volume = similar(field)

    use_isotropic_operator = parameters isa IsotropicMatern && !is_immersed_grid(grid)

    modified_helmholtz_operator = if use_isotropic_operator
        IsotropicModifiedHelmholtzOperator(lookup_operator(:∇², loc...))
    elseif metric_separability(grid) isa SeparableMetrics
        SeparableModifiedHelmholtzOperator(
            directional_operators(SeparableMetrics(), loc, active)
        )
    else
        NonSeparableModifiedHelmholtzOperator(
            directional_operators(NonseparableMetrics(), loc, active),
            volume_reciprocal_operator,
        )
    end

    run_kernel!(
        _inv_sqrt_volume_kernel!,
        grid,
        inverse_sqrt_volume,
        grid,
        volume_reciprocal_operator,
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
