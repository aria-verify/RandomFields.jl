struct RandomFieldGenerator{G,L,T,F,H,S}
    grid::G
    location::L
    n_inverse_apply::Int
    require_half_order::Bool
    weights::Tuple{Vararg{T}}
    field_buffer_a::F
    field_buffer_b::F
    white_noise_scale::F
    modified_helmholtz_operator::H
    solver::S
    n_sqrt_quadrature_points::Int
end

function apply_modified_helmholtz_operator!(
    result, operand, generator::RandomFieldGenerator, shift_coefficient=1.0
)
    apply!(
        result,
        operand,
        generator.modified_helmholtz_operator,
        generator.grid,
        shift_coefficient,
        generator.weights,
    )
    return nothing
end

function RandomFieldGenerator(
    field,
    parameters::AbstractMaternParameters;
    reltol=1e-7,
    maxiter=prod(size(field)),
    n_sqrt_quadrature_points=32,
)
    grid = field.grid
    loc = location(field)
    dimension = dimension_count(grid)

    check_parameter_dimension(parameters, dimension)

    alpha = matern_spde_alpha(parameters.smoothness, dimension)
    n_inverse_apply, require_half_order = alpha ÷ 2, isodd(alpha)
    scale = variance_matching_constant(parameters, alpha, dimension)
    weights = derivative_weights(parameters, dimension)

    field_buffer_a, field_buffer_b = similar(field), similar(field)

    white_noise_scale = similar(field)

    use_isotropic_operator =
        parameters isa IsotropicMatern && !(grid isa ImmersedBoundaryGrid)

    modified_helmholtz_operator = if use_isotropic_operator
        IsotropicModifiedHelmholtzOperator(lookup_operator(:∇², loc...))
    else
        AnisotropicModifiedHelmholtzOperator(directional_operators(grid, loc))
    end

    run_kernel!(
        _white_noise_scale_kernel!,
        grid,
        white_noise_scale,
        grid,
        scale,
        lookup_operator(:V⁻¹, loc...),
    )

    solver = ConjugateGradientSolver(
        apply_modified_helmholtz_operator!; template_field=field, reltol, maxiter
    )

    return RandomFieldGenerator(
        grid,
        loc,
        n_inverse_apply,
        require_half_order,
        weights,
        field_buffer_a,
        field_buffer_b,
        white_noise_scale,
        modified_helmholtz_operator,
        solver,
        n_sqrt_quadrature_points,
    )
end
