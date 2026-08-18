struct RandomFieldGenerator{G,L,T,F,H,S}
    grid::G
    location::L
    k::Int
    half::Bool
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
    fill_halo_regions!(operand)
    apply!(
        result,
        operand,
        generator.modified_helmholtz_operator,
        generator.grid,
        shift_coefficient,
        generator.weights,
    )
    return result
end

get_weights(p::IsotropicMatern, dimension) = ntuple(_ -> p.length_scale^2, dimension)
get_weights(p::AnisotropicMatern, dimension) = ntuple(n -> p.length_scale[n]^2, dimension)

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
    k, half = alpha ÷ 2, isodd(alpha)

    scale = variance_matching_constant(parameters, alpha, dimension)

    weights = get_weights(parameters, dimension)

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
        k,
        half,
        weights,
        field_buffer_a,
        field_buffer_b,
        white_noise_scale,
        modified_helmholtz_operator,
        solver,
        n_sqrt_quadrature_points,
    )
end
