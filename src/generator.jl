"Precomputed quantities and buffer storage for generating Gaussian random fields"
struct RandomFieldGenerator{G,L,T,F,H,S}
    "Oceananigans grid fields will be generated on"
    grid::G
    "Staggered grid location tuple fields will be generated on"
    location::L
    "Number of times inverse of modified Helmholtz linear operator is applied to generate field"
    n_inverse_apply::Int
    "Whether the inverse half-order modified Helmholtz linear operator needs to be applied to generate field"
    require_half_order::Bool
    "Length-scale based weights for second derivative terms in modified Helmholtz operator"
    weights::Tuple{Vararg{T}}
    "Field buffer used to store intermediate quantities during computation"
    field_buffer_a::F
    "Field buffer used to store intermediate quantities during computation"
    field_buffer_b::F
    "Field used to store precomputed per-cell scale factors for white noise generation"
    white_noise_scale::F
    "Linear operator corresponding to modified Helmholtz equation for the parameters of interest"
    modified_helmholtz_operator::H
    "Solver for linear system in modified Helmholtz operator "
    solver::S
    "Number of quadrature points to use in integral approximation to square root of linear operator"
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

"""
$(SIGNATURES)

Construct a random field generator for template `field` with covariance parameters `parameters`.

The linear systems are solved using a conjugate gradient solver with relative tolerance `reltol`
and maximum number of iterations `maxiter` and the half-order (square-root) linear operator
inverse is approximated using quadrature of an integral representation with `n_sqrt_quadrature_points`
quadrature points if relevant.
"""
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
