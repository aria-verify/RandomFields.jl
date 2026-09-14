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
    field_buffer::F
    "Field used to store precomputed per-cell scale factors for white noise generation"
    white_noise_scale::F
    "Linear operator corresponding to modified Helmholtz equation for the parameters of interest"
    modified_helmholtz_operator::H
    "Solver for linear system in modified Helmholtz operator "
    solver::S
end

"""
$(SIGNATURES)

Construct a random field generator for template `field` with covariance parameters `parameters`.

The linear systems are solved with a solver of type `solver_type` with additional keyword
arguments `solver_kwargs` passed to its constructor along with a function for applying
the modified Helmholtz linear operator for a given shift coefficient and a template field
for constructing buffers for use in intermediate computations in the solver.
"""
function RandomFieldGenerator(
    field, parameters::AbstractMaternParameters; solver_type=CGSolver, solver_kwargs...
)
    grid = field.grid
    loc = location(field)
    dimension = dimension_count(grid)

    check_parameter_dimension(parameters, dimension)

    alpha = matern_spde_alpha(parameters.smoothness, dimension)
    n_inverse_apply, require_half_order = alpha ÷ 2, isodd(alpha)
    scale = variance_matching_constant(parameters, alpha, dimension)
    weights = derivative_weights(parameters, dimension)

    field_buffer, white_noise_scale = similar(field), similar(field)

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

    function wrapped_apply!(solution, rhs, shift_coefficient)
        return apply!(
            solution, rhs, modified_helmholtz_operator, grid, shift_coefficient, weights
        )
    end

    solver = solver_type(wrapped_apply!, field; solver_kwargs...)

    return RandomFieldGenerator(
        grid,
        loc,
        n_inverse_apply,
        require_half_order,
        weights,
        field_buffer,
        white_noise_scale,
        modified_helmholtz_operator,
        solver,
    )
end
