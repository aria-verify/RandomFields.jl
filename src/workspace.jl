struct GRFWorkspace{F,S,G,T,D,P}
    grid::G
    location::Tuple{DataType,DataType,DataType}

    parameters::P
    scale::T
    k::Int
    half::Bool

    field_buffer_a::F
    field_buffer_b::F
    inverse_sqrt_volume::F

    weights::NTuple{D,T}
    use_fused_isotropic_path::Bool

    laplacian_operator::Any                 # built-in ∇² at field's location (fused isotropic path)
    volume_reciprocal_operator::Any         # reciprocal cell-volume op at field's location
    separable_operators::Any                # used when metric_separability(grid) === SeparableMetrics()
    nonseparable_operators::Any             # used otherwise

    solver::S
end

function apply_matern_operator!(result, u, ws::GRFWorkspace, shift_coefficient=1.)
    fill_halo_regions!(u)
    if ws.use_fused_isotropic_path
        run_kernel!(
            _fused_isotropic_operator_kernel!,
            ws.grid,
            result,
            ws.grid,
            shift_coefficient,
            ws.weights[1],
            ws.laplacian_operator,
            u,
        )
    elseif metric_separability(grid) isa SeparableMetrics
        run_kernel!(
            _separable_operator_kernel!,
            ws.grid,
            result,
            ws.grid,
            shift_coefficient,
            ws.weights,
            ws.separable_operators,
            ws.location,
            u,
        )
    else
        run_kernel!(
            _nonseparable_operator_kernel!,
            ws.grid,
            result,
            ws.grid,
            shift_coefficient,
            ws.weights,
            ws.nonseparable_operators,
            ws.location,
            ws.volume_reciprocal_operator,
            u,
        )
    end
    return result
end

get_weights(p::IsotropicMatern, dimension) = ntuple(_ -> p.length_scale^2, dimension)
get_weights(p::AnisotropicMatern, dimension) = ntuple(n -> p.length_scale[n]^2, dimension)

function GRFWorkspace(field, parameters::MaternParameters; reltol=1e-7, maxiter=prod(size(field)))
    
    grid = field.grid
    loc = location(field)
    active = active_dimensions(grid)
    dimension = dimension_count(grid)
    T = eltype(field)

    check_parameter_dimension(parameters, dimension)

    alpha = matern_spde_alpha(parameters, dimension)
    k, half = alpha ÷ 2, isodd(alpha)

    scale = variance_matching_constant(parameters, alpha, dimension)

    field_buffer_a, field_buffer_b = similar(field), similar(field)

    volume_reciprocal_operator = lookup_operator(:V⁻¹, loc...)
    inverse_sqrt_volume = similar(field)

    weights = get_weights(parameters, dimension)
    use_fused_isotropic_path = parameters isa IsotropicMatern && !is_immersed_grid(grid)

    run_kernel!(
        _inv_sqrt_volume_kernel!,
        grid,
        inverse_sqrt_volume,
        grid,
        volume_reciprocal_operator,
    )

    solver = ConjugateGradientSolver(
        apply_matern_operator!;
        template_field=field,
        reltol,
        maxiter,
    )

    return GRFWorkspace{typeof(field),typeof(solver),typeof(grid),T,dimension,typeof(parameters)}(
        grid,
        loc,
        parameters,
        scale,
        k,
        half,
        field_buffer_a,
        field_buffer_b,
        inverse_sqrt_volume,
        weights,
        use_fused_isotropic_path,
        lookup_operator(:∇², loc...),
        volume_reciprocal_operator,
        directional_operators(SeparableMetrics(), loc, active),
        directional_operators(NonseparableMetrics(), loc, active),
        solver,
    )
end
