mutable struct GRFWorkspace{F,S,G,T,D,P}
    grid::G
    location::Tuple{DataType,DataType,DataType}
    dimension_count::Int

    parameters::P
    scale::T
    k::Int
    half::Bool

    buffer_a::F
    buffer_b::F
    accumulator::F
    inverse_sqrt_volume::F
    active_is_a::Bool

    shift_coefficient::T
    weights::NTuple{D,T}
    use_fused_isotropic_path::Bool

    laplacian_operator::Any                 # built-in ∇² at field's location (fused isotropic path)
    volume_reciprocal_operator::Any         # reciprocal cell-volume op at field's location
    separable_operators::Any                # used when metric_separability(grid) === SeparableMetrics()
    nonseparable_operators::Any             # used otherwise

    solver::S
end

current_solution(ws::GRFWorkspace) = ws.active_is_a ? ws.buffer_a : ws.buffer_b
next_solution(ws::GRFWorkspace) = ws.active_is_a ? ws.buffer_b : ws.buffer_a
swap_solution_buffers!(ws::GRFWorkspace) = (ws.active_is_a=(!ws.active_is_a); ws)

function apply_matern_operator!(result, u, ws::GRFWorkspace)
    fill_halo_regions!(u)
    if ws.use_fused_isotropic_path
        run_kernel!(
            _fused_isotropic_operator_kernel!,
            ws.grid,
            result,
            ws.grid,
            ws.shift_coefficient,
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
            ws.shift_coefficient,
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
            ws.shift_coefficient,
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

    two_alpha = matern_spde_two_alpha(parameters, dimension)
    k, half = two_alpha ÷ 2, isodd(two_alpha)
    alpha = two_alpha / 2

    scale = variance_matching_constant(parameters, alpha, dimension)

    buffer_a, buffer_b, accumulator = similar(field), similar(field), similar(field)

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
        dimension,
        parameters,
        scale,
        k,
        half,
        buffer_a,
        buffer_b,
        accumulator,
        inverse_sqrt_volume,
        true,
        one(T),
        weights,
        use_fused_isotropic_path,
        lookup_operator(:∇², loc...),
        volume_reciprocal_operator,
        directional_operators(SeparableMetrics(), loc, active),
        directional_operators(NonseparableMetrics(), loc, active),
        solver,
    )
end
