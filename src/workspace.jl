mutable struct GRFWorkspace{F,S,G,T,D}
    grid::G
    location::Tuple{DataType,DataType,DataType}
    dimension_count::Int

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
    grid = ws.grid
    if ws.use_fused_isotropic_path
        run_kernel!(
            _fused_isotropic_operator_kernel!,
            grid,
            result,
            grid,
            ws.shift_coefficient,
            ws.weights[1],
            ws.laplacian_operator,
            u,
        )
    elseif metric_separability(grid) isa SeparableMetrics
        run_kernel!(
            _separable_operator_kernel!,
            grid,
            result,
            grid,
            ws.shift_coefficient,
            ws.weights,
            ws.separable_operators,
            ws.location,
            u,
        )
    else
        run_kernel!(
            _nonseparable_operator_kernel!,
            grid,
            result,
            grid,
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

function GRFWorkspace(field; reltol=1e-7, maxiter=prod(size(field)))
    grid = field.grid
    loc = location(field)
    active = active_dimensions(grid)
    dimension = dimension_count(grid)
    T = eltype(field)

    buffer_a, buffer_b, accumulator = similar(field), similar(field), similar(field)

    volume_reciprocal_operator = lookup_operator(:V⁻¹, loc...)
    inverse_sqrt_volume = similar(field)

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

    return GRFWorkspace{typeof(field),typeof(solver),typeof(grid),T,dimension}(
        grid,
        loc,
        dimension,
        buffer_a,
        buffer_b,
        accumulator,
        inverse_sqrt_volume,
        true,
        one(T),
        ntuple(_ -> one(T), dimension),
        false,
        lookup_operator(:∇², loc...),
        volume_reciprocal_operator,
        directional_operators(SeparableMetrics(), loc, active),
        directional_operators(NonseparableMetrics(), loc, active),
        solver,
    )
end
