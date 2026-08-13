struct SeparableDirectionalOperator{M, Inner, Delta}
    inner::Inner   # derivative op, Loc_m -> flipped(Loc_m)
    delta::Delta   # grid spacing op at Loc (output location)
end
dimension_of(::SeparableDirectionalOperator{M}) where M = M

struct NonseparableDirectionalOperator{M, Inner, Area}
    inner::Inner   # derivative op, Loc_m -> flipped(Loc_m)
    area::Area     # face-area op, evaluated at the same flipped location as inner
end
dimension_of(::NonseparableDirectionalOperator{M}) where M = M

function directional_operators(::SeparableMetrics, Loc, active)
    ops = ()
    for m in 1:3
        active[m] || continue
        flipped = flip_location_in_dimension(Loc, m)
        inner = lookup_operator(Symbol(:∂, DIMENSION_SYMBOLS[m]), flipped...)
        delta = lookup_operator(Symbol(:Δ, DIMENSION_SYMBOLS[m]), Loc...)
        ops = (ops..., SeparableDirectionalOperator{m, typeof(inner), typeof(delta)}(inner, delta))
    end
    return ops
end

function directional_operators(::NonseparableMetrics, Loc, active)
    ops = ()
    for m in 1:3
        active[m] || continue
        flipped = flip_location_in_dimension(Loc, m)
        inner = lookup_operator(Symbol(:∂, DIMENSION_SYMBOLS[m]), flipped...)
        area  = lookup_operator(Symbol(:A, DIMENSION_SYMBOLS[m]), flipped...)
        ops = (ops..., NonseparableDirectionalOperator{m, typeof(inner), typeof(area)}(inner, area))
    end
    return ops
end

# Zero out a directional derivative evaluated at (i,j,k) if either of the two
# Center cells bounding that point (one step back along dimension m) is solid.
# Correct for a field that is Center-located in dimension m; see note in prose
# above for the Face-located + immersed generalization.
@inline function masked_evaluate(op, m, i, j, k, grid, u)
    near = shift_index(m, i, j, k, -1)
    blocked = is_immersed_cell(near..., grid) | is_immersed_cell(i, j, k, grid)
    return blocked ? zero(eltype(u)) : op(i, j, k, grid, u)
end

@inline separable_weighted_second_derivative_sum(i, j, k, grid, u, ::Tuple{}, ::Tuple{}, Loc) = zero(eltype(u))
@inline function separable_weighted_second_derivative_sum(i, j, k, grid, u, weights, operators, Loc)
    w, rest_w   = first(weights), Base.tail(weights)
    op, rest_op = first(operators), Base.tail(operators)
    m = dimension_of(op)
    near, far = bounding_pair(m, Loc[m], i, j, k)
    flux_near = masked_evaluate(op.inner, m, near..., grid, u)
    flux_far  = masked_evaluate(op.inner, m, far...,  grid, u)
    term = w * (flux_far - flux_near) / op.delta(i, j, k, grid)
    return term + separable_weighted_second_derivative_sum(i, j, k, grid, u, rest_w, rest_op, Loc)
end

@inline nonseparable_weighted_flux_sum(i, j, k, grid, u, ::Tuple{}, ::Tuple{}, Loc) = zero(eltype(u))
@inline function nonseparable_weighted_flux_sum(i, j, k, grid, u, weights, operators, Loc)
    w, rest_w   = first(weights), Base.tail(weights)
    op, rest_op = first(operators), Base.tail(operators)
    m = dimension_of(op)
    near, far = bounding_pair(m, Loc[m], i, j, k)
    flux_near = op.area(near..., grid) * masked_evaluate(op.inner, m, near..., grid, u)
    flux_far  = op.area(far...,  grid) * masked_evaluate(op.inner, m, far...,  grid, u)
    term = w * (flux_far - flux_near)
    return term + nonseparable_weighted_flux_sum(i, j, k, grid, u, rest_w, rest_op, Loc)
end