abstract type AbstractDirectionalOperator{M,I} end

dimension_of(::AbstractDirectionalOperator{M}) where {M} = M

struct SeparableDirectionalOperator{M,I,D} <: AbstractDirectionalOperator{M,I}
    inner::I   # derivative op, Loc_m -> flipped(Loc_m)
    delta::D   # grid spacing op at Loc (output location)
end

struct NonseparableDirectionalOperator{M,I,A} <: AbstractDirectionalOperator{M,I}
    inner::I   # derivative op, Loc_m -> flipped(Loc_m)
    area::A    # face-area op, evaluated at the same flipped location as inner
end

function directional_operators(metrics_are_separable, loc, active)
    ops = ()
    for m in 1:3
        active[m] || continue
        flipped = flip_location_in_dimension(loc, m)
        inner = lookup_operator(Symbol(:∂, DIMENSION_SYMBOLS[m]), flipped...)
        new_op = if metrics_are_separable
            delta = lookup_operator(Symbol(:Δ, DIMENSION_SYMBOLS[m]), loc...)
            SeparableDirectionalOperator{m,typeof(inner),typeof(delta)}(inner, delta)
        else
            area = lookup_operator(Symbol(:A, DIMENSION_SYMBOLS[m]), flipped...)
            NonseparableDirectionalOperator{m,typeof(inner),typeof(area)}(inner, area)
        end
        ops = (ops..., new_op)
    end
    return ops
end

# Zero out a directional derivative evaluated at (i,j,k) if either of the two
# Center cells bounding that point (one step back along dimension m) is solid.
# Correct for a field that is Center-located in dimension m; see note in prose
# above for the Face-located + immersed generalization.
@inline function masked_evaluate(op, m, i, j, k, grid, operand)
    near = shift_index(m, i, j, k, -1)
    blocked = is_immersed_cell(near..., grid) | is_immersed_cell(i, j, k, grid)
    return blocked ? zero(eltype(operand)) : op(i, j, k, grid, operand)
end

@inline function separable_weighted_second_derivative_sum(
    i, j, k, grid::G, operand::Field{L1,L2,L3}, weights::NTuple{D}, operators::NTuple{D, SeparableDirectionalOperator}
) where {G,L1,L2,L3,D}
    second_derivative_sum = zero(eltype(operand))
    @unroll for d in 1:D
        w, op = weights[d], operators[d]
        m = dimension_of(op)
        near, far = bounding_pair(Val(m), (L1, L2, L3)[m], i, j, k)
        flux_near = masked_evaluate(op.inner, Val(m), near..., grid, operand)
        flux_far = masked_evaluate(op.inner, Val(m), far..., grid, operand)
        second_derivative_sum += w * (flux_far - flux_near) / op.delta(i, j, k, grid)
    end
    return second_derivative_sum
end

@inline function nonseparable_weighted_flux_sum(
    i, j, k, grid::G, operand::Field{L1,L2,L3}, weights::NTuple{D}, operators::NTuple{D, NonseparableDirectionalOperator}
) where {G,L1,L2,L3,D}
    flux_sum = zero(eltype(operand))
    @unroll for d in 1:D
        w, op = weights[d], operators[d]
        m = dimension_of(op)
        near, far = bounding_pair(Val(m), (L1, L2, L3)[m], i, j, k)
        flux_near =
            op.area(near..., grid) * masked_evaluate(op.inner, Val(m), near..., grid, operand)
        flux_far =
            op.area(far..., grid) * masked_evaluate(op.inner, Val(m), far..., grid, operand)
        flux_sum += w * (flux_far - flux_near)
    end
    return flux_sum
end