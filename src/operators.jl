abstract type AbstractDirectionalOperator{M,I} end

dimension_of(::AbstractDirectionalOperator{M}) where {M} = M

struct SeparableDirectionalOperator{M,I,D} <: AbstractDirectionalOperator{M,I}
    "Derivative operation at dimension M flipped operand location"
    inner::I
    "Grid spacing operation at operand location"
    delta::D 
end

struct NonSeparableDirectionalOperator{M,I,A,V} <: AbstractDirectionalOperator{M,I}
    "Derivative operation at dimension M flipped operand location"
    inner::I
    "Face area operation at dimension M flipped operand location"
    area::A
    "Cell volume operation at operand location"
    volume::V
end

function directional_operators(grid, loc)
    ops = ()
    active = active_dimensions(grid)
    for m in 1:3
        active[m] || continue
        flipped = flip_location_in_dimension(loc, m)
        inner = lookup_operator(Symbol(:∂, DIMENSION_SYMBOLS[m]), flipped...)
        new_op = if separable_metrics(grid)
            delta = lookup_operator(Symbol(:Δ, DIMENSION_SYMBOLS[m]), loc...)
            SeparableDirectionalOperator{m,typeof(inner),typeof(delta)}(inner, delta)
        else
            area = lookup_operator(Symbol(:A, DIMENSION_SYMBOLS[m]), flipped...)
            volume = lookup_operator(:V, loc...)
            NonSeparableDirectionalOperator{m,typeof(inner),typeof(area),typeof(volume)}(
                inner, area, volume
            )
        end
        ops = (ops..., new_op)
    end
    return ops
end


"""
    masked_evaluate(op, m, i, j, k, grid, operand)

Zero out a directional derivative  `op` evaluated at `(i,j,k)` on `grid` with `operand`
if either of the two cells bounding that point (along dimension `m`) is solid.
"""
@inline function masked_evaluate(op, m, i, j, k, grid, operand)
    near = shift_index(m, i, j, k, -1)
    blocked = is_immersed_cell(near..., grid) | is_immersed_cell(i, j, k, grid)
    return blocked ? zero(eltype(operand)) : op(i, j, k, grid, operand)
end

@inline function second_derivative_term(
    i, j, k, grid, op::SeparableDirectionalOperator, ::Tuple, ::Tuple, flux_far, flux_near
)
    (flux_far - flux_near) / op.delta(i, j, k, grid)
end

@inline function second_derivative_term(
    i,
    j,
    k,
    grid,
    op::NonSeparableDirectionalOperator,
    far::Tuple,
    near::Tuple,
    flux_far,
    flux_near,
)
    (flux_far * op.area(far..., grid) - flux_near * op.area(near..., grid)) /
    op.volume(i, j, k, grid)
end

@inline function weighted_second_derivative_sum(
    i,
    j,
    k,
    grid::G,
    operand::Field{L1,L2,L3},
    weights::NTuple{D},
    operators::NTuple{D,AbstractDirectionalOperator},
) where {G,L1,L2,L3,D}
    second_derivative_sum = zero(eltype(operand))
    @unroll for d in 1:D
        w, op = weights[d], operators[d]
        m = dimension_of(op)
        near, far = bounding_pair(Val(m), (L1, L2, L3)[m], i, j, k)
        flux_near = masked_evaluate(op.inner, Val(m), near..., grid, operand)
        flux_far = masked_evaluate(op.inner, Val(m), far..., grid, operand)
        second_derivative_sum +=
            w * second_derivative_term(i, j, k, grid, op, far, near, flux_far, flux_near)
    end
    return second_derivative_sum
end

abstract type AbstractModifiedHelmholtzOperator end

struct IsotropicModifiedHelmholtzOperator{L} <: AbstractModifiedHelmholtzOperator
    laplacian_operator::L
end

function kernel(::IsotropicModifiedHelmholtzOperator)
    _isotropic_modified_helmholtz_operator_kernel!
end
differential_operators(op::IsotropicModifiedHelmholtzOperator) = op.laplacian_operator

struct AnisotropicModifiedHelmholtzOperator{D} <: AbstractModifiedHelmholtzOperator
    differential_operators::D
end

function kernel(::AnisotropicModifiedHelmholtzOperator)
    _anisotropic_modified_helmholtz_operator_kernel!
end
function differential_operators(op::AnisotropicModifiedHelmholtzOperator)
    op.differential_operators
end

function apply!(
    result,
    operand,
    operator::AbstractModifiedHelmholtzOperator,
    grid,
    shift_coefficient,
    weights,
)
    run_kernel!(
        kernel(operator),
        grid,
        result,
        operand,
        grid,
        shift_coefficient,
        weights,
        differential_operators(operator),
    )
end