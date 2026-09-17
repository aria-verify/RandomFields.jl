using KernelAbstractions.Extras.LoopInfo: @unroll

abstract type AbstractDirectionalOperator{M,I} end

dimension_of(::AbstractDirectionalOperator{M}) where {M} = M

"""
Directional differential operator on a dimension `M` for grids with separable metrics.
"""
struct SeparableDirectionalOperator{M,I,D} <: AbstractDirectionalOperator{M,I}
    "Derivative operation at dimension M flipped operand location"
    inner::I
    "Grid spacing operation at operand location"
    delta::D
end

"""
Directional differential operator on a dimension `M` for grids with non-separable metrics.
"""
struct NonSeparableDirectionalOperator{M,I,A,V} <: AbstractDirectionalOperator{M,I}
    "Derivative operation at dimension M flipped operand location"
    inner::I
    "Face area operation at dimension M flipped operand location"
    area::A
    "Cell volume operation at operand location"
    volume::V
end

"""
Construct directional differential operators for a field on a grid `grid` and with
staggered grid locations tuple `loc`.
"""
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
Zero out a directional derivative  `op` evaluated at `(i,j,k)` on `grid` with `operand`
if either of the two cells bounding that point (along dimension `m`) is solid.
"""
@inline function masked_evaluate(op, m, i, j, k, grid, operand)
    near = shift_index(m, i, j, k, -1)
    blocked = immersed_cell(near..., grid) | immersed_cell(i, j, k, grid)
    return blocked ? zero(eltype(operand)) : op(i, j, k, grid, operand)
end

"""
    $(FUNCTIONNAME)(i, j, k, grid, op, far, near, flux_far, flux_near)

Compute second derivative term for a field on grid `grid` with direction differential operator
`op` at indices `(i, j, k)` given the computed flux terms `flux_near` and `flux_far` at near
and far offset index tuples `far` and `near`.
"""
function second_derivative_term end
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

"""
Compute the weighted second derivative sum for a field `operand` on grid `grid` at
indices `(i, j, k)` with per-dimension derivative weights `weights` and directional
derivative operators `operators`.
"""
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
    return _isotropic_modified_helmholtz_operator_kernel!
end
differential_operators(op::IsotropicModifiedHelmholtzOperator) = op.laplacian_operator

struct AnisotropicModifiedHelmholtzOperator{D} <: AbstractModifiedHelmholtzOperator
    differential_operators::D
end

function kernel(::AnisotropicModifiedHelmholtzOperator)
    return _anisotropic_modified_helmholtz_operator_kernel!
end
function differential_operators(op::AnisotropicModifiedHelmholtzOperator)
    return op.differential_operators
end

function apply!(
    result,
    operand,
    operator::AbstractModifiedHelmholtzOperator,
    grid,
    weights,
    shift=zero(eltype(grid)),
)
    active_cells_map = get_active_cells_map(grid, Val(:xyz))
    run_kernel!(
        kernel(operator),
        grid,
        result,
        operand,
        grid,
        shift,
        weights,
        differential_operators(operator);
        active_cells_map,
    )
    fill_halo_regions!(result)
    isnothing(active_cells_map) && mask_immersed_field!(result)
    return nothing
end

function div_by_sqrt_cell_volumes!(result, operand, sqrt_volumes)
    grid = result.grid
    active_cells_map = get_active_cells_map(grid, Val(:xyz))
    run_kernel!(
        _div_by_sqrt_cell_volumes_kernel!,
        grid,
        result,
        operand,
        sqrt_volumes;
        active_cells_map,
    )
    fill_halo_regions!(result)
    return nothing
end

function div_by_sqrt_cell_volumes!(field, sqrt_volumes)
    return div_by_sqrt_cell_volumes!(field, field, sqrt_volumes)
end

function mul_by_sqrt_cell_volumes!(result, operand, sqrt_volumes)
    grid = result.grid
    active_cells_map = get_active_cells_map(grid, Val(:xyz))
    run_kernel!(
        _mul_by_sqrt_cell_volumes_kernel!,
        grid,
        result,
        operand,
        sqrt_volumes;
        active_cells_map,
    )
    fill_halo_regions!(result)
    return nothing
end

function mul_by_sqrt_cell_volumes!(field, sqrt_volumes)
    return mul_by_sqrt_cell_volumes!(field, field, sqrt_volumes)
end

function symmetric_apply!(
    result,
    operand,
    operator::AbstractModifiedHelmholtzOperator,
    grid,
    weights,
    sqrt_cell_volumes,
    shift=zero(eltype(grid)),
)
    # Modified Helmholtz operator A is symmetric with respect to the cell volume weighted
    # inner product that is AᵀV = VA where V is a diagonal matrix of the cell volumes
    # To form an operator which is symmetric with respect to Euclidean inner product we
    # compute sqrt(V) * A * sqrt(V)⁻¹
    div_by_sqrt_cell_volumes!(operand, sqrt_cell_volumes)
    apply!(result, operand, operator, grid, weights, shift)
    mul_by_sqrt_cell_volumes!(result, sqrt_cell_volumes)
    # Undo scaling of operand
    mul_by_sqrt_cell_volumes!(operand, sqrt_cell_volumes)
    return nothing
end
