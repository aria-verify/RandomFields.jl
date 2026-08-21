separable_metrics(::RectilinearGrid) = true
separable_metrics(::LatitudeLongitudeGrid) = false
separable_metrics(grid::ImmersedBoundaryGrid) = separable_metrics(grid.underlying_grid)
separable_metrics(::Oceananigans.Grids.AbstractGrid) = false

active_dimensions(grid) = ntuple(m -> topology(grid)[m] !== Flat, Val(3))
dimension_count(grid) = count(active_dimensions(grid))

const DIMENSION_SYMBOLS = (:x, :y, :z)

location_character(::Type{Center}) = 'ᶜ'
location_character(::Type{Face}) = 'ᶠ'
flip_location(::Type{Center}) = Face
flip_location(::Type{Face}) = Center

function lookup_operator(prefix::Symbol, L1, L2, L3)
    getfield(
        Oceananigans.Operators,
        Symbol(
            prefix, location_character(L1), location_character(L2), location_character(L3)
        ),
    )
end

function flip_location_in_dimension(loc, m)
    ntuple(n -> n == m ? flip_location(loc[n]) : loc[n], Val(3))
end

"""
    $FUNCTIONNAME(m, i, j, k, offset)

Shift indices `(i, j, k)` by `offset` cells along dimension `m`.
"""
function shift_index end

@inline shift_index(::Val{1}, i, j, k, offset) = (i + offset, j, k)
@inline shift_index(::Val{2}, i, j, k, offset) = (i, j + offset, k)
@inline shift_index(::Val{3}, i, j, k, offset) = (i, j, k + offset)

"""
    $FUNCTIONNAME(m, i, j, k)

The two indices, in (near, far) order, of the flipped-location evaluation points
that bound a field's own location at indices `(i, j, k)` in dimension `m`. 
Assumes `loc` is the *output* (field's own) location in dimension `m`.
"""
function bounding_pair end

@inline bounding_pair(m, ::Type{Center}, i, j, k) = ((i, j, k), shift_index(m, i, j, k, 1))
@inline bounding_pair(m, ::Type{Face}, i, j, k) = (shift_index(m, i, j, k, -1), (i, j, k))
