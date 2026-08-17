separable_metrics(::RectilinearGrid) = true
separable_metrics(::LatitudeLongitudeGrid) = false
separable_metrics(grid::ImmersedBoundaryGrid) = metric_separability(grid.underlying_grid)
separable_metrics(::Oceananigans.Grids.AbstractGrid) = false  # safe default for other grids

is_immersed_cell(i, j, k, grid) = false
function is_immersed_cell(i, j, k, grid::ImmersedBoundaryGrid)
    Oceananigans.ImmersedBoundaries.immersed_cell(i, j, k, grid)
end

# --- active dimensions ----------------------------------------------------------

active_dimensions(grid) = ntuple(m -> topology(grid)[m] !== Flat, 3)
dimension_count(grid) = count(active_dimensions(grid))

const DIMENSION_SYMBOLS = (:x, :y, :z)

# --- location <-> built-in operator name plumbing --------------------------------

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

flip_location_in_dimension(loc, m) = ntuple(n -> n == m ? flip_location(loc[n]) : loc[n], 3)

# index shift by `offset` cells along dimension `m`
@inline shift_index(m, i, j, k, offset) =
    if m == 1
        (i + offset, j, k)
    elseif m == 2
        (i, j + offset, k)
    else
        (i, j, k + offset)
    end

"""
    bounding_pair(m, loc, i, j, k)

The two indices, in (near, far) order, of the flipped-location evaluation points
that bound a field's own location at indices `(i, j, k)` in dimension `m`. 
Assumes `loc` is the *output* (field's own) location in dimension `m`.
"""
function bounding_pair end

@inline bounding_pair(m, ::Type{Center}, i, j, k) = (
    shift_index(m, i, j, k, 0), shift_index(m, i, j, k, 1)
)
@inline bounding_pair(m, ::Type{Face}, i, j, k) = (
    shift_index(m, i, j, k, -1), shift_index(m, i, j, k, 0)
)
