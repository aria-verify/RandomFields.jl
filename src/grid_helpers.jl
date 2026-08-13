# --- metric separability trait ------------------------------------------------

abstract type MetricSeparability end

struct SeparableMetrics <: MetricSeparability end
struct NonseparableMetrics <: MetricSeparability end

metric_separability(::RectilinearGrid) = SeparableMetrics()
metric_separability(::LatitudeLongitudeGrid) = NonseparableMetrics()
metric_separability(grid::ImmersedBoundaryGrid) = metric_separability(grid.underlying_grid)
metric_separability(::Oceananigans.Grids.AbstractGrid) = NonseparableMetrics()  # safe default for other grids

is_immersed_grid(::Oceananigans.Grids.AbstractGrid) = false
is_immersed_grid(::ImmersedBoundaryGrid) = true

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

flip_location_in_dimension(Loc, m) = ntuple(n -> n == m ? flip_location(Loc[n]) : Loc[n], 3)

# index shift by `offset` cells along dimension `m`
@inline shift_index(m, i, j, k, offset) =
    if m == 1
        (i + offset, j, k)
    elseif m == 2
        (i, j + offset, k)
    else
        (i, j, k + offset)
    end

# the two indices, in (near, far) order, of the flipped-location evaluation points
# that bound `field`'s own location at index i in dimension m. Assumes `Loc_m` is
# the *output* (field's own) location in dimension m.
@inline function bounding_pair(m, Loc_m, i, j, k)
    if Loc_m == Center
        (shift_index(m, i, j, k, 0), shift_index(m, i, j, k, 1))
    else
        (shift_index(m, i, j, k, -1), shift_index(m, i, j, k, 0))
    end
end
