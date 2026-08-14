using RandomFields
using Oceananigans
using Random
using Test
using Aqua

const RANDOM_SEED = 8095841277631034267
const DEFAULT_DIMENSION_EXTENTS = (;
    x=(0.0, 1.0), y=(0.0, 1.0), z=(-1.0, 0.0), longitude=(0.0, 30.0), latitude=(0.0, 30.0)
)

dimension_symbols(::Type{RectilinearGrid}) = (:x, :y, :z)
dimension_symbols(::Type{LatitudeLongitudeGrid}) = (:longitude, :latitude, :z)

function make_test_grid(
    grid_type, dimension, float_type, non_flat_topology=Bounded, halo_size=4, grid_size=8
)
    size = Tuple(grid_size for _ in 1:dimension)
    topology = Tuple(d > dimension ? Flat : non_flat_topology for d in 1:3)
    halo = Tuple(halo_size for _ in 1:dimension)
    extents = NamedTuple(
        k => DEFAULT_DIMENSION_EXTENTS[k] for k in dimension_symbols(grid_type)[1:dimension]
    )
    return grid_type(CPU(), float_type; extents..., size, topology, halo)
end

function make_test_grids(
    dimensions=1:3,
    grid_types=(RectilinearGrid, LatitudeLongitudeGrid),
    float_types=(Float32, Float64),
)
    (
        make_test_grid(g, d, f) for d in dimensions, g in grid_types, f in float_types if
        # Skip one-dimensional lat-lon grid as some operators appear to not be defined
        g !== LatitudeLongitudeGrid || d >= 2
    )
end

@testset "RandomFields.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(RandomFields)
    end
    @testset "generate! on grid $(summary(grid))" for grid in make_test_grids()
        rng = Xoshiro(RANDOM_SEED)
        field = CenterField(grid)
        v = randn(rng, size(field))
        dimension = RandomFields.dimension_count(grid)
        parameters = IsotropicMatern(
            length_scale=1.0, output_scale=1.0, smoothness=(dimension / 2) + 1
        )
        workspace = GRFWorkspace(field, parameters)
        generate!(field, workspace, v)
        @test any(field .!= 0)
        field_2 = CenterField(grid)
        generate!(field_2, workspace, v)
        @test all(field .== field_2)
    end
end
