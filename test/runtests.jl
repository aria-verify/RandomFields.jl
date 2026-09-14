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
    return (
        make_test_grid(g, d, f) for d in dimensions, g in grid_types, f in float_types if
        # Skip one-dimensional lat-lon grid as some operators appear to not be defined
        g !== LatitudeLongitudeGrid || d >= 2
    )
end

function make_test_parameters(dimension, T)
    return (
        IsotropicMatern(;
            length_scale=one(T), output_scale=one(T), smoothness=T((dimension / 2) + 1)
        ),
        AnisotropicMatern(;
            length_scale=ntuple(_ -> one(T), dimension),
            output_scale=one(T),
            smoothness=T((dimension / 2) + 1),
        ),
    )
end

@testset "RandomFields.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(RandomFields)
    end
    @testset "$(
        "generate! on grid $(summary(grid)) with parameters $(parameters)"
    )" for grid in make_test_grids(),
        parameters in make_test_parameters(RandomFields.dimension_count(grid), eltype(grid))

        rng = Xoshiro(RANDOM_SEED)
        field = CenterField(grid)
        noise = randn(rng, size(field))
        generator = RandomFieldGenerator(field, parameters)
        generate!(field, generator, noise)
        @test any(field .!= 0)
        field_2 = CenterField(grid)
        generate!(field_2, generator, noise)
        @test all(field .== field_2)
    end
    @testset "$(
        "Solver $(solver_type) on grid $(summary(grid)) with parameters $(parameters) inverts apply!"
    )" for solver_type in (CGSolver, SparseSolver),
        grid in make_test_grids(),
        parameters in make_test_parameters(RandomFields.dimension_count(grid), eltype(grid))

        T = eltype(grid)
        rng = Xoshiro(RANDOM_SEED)
        field = CenterField(grid)
        noise = randn(rng, size(field))
        solver_kwargs = if (solver_type <: RandomFields.AbstractIterativeSolver)
            (; reltol=sqrt(eps(T)))
        else
            (;)
        end
        generator = RandomFieldGenerator(field, parameters; solver_type, solver_kwargs...)
        white_noise, reconstructed_white_noise = similar(field), similar(field)
        RandomFields.generate_white_noise!(white_noise, noise, generator.white_noise_scale)
        RandomFields.apply_inverse!(field, white_noise, generator.solver)
        RandomFields.apply!(
            reconstructed_white_noise,
            field,
            generator.modified_helmholtz_operator,
            grid,
            1.0,
            generator.weights,
        )
        tolerance = 1000 * (
            if solver_type <: RandomFields.AbstractIterativeSolver
                solver_kwargs.reltol
            else
                eps(T)
            end
        )
        @test maximum(abs, white_noise - reconstructed_white_noise) /
              maximum(abs, white_noise) < tolerance
    end
end
