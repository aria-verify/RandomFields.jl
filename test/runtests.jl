using RandomFields
using Test
using Aqua


@testset "RandomFields.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(RandomFields)
    end
    @testset "generate! on grid $(summary(grid))" for grid in make_test_grids()
end
