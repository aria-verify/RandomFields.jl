using RandomFields
using Test
using Aqua
using JET

@testset "RandomFields.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(RandomFields)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(RandomFields; target_defined_modules = true)
    end
    # Write your tests here.
end
