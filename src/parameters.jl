function check_matern_parameters_and_get_type(length_scale, output_scale, smoothness)
    all(>(0), length_scale) || throw(ArgumentError("length_scale (entries) must be positive"))
    output_scale > 0 || throw(ArgumentError("output_scale must be positive"))
    smoothness > 0 || throw(ArgumentError("smoothness must be positive"))
    isinteger(2 * smoothness) || throw(ArgumentError("smoothness must be integer or half-integer"))
    promote_type(typeof(length_scale), typeof(output_scale), typeof(smoothness))
end


abstract type MaternParameters end

struct IsotropicMatern{T<:Real} <: MaternParameters
    length_scale::T
    output_scale::T
    smoothness::T
end

function IsotropicMatern(; length_scale, output_scale, smoothness)
    T = check_matern_parameters_and_get_type(length_scale, output_scale, smoothness)
    return IsotropicMatern{T}(length_scale, output_scale, smoothness)
end

struct AnisotropicMatern{T<:Real,N} <: MaternParameters
    length_scale::NTuple{N,T}
    output_scale::T
    smoothness::T
end

function AnisotropicMatern(length_scale; output_scale, smoothness)
    T = check_matern_parameters_and_get_type(length_scale, output_scale, smoothness)
    return AnisotropicMatern{T, length(length_scale)}(
        Tuple(T.(length_scale)), T(output_scale), T(smoothness)
    )
end

length_scale_product(p::IsotropicMatern, dimension) = p.length_scale^dimension
length_scale_product(p::AnisotropicMatern, dimension) = prod(p.length_scale)

check_parameter_dimension(::IsotropicMatern, dimension) = nothing

function check_parameter_dimension(::AnisotropicMatern{T,N}, dimension) where {T,N}
    N == dimension || throw(
        ArgumentError(
            "length_scale has $N entries but grid has $dimension active dimensions"
        ),
    )
end

function variance_matching_constant(parameters::MaternParameters, alpha, dimension)
    return sqrt(
        parameters.output_scale^2 *
        length_scale_product(parameters, dimension) *
        gamma(alpha) *
        (4π)^(dimension / 2) / gamma(parameters.smoothness),
    )
end
