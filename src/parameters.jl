abstract type MaternParameters end

struct IsotropicMatern{T<:Real} <: MaternParameters
    length_scale::T
    output_scale::T
    smoothness::T
end

function IsotropicMatern(; length_scale, output_scale, smoothness)
    length_scale > 0 || throw(ArgumentError("length_scale must be positive"))
    output_scale > 0 || throw(ArgumentError("output_scale must be positive"))
    smoothness > 0 || throw(ArgumentError("smoothness must be positive"))
    T = promote_type(typeof(length_scale), typeof(output_scale), typeof(smoothness))
    return IsotropicMatern{T}(length_scale, output_scale, smoothness)
end

struct AnisotropicMatern{T<:Real,N} <: MaternParameters
    length_scale::NTuple{N,T}
    output_scale::T
    smoothness::T
end

function AnisotropicMatern(length_scale; output_scale, smoothness)
    all(>(0), length_scale) || throw(ArgumentError("length_scale entries must be positive"))
    output_scale > 0 || throw(ArgumentError("output_scale must be positive"))
    smoothness > 0 || throw(ArgumentError("smoothness must be positive"))
    T = promote_type(eltype(length_scale), typeof(output_scale), typeof(smoothness))
    return AnisotropicMatern{T,length(length_scale)}(
        Tuple(T.(length_scale)), T(output_scale), T(smoothness)
    )
end

length_scale_product(p::IsotropicMatern, dimension_count) = p.length_scale^dimension_count
length_scale_product(p::AnisotropicMatern, dimension_count) = prod(p.length_scale)

check_parameter_dimension(::IsotropicMatern, dimension_count) = nothing
function check_parameter_dimension(p::AnisotropicMatern{T,N}, dimension_count) where {T,N}
    N == dimension_count || throw(
        ArgumentError(
            "length_scale has $N entries but grid has $dimension_count active dimensions",
        ),
    )
end

function variance_matching_constant(parameters::MaternParameters, alpha, dimension_count)
    ν = parameters.smoothness
    return sqrt(
        parameters.output_scale^2 *
        length_scale_product(parameters, dimension_count) *
        gamma(alpha) *
        (4π)^(dimension_count / 2) / gamma(ν),
    )
end
