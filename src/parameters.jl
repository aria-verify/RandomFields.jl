function check_matern_parameters(length_scale, output_scale, smoothness)
    all(>(0), length_scale) ||
        throw(ArgumentError("length_scale (entries) must be positive"))
    output_scale > 0 || throw(ArgumentError("output_scale must be positive"))
    smoothness > 0 || throw(ArgumentError("smoothness must be positive"))
    isinteger(2 * smoothness) ||
        throw(ArgumentError("smoothness must be integer or half-integer"))
end

abstract type AbstractMaternParameters{T} end

"""
Parameters for an isotropic Matérn covariance function with a single length-scale parameter.
"""
struct IsotropicMatern{T} <: AbstractMaternParameters{T}
    "Scalar length scale parameter."
    length_scale::T
    "Scalar output scale parameter."
    output_scale::T
    "Scalar smoothness index - should be integer or half-integer."
    smoothness::T
    function IsotropicMatern{T}(length_scale::T, output_scale::T, smoothness::T) where {T}
        check_matern_parameters(length_scale, output_scale, smoothness)
        return new(length_scale, output_scale, smoothness)
    end
end

function IsotropicMatern(; length_scale::T, output_scale::T, smoothness::T) where {T}
    IsotropicMatern{T}(length_scale, output_scale, smoothness)
end

"""
Parameters for an anisotropic Matérn covariance function with a per-dimension length-scale parameters.
"""
struct AnisotropicMatern{T,N} <: AbstractMaternParameters{T}
    "Tuple of length scale parameters, one per active dimension."
    length_scale::NTuple{N,T}
    "Scalar output scale parameter."
    output_scale::T
    "Scalar smoothness index - should be integer or half-integer."
    smoothness::T
    function AnisotropicMatern{T,N}(
        length_scale::NTuple{N,T}, output_scale::T, smoothness::T
    ) where {T,N}
        check_matern_parameters(length_scale, output_scale, smoothness)
        return new(length_scale, output_scale, smoothness)
    end
end

function AnisotropicMatern(;
    length_scale::NTuple{N,T}, output_scale::T, smoothness::T
) where {T,N}
    AnisotropicMatern{T,N}(length_scale, output_scale, smoothness)
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

function variance_matching_constant(
    parameters::AbstractMaternParameters{T}, alpha::Int, dimension::Int
) where {T}
    return T(
        sqrt(
            parameters.output_scale^2 *
            length_scale_product(parameters, dimension) *
            gamma(alpha) *
            (4π)^(dimension / 2) / gamma(parameters.smoothness),
        ),
    )
end

derivative_weights(p::IsotropicMatern, dimension) = ntuple(_ -> p.length_scale^2, dimension)
derivative_weights(p::AnisotropicMatern, dimension) = ntuple(n -> p.length_scale[n]^2, dimension)

function matern_spde_alpha(smoothness, dimension)
    two_alpha = round(Int, 2 * smoothness) + dimension
    two_alpha > 0 || throw(ArgumentError("smoothness + dimension/2 must be positive"))
    alpha = two_alpha / 2
    isinteger(alpha) || throw(ArgumentError("smoothness + dimension/2 must be integer"))
    return round(Int, alpha)
end