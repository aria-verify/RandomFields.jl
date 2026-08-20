@kernel function _isotropic_modified_helmholtz_operator_kernel!(
    result, operand, grid, shift_coefficient, weights, laplacian
)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] =
        shift_coefficient * operand[i, j, k] -
        weights[1] * laplacian(i, j, k, grid, operand)
end

@kernel function _anisotropic_modified_helmholtz_operator_kernel!(
    result, operand, grid, shift_coefficient, weights, operators
)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] =
        shift_coefficient * operand[i, j, k] -
        weighted_second_derivative_sum(i, j, k, grid, operand, weights, operators)
end

@kernel function _discretize_white_noise_kernel!(
    result, standard_normal_noise, white_noise_scale
)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = standard_normal_noise[i, j, k] * white_noise_scale[i, j, k]
end

@kernel function _accumulate_weighted_kernel!(accumulator, addend, weight)
    i, j, k = @index(Global, NTuple)
    @inbounds accumulator[i, j, k] += weight * addend[i, j, k]
end

@kernel function _white_noise_scale_kernel!(out, grid, scale, volume_reciprocal)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = scale * sqrt(volume_reciprocal(i, j, k, grid))
end

function run_kernel!(kernel, grid, args...; kwargs...)
    launch!(architecture(grid), grid, :xyz, kernel, args...; kwargs...)
    return nothing
end

function accumulate_weighted!(acc, addend, w)
    (run_kernel!(_accumulate_weighted_kernel!, acc.grid, acc, addend, w); acc)
end