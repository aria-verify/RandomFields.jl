@kernel function _fused_isotropic_operator_kernel!(
    result, u, grid, location, shift_coefficient, weights, laplacian,
)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] =
        shift_coefficient * u[i, j, k] - weights[1] * laplacian(i, j, k, grid, u)
end

@kernel function _separable_operator_kernel!(
    result, u, grid, location, shift_coefficient, weights, operators,
)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] =
        shift_coefficient * u[i, j, k] - separable_weighted_second_derivative_sum(
            i, j, k, grid, u, weights, operators, location
        )
end

@kernel function _nonseparable_operator_kernel!(
    result, u, grid, location, shift_coefficient, weights, operators, volume_reciprocal,
)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] =
        shift_coefficient * u[i, j, k] -
        nonseparable_weighted_flux_sum(i, j, k, grid, u, weights, operators, location) *
        volume_reciprocal(i, j, k, grid)
end

@kernel function _discretize_white_noise_kernel!(result, grid, scale, inverse_sqrt_volume, v)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = if is_immersed_cell(i, j, k, grid)
        zero(eltype(result))
    else
        scale * v[i, j, k] * inverse_sqrt_volume[i, j, k]
    end
end

@kernel function _mask_immersed_kernel!(field, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds is_immersed_cell(i, j, k, grid) && (field[i, j, k] = 0)
end

@kernel function _accumulate_weighted_kernel!(accumulator, addend, weight)
    i, j, k = @index(Global, NTuple)
    @inbounds accumulator[i, j, k] += weight * addend[i, j, k]
end

@kernel function _zero_kernel!(field)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] = 0
end

@kernel function _copy_masked_kernel!(destination, grid, source)
    i, j, k = @index(Global, NTuple)
    @inbounds destination[i, j, k] =
        is_immersed_cell(i, j, k, grid) ? zero(eltype(destination)) : source[i, j, k]
end

@kernel function _inv_sqrt_volume_kernel!(out, grid, volume_reciprocal)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = sqrt(volume_reciprocal(i, j, k, grid))
end

# thin dispatch wrappers around launch!
function run_kernel!(kernel, grid, args...)
    (launch!(architecture(grid), grid, :xyz, kernel, args...); nothing)
end

zero_field!(field) = (run_kernel!(_zero_kernel!, field.grid, field); field)
function mask_immersed_values!(field)
    (run_kernel!(_mask_immersed_kernel!, field.grid, field, field.grid); field)
end
function accumulate_weighted!(acc, addend, w)
    (run_kernel!(_accumulate_weighted_kernel!, acc.grid, acc, addend, w); acc)
end
function copy_masked_result!(destination, source)
    run_kernel!(
        _copy_masked_kernel!, destination.grid, destination, destination.grid, source
    )
    fill_halo_regions!(destination)
    return destination
end
