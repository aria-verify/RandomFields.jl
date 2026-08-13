@kernel function _fused_isotropic_operator_kernel!(
    result, grid, shift_coefficient, length_scale_squared, laplacian, u
)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] =
        shift_coefficient * u[i, j, k] - length_scale_squared * laplacian(i, j, k, grid, u)
end

@kernel function _separable_operator_kernel!(
    result, grid, shift_coefficient, weights, operators, location, u
)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] =
        shift_coefficient * u[i, j, k] - separable_weighted_second_derivative_sum(
            i, j, k, grid, u, weights, operators, location
        )
end

@kernel function _nonseparable_operator_kernel!(
    result, grid, shift_coefficient, weights, operators, location, volume, u
)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] =
        shift_coefficient * u[i, j, k] -
        nonseparable_weighted_flux_sum(i, j, k, grid, u, weights, operators, location) /
        volume(i, j, k, grid)
end

@kernel function _discretize_white_noise_kernel!(rhs, grid, scale, inverse_sqrt_volume, v)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = if is_immersed_cell(i, j, k, grid)
        zero(eltype(rhs))
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
