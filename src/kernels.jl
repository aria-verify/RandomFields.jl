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
    result, grid, white_noise_scale, standard_normal_noise
)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = if is_immersed_cell(i, j, k, grid)
        zero(eltype(result))
    else
        standard_normal_noise[i, j, k] * white_noise_scale[i, j, k]
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

@kernel function _white_noise_scale_kernel!(out, grid, scale, volume_reciprocal)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = scale * sqrt(volume_reciprocal(i, j, k, grid))
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
