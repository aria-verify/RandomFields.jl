using Oceananigans: architecture
using Oceananigans.Utils: get_active_cells_map, launch!
using KernelAbstractions: @index, @kernel

@kernel function _isotropic_modified_helmholtz_operator_kernel!(
    result, operand, grid, shift, weights, laplacian
)
    i, j, k = @index(Global, NTuple)
    T = eltype(result)
    @inbounds result[i, j, k] =
        (one(T) + T(shift)) * operand[i, j, k] -
        weights[1] * laplacian(i, j, k, grid, operand)
end

@kernel function _anisotropic_modified_helmholtz_operator_kernel!(
    result, operand, grid, shift, weights, operators
)
    i, j, k = @index(Global, NTuple)
    T = eltype(result)
    @inbounds result[i, j, k] =
        (one(T) + T(shift)) * operand[i, j, k] -
        weighted_second_derivative_sum(i, j, k, grid, operand, weights, operators)
end

@kernel function _accumulate_weighted_kernel!(accumulator, addend, weight)
    i, j, k = @index(Global, NTuple)
    @inbounds accumulator[i, j, k] += weight * addend[i, j, k]
end

@kernel function _compute_sqrt_cell_volumes!(out, grid, volume)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = sqrt(volume(i, j, k, grid))
end

@kernel function _mul_by_sqrt_cell_volumes_kernel!(result, operand, sqrt_cell_volumes)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = operand[i, j, k] * sqrt_cell_volumes[i, j, k]
end

@kernel function _div_by_sqrt_cell_volumes_kernel!(result, operand, sqrt_cell_volumes)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = operand[i, j, k] / sqrt_cell_volumes[i, j, k]
end

@kernel function _scale_noise_kernel!(result, noise, scale)
    i, j, k = @index(Global, NTuple)
    @inbounds result[i, j, k] = noise[i, j, k] * scale
end

function run_kernel!(kernel, grid, args...; kwargs...)
    launch!(architecture(grid), grid, :xyz, kernel, args...; kwargs...)
    return nothing
end
