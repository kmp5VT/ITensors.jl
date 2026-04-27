using Base: ReshapedArray
using NDTensors.Expose: expose
using NDTensors: NDTensors, BlockSparseTensor, ContractAlgorithm, Dense, DenseTensor,
    NativeContract, TensorAndContractionPlan, array, blockdim, blockdims, contract!, data,
    datatype, default_contract_algorithm, eachnzblock, inds, is_applicable, nblocks,
    nzblocks, with_contract_algorithm
using cuTENSOR: cuTENSOR, CuArray, CuTensor

# ============================================================
# cuTENSOR block-sparse contraction
# ============================================================

"""
    cuTENSORBlockSparse <: NDTensors.ContractAlgorithm

Algorithm tag for cuTENSOR's batched block-sparse contraction path.
Applies to two `BlockSparseTensor`s whose backing data type is a
`CuArray`. Selected via [`with_cutensorblocksparse`](@ref).
"""
struct cuTENSORBlockSparse <: ContractAlgorithm end

NDTensors.is_applicable(::cuTENSORBlockSparse, ::Type, ::Type) = false
function NDTensors.is_applicable(
        ::cuTENSORBlockSparse,
        T1::Type{<:BlockSparseTensor},
        T2::Type{<:BlockSparseTensor}
    )
    return datatype(T1) <: CuArray && datatype(T2) <: CuArray
end

function NDTensors.contract!(
        ::cuTENSORBlockSparse,
        dest::TensorAndContractionPlan{T},
        labelsR,
        tensor1::BlockSparseTensor,
        labelstensor1,
        tensor2::BlockSparseTensor,
        labelstensor2,
        α::Number = one(eltype(dest.tensor)),
        β::Number = zero(eltype(dest.tensor))
    ) where {T <: BlockSparseTensor}
    R = dest.tensor
    cuR = to_cuTensorBS(R, labelsR)
    cuT1 = to_cuTensorBS(tensor1, labelstensor1)
    cuT2 = to_cuTensorBS(tensor2, labelstensor2)
    try
        cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
    catch e
        e isa cuTENSOR.CUTENSORError || rethrow()
        # cuTENSOR couldn't run this contraction (typically an unsupported
        # operation). Surface what was suppressed, then fall back to the
        # native block-sparse path. Non-CUTENSORError exceptions (OOM,
        # driver mismatch, version regression, internal bugs) propagate.
        @warn "cuTENSOR block-sparse contract failed; falling back to NativeContract." exception =
            (
            e,
            catch_backtrace(),
        )
        return contract!(
            NativeContract(),
            dest,
            labelsR,
            tensor1,
            labelstensor1,
            tensor2,
            labelstensor2,
            α,
            β
        )::T
    end
    return R::T
end

"""
    with_cutensorblocksparse(f, enable::Bool = true)

Run `f()` with cuTENSOR's batched block-sparse contraction backend
preferred for any contractions inside the block. When `enable = false`,
runs `f()` unchanged.

Equivalent to `with_contract_algorithm(f, cuTENSORBlockSparse())`. The
preference applies only when [`is_applicable`](@ref NDTensors.is_applicable)
returns `true` for the inputs at hand (CUDA-backed `BlockSparseTensor`s);
other contractions inside the scope fall through to the default.
"""
function with_cutensorblocksparse(f, enable::Bool = true)
    return enable ? with_contract_algorithm(f, cuTENSORBlockSparse()) : f()
end

# Build a cuTENSOR.CuTensorBS from a NDTensors BlockSparseTensor, with
# `labels` becoming the cuTENSOR mode labels at construction time.
function to_cuTensorBS(T::BlockSparseTensor, labels)
    blocks = map(eachnzblock(T)) do b
        offset = NDTensors.offset(T, b)
        len = prod(blockdims(T, b))
        return @view data(T)[(offset + 1):(offset + len)]
    end
    block_extents = [[blockdim(idx, i) for i in 1:nblocks(idx)] for idx in inds(T)]
    nzblock_coords = [Int64.(x.data) for x in nzblocks(T)]
    block_per_mode = length.(block_extents)
    return cuTENSOR.CuTensorBS(
        blocks, block_per_mode, block_extents, nzblock_coords, collect(labels)
    )
end

# ============================================================
# cuTENSOR dense contraction
# ============================================================

"""
    cuTENSORDense <: NDTensors.ContractAlgorithm

Algorithm tag for cuTENSOR's dense contraction path. Applies to two
`DenseTensor`s whose backing data type is a `CuArray`. Set as the
default for that input shape when this extension is loaded (via the
`default_contract_algorithm` overload below), so dense CUDA contractions
automatically use cuTENSOR.
"""
struct cuTENSORDense <: ContractAlgorithm end

NDTensors.is_applicable(::cuTENSORDense, ::Type, ::Type) = false
function NDTensors.is_applicable(
        ::cuTENSORDense,
        T1::Type{<:DenseTensor{<:Any, <:Any, <:Dense{<:Any, <:CuArray}}},
        T2::Type{<:DenseTensor{<:Any, <:Any, <:Dense{<:Any, <:CuArray}}}
    )
    return true
end

# Loading this extension makes `cuTENSORDense` the default for dense CUDA
# contractions (matching the behavior of the previous `Exposed{<:CuArray,
# <:DenseTensor}` direct-dispatch method).
function NDTensors.default_contract_algorithm(
        ::Type{<:DenseTensor{<:Any, <:Any, <:Dense{<:Any, <:CuArray}}},
        ::Type{<:DenseTensor{<:Any, <:Any, <:Dense{<:Any, <:CuArray}}}
    )
    return cuTENSORDense()
end

# Handle CuArrays cuTENSOR.jl can't accept directly (non-zero offsets,
# reshaped views).
to_zero_offset_cuarray(a::CuArray) = iszero(a.offset) ? a : copy(a)
to_zero_offset_cuarray(a::ReshapedArray) = copy(expose(a))

function NDTensors.contract!(
        ::cuTENSORDense,
        R::DenseTensor,
        labelsR,
        T1::DenseTensor,
        labelsT1,
        T2::DenseTensor,
        labelsT2,
        α::Number = one(eltype(R)),
        β::Number = zero(eltype(R))
    )
    zoffR = iszero(array(R).offset)
    arrayR = zoffR ? array(R) : copy(array(R))
    arrayT1 = to_zero_offset_cuarray(array(T1))
    arrayT2 = to_zero_offset_cuarray(array(T2))
    # Promote inputs to a common type. cuTENSOR contraction only performs
    # limited promotions of input element types, see e.g.
    # https://github.com/JuliaGPU/CUDA.jl/blob/v5.4.2/lib/cutensor/src/types.jl#L11-L19
    elt = promote_type(eltype.((arrayR, arrayT1, arrayT2))...)
    if elt !== eltype(arrayR)
        return error(
            "In cuTENSOR contraction, input tensors have element types `$(eltype(arrayT1))` and `$(eltype(arrayT2))` while the output has element type `$(eltype(arrayR))`."
        )
    end
    arrayT1 = convert(CuArray{elt}, arrayT1)
    arrayT2 = convert(CuArray{elt}, arrayT2)
    cuR = CuTensor(arrayR, collect(labelsR))
    cuT1 = CuTensor(arrayT1, collect(labelsT1))
    cuT2 = CuTensor(arrayT2, collect(labelsT2))
    try
        cuTENSOR.mul!(cuR, cuT1, cuT2, α, β)
    catch e
        e isa cuTENSOR.CUTENSORError || rethrow()
        # cuTENSOR couldn't run this contraction (typically an unsupported
        # operation). Surface what was suppressed, then fall back to the
        # native (cuBLAS-loop) path. Non-CUTENSORError exceptions (OOM,
        # driver mismatch, version regression, internal bugs) propagate.
        @warn "cuTENSOR dense contract failed; falling back to NativeContract." exception =
            (
            e,
            catch_backtrace(),
        )
        contract!(NativeContract(), R, labelsR, T1, labelsT1, T2, labelsT2, α, β)
        return R
    end
    if !zoffR
        array(R) .= cuR.data
    end
    return R
end
