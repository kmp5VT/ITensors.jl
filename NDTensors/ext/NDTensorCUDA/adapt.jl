#
# Used to adapt `EmptyStorage` types
#

to_vector_type(arraytype::Type{CuArray}) = CuVector
to_vector_type(arraytype::Type{CuArray{T}}) where {T} = CuVector{T}

@inline function NDTensors.cu(xs; unified::Bool=false)
  ElT = eltype(xs)
  return NDTensors.adapt_structure(
    CuVector{ElT,(unified ? CUDA.Mem.UnifiedBuffer : CUDA.Mem.DeviceBuffer)}, xs
  )
end