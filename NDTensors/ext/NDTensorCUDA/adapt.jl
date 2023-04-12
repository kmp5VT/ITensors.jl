#
# Used to adapt `EmptyStorage` types
#

struct NDTensorCuArrayAdaptor{B} end
## TODO make this work for unified. This works but overwrites CUDA's adapt_storage. This fails for emptystorage...
@inline function NDTensors.cu(xs; unified::Bool=false)
  return fmap(
    x -> adapt(NDTensorCuArrayAdaptor{unified ? Mem.UnifiedBuffer : Mem.DeviceBuffer}(), x),
    xs,
  )
end

function Adapt.adapt_storage(
  ::NDTensorCuArrayAdaptor{B}, xs::AbstractArray{T,N}
) where {T,N,B}
  return isbits(xs) ? xs : CuArray{T,N,B}(xs)
end

function NDTensors.adapt_storagetype(
  ::NDTensorCuArrayAdaptor{B}, xs::Type{EmptyStorage{ElT,StoreT}}
) where {ElT,StoreT,B}
  return NDTensors.emptytype(NDTensors.adapt_storagetype(CuVector{ElT,B}, StoreT))
end
