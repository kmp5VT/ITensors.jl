# TypeParameterAccessors definitions
using CUDA: CUDA, CuArray
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
using NDTensors.GPUArraysCoreExtensions: storagemode

function TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(storagemode))
  return Position(3)
end

function TypeParameterAccessors.default_type_parameters(::Type{<:CuArray})
  return (Float64, 1, CUDA.Mem.DeviceBuffer)
end
