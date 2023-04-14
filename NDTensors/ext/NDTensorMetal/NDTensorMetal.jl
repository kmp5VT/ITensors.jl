module NDTensorMetal

using NDTensors
using Functors
using Adapt

if isdefined(Base, :get_extension)
  using Metal
else
  using ..Metal
end

include("imports.jl")

include("adapt.jl")
include("set_types.jl")
include("array.jl")
end