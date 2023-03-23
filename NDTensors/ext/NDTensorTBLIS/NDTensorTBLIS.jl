module NDTensorTBLIS

using NDTensors
using LinearAlgebra
isdefined(Base, :get_extension) ? (using TBLIS) : (using ..TBLIS)

import NDTensors.contract!

include("contract.jl")
end