## TODO still working to make this implementation simplified
struct UnallocatedZeros{ElT,N,Axes,Alloc<:AbstractArray{ElT,N}} <:
       FillArrays.AbstractZeros{ElT,N,Axes}
  z::FillArrays.Zeros{ElT,N,Axes}
  ## TODO use `set_parameters` as constructor to these types
end

parent(Z::UnallocatedZeros) = Z.z

# function complex(z::UnallocatedZeros)
#   ElT = complex(eltype(z))
#   N = ndims(z)
#   AllocT = similartype(alloctype(z), ElT)
#   return UnallocatedZeros{ElT,N,AllocT}(dims(z))
# end