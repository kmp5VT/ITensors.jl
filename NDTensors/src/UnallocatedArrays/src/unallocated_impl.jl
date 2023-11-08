for Typ in (:UnallocatedFill, :UnallocatedZeros)
  ## Here are functions specifically defined for UnallocatedArrays
  ## not implemented by FillArrays
  ## TODO use set_parameters/get_parameters instead of alloctype and other 
  ## type info functions.
  ## TODO determine min number of functions needed to be forwarded
  @eval begin
    alloctype(A::$Typ) = alloctype(typeof(A))
    alloctype(Atype::Type{<:$Typ}) = get_parameter(Atype, Position{4}())

    ## something is wrong when this function is
    ## called an error is thrown.
    # function getindex(A::$Typ, i...) 
    #   parent(A)[i...]
    # end

    Array(A::$Typ) = alloctype(typeof(A))

    copy(A::$Typ) = A
    ## TODO Implement vec
    # Base.vec(Z::UnallocatedZeros) = typeof(Z)(length(Z))
  end

  ## TODO forwarding functions to fillarrays
  for fun in (:size, :length, :convert, :sum, :getindex_value)
    @eval FillArrays.$fun(A::$Typ) = $fun(parent(A))
  end
  ## TODO Here I am defining LinearAlgebra functions in one sweep
  for fun in (:norm,)
    @eval LinearAlgebra.$fun(A::$Typ) = $fun(parent(A))
  end
end
