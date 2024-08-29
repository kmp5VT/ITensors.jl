using ITensors: Index

abstract type mttkrp_algorithm end
struct KRP<:mttkrp_algorithm
end

## This version assumes we have the exact target and can form the tensor
function mttkrp(::KRP, factors, target, rank::Index, fact::Int)
  ## form the tensor which will be written
  m = similar(factors[fact])

  factor_portion = factors[1:end .!= fact]
  for i in 1:dim(rank)
    array(m)[i,:] = array(target * contract(map(x -> itensor(array(x)[i,:], ind(x, 2)), factor_portion)))
  end
  return m
end


struct direct<:mttkrp_algorithm
end

function mttkrp(::direct, factors, target, rank::Index, fact::Int)
  m = similar(factors[fact])

  factor_portion = factors[1:end .!= fact]
  for i in 1:dim(rank)
    mtkrp = target
    for ten in factor_portion
      mtkrp = itensor(array(ten)[i,:], ind(ten, 2)) * mtkrp
    end
    array(m)[i,:] = data(mtkrp)
  end
  return m
end