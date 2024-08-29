using ITensors
using Random

struct CPD
  target
  factors::Vector{ITensor}
  λ::ITensor
  ## potentially this could be a dict of 
  ## extra arguments which could be queried 
  mttkrp_alg::mttkrp_algorithm
end

CPD(target, factors, λ) = CPD(target, factors, λ, KRP())

Base.eltype(cp::CPD) = return eltype(cp.λ)

## Right now this only works for a single itensor.
## However, it should be possible to make it for a arbitrary tensor network.
function random_CPD(target::ITensor, rank::Index; algorithm = nothing)
  rng = MersenneTwister(3);
  elt = eltype(target)
  cp = Vector{ITensor}([])
  l = nothing

  for i in inds(target)
    rtensor, l = row_norm(random_itensor(rng, elt, rank, i), i)
    push!(cp, rtensor)
  end
  return isnothing(algorithm) ? CPD(target, cp, l) : CPD(target, cp, l, algorithm)
end

## For this one, we assume there is one loop and 
## each tensor at index i is only connected to the tensor at index i+1 and 
## i -1. if i - 1 == 0, then the tensor is at the end of the list
##   b     c
##   |  x  | 
## a-1 --- 2 - d
##  w|    z| 
## e-4 --- 3 - h
##   |  y  | 
##   f     g
function random_CPD_square_network(target::Vector{ITensor}, rank::Index)
  rng = MersenneTwister(3);
  elt = eltype(target[1])
  cp = Vector{ITensor}([])
  num_tensors = length(target)
  l = nothing
  is = uniqueinds(target[1], target[2], target[num_tensors])
  for i in is
    push!(cp, column_norm(random_itensor(rng, elt, (rank, i)), i)[1])
  end

  for ten in 2:(num_tensors - 1)
    is = uniqueinds(target[ten], target[ten - 1], target[ten + 1])
    for i in is
      push!(cp, column_norm(random_itensor(rng, elt, rank, i), i)[1])
    end
  end

  is = uniqueinds(target[num_tensors], target[num_tensors - 1], target[1])
  for i in is
    ten, l = column_norm(random_itensor(rng, elt, rank, i), i)
    push!(cp, ten)
  end

  return CPD(target, cp, l)
end
