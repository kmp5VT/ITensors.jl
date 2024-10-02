using ITensors: Index
using ITensors.NDTensors: data

abstract type MttkrpAlgorithm end
struct KRP <: MttkrpAlgorithm end

function post_solve(::MttkrpAlgorithm, factors, λ, cp, rank::Index, fact::Integer) end

## This version assumes we have the exact target and can form the tensor
## This forms the khatri-rao product for a single value of r and immediately
## contracts it with the target tensor. This is relatively expensive because the KRP will be
## order $d - 1$ where d is the number of modes in the target tensor.
## This process could be distributed.
function mttkrp(::KRP, factors, cp, rank::Index, fact::Int)
  ## form the tensor which will be written
  m = similar(factors[fact])

  factor_portion = factors[1:end .!= fact]
  for i in 1:dim(rank)
    array(m)[i, :] = array(
      cp.target * contract(map(x -> itensor(array(x)[i, :], ind(x, 2)), factor_portion))
    )
  end
  return m
end

struct direct <: MttkrpAlgorithm end

## This code skips computing the khatri-rao product by incrementally 
## contracting the factor matrices into the tensor for each value of r
## This process could be distributed.
function mttkrp(::direct, factors, cp, rank::Index, fact::Int)
  m = similar(factors[fact])

  factor_portion = factors[1:end .!= fact]
  for i in 1:dim(rank)
    mtkrp = cp.target
    for ten in factor_portion
      mtkrp = itensor(array(ten)[i, :], ind(ten, 2)) * mtkrp
    end
    array(m)[i, :] = data(mtkrp)
  end
  return m
end

struct square_lattice <: MttkrpAlgorithm end

## In this one we have essentially contracted all of the factors in
## and that is stored in cp.additional_items[:partial_mtkrp]
## it looks like this for the square lattice
##   r     r
##   |  x  | 
##   1 --- 2 
##  w|    z| 
##   4 --- 3 
##   |  y  | 
##   r     r
## Where r is the hyperedge. We can choose to contract all of these except the \c fact
## To form the environment for each r. Then we just need to contract in the 
##   b     r              
##   |  x  |              f ------- r
## a-1 --- 2            b |  
##  w|    z|   -->    a - 1 --- E - r
##   4 --- 3                w,x
##   |  y  | 
##   r     r
## where E is the contracted environement for each r. Then we will contract the bth factor matrix in

function mttkrp(::square_lattice, factors, cp, rank::Index, fact::Int)
  m = similar(factors[fact])
  fact_even = iseven(fact)
  other = fact_even ? fact - 1 : fact + 1
  effective_fact = fact_even ? fact ÷ 2 : fact ÷ 2 + 1

  ## This forms the second factor in the pair times the 
  p = factors[other] * cp.target[effective_fact]

  factor_portion = cp.additional_items[:partial_mtkrp][1:end .!= effective_fact]
  for i in 1:dim(rank)
    E = contract(map(x -> itensor(array(x)[i, :, :], inds(x)[2:end]), factor_portion))
    array(m)[i, :] = array(itensor(array(p)[i, :, :, :], inds(p)[2:end]) * E)
  end

  return m
end

function post_solve(::square_lattice, factors, λ, cp, rank::Index, fact::Integer)
  ## Once done with 2 factor, multiply the two into the target and set in the partial_mtkrp list
  if iseven(fact)
    effective_fact = fact ÷ 2
    for i in 1:dim(rank)
      array(cp.additional_items[:partial_mtkrp][effective_fact])[i, :, :] = array(
        itensor(array(factors[fact])[i, :], ind(factors[fact], 2)) * (
          itensor(array(factors[fact - 1])[i, :], ind(factors[fact - 1], 2)) *
          cp.target[effective_fact]
        ),
      )
    end
  end
end

struct network_solver <: MttkrpAlgorithm end
