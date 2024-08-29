function als_optimize(cp::CPD, rank::Index; maxiters = nothing)
  iter = 0
  maxiters = isnothing(maxiters) ? 1 : maxiters
  part_grammian = cp.factors .* conj(prime.(cp.factors; tags=tags(rank)))
  num_factors = length(cp.factors)
  @show norm(cp.target - reconstruct(cp)) / norm(cp.target)
  λ = copy(cp.λ)
  factors = copy(cp.factors)
  while iter < maxiters
    for fact in 1:num_factors
      ## compute the matrized tensor time khatri rao product with a provided algorithm.
      m = mttkrp(cp.mttkrp_alg, factors, cp.target, rank, fact)
      ## compute the grammian which requires the hadamard product
      grammian = similar(part_grammian[1])
      fill!(grammian, one(eltype(cp)))
      for i in 1:num_factors
        if i == fact
          continue
        end
        grammian = hadamard_product(grammian, part_grammian[i])
      end
      # potentially save the MTTKRP for the loss function
      factors[fact], λ = row_norm(itensor(array(grammian) \ array(m), inds(m)), ind(m, 2))
      part_grammian[fact] = factors[fact] * conj(prime(factors[fact]; tags=tags(rank)))
    end
    if check_converge()
      break
    end
    iter += 1
  end
  @show norm(cp.target - reconstruct(factors, λ)) / norm(cp.target)
  return CPD(cp.target, factors, λ, cp.mttkrp_alg)
end