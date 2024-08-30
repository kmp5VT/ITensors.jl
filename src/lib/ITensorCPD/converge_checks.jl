using ITensors: hadamard_product!
abstract type ConvergeAlg end

mutable struct NoCheck <: ConvergeAlg
  counter::Int
  max_counter::Int
  fit::Number

  NoCheck(max) = new(0, max, -1)
end

function check_converge(check::NoCheck, ::CPD, partial_gram)
  if check.counter ≥ check.max_counter
    return true
  end
  check.counter += 1
  return false
end

function save_mttkrp(::ConvergeAlg, ::ITensor)
end

mutable struct FitCheck <: ConvergeAlg
  iter::Int
  counter::Int
  tolerance::Number
  max_counter::Int
  ref_norm::Number
  MttKRP::ITensor
  fit::Number
  final_fit::Number 

  FitCheck(tol, max, norm) = new(0, 0, tol, max, norm, ITensor(), 1, 0)
end

save_mttkrp(fit::FitCheck, mttkrp::ITensor) = fit.MttKRP = mttkrp

function check_converge(check::FitCheck, cp::CPD, partial_gram; verbose = true)
  check.iter += 1
  rank = ind(cp.λ, 1)
  # inner_prod = (check.MttKRP * conj(prime(cp.factors[end]; tags=tags(rank))) * (cp.λ * δ(rank')))[]
  inner_prod = sum(hadamard_product(check.MttKRP, cp.factors[end]) * cp.λ)
  fact_square = norm_factors(partial_gram, cp.λ)
  normResidual = sqrt(abs(check.ref_norm * check.ref_norm + fact_square - 2 * abs(inner_prod)));
  curr_fit = 1. - (normResidual / check.ref_norm);
  Δfit = abs(check.fit - curr_fit)
  check.fit = curr_fit

  if (verbose) 
    println("$(dim(rank))\t $(check.iter) \t $(curr_fit) \t $(Δfit)")
  end

  if Δfit < check.tolerance
    check.counter += 1
    if check.counter >= 2
      check.iter = 0
      check.counter = 0
      check.final_fit = check.fit
      check.fit = 0
      return true
    end
  else
    check.counter = 0
  end
  return false
end

function norm_factors(partial_gram::Vector, λ::ITensor)
  had = copy(partial_gram[1])
  for i in 2:length(partial_gram)
    hadamard_product!(had, had, partial_gram[i])
  end
  return (had * (λ * prime(λ)))[]
end