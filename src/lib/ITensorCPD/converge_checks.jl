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
  counter::Int
  max_counter::Int
  ref_norm::Number
  MttKRP::ITensor
  fit::Number

  FitCheck(max, norm) = new(0, max, norm, ITensor(), 1)
end

save_mttkrp(fit::FitCheck, mttkrp::ITensor) = fit.MttKRP = mttkrp

function check_converge(check::FitCheck, cp::CPD, partial_gram)

end