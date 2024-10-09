using ITensorNetworks: ITensorNetwork, vertices

function tn_cp_contract(tn::ITensorNetwork, cp::CPD)
  tnp = copy(tn)
  r = ind(cp[],1)
  ## Go through all the nodes
  contracted_cps = Int[]
  for v in vertices(tnp)
    ## For each node find the indices that are not connected to other tensors in the node
    iss = uniqueinds(tnp, v)
    for is in iss
      cp_pos = findfirst(x -> x == is, ind.(cp.factors, 2))
      @show cp_pos
      isnothing(cp_pos) && continue
      tnp[v] = ITensorCPD.had_contract(cp[cp_pos], tnp[v], r)
      push!(contracted_cps, cp_pos)
    end
  end
  v = [cp[x] for x in contracted_cps]
  return tnp, filter!(x -> x ∉ v, cp.factors)
end

function cp_cp_contract(cp1::CPD, cp2::CPD)
  r1 = ind(cp1[], 1)
  r2 = ind(cp2[], 1)
  for c1 in cp1.factors
    pos = findfirst(x -> x == ind(c1,2), ind.(cp2.factors, 2))
    @show pos
  end
end