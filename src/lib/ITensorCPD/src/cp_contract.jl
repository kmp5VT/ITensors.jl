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
      #@show cp_pos
      isnothing(cp_pos) && continue
      tnp[v] = ITensorCPD.had_contract(cp[cp_pos], tnp[v], r)
      push!(contracted_cps, cp_pos)
    end
  end
  v = [cp[x] for x in contracted_cps]
  return tnp, filter!(x -> x ∉ v, cp.factors)
end

function cp_cp_contract(cp1::Vector{ITensor}, cp2::Vector{ITensor})
  r1 = ind(cp1[1], 1)
  r2 = ind(cp2[1], 1)
  ## TODO check to see if eltypes are equivalent
  elt = eltype(cp1[1])
  inner = ITensor(elt, r1, r2)
  fill!(inner, one(elt))
  inner_pos_cp1, inner_pos_cp2 = Vector{Int}(), Vector{Int}()
  for i in 1:length(cp1)
    c1 = cp1[i]
    pos = findfirst(x -> x == ind(c1,2), ind.(cp2, 2))
    if isnothing(pos)
      continue
    end
    data(inner) .*= data(c1 * cp2[pos])
    push!(inner_pos_cp1, i)
    push!(inner_pos_cp2, pos)
  end
  v1 = [cp1[x] for x in inner_pos_cp1]
  v2 = [cp2[x] for x in inner_pos_cp2]
  return inner, filter(x -> x ∉ v1, cp1), filter(x -> x ∉ v2, cp2)
end