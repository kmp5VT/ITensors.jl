using ITensorNetworks
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: IndsNetwork, delta_network, edges, src, dst, degree, insert_linkinds
using ITensors
#include("$(@__DIR__)/../ITensorCPD.jl")
using ITensorCPD:
ITensorCPD,
  als_optimize,
  direct,
  random_CPD,
  random_CPD_square_network,
  row_norm,
  reconstruct,
  had_contract

  function ising_network(
    eltype::Type, s::IndsNetwork, beta::Number; h::Number=0.0, szverts=nothing
  )
    s = insert_linkinds(s; link_space=2)
    # s = insert_missing_internal_inds(s, edges(s); internal_inds_space=2)
    tn = delta_network(eltype, s)
    if (szverts != nothing)
      for v in szverts
        tn[v] = diagITensor(eltype[1, -1], inds(tn[v]))
      end
    end
    for edge in edges(tn)
      v1 = src(edge)
      v2 = dst(edge)
      i = commoninds(tn[v1], tn[v2])[1]
      deg_v1 = degree(tn, v1)
      deg_v2 = degree(tn, v2)
      f11 = exp(beta * (1 + h / deg_v1 + h / deg_v2))
      f12 = exp(beta * (-1 + h / deg_v1 - h / deg_v2))
      f21 = exp(beta * (-1 - h / deg_v1 + h / deg_v2))
      f22 = exp(beta * (1 - h / deg_v1 - h / deg_v2))
      q = eltype[f11 f12; f21 f22]
      w, V = eigen(q)
      w = map(sqrt, w)
      sqrt_q = V * ITensors.Diagonal(w) * inv(V)
      t = itensor(sqrt_q, i, i')
      tn[v1] = tn[v1] * t
      tn[v1] = noprime!(tn[v1])
      t = itensor(sqrt_q, i', i)
      tn[v2] = tn[v2] * t
      tn[v2] = noprime!(tn[v2])
    end
    return tn
  end

  function replace_inner_w_prime_loop(tn)
    ntn = deepcopy(tn)
    for i in 1:(length(tn) - 1)
      cis = inds(tn[i])
      is = commoninds(tn[i], tn[i + 1])
      nis = [i ∈ is ? i' : i for i in cis]
      replaceinds!(ntn[i], cis, nis)
      cis = inds(tn[i + 1])
      nis = [i ∈ is ? i' : i for i in cis]
      replaceinds!(ntn[i + 1], cis, nis)
    end
  
    i = length(tn)
    cis = inds(tn[i])
    is = commoninds(tn[i], tn[1])
    nis = [i ∈ is ? i' : i for i in cis]
    replaceinds!(ntn[i], cis, nis)
    cis = inds(tn[1])
    nis = [i ∈ is ? i' : i for i in cis]
    replaceinds!(ntn[1], cis, nis)
    return ntn
  end

  function norm_of_loop(s1::ITensorNetwork)
    sising = s1.data_graph.vertex_data.values
    sisingp = replace_inner_w_prime_loop(sising)
  
    sqrs = sising[1] * sisingp[1]
    for i in 2:length(sising)
      sqrs = sqrs * sising[i] * sisingp[i]
    end
    return sqrt(sqrs[])
  end

  function ring_inds(start::Int, nx::Int, ny::Int)
    inds = Vector{Tuple{Int,Int}}()
    for y in start:(ny-start+1)
        push!(inds, (start,y))
    end
    for x in start+1:(nx-start+1)
        push!(inds, (x, ny-start+1))
    end
    for y in (ny-start +1):-1:start
        push!(inds, (nx-start+1, y))
    end
    for x in (nx-start+1):-1:start+1
    push!(inds, (x, start))
    end
    return Tuple(unique!(inds))
  end

ring_inds(4, 7,8)

  function contract_3_loops(r1, r2, r3, tn, i)
    #core = subgraph(tn, ((3,3),(3,4)))
    s1 = subgraph(tn, ring_inds(4, 9, 10))
    s2 = subgraph(tn, ring_inds(3, 9, 10))
    s3 = subgraph(tn, ring_inds(2, 9, 10))
    s4 = subgraph(tn, ring_inds(1, 9, 10))
  
  
    #fit = ITensorCPD.FitCheck(1e-4,100, norm_of_loop(s3))
    cp_guess = ITensorCPD.random_CPD_ITensorNetwork(s3, r3)
    cpopt_s3 = ITensorCPD.als_optimize(cp_guess, r3; maxiters=10);
    outer, inner = ITensorCPD.tn_cp_contract(s4, cpopt_s3);
    t3 = itensor(array(ITensorCPD.had_contract(outer.data_graph.vertex_data.values, r3)) .* array(cpopt_s3[]), r3) 

    #fit = ITensorCPD.FitCheck(1e-4, 100, norm_of_loop(s2))
    cp_guess = ITensorCPD.random_CPD_ITensorNetwork(s2, r2)
    cpopt_s2 = ITensorCPD.als_optimize(cp_guess, r2; maxiters=10);
    # contract cpopt with inner
    had, _, inner = ITensorCPD.cp_cp_contract(inner, cpopt_s2.factors)
    t2 = itensor(array(t3 * had) .* array(cpopt_s2[]), r2)

    #fit = ITensorCPD.FitCheck(1e-4, 100, norm_of_loop(s1))
    cp_guess = ITensorCPD.random_CPD_ITensorNetwork(s1, r1)
    cpopt_s1 = ITensorCPD.als_optimize(cp_guess, r1; maxiters=10)
    had, _, inner = ITensorCPD.cp_cp_contract(inner, cpopt_s1.factors)

    t1 = ITensorCPD.had_contract([inner..., tn[5,5], tn[5,6]], r1)
    val = (t2 * had * itensor(array(t1) .* array(cpopt_s1[]), r1))[]
    return vals[i] = val
  end
  
  ##############################################
  nx = 9
  ny = 10

  s = IndsNetwork(named_grid((nx, ny)); link_space=2);
  vals = [0.0, 0.0]
cp_szsz = Vector{Vector{Float64}}([])
cp_szsz_old = copy(cp_szsz)
ranks = [1, 6, 15]
cp_guess = nothing
for rank in 1:length(ranks)
 push!(cp_szsz, Vector{Float64}(undef, 0))
  r1 = Index(ranks[rank], "CP_rank1")
  r2 = Index(ranks[rank], "CP_rank2")
  r3 = Index(ranks[rank], "CP_rank3")
  for beta in [1.0 - i for i in 0:0.01:1]
   for i in 1:2
      if i == 1
        tn = ising_network(Float64, s, beta; h)
      else
        tn = ising_network(Float64, s, beta; h, szverts=[(3, 3), (3, 4)])
      end
      #if isnothing(env)
      val = contract_3_loops(r1, r2, r3, tn, i)
      #contract_loops(r1, r2, tn, i; old_contract = false)
      # contract_loop_exact_core(r1, r2, tn, i)
    end
    cp = vals[2] / vals[1]
    push!(cp_szsz[rank], cp)
  end
end
