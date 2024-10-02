using Test
include("$(@__DIR__)/../ITensorCPD.jl")
using .ITensorCPD:
  als_optimize, direct, random_CPD, random_CPD_square_network, row_norm, reconstruct, had_contract
using ITensors: Index, ITensor, array, contract, dim, norm, random_itensor

@testset "Norm Row test, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
  i, j = Index.((20, 30))
  A = random_itensor(elt, i, j)
  Ainorm, lam = row_norm(A, i)
  for id in 1:dim(i)
    @test real(one(elt)) ≈ sum(array(Ainorm .^ 2)[:, id])
  end

  Ajnorm, lam = row_norm(A, j)
  for id in 1:dim(i)
    @test real(one(elt)) ≈ sum(array(Ajnorm .^ 2)[id, :])
  end

  Aijnorm, lam = row_norm(A, i, j)
  @test real(one(elt)) ≈ sum(array(Aijnorm .^ 2))
end

@testset "reconstruct, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
  i, j = Index.((20, 30))
  r = Index(10, "CP_rank")
  A, B = random_itensor.(elt, ((r, i), (r, j)))
  λ = ITensor(randn(elt, dim(r)), r)
  exact = fill!(ITensor(elt, i, j), zero(elt))
  for R in 1:dim(r)
    for I in 1:dim(i)
      for J in 1:dim(j)
        exact[I, J] += λ[R] * A[R, I] * B[R, J]
      end
    end
  end
  recon = reconstruct([A, B], λ)
  @test 1.0 - norm(array(exact - recon)) / norm(exact) ≈ 1.0 rtol = eps(real(elt))

  k = Index.(40)
  A, B, C = random_itensor.(elt, ((r, i), (r, j), (r, k)))
  λ = ITensor(randn(elt, dim(r)), r)
  exact = fill!(ITensor(elt, i, j, k), zero(elt))
  for R in 1:dim(r)
    for I in 1:dim(i)
      for J in 1:dim(j)
        for K in 1:dim(k)
          exact[I, J, K] += λ[R] * A[R, I] * B[R, J] * C[R, K]
        end
      end
    end
  end
  recon = reconstruct([A, B, C], λ)

  @test 1.0 - norm(array(exact - recon)) / norm(exact) ≈ 1.0 rtol = eps(real(elt))
end

## Complex optimization still does not work
@testset "Standard CPD, elt=$elt" for elt in [Float32, Float64]
  ## Working here
  i, j, k = Index.((20, 30, 40))
  r = Index(400, "CP_rank")
  A = random_itensor(elt, i, j, k)
  cp_A = random_CPD(A, r)

  opt_A = als_optimize(cp_A, r; maxiters=100)
  @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-7

  check = ITensorCPD.FitCheck(1e-15, 100, norm(A))
  opt_A = als_optimize(cp_A, r, check)
  @test norm(reconstruct(opt_A) - A) / norm(A) ≤ 1.0 - ITensorCPD.fit(check)

  cp_A = random_CPD(A, r; algorithm=direct())
  opt_A = als_optimize(cp_A, r; maxiters=100)
  @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-7

  check = ITensorCPD.FitCheck(1e-15, 100, norm(A))
  cp_A = random_CPD(A, r; algorithm=direct())
  opt_A = als_optimize(cp_A, r, check)
  @test norm(reconstruct(opt_A) - A) / norm(A) ≤ 1.0 - ITensorCPD.fit(check)
end

@testset "Lattice CPD, elt=$elt" for elt in [Float32, Float64]
  a, b, c, d, e, f, g, h =
    Index.((5, 5, 5, 5, 5, 5, 5, 5), ("a", "b", "c", "d", "e", "f", "g", "h"))
  w, x, y, z = Index.((5, 5, 5, 5), ("w", "x", "y", "z"))

  line = [random_itensor(elt, a, b, x), random_itensor(elt, c, d, x)]
  r = Index(40, "CP_rank")
  CP = random_CPD_square_network(line, r)
  check = ITensorCPD.FitCheck(1e-10, 100, sqrt((contract(line) * contract(line))[]))

  opt_A = ITensorCPD.als_optimize(CP, r, check)
  @test norm(ITensorCPD.reconstruct(opt_A) - contract(line)) / norm(line) ≤
    ITensorCPD.fit(check)

  square = [
    random_itensor(elt, a, b, x, w),
    random_itensor(elt, c, d, x, z),
    random_itensor(elt, g, h, z, y),
    random_itensor(elt, e, f, y, w),
  ]
  r = Index(2000, "CP_rank")
  CP = random_CPD_square_network(square, r)
  ## TODO write better code to take norm of square lattice
  check = ITensorCPD.FitCheck(1e-3, 30, sqrt((contract(square) * contract(square))[]))
  @time opt_A = ITensorCPD.als_optimize(CP, r, check)
  @test norm(ITensorCPD.reconstruct(opt_A) - contract(square)) /
        sqrt((contract(square) * contract(square))[]) ≤ 1.0 - ITensorCPD.fit(check)
end

# ## Study 2d eising model case with this solver.
# ## Try different beta values. Does it work better with larger beta.
# using ITensors
# include("$(@__DIR__)/../ITensorCPD.jl")
# using .ITensorCPD: als_optimize, direct, random_CPD, random_CPD_square_network, row_norm, reconstruct

# N = 4
# sites = Index.((N,N,N,N))
# β = 0.001
# d = Vector{Float64}(undef, N*N)
# for i in 1:N
#   for j in 1:N
#     d[(i-1) * N + j] = exp(-β* i * j)
#   end
# end
# its = (itensor(d, sites[1], sites[2]), itensor(d, sites[2], sites[3]), itensor(d, sites[3], sites[4]), itensor(d, sites[4], sites[1]))

# contract(its)[]

## goal is to make 
##       d2        d3      d4  
##        |  d11   |  d22  |
##    d1--a  ----  b   ---  c -- d5
##     d88|        |1d      |d33 
## d12 -- i ---4d    2d---  e -- d6
##     d77|      3d|        |d44
##   d11--h  ---   g   --   f -- d7
##        |  d66   |   d55  |    
##       d10      d9        d8  
#@testset "ising_model" begin
using ITensorNetworks: IndsNetwork, delta_network, edges, src, dst, degree, insert_linkinds
using ITensors
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

## This function primes the internal indices of a 2d
## network. Doing so allows one to compute the norm
## of the ising network more easily
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

using ITensorNetworks
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

nx = 5
ny = 6

beta = 1.0
h = 0.0

vals = [0.0, 0.0]
r = Index(1, "CP_rank")
for i in 1:2
  if i == 1
    tn = ising_network(Float64, IndsNetwork(named_grid((nx, ny))), beta; h)
  else
    tn = ising_network(
      Float64, IndsNetwork(named_grid((nx, ny))), beta; h, szverts=[(3, 3), (3, 4)]
    )
  end
  s1 = subgraph(
    tn, ((2, 2), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5), (4, 4), (4, 3), (4, 2), (3, 2))
  )

  sising = s1.data_graph.vertex_data.values
  sisingp = replace_inner_w_prime_loop(sising)

  sqrs = sising[1] * sisingp[1]
  for i in 2:length(sising)
    sqrs = sqrs * sising[i] * sisingp[i]
  end
  sqrt(sqrs[])

  fit = ITensorCPD.FitCheck(1e-10, 100, sqrt(sqrs[]))
  cp = ITensorCPD.random_CPD_square_network(sising, r)
  cpopt = ITensorCPD.als_optimize(cp, r, fit)
  core = [cpopt[4], cpopt[15], cpopt[20], cpopt[6], cpopt[9], cpopt[13]]

  es = [
    cpopt[2] * tn[1, 1] * tn[1, 2],
    cpopt[3] * tn[1, 3],
    cpopt[5] * tn[1, 4],
    cpopt[7] * tn[1, 5] * tn[1, 6],
    cpopt[8] * tn[2, 6],
    cpopt[10] * tn[3, 6],
    cpopt[12] * tn[4, 6] * tn[5, 6],
    cpopt[11] * tn[5, 5],
    cpopt[14] * tn[5, 4],
    cpopt[16] * tn[5, 3],
    cpopt[18] * tn[5, 2] * tn[5, 1],
    cpopt[17] * tn[4, 1],
    cpopt[19] * tn[3, 1],
    cpopt[1] * tn[2, 1],
  ]

  is = ind.(core, 2)
  c = ITensor(Float64, is)
  for i in 1:dim(r)
    l = contract([itensor(array(x)[i, :, :], inds(x)[2:end]) for x in es])[] * cpopt[][i]
    c += l * contract([itensor(array(x)[i, :], ind(x, 2)) for x in core])
  end
  # (c * tn[3,3] * tn[3,4])[]
  # (c * tn[3,3] * tn[3,4])[]
  vals[i] = (c * tn[3, 3] * tn[3, 4])[]
end
cp = vals[2] / vals[1]

# cp = 4.3732548624393e10 / 8.3001368444895e10

tn = ising_network(Float64, IndsNetwork(named_grid((nx, ny))), beta; h);
tnO = ising_network(
  Float64, IndsNetwork(named_grid((nx, ny))), beta; h, szverts=[(3, 3), (3, 4)]
);

using OMEinsumContractionOrders: OMEinsumContractionOrders
using ITensorNetworks: contraction_sequence
seq = contraction_sequence(tn; alg="sa_bipartite");
szsz = contract(tnO; sequence=seq)[] / contract(tn; sequence=seq)[]
szsz - cp

szsz_bp = scalar(tnO; alg="bp") / scalar(tn; alg="bp")
szsz - szsz_bp

##############################################
nx = 7
ny = 8

beta = 1.0
h = 0.0

h = 0
s = IndsNetwork(named_grid((nx, ny)); link_space=2);

function contract_loops(r1, r2, tn, i; check_svd::Bool=false)
    s1 = subgraph(
      tn,
      (
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (3, 7),
        (4, 7),
        (5, 7),
        (6, 7),
        (6, 6),
        (6, 5),
        (6, 4),
        (6, 3),
        (6, 2),
        (5, 2),
        (4, 2),
        (3, 2),
      ),
    )

    sising = s1.data_graph.vertex_data.values
    sisingp = replace_inner_w_prime_loop(sising)

    sqrs = sising[1] * sisingp[1]
    for i in 2:length(sising)
      sqrs = sqrs * sising[i] * sisingp[i]
    end
    sqrt(sqrs[])

    fit = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]))
    cp = ITensorCPD.random_CPD_square_network(sising, r1)
    @time cpopt = ITensorCPD.als_optimize(cp, r1, fit)
    ## contract s1 with outer layer   
    core = [
      cpopt[4],
      cpopt[6],
      cpopt[8],
      cpopt[10],
      cpopt[13],
      cpopt[15],
      cpopt[17],
      cpopt[21],
      cpopt[23],
      cpopt[25],
      cpopt[27],
      cpopt[32],
      cpopt[34],
      cpopt[36],
    ]

    es = [
      cpopt[2] * tn[1, 2] * tn[1, 1]
      cpopt[3] * tn[1, 3]
      cpopt[5] * tn[1, 4]
      cpopt[7] * tn[1, 5]
      cpopt[9] * tn[1, 6]
      cpopt[11] * tn[1, 7] * tn[1, 8]
      cpopt[12] * tn[2, 8]
      cpopt[14] * tn[3, 8]
      cpopt[16] * tn[4, 8]
      cpopt[18] * tn[5, 8]
      cpopt[20] * tn[6, 8] * tn[7, 8]
      cpopt[19] * tn[7, 7]
      cpopt[22] * tn[7, 6]
      cpopt[24] * tn[7, 5]
      cpopt[26] * tn[7, 4]
      cpopt[28] * tn[7, 3]
      cpopt[30] * tn[7, 2] * tn[7, 1]
      cpopt[29] * tn[6, 1]
      cpopt[31] * tn[5, 1]
      cpopt[33] * tn[4, 1]
      cpopt[35] * tn[3, 1]
      cpopt[1] * tn[2, 1]
    ]

    v = Vector{Float64}(undef, dim(r1))
    for i in 1:dim(r1)
      v[i] =
        contract([itensor(array(x)[i, :, :], inds(x)[2:end]) for x in es])[] * cpopt[][i]
    end
    v = itensor(v, r1)

    s2 = subgraph(
      tn, ((3, 3), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6), (5, 5), (5, 4), (5, 3), (4, 3))
    )

    sising = s2.data_graph.vertex_data.values
    sisingp = replace_inner_w_prime_loop(sising)

    sqrs = sising[1] * sisingp[1]
    for i in 2:length(sising)
      sqrs = sqrs * sising[i] * sisingp[i]
    end
    sqrt(sqrs[])

    fit = ITensorCPD.FitCheck(1e-3, 10, sqrt(sqrs[]))
    cp = ITensorCPD.random_CPD_square_network(sising, r2)
    @time cpopt = ITensorCPD.als_optimize(cp, r2, fit)

    es = [
      cpopt[2] * core[1]
      cpopt[3] * core[2]
      cpopt[5] * core[3]
      cpopt[7] * core[4]
      cpopt[8] * core[5]
      cpopt[10] * core[6]
      cpopt[11] * core[8]
      cpopt[12] * core[7]
      cpopt[14] * core[9]
      cpopt[16] * core[10]
      cpopt[17] * core[12]
      cpopt[18] * core[11]
      cpopt[19] * core[13]
      cpopt[1] * core[14]
    ]

    core = [cpopt[4], cpopt[6], cpopt[9], cpopt[13], cpopt[15], cpopt[20]]

    had = copy(es[1])
    for j in 2:length(es)
      had = ITensors.hadamard_product(had, es[j])
    end

    if check_svd
      U,S,V = svd(had, r1)
      return S
    end
    is = ind.(core, 2)
    #env = ITensor(Float64, is);
    l = had * v
    val = 0
    for i in 1:dim(r2)
      #env += cpopt[][i] * l[i] * contract([itensor(array(x)[i,:], ind(x,2)) for x in core])
      val += (cpopt[][i] * (l[i] * contract([
        itensor(array(x)[i, :], ind(x, 2)) for x in core
      ])) * tn[4, 4] * tn[4, 5])[]
    end
    vals[i] = val
    #end
    #env = itensor(ITensors.NDTensors.data(env), noncommoninds(tn[4,4], tn[4,5]))
    #vals[i] = (env * tn[4,4] * tn[4,5])[]
end

function contract_loop_exact_core(r1, r2, tn, i; check_svd::Bool=false)
  s1 = subgraph(
      tn,
      (
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (3, 7),
        (4, 7),
        (5, 7),
        (6, 7),
        (6, 6),
        (6, 5),
        (6, 4),
        (6, 3),
        (6, 2),
        (5, 2),
        (4, 2),
        (3, 2),
      ),
    )

    sising = s1.data_graph.vertex_data.values
    sisingp = replace_inner_w_prime_loop(sising)

    sqrs = sising[1] * sisingp[1]
    for i in 2:length(sising)
      sqrs = sqrs * sising[i] * sisingp[i]
    end
    sqrt(sqrs[])

    fit = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]))
    cp = ITensorCPD.random_CPD_square_network(sising, r1)
    @time cpopt = ITensorCPD.als_optimize(cp, r1, fit)
    ## contract s1 with outer layer   
    core = [
      cpopt[4],
      cpopt[6],
      cpopt[8],
      cpopt[10],
      cpopt[13],
      cpopt[15],
      cpopt[17],
      cpopt[21],
      cpopt[23],
      cpopt[25],
      cpopt[27],
      cpopt[32],
      cpopt[34],
      cpopt[36],
    ]

    es = [
      cpopt[2] * tn[1, 2] * tn[1, 1]
      cpopt[3] * tn[1, 3]
      cpopt[5] * tn[1, 4]
      cpopt[7] * tn[1, 5]
      cpopt[9] * tn[1, 6]
      cpopt[11] * tn[1, 7] * tn[1, 8]
      cpopt[12] * tn[2, 8]
      cpopt[14] * tn[3, 8]
      cpopt[16] * tn[4, 8]
      cpopt[18] * tn[5, 8]
      cpopt[20] * tn[6, 8] * tn[7, 8]
      cpopt[19] * tn[7, 7]
      cpopt[22] * tn[7, 6]
      cpopt[24] * tn[7, 5]
      cpopt[26] * tn[7, 4]
      cpopt[28] * tn[7, 3]
      cpopt[30] * tn[7, 2] * tn[7, 1]
      cpopt[29] * tn[6, 1]
      cpopt[31] * tn[5, 1]
      cpopt[33] * tn[4, 1]
      cpopt[35] * tn[3, 1]
      cpopt[1] * tn[2, 1]
    ]

    v = Vector{Float64}(undef, dim(r1))
    for i in 1:dim(r1)
      v[i] =
        contract([itensor(array(x)[i, :, :], inds(x)[2:end]) for x in es])[] * cpopt[][i]
    end
    v = itensor(v, r1)

    next_layer = [
    core[1] * tn[3,3] # * core[14]
    core[2] * tn[3,4]
    core[3] * tn[3,5]
    core[4] * tn[3,6] # * core[5]
    core[6] * tn[4,6]
    core[7] * tn[5,6] # * core[8]
    core[9] * tn[5,5]
    core[10] * tn[5,4]
    core[11] * tn[5,3] # * core[12]
    core[13] * tn[4,3]
    ]

    result = 0;
    for rank in 1:dim(r1)
      A = contract([itensor(array(x)[rank, :, :,:], inds(x)[2:end]) for x in next_layer[1:2]])
      B = contract([itensor(array(x)[rank, :, :,:], inds(x)[2:end]) for x in next_layer[8:10]])
      A = A * itensor(array(core[14])[rank,:], inds(core[14])[2])
      B = B * itensor(array(core[12])[rank,:], inds(core[12])[2])
      C1 = A * B * tn[4,4]

      A = contract([itensor(array(x)[rank, :, :,:], inds(x)[2:end]) for x in next_layer[3:4]])
      B = contract([itensor(array(x)[rank, :, :,:], inds(x)[2:end]) for x in next_layer[5:7]])
      A = A * itensor(array(core[5])[rank,:], inds(core[5])[2])
      B = B * itensor(array(core[8])[rank,:], inds(core[8])[2])
      C2 = A * B * tn[4,5]
      result += (C1 * C2)[] * v[rank]
    end
    vals[i] = result
end

vals = [0.0, 0.0]
cp_szsz = Vector{Vector{Float64}}([])
ranks = [2,6,15]
for rank in 1:length(ranks)
  push!(cp_szsz, Vector{Float64}(undef, 0))
  r1 = Index(ranks[rank], "CP_rank")
  r2 = Index(ranks[rank], "CP_rank")
  for beta in [1.0 - i for i in 0:0.01:1]
    for i in 1:2
      if i == 1
        tn = ising_network(Float64, s, beta; h)
      else
        tn = ising_network(
          Float64, s, beta; h, szverts=[(4, 4), (4, 5)]
        )
      end
      #if isnothing(env)
      # contract_loops(r1, r2, tn, i)
      contract_loop_exact_core(r1,r2,tn,i)
    end
    cp = vals[2] / vals[1]
    push!(cp_szsz[rank], cp)
  end
end

theor = [
  0.0,
  0.00996425164601078,
  0.020011770018868447,
  0.030031532816110484,
  0.04010680676458378,
  0.05023759186428833,
  0.06034062138837726,
  0.0705546732149287,
  0.08087974734394265,
  0.09120482147295661,
  0.10169642905566434,
  0.11224354778960333,
  0.12301271112846734,
  0.13369860774048448,
  0.14468981568427353,
  0.15584755708175635,
  0.16708856520608606,
  0.1785516179353408,
  0.1901534485426737,
  0.20200507933054723,
  0.2140787547233458,
  0.22648549702353193,
  0.23900326162618057,
  0.251909604287448,
  0.2650379915536405,
  0.27861046802968303,
  0.2925160114131131,
  0.3067823772795464,
  0.32143732120459845,
  0.3365363543395006,
  0.3521904989867153,
  0.36839975514624257,
  0.3852196339693137,
  0.40265013545592865,
  0.42088554863539684,
  0.4399536290833339,
  0.45988213237535547,
  0.4809208586920022,
  0.5034028749406616,
  0.52716164766764,
  0.5530298441414061,
  0.5812572645425007,
  0.6130651541980114,
  0.6502298699473386,
  0.7011058400507864,
  0.7563949466771192,
  0.7905620602599583,
  0.817068634972884,
  0.8388845174067683,
  0.8571476861618521,
  0.8727463196578356,
  0.8862355294070312,
  0.8981149157705204,
  0.9083844787483031,
  0.9174327963989981,
  0.9255374244787617,
  0.9327538741388253,
  0.9390821453791887,
  0.9447997939560082,
  0.9499623310205152,
  0.9544587342702471,
  0.9587330929150539,
  0.9623413177450857,
  0.9657274979701924,
  0.9688361224391429,
  0.9715561688494745,
  0.9739431483524186,
  0.9762746167041314,
  0.9782730181484567,
  0.9801603972903195,
  0.9818812429784884,
  0.9834355552129637,
  0.9847678228425139,
  0.9860445793208328,
  0.9872658246479205,
  0.9883205365213144,
  0.989319737243477,
  0.9900968933607146,
  0.9909295606291835,
  0.9917067167464211,
  0.9924838728636587,
  0.9929279620735088,
  0.9935385847370526,
  0.9940936962493652,
  0.9945932966104465,
  0.9949818746690653,
  0.9954259638789154,
  0.9958700530887654,
  0.996203119996153,
  0.9965361869035405,
  0.9967582315084655,
  0.9969802761133906,
  0.9972023207183156,
  0.9974243653232406,
  0.9976464099281657,
  0.9977574322306282,
  0.9978684545330907,
  0.9983125437429408,
  0.9984235660454033,
  0.9984235660454033,
  0.9985345883478658
]
beta = 0.41
tn = ising_network(Float64, s, beta; h);
using NDTensors
r1 = Index(10, "CP_rank")
r2 = Index(10, "CP_rank")
evals = contract_loops(r1, r2, tn, 1; check_svd=false)

betas = [1.0 - i for i in 0:0.01:1]
# cp = 4.3732548624393e10 / 8.3001368444895e10

full_szsz = Vector{Float64}([])
bp_szsz = Vector{Float64}([])
using ITensorNetworks: BeliefPropagationCache, update, environment
for beta in betas
  tn = ising_network(Float64, IndsNetwork(named_grid((nx, ny))), beta; h)
  tnO = ising_network(
    Float64, IndsNetwork(named_grid((nx, ny))), beta; h, szverts=[(4, 4), (4, 5)]
  )

  using OMEinsumContractionOrders: OMEinsumContractionOrders
  using ITensorNetworks: contraction_sequence
  seq = contraction_sequence(tn; alg="sa_bipartite")
  szsz = @time contract(tnO; sequence=seq)[] / contract(tn; sequence=seq)[]
  push!(full_szsz, szsz)

  vs_centre = [(4, 4), (4, 5)]
  s = IndsNetwork(named_grid((nx, ny)); link_space=2)
  tn = ising_network(Float64, s, beta; h)
  tnO = ising_network(Float64, s, beta; h, szverts=vs_centre)
  tn_bpc = BeliefPropagationCache(tn)
  tn_bpc = update(tn_bpc; maxiter=50)
  envs = environment(tn_bpc, vs_centre)

  numer = contract([[tn[v] for v in vs_centre]; envs]; sequence="automatic")
  denom = contract([[tnO[v] for v in vs_centre]; envs]; sequence="automatic")

  szsz_bp_vbetter = denom[] / numer[]
  push!(bp_szsz, szsz_bp_vbetter)
end
inf = 0.5530853552926374
szsz - inf

using Plots
plot(betas, theor[end:-1:1]; label="Infinite Lattice")
plot!(betas, full_szsz; label="Exact Contraction")
plot!(betas, bp_szsz; label="BP Contraction")
plot!(betas, cp_szsz[1]; label="CP rank 2")
plot!(betas, cp_szsz[2]; label="CP rank 6")
plot!(betas, cp_szsz[3]; label="CP rank 15")
plot!(; xlabel="Inverse Temparature", ylabel="SZ Correlation", legend=:bottomright, title="CPD contraction of 2D network")
savefig("../CP_ising_2site_exact_center.pdf")

plot(betas, abs.(full_szsz .- cp_szsz[1]) ; label="CP rank 1")
plot!(betas, abs.(full_szsz .- cp_szsz[2]) ; label="CP rank 6")
plot!(betas, abs.(full_szsz .- cp_szsz[3]) ; label="CP rank 10")
plot!(betas, abs.(full_szsz .- cp_szsz[4]) ; label="CP rank 15")
plot!(betas, abs.(full_szsz .- bp_szsz) ; label="BP")
plot!(; xlabel="Inverse Temparature", ylabel="SZ Correlation error")
savefig("../CP_ising_2site_error_exact_center.pdf")

szsz - szsz_bp
cp - szsz_bp


beta = 0.41
tn = ising_network(Float64, s, beta; h);
using NDTensors
r1 = Index(10, "CP_rank")
r2 = Index(10, "CP_rank")
s1 = subgraph(
                tn,
                (
                  (2, 2),
                  (2, 3),
                  (2, 4),
                  (2, 5),
                  (2, 6),
                  (2, 7),
                  (3, 7),
                  (4, 7),
                  (5, 7),
                  (6, 7),
                  (6, 6),
                  (6, 5),
                  (6, 4),
                  (6, 3),
                  (6, 2),
                  (5, 2),
                  (4, 2),
                  (3, 2),
                ),
              )

sising = s1.data_graph.vertex_data.values
sisingp = replace_inner_w_prime_loop(sising)

sqrs = sising[1] * sisingp[1]
for i in 2:length(sising)
  sqrs = sqrs * sising[i] * sisingp[i]
end
sqrt(sqrs[])

fit = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]))
cp = ITensorCPD.random_CPD_square_network(sising, r1)
@time cpopt = ITensorCPD.als_optimize(cp, r1, fit)

A = s1[2,2]
Ar = cpopt[1] * s1[2,2]
ITensorCPD.had_contract(Ar, cpopt[2], r1)

r1 = Index(100, "CP_rank")
cp = ITensorCPD.random_CPD_ITensorNetwork(s1, r1);
@time ITensorCPD.als_optimize(cp, r1, fit);

cp = ITensorCPD.random_CPD_square_network(sising, r1);
@time cpopt = ITensorCPD.als_optimize(cp, r1, fit);