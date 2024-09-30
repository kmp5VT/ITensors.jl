using Test
include("$(@__DIR__)/../ITensorCPD.jl")
using .ITensorCPD:
  als_optimize, direct, random_CPD, random_CPD_square_network, row_norm, reconstruct
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

vals = [0.0, 0.0]
cp_szsz = Vector{Vector{Float64}}([])
ranks = [1, 5, 10, 15]
h = 0
for rank in 1:length(ranks)
  push!(cp_szsz, Vector{Float64}(undef, 0))
  r1 = Index(ranks[rank], "CP_rank")
  r2 = Index(ranks[rank], "CP_rank")
  for beta in [1.0 - i for i in 0:0.01:1]
    for i in 1:2
      if i == 1
        tn = ising_network(Float64, IndsNetwork(named_grid((nx, ny))), beta; h)
      else
        tn = ising_network(
          Float64, IndsNetwork(named_grid((nx, ny))), beta; h, szverts=[(4, 4), (4, 5)]
        )
      end
      #if isnothing(env)
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
      cpopt = ITensorCPD.als_optimize(cp, r1, fit)
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
      cpopt = ITensorCPD.als_optimize(cp, r2, fit)

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
    cp = vals[2] / vals[1]
    push!(cp_szsz[rank], cp)
  end
end

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
plot(betas, full_szsz; label="Exact Contraction")
plot!(betas, bp_szsz; label="BP Contraction")
plot!(betas, cp_szsz[1]; label="CP rank 1")
plot!(betas, cp_szsz[2]; label="CP rank 5")
plot!(betas, cp_szsz[3]; label="CP rank 10")
plot!(betas, cp_szsz[4]; label="CP rank 15")
plot!(; xlabel="Inverse Temparature", ylabel="SZ Correlation")
savefig("../CP_ising_2site.pdf")

plot(betas, abs.(full_szsz .- cp_szsz[1]) ./ full_szsz .* 100; label="CP rank 1")
plot!(betas, abs.(full_szsz .- cp_szsz[2]) ./ full_szsz .* 100; label="CP rank 5")
plot!(betas, abs.(full_szsz .- cp_szsz[3]) ./ full_szsz .* 100; label="CP rank 10")
plot!(betas, abs.(full_szsz .- cp_szsz[4]) ./ full_szsz .* 100; label="CP rank 15")
plot!(betas, abs.(full_szsz .- bp_szsz) ./ full_szsz .* 100; label="BP")
plot!(; xlabel="Inverse Temparature", ylabel="SZ Correlation error")
savefig("../CP_ising_2site_error.pdf")

szsz - szsz_bp
cp - szsz_bp
