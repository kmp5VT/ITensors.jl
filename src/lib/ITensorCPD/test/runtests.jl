using Test
include("$(@__DIR__)/../ITensorCPD.jl")
using .ITensorCPD: als_optimize, direct, random_CPD, random_CPD_square_network, row_norm, reconstruct
using ITensors: Index, ITensor, array, contract, dim, norm, random_itensor

@testset "Norm Row test, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
  i,j = Index.((20,30))
  A = random_itensor(elt, i,j)
  Ainorm, lam = row_norm(A, i)
  for id in 1:dim(i)
    @test real(one(elt)) ≈ sum(array(Ainorm .^2)[:,id])
  end

  Ajnorm, lam = row_norm(A, j)
  for id in 1:dim(i)
    @test real(one(elt)) ≈ sum(array(Ajnorm .^2)[id,:])
  end

  Aijnorm, lam = row_norm(A, i,j);
  @test real(one(elt)) ≈ sum(array(Aijnorm .^2))
end

@testset "reconstruct, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
  i,j = Index.((20,30))
  r = Index(10, "CP_rank")
  A, B = random_itensor.(elt, ((r,i),(r,j)));
  λ = ITensor(randn(elt, dim(r)), r)
  exact = fill!(ITensor(elt, i,j), zero(elt))
  for R in 1:dim(r)
    for I in 1:dim(i)
      for J in 1:dim(j)
        exact[I,J] += λ[R] * A[R,I] * B[R, J]
      end
    end
  end
  recon = reconstruct([A,B], λ);
  @test 1.0 - norm(array(exact - recon)) / norm(exact) ≈ 1.0 rtol = eps(real(elt))

  k = Index.(40)
  A, B, C = random_itensor.(elt, ((r,i),(r,j), (r,k)));
  λ = ITensor(randn(elt, dim(r)), r)
  exact = fill!(ITensor(elt, i,j,k), zero(elt))
  for R in 1:dim(r)
    for I in 1:dim(i)
      for J in 1:dim(j)
        for K in 1:dim(k)
          exact[I,J,K] += λ[R] * A[R,I] * B[R, J] * C[R,K]
        end
      end
    end
  end
  recon = reconstruct([A,B,C], λ)

  @test 1.0 - norm(array(exact - recon)) / norm(exact) ≈ 1.0 rtol = eps(real(elt))
end

## Complex optimization still does not work
@testset "Standard CPD, elt=$elt" for elt in [Float32, Float64]
  ## Working here
  i,j,k = Index.((20,30,40))
  r = Index(400, "CP_rank")
  A = random_itensor(elt, i,j,k);
  cp_A = random_CPD(A, r);

  opt_A = als_optimize(cp_A, r; maxiters=100);
  @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-7

  check = ITensorCPD.FitCheck(1e-15, 100, norm(A))
  opt_A = als_optimize(cp_A, r, check);
  @test norm(reconstruct(opt_A) - A) / norm(A) ≤ 1.0 - ITensorCPD.fit(check)

  cp_A = random_CPD(A, r; algorithm=direct());
  opt_A = als_optimize(cp_A, r; maxiters=100);
  @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-7

  check = ITensorCPD.FitCheck(1e-15, 100, norm(A))
  cp_A = random_CPD(A, r; algorithm=direct());
  opt_A = als_optimize(cp_A, r, check);
  @test norm(reconstruct(opt_A) - A) / norm(A) ≤ 1.0 - ITensorCPD.fit(check)
end

@testset "Lattice CPD, elt=$elt" for elt in [Float32, Float64]
  a,b,c,d,e,f,g,h = Index.((5,5,5,5,5,5,5,5), ("a","b","c","d","e","f","g","h"));
  w,x,y,z = Index.((5,5,5,5), ("w","x","y","z"));

  line = [random_itensor(elt, a,b,x), random_itensor(elt, c,d,x)];
  r = Index(40,"CP_rank")
  CP = random_CPD_square_network(line, r);
  check = ITensorCPD.FitCheck(1e-10, 100, sqrt((contract(line) * contract(line))[]))

  opt_A = ITensorCPD.als_optimize(CP, r, check);
  @test norm(ITensorCPD.reconstruct(opt_A) - contract(line)) / norm(line) ≤ ITensorCPD.fit(check)


  square = [random_itensor(elt, a,b,x,w), random_itensor(elt, c,d,x,z), random_itensor(elt, g,h,z,y), random_itensor(elt, e,f,y,w)];
  r = Index(2000,"CP_rank")
  CP = random_CPD_square_network(square, r);
  ## TODO write better code to take norm of square lattice
  check = ITensorCPD.FitCheck(1e-3, 30, sqrt((contract(square) * contract(square))[]))
  @time opt_A = ITensorCPD.als_optimize(CP, r, check);
  @test norm(ITensorCPD.reconstruct(opt_A) - contract(square)) / sqrt((contract(square) * contract(square))[]) ≤ 1.0 - ITensorCPD.fit(check)
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
    s = insert_linkinds(s; link_space = 2)
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
    for i in 1:length(tn) - 1
      cis = inds(tn[i])
      is = commoninds(tn[i], tn[i+1])
      nis = [i ∈ is ? i' : i for i in cis]
      replaceinds!(ntn[i], cis, nis)
      cis = inds(tn[i+1])
      nis = [i ∈ is ? i' : i for i in cis]
      replaceinds!(ntn[i+1], cis, nis)
    end

    i = length(tn)
    cis = inds(tn[i])
    is = commoninds(tn[i], tn[1])
    nis = [i ∈ is ? i' : i for i in cis]
    replaceinds!(ntn[i], cis, nis)
    cis = inds(tn[1])
    nis = [i ∈ is ? i' : i for i in cis]
    @show nis
    replaceinds!(ntn[1], cis, nis)
    return ntn
  end

using ITensorNetworks
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

nx = 7
ny = 7

beta = 0.1
h = 1.0
tn = ising_network(Float64, IndsNetwork(named_grid((nx, ny))), beta; h)

# s = subgraph(g, (
#   (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), 
#   (2,10), (3,10), (4,10), (5,10), (6,10), (7,10), (8,10), (9,10), (10, 10), 
#   (10, 9), (10,8), (10,7), (10,6), (10,5), (10,4), (10,3), (10, 2), (10,1), 
#   (9,1), (8,1), (7,1), (6,1), (5,1), (4,1), (3,1), (2,1)));

# s = subgraph(tn, (
# (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9), 
# (3,9), (4,9), (5,9), (6,9), (7,9), (8,9), (9,9), 
# (9,8), (9,7), (9,6), (9,5), (9,4), (9,3), (9, 2), 
# (8,2), (7,2), (6,2), (5,2), (4,2), (3,2)
# ));
# s = subgraph(tn, ((2,2), (2,3), 
#                   (3,3), (3,2)))
s1 = subgraph(tn, ((2,2), (2,3), (2,4),(2,5), (2,6),
                  (3,6), (4,6), (5,6), (6,6),
                  (6,5), (6,4), (6,3), (6,2),
                  (5,2), (4,2), (3,2)))
s2 = subgraph(tn, ((3,3), (3,4), (3,5),
                  (4,5), (5,5),
                  (5,4), (5,3),
                  (4,3)))

sising = s1.data_graph.vertex_data.values ;
sisingp = replace_inner_w_prime_loop(sising);

sqrs = sising[1] * sisingp[1]
for i in 2:length(sising)
    sqrs = sqrs * sising[i] * sisingp[i]
end
sqrt(sqrs[])

fit = ITensorCPD.FitCheck(1e-15, 100, sqrt(sqrs[]));
r = Index(20, "CP_rank")
cp = ITensorCPD.random_CPD_square_network(sising, r);
cpopt = ITensorCPD.als_optimize(cp, r, fit);
