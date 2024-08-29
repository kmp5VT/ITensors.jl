using Test
include("$(@__DIR__)/../ITensorCPD.jl")
using .ITensorCPD: row_norm, reconstruct
using ITensors: Index, ITensor, array, dim, norm, random_itensor

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

@testset "Standard CPD, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
## Working here
end

f = 3
m = similar(cp.factors[f])

factor_portion = cp.factors[1:end .!= f]
for i in 1:dim(r)
  array(m)[i,:] = array(A * contract(map(x -> itensor(array(x)[i,:], ind(x, 2)), factor_portion)))
end

exact = fill!(similar(cp.factors[f]), 0.0)
for R in 1:dim(r)
  for I in 1:dim(i)
    for J in 1:dim(j)
      for K in 1:dim(k)
        if f == 1
          exact[R,I] += A[I,J,K] * cp.factors[2][R,J] * cp.factors[3][R,K]
        elseif f == 2
          exact[R,J] += A[I,J,K] * cp.factors[1][R,I] * cp.factors[3][R,K]
        else
          exact[R,K] += A[I,J,K] * cp.factors[1][R,I] * cp.factors[2][R,J]
        end
      end
    end
  end
end
array(m) - array(exact)


i,j,k = Index.((2,3,4))
r = Index(5, "CP_rank")
A = random_itensor(i,j,k)
cp = random_CPD(A, r)

a = cp.factors .* prime.(cp.factors; tags="CP_rank")
b = array.(cp.factors) .* transpose.(array.(cp.factors))

array(a[3]) - array(b[3])

i,j,k,l = Index.((20,4,5,6))
r = Index(40, "rank")
v = randomITensor(i,j,k,l)
@btime als_optimize(random_CPD(v, r, ;algorithm = direct()), r; maxiters=100);
@btime als_optimize(random_CPD(v, r,), r; maxiters=100);

exact = fill!(similar(facs[2]), 0.0)
for R in 1:dim(r)
  for I in 1:dim(i)
    for J in 1:dim(j)
      for K in 1:dim(k)
        exact[R,J] += v[I,J,K] * facs[1][R,I] * facs[3][R,K]
      end
    end
  end
end
using BenchmarkTools
@benchmark mttkrp(direct(),cp_v.factors, v, r, 2)
norm(exact - mkp)

a,b,c,d,e,f,g,h = Index.((3,3,3,3,3,3,3,3), ("a","b","c","d","e","f","g","h"))
# w,x,y,z = Index.((5,5,5,5), ("w","x","y","z"))
# square = [random_itensor(a,b,x,w), random_itensor(c,d,x,z), random_itensor(g,h,z,y), random_itensor(e,f,y,w)]
# r = Index(2,"rank")
# CP = random_CPD_square_network(square, r)


# facs = cp_v.factors
# p = fill!(Array{Float64}(undef, 2,3,3), 1.0)
# m = itensor(similar(p), r,j,k)
# for i in 1:2
#   for j in 1:3
#     for k in 1:3
#       p[i,j,k] = array(facs[2])[i,j] * array(facs[3])[i,k]
#     end
#   end
# end
# factor_portion = facs[1:end .!= 1]

# v = random_itensor(i,j)
# U,S,V = svd(v, i)
# reconstruct(CPD(v, permute!.([tensor(U),tensor(V)], (2,1)), S))

# for i in 1:dim(r)
#   array(m)[i,:,:] = array(contract(map(x -> itensor(array(x)[i,:], ind(x, 2)), factor_portion)))
# end
# # delta(r) * 

# fill!(Array{Float64}(undef, 2,1,1), 1.0)


# sum((v.tensor ./ (sqrt.(itensor(tensor(v.tensor .* v.tensor)) * delta(i)) * delta(i)).tensor)[1,:])

# c = random_CPD(v, r)

# n = itensor(c.factors[1].tensor .* c.factors[1].tensor, inds(c[1]))
# m = n * ITensor(one(Float64), i)

# norm(p)
# CP.factors

