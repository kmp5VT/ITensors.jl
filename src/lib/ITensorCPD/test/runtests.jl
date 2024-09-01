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
  opt_A = ITensorCPD.als_optimize(CP, r, check);
  @test norm(ITensorCPD.reconstruct(opt_A) - contract(square)) / sqrt((contract(square) * contract(square))[]) ≤ 1.0 - ITensorCPD.fit(check)
end