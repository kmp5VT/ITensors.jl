using ITensors: ITensor, Index
function had_contract(A::ITensor, B::ITensor, had::Index; α = true)
  if had ∉ commoninds(A, B)
    return α .* A * B
  else
    position_of_had_A = findall(x -> x==had, inds(A))[]
    position_of_had_B = findall(x -> x==had, inds(B))[]
    slices_A = eachslice(array(A); dims=position_of_had_A)
    slices_B = eachslice(array(B); dims=position_of_had_B)

    @assert length(slices_A) == length(slices_B)
    inds_c = noncommoninds(A,B)
    elt = promote_type(eltype(A), eltype(B))
    C = ITensor(elt, vcat(had, inds_c...))
    ## Right now I have to fill C with zeros because I hate empty tensor
    fill!(C, zero(elt))
    slices_C = eachslice(array(C); dims=1)
    for i in 1:length(slices_A)
      a_inds = [ind(A, x) for x in 1:ndims(A) if x != position_of_had_A]
      b_inds = [ind(B, x) for x in 1:ndims(B) if x != position_of_had_B]
      ITensors.contract!(itensor(slices_C[i], inds_c), itensor(slices_A[i], a_inds),  itensor(slices_B[i], b_inds), elt(α))
    end
    return C
  end
end