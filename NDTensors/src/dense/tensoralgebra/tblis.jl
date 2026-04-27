using LinearAlgebra: LinearAlgebra

"""
    TBLIS <: ContractAlgorithm

Algorithm tag for the TBLIS dense contraction backend. Selected via
[`with_tblis`](@ref). The implementation lives in `NDTensorsTBLISExt`;
this file declares the tag plus applicability so the selection layer
can route correctly without `TBLIS.jl` being loaded.

TBLIS has bad/slow complex support, so [`is_applicable`](@ref) restricts
this algorithm to dense inputs with matching `Float32` / `Float64`
element types; other contractions in a `with_tblis` scope fall through
to the default.
"""
struct TBLIS <: ContractAlgorithm end

is_applicable(::TBLIS, ::Type, ::Type) = false
function is_applicable(
        ::TBLIS,
        T1::Type{<:DenseTensor{<:LinearAlgebra.BlasReal}},
        T2::Type{<:DenseTensor{<:LinearAlgebra.BlasReal}}
    )
    return eltype(T1) === eltype(T2)
end

"""
    with_tblis(f, enable::Bool = true)

Run `f()` with the TBLIS dense contraction backend preferred for any
contractions inside the block. When `enable = false`, runs `f()`
unchanged.

Equivalent to `with_contract_algorithm(f, TBLIS())`. Requires
`NDTensorsTBLISExt` to be loaded (i.e. `using TBLIS`); the scope is
otherwise inert (no `TBLIS` `contract!` method exists). Applies only to
real-eltype `DenseTensor` × `DenseTensor` contractions where both
inputs share the same `Float32` or `Float64` eltype; other contractions
inside the scope fall through to the default.
"""
function with_tblis(f, enable::Bool = true)
    return enable ? with_contract_algorithm(f, TBLIS()) : f()
end
