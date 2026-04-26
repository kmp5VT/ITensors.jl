using Base.ScopedValues: @with, ScopedValue

"""
    abstract type ContractAlgorithm

Supertype for tags that select an algorithm for tensor contraction.

Concrete subtypes are dispatched on by `contract!` to choose between
implementations (e.g. the native NDTensors block loop, an external library
like cuTENSOR, ...). The supertype-default `is_applicable` returns `true`,
so each concrete algorithm declares its support set with explicit
`is_applicable` overloads.

See also: [`DefaultContract`](@ref), [`NativeContract`](@ref),
[`select_contract_algorithm`](@ref), [`with_contract_algorithm`](@ref).
"""
abstract type ContractAlgorithm end

"""
    DefaultContract <: ContractAlgorithm

Sentinel meaning "no algorithm chosen — please auto-pick" (the role MAK's
`DefaultAlgorithm` and TensorOperations' `DefaultBackend` play). Triggers
[`select_contract_algorithm`](@ref) to consult the current scope and the
trait-dispatched [`default_contract_algorithm`](@ref) fallback.
"""
struct DefaultContract <: ContractAlgorithm end

"""
    NativeContract <: ContractAlgorithm

Tag for the native NDTensors contract path. `_contract!` methods
dispatched on `::NativeContract` carry the existing per-tensor-type
implementations (block-sparse loop, dense GEMM, diag, ...).
"""
struct NativeContract <: ContractAlgorithm end

"""
    CURRENT_CONTRACT_ALGORITHM::ScopedValue{ContractAlgorithm}

Holds the user's currently-scoped contract-algorithm preference. Default
is `DefaultContract()` (no preference). Set via
[`with_contract_algorithm`](@ref).
"""
const CURRENT_CONTRACT_ALGORITHM = ScopedValue{ContractAlgorithm}(DefaultContract())

"""
    with_contract_algorithm(f, alg::ContractAlgorithm)

Run `f()` with `alg` as the scoped contract-algorithm preference.

The scoped algorithm is a *preference*, not a command — it applies only
when [`is_applicable`](@ref) returns `true` for the inputs at hand.

# Example

```julia
with_contract_algorithm(MyAlg()) do
    return A * B    # uses MyAlg if applicable; otherwise default
end
```
"""
function with_contract_algorithm(f, alg::ContractAlgorithm)
    return @with CURRENT_CONTRACT_ALGORITHM => alg f()
end

"""
    is_applicable(alg::ContractAlgorithm, t1, t2) -> Bool
    is_applicable(alg::ContractAlgorithm, T1::Type, T2::Type) -> Bool

Whether `alg` can handle a contraction of `t1` and `t2`. Backends opt in
by overloading the type form for their supported input types; the value
form forwards to the type form by default. Backends that need runtime
information (e.g. block counts) can overload the value form directly.

The supertype default returns `true` — each concrete algorithm declares
its own support set with an explicit reject-everything overload plus
specific accepts, e.g.:

    is_applicable(::MyAlg, ::Type, ::Type) = false
    is_applicable(::MyAlg, ::Type{<:MyTensorType}, ::Type{<:MyTensorType}) = true
"""
is_applicable(alg::ContractAlgorithm, t1, t2) =
    is_applicable(alg, typeof(t1), typeof(t2))
is_applicable(::ContractAlgorithm, ::Type, ::Type) = true

"""
    default_contract_algorithm(t1, t2) -> ContractAlgorithm
    default_contract_algorithm(T1::Type, T2::Type) -> ContractAlgorithm

The trait-dispatched default algorithm for inputs of these types, used
when no explicit algorithm is passed and no scoped preference applies.
The supertype default returns [`NativeContract`](@ref) — i.e., NDTensors'
own contract machinery handles anything not claimed by another backend.
Extensions can overload to return a different default for specific
input types (e.g. an extension might register
`default_contract_algorithm(::Type{<:T}, ::Type{<:T}) = MyBackend()`).
"""
default_contract_algorithm(t1, t2) =
    default_contract_algorithm(typeof(t1), typeof(t2))
default_contract_algorithm(::Type, ::Type) = NativeContract()

"""
    select_contract_algorithm(alg::ContractAlgorithm, t1, t2) -> ContractAlgorithm

Resolve the contract algorithm for the call. Precedence:

 1. Explicit non-default `alg` argument wins.
 2. Otherwise: if the scoped preference is applicable, use it.
 3. Otherwise: fall back to [`default_contract_algorithm`](@ref).
"""
select_contract_algorithm(alg::ContractAlgorithm, t1, t2) = alg

function select_contract_algorithm(::DefaultContract, t1, t2)
    return _select_contract_algorithm(CURRENT_CONTRACT_ALGORITHM[], t1, t2)
end

_select_contract_algorithm(::DefaultContract, t1, t2) = default_contract_algorithm(t1, t2)

function _select_contract_algorithm(alg::ContractAlgorithm, t1, t2)
    !is_applicable(alg, t1, t2) && return default_contract_algorithm(t1, t2)
    return alg
end

"""
    TensorAndContractionPlan{T<:Tensor, P}

Wrapper that bundles a contraction output tensor with auxiliary context
needed downstream — currently a contraction plan for block-sparse
contractions. The `contraction_output` overload for
`BlockSparseTensor` × `BlockSparseTensor` returns a
`TensorAndContractionPlan`, letting the bundled plan flow through the
in-place `contract!` chain without changing the entry-point signature
across tensor types.
"""
struct TensorAndContractionPlan{T <: Tensor, P}
    tensor::T
    contraction_plan::P
end

# In-place entries that pick the algorithm and dispatch on the tag.
# `::typeof(dest)` / `::T` annotations form a function-barrier so the
# algorithm-tagged dispatch downstream is type-stable on the return.

function contract!(
        dest::Tensor, lR, t1::Tensor, l1, t2::Tensor, l2,
        α::Number = one(Bool), β::Number = zero(Bool)
    )
    alg = select_contract_algorithm(DefaultContract(), t1, t2)
    return contract!(alg, dest, lR, t1, l1, t2, l2, α, β)::typeof(dest)
end

function contract!(
        dest::TensorAndContractionPlan{T}, lR, t1::Tensor, l1, t2::Tensor, l2
    ) where {T <: Tensor}
    alg = select_contract_algorithm(DefaultContract(), t1, t2)
    return contract!(alg, dest, lR, t1, l1, t2, l2)::T
end
