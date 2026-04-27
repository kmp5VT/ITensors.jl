using Base.ScopedValues: @with, ScopedValue

# The shape of this scaffold borrows from two prior-art conceptual
# patterns in the Julia tensor / linear-algebra ecosystem:
#
# - MatrixAlgebraKit.jl's `select_algorithm(f, A, alg; kwargs...)` with
#   an `AbstractAlgorithm` supertype and a trait-dispatched
#   `default_algorithm(f, ::Type)` fallback.
# - TensorOperations.jl's `select_backend(tensorfun, tensors...)` with
#   an `AbstractBackend` supertype, a `DefaultBackend` sentinel, and
#   per-tensor-type overloads.
#
# Both let callers pass an explicit choice or fall through to
# auto-selection by input type. Neither covers a *scoped default* — i.e.
# preferring an algorithm for a block of code without threading it
# through every call inside. The addition here is the
# `CURRENT_CONTRACT_ALGORITHM::ScopedValue` and `with_contract_algorithm`
# layer, plus the per-algorithm `is_applicable` predicate that lets a
# scoped preference fall through cleanly when it can't handle the
# inputs at hand. `select_contract_algorithm` glues them with a fixed
# precedence: explicit arg → scoped-if-applicable → trait-dispatched
# default.

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

Tag for the native NDTensors contract path. `contract!` methods
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

Whether `alg` can handle a contraction of `t1` and `t2`. The value form
forwards to the type form by default; backends that need runtime
information (e.g. block counts, sizes) can overload the value form
directly.

The supertype default at `(::ContractAlgorithm, ::Type, ::Type)` returns
`true` — i.e. "assume applicable unless the algorithm says otherwise."
Concrete narrow algorithms typically opt out for everything they can't
handle by adding an explicit reject-everything overload at their own
type, then opt back in for the input types they do handle:

    is_applicable(::MyAlg, ::Type, ::Type) = false
    is_applicable(::MyAlg, ::Type{<:MyTensorType}, ::Type{<:MyTensorType}) = true

Universally-applicable algorithms can simply inherit the supertype
default and skip the per-algorithm reject overload.
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
