# ITensor

## Description

```@docs
ITensor
```

## Constructors

### Dense Constructors

```@docs
ITensor(::Type{<:Number}, ::ITensors.Indices)
ITensor(::Type{<:Number}, ::UndefInitializer, ::ITensors.Indices)
ITensor(::Type{<:Number}, ::Number, ::ITensors.Indices)
ITensor(::ITensors.AliasStyle, ::Type{<:Number}, ::Array{<:Number}, ::ITensors.Indices{Index{Int}}; kwargs...)
randomITensor(::Type{<:Number}, ::ITensors.Indices)
onehot
```

### Dense View Constructors

```@docs
itensor(::Array{<:Number}, ::ITensors.Indices)
dense(::ITensor)
```

### QN BlockSparse Constructors

```@docs
ITensor(::Type{<:Number}, ::QN, ::ITensors.Indices)
ITensor(::ITensors.AliasStyle, ::Type{<:Number}, ::Array{<:Number}, ::ITensors.QNIndices; tol=0)
ITensor(::Type{<:Number}, ::UndefInitializer, ::QN, ::ITensors.Indices)
```

### Diagonal constructors

```@docs
diagITensor(::Type{<:Number}, ::ITensors.Indices)
diagITensor(::ITensors.AliasStyle, ::Type{<:Number}, ::Vector{<:Number}, ::ITensors.Indices)
diagITensor(::ITensors.AliasStyle, ::Type{<:Number}, ::Number, ::ITensors.Indices)
delta(::Type{<:Number}, ::ITensors.Indices)
```

### QN Diagonal constructors

```@docs
diagITensor(::Type{<:Number}, ::QN, ::ITensors.Indices)
delta(::Type{<:Number}, ::QN, ::ITensors.Indices)
```

### Convert to Array

```@docs
Array{ElT, N}(::ITensor, ::ITensors.Indices) where {ElT, N}
array(::ITensor, ::Any...)
matrix(::ITensor, ::Any...)
vector(::ITensor, ::Any...)
array(::ITensor)
matrix(::ITensor)
vector(::ITensor)
```

### Copy constructors
```@docs
copyto!
```

## Properties

```@docs
storage(::ITensor)
order(::ITensor)
inds(::ITensor)
ind(::ITensor, ::Int)
maxdim(::ITensor)
mindim(::ITensor)
dim(::ITensor)
dim(::ITensor, ::Int)
dims(::ITensor)
dir(::ITensor, ::Index)
ishermitian(::ITensor; kwargs...)
```

## Iterators

```@docs
CartesianIndices(::ITensor)
eachindval(::ITensor)
iterate(::ITensor, args...)
eachnzblock(::ITensor)
eachindex(::ITensor)
```

## ITensor accessors/mutators

```@docs
complex(::ITensor)
scalar(::ITensor)
fill!(::ITensor, ::Number)
getindex(::ITensor, ::Any...)
setindex!(::ITensor, ::Number, ::Int...)
```

## ITensor index functions

### Collecting/comparing tensor indices

```@docs
hasinds
hascommoninds
commoninds
commonind
noncommoninds
noncommonind
uniqueinds
uniqueind
unioninds
unionind
anyhastags
allhastags
```

### [Modifying tags and index sets](@id Priming_and_tagging_ITensor)

```@docs
prime(::ITensor, ::Any...)
setprime(::ITensor, ::Any...)
noprime(::ITensor, ::Any...)
mapprime(::ITensor, ::Any...)
swapprime(::ITensor, ::Any...)
addtags(::ITensor, ::Any...)
removetags(::ITensor, ::Any...)
replacetags(::ITensor, ::Any...)
settags(::ITensor, ::Any...)
swaptags(::ITensor, ::Any...)
replaceind(::ITensor, ::Any...)
replaceinds(::ITensor, ::Any...)
swapind(::ITensor, ::Any...)
swapinds(::ITensor, ::Any...)
```

## Math operations

```@docs
dag(::ITensor; kwargs...)
*(::ITensor, ::ITensor)
normalize!(::ITensor)
exp(::ITensor, ::Any, ::Any)
nullspace(::ITensor, ::Any...)
axpy!
scale!
mul!
```

## Permutations operations

```@docs
permute(::ITensor, ::Any)
transpose(::ITensor)
adjoint(::ITensor)
```
