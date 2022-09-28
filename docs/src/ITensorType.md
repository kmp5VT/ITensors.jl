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

## Properties

```@docs
storage(::ITensor)
order(::ITensor)
inds(::ITensor)
maxdim(::ITensor)
mindim(::ITensor)
dim(::ITensor)
dim(::ITensor, ::Int)
dims(::ITensor)
size(::ITensor)
dir(::ITensor, ::Index)
ishermitian(::ITensor; kwargs...)
isemptystorage(::ITensor)
```

## Getting and setting elements

```@docs
getindex(::ITensor, ::Any...)
setindex!(::ITensor, ::Number, ::Int...)
```

## [Priming and tagging](@id Priming_and_tagging_ITensor)

```@docs
ind(::ITensor, ::Int)
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
```

## Index collections set operations

```@docs
commoninds
commonind
uniqueinds
uniqueind
noncommoninds
noncommonind
unioninds
unionind
hascommoninds
```

## Index Manipulations

```@docs
replaceind(::ITensor, ::Any...)
replaceinds(::ITensor, ::Any...)
swapind(::ITensor, ::Any...)
swapinds(::ITensor, ::Any...)
```

## Math operations

```@docs
*(::ITensor, ::ITensor)
dag(T::ITensor; kwargs...)
exp(::ITensor, ::Any, ::Any)
nullspace(::ITensor, ::Any...)
```

## Memory operations

```@docs
permute(::ITensor, ::Any)
dense(::ITensor)
```
