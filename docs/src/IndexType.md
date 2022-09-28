# Index

## Description

```@docs
Index
ITensors.QNIndex
```

## Constructors

```@docs
Index(::Int)
Index(::Int, ::Union{AbstractString, TagSet})
Index(::Pair{QN, Int}...)
Index(::Vector{Pair{QN, Int}})
Index(::Vector{Pair{QN, Int}}, ::Union{AbstractString, TagSet})
copy(::Index)
sim(::Index)
```

## Properties

```@docs
id(::Index)
hasid(::Index, ::ITensors.IDType)
tags(::Index)
hastags(::Index, ::Union{AbstractString,TagSet})
plev(::Index)
hasplev(::Index, ::Int)
hasind(::Index)
dim(::Index)
==(::Index, ::Index)
dir(::Index)
hasqns(::Index)
```

## Operations

```@docs
dag(::Index)
removeqns(::Index)
```

## Priming and tagging methods

```@docs
prime(::Index, ::Int)
adjoint(::Index)
^(::Index, ::Int)
setprime(::Index, ::Int)
noprime(::Index)
settags(::Index, ::Any)
addtags(::Index, ::Any)
removetags(::Index, ::Any)
replacetags(::Index, ::Any, ::Any)
```

## Iterating

```@docs
eachval(::Index)
eachindval(::Index)
```
