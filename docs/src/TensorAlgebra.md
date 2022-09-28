# ITensor Algebra

## Description
It is often necessary to perform tensor algebra (vector, matrix or higher order)
operations on a tensor or set of tensors. There are a number of acceptable
operations offered. Tensor algebra is a current area of development and research
for the ITensor team.

## Matrix Decompositions
These can be found in `src/tensor_operations/matrix_decomposition.jl`

```@docs
svd(::ITensor, ::Any...)
eigen(::ITensor, ::Any, ::Any)
factorize(::ITensor, ::Any...)
```

## Tensor Contractions
