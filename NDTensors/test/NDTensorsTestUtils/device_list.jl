using Pkg: Pkg
using NDTensors: NDTensors

if "cuda" in ARGS || "all" in ARGS
  ## Right now adding CUDA during Pkg.test results in a
  ## compat issues. I am adding it back to test/Project.toml
  Pkg.add("CUDA")
  using CUDA
end
if "rocm" in ARGS || "all" in ARGS
  ## Warning AMDGPU does not work in Julia versions below 1.8
  Pkg.add("AMDGPU")
  using AMDGPU
end
if "metal" in ARGS || "all" in ARGS
  ## Warning Metal does not work in Julia versions below 1.8
  Pkg.add("Metal")
  using Metal
end
if "cutensor" in ARGS || "all" in ARGS
  if in("TensorOperations", map(v -> v.name, values(Pkg.dependencies())))
    Pkg.rm("TensorOperations")
  end
  Pkg.add("cuTENSOR")
  Pkg.add("CUDA")
  using CUDA, cuTENSOR
end
if isempty(ARGS) || VERSION > v"1.7"
  Pkg.add("JLArrays")
  using JLArrays: jl
end

function devices_list(test_args)
  devs = Vector{Function}(undef, 0)
  if isempty(test_args) || "base" in test_args
    push!(devs, NDTensors.cpu)
    ## Skip jl on lower versions of Julia for now
    ## all linear algebra is failing on Julia 1.6 with JLArrays
    if VERSION > v"1.7"
      push!(devs, jl)
    end
  end

  if "cuda" in test_args || "cutensor" in test_args || "all" in test_args
    if CUDA.functional()
      push!(devs, NDTensors.CUDAExtensions.cu)
    else
      println(
        "Warning: CUDA.jl is not functional on this architecture and tests will be skipped."
      )
    end
  end

  if "rocm" in test_args || "all" in test_args
    push!(devs, NDTensors.AMDGPUExtensions.roc)
  end

  if "metal" in test_args || "all" in test_args
    push!(devs, NDTensors.MetalExtensions.mtl)
  end

  return devs
end
