# Validate the JIT implementation

The JIT tests check the comgr code-object builder, the source bundle reader
and the TensileLite loader. The JIT headers are not installed. The
tests include them from `library/src/amd_detail`. They build the gfx950 source
bundle committed in [`data`](data/README.md), so they need neither Python nor a
generator.

## Build and run from a checkout

From the repository root, with `project_build` set as in the
[JIT build instructions](../../../JIT.md#build):

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_HOST=ON -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
ctest --test-dir "$project_build/clients/tests/jit" -L jit-cpu --output-on-failure
ctest --test-dir "$project_build/clients/tests/jit" -L jit-gpu --output-on-failure
```

`-L jit-cpu` runs the tests that need no GPU. `-L jit-gpu` runs the tests that
need device 0, which must be a gfx950. Each test empties its own directory
under `clients/tests/jit/scratch` in the build directory before it runs.

The CTest tests are:

- `jit-cpu`: `jit-source-bundle` and `jit-builder`. A build with
  `HIPBLASLT_ENABLE_JIT=OFF` has `jit-source-bundle`.
- `jit-gpu`: `jit-loader`, when `GPU_TARGETS` include gfx950. A build with
  `HIPBLASLT_ENABLE_YAML=ON` does not have it, because the committed library
  entry is MsgPack.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | The source bundle reader: relative paths, symbolic links that escape the bundle, size limits, and a library entry that is missing, empty or not named `library/TensileLibrary.dat` |
| `jit-builder` | The comgr builder building the committed bundle's assembly and a HIP helper unit for the bundle's target, without a GPU; the code object defines both kernels and has the builder's code-object version |
| `jit-loader` | The committed bundle built with comgr for device 0 and loaded through the Tensile loader; its library selects its solution for the FP16 GEMM it was generated for and nothing for a transposed A. Launches no kernel |

## Test arguments

`hipblaslt-jit-builder-test` and `hipblaslt-jit-loader-test` take a source
bundle directory and a scratch directory; CTest passes `data/gfx950/plain`. The
loader test checks an FP16 problem with M=256, N=128 and K=512.
`hipblaslt-jit-source-bundle-test` takes a scratch directory.
