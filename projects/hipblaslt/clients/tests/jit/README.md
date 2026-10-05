# Validate the JIT implementation

The JIT tests check the comgr code-object builder, the source bundle reader,
the TensileLite loader and, through the replay backend, the internal entry
points that run a JIT solution with the GEMM APIs. The JIT headers are not
installed. The tests include them from `library/src/amd_detail`. They build
the gfx950 source bundles committed in [`data`](data/README.md), so they need
neither Python nor a generator.

## Build and run from a checkout

From the repository root, with `project_build` set as in the
[JIT build instructions](../../../JIT.md#build):

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_JIT_TESTING=ON -DHIPBLASLT_BUILD_TESTING=ON \
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
  `HIPBLASLT_ENABLE_JIT=OFF` has `jit-source-bundle` and `jit-disabled`.
- `jit-gpu`: `jit-loader`, and `jit-end-to-end`, `jit-api-streamk`,
  `jit-api-amax` and `jit-api-alpha-zero` in a build with
  `HIPBLASLT_JIT_TESTING=ON`, when `GPU_TARGETS` include gfx950. A build with
  `HIPBLASLT_ENABLE_YAML=ON` has none of them, because the committed library
  entries are MsgPack.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | The source bundle reader: relative paths, symbolic links that escape the bundle, size limits, and a library entry that is missing, empty or not named `library/TensileLibrary.dat` |
| `jit-builder` | The comgr builder building the committed bundle's assembly and a HIP helper unit for the bundle's target, without a GPU; the code object defines both kernels and has the builder's code-object version |
| `jit-loader` | The committed bundle built with comgr for device 0 and loaded through the Tensile loader; its library selects its solution for the FP16 GEMM it was generated for and nothing for a transposed A. Launches no kernel |
| `jit-end-to-end` | The committed bundle replayed, built with comgr and loaded, then run through `hipblasLtMatmul` and `hipblaslt_ext::Gemm` with D checked against a host reference; a problem the bundle does not solve is not supported |
| `jit-api-streamk`, `jit-api-amax` | The bundle of that name replayed and run through `hipblasLtMatmul` and `hipblaslt_ext::Gemm`: copied algorithms, forged tokens and indices rejected, the workspace rules, repeated runs with changed inputs, a second solution beside the first, and a rejected reinitialization that keeps the prepared solution; D, and the amax output, checked against a host reference |
| `jit-api-alpha-zero` | Alpha=0 with null A and B and a nonzero K still computes beta*C and the amax output through both APIs |
| `jit-disabled` | The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library |

The `GemmPointerCheck` tests in `hipblaslt-test` check that `Gemm::setProblem`
rejects a null A or B when alpha is nonzero, also with K=0, and accepts them
when alpha is zero, in builds with and without JIT.

## Test arguments

`hipblaslt-jit-end-to-end-test` takes a source bundle directory; CTest passes
`data/gfx950/plain`. It creates the replay backend with
`jit::replay::createBackend` from `hipblaslt-jit-replay.hpp`, so generation
runs no generator, and solves an FP16 problem with M=256, N=128 and K=512.
`hipblaslt-jit-builder-test` and `hipblaslt-jit-loader-test` take the same
bundle and a scratch directory; the loader test uses the same problem.
`hipblaslt-jit-source-bundle-test` takes a scratch directory.

`hipblaslt-jit-api-test --replay BUNDLE` runs the API checks on the solution
the replay backend replays from `BUNDLE`. `--m`, `--n`, `--k`, `--trans-b`,
`--amax` and `--alpha-zero` shape the problem, `--workspace-fallback 1` expects
a run with too little workspace to succeed, as Stream-K's does, and
`--second-replay` names the bundle of the second solution. CTest runs the
Stream-K case with `TENSILE_PERSISTENT_FIXED_GRID=16` and
`TENSILE_PERSISTENT_DYNAMIC_GRID=0`.
