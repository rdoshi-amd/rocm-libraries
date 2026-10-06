# Validate the JIT implementation

The JIT tests check the comgr code-object builder, the source bundle reader,
the TensileLite loader and, through the replay backend, the internal entry
points that run a JIT solution with the GEMM APIs. The JIT headers are not
installed. The tests include them from `library/src/amd_detail`. They build
the gfx90a, gfx942 and gfx950 kernel assembly committed in
[`data`](data/README.md), as the bundles that the `jit-bundles` test writes
with their library entries and manifests, so they need neither Python nor a
generator.

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
need device 0, which must be a gfx90a, gfx942 or gfx950; they run the bundles of
its architecture. Each test empties its own directory
under `clients/tests/jit/scratch` in the build directory before it runs.

The CTest tests are:

- `jit-cpu`: `jit-bundles`, `jit-source-bundle` and `jit-builder`. A build
  with `HIPBLASLT_ENABLE_JIT=OFF` has `jit-source-bundle` and `jit-disabled`.
  CTest runs `jit-bundles` before each test that reads a bundle.
- `jit-gpu`: `jit-loader`, and `jit-end-to-end` in a build with
  `HIPBLASLT_JIT_TESTING=ON`, when `GPU_TARGETS` include an architecture with
  committed bundles. A build with
  `HIPBLASLT_ENABLE_YAML=ON` has neither, because the library entry is
  MsgPack.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-bundles` | Writes each bundle that `bundle_writer.cpp` describes under `clients/tests/jit/scratch/bundles/<architecture>` in the build directory: the committed assembly, and a library entry and manifest built from it and from the description of each solution. `plain` has one solution; `plain-pair` has two that run the plain kernel, the first for K a multiple of 512 and the second for any K |
| `jit-source-bundle` | The source bundle reader: assembly, HIP sources and headers sorted by name, relative paths, symbolic links that escape the bundle, size limits, and a library entry that is missing, empty or not named `library/TensileLibrary.dat` |
| `jit-builder` | The comgr builder, without a GPU, building the hand-written HIP kernel `builder_test_kernel.hip` for each written bundle's target, then each bundle's assembly linked with that kernel into one code object; each code object defines its kernels and has the builder's code-object version, and a kernel name it does not define fails the build |
| `jit-loader` | `plain` and `plain-pair` built with comgr for device 0 and loaded through the TensileLite loader. `plain` selects its solution for the FP16 GEMM it was generated for; `plain-pair` selects its first solution for K=512 and its second for K=256; neither selects anything for a transposed A. The loader rejects an entry with solutions 0 and 2, a solution whose kernel was not built, and a built kernel no solution names. Launches no kernel |
| `jit-end-to-end` | `plain-pair` replayed, built with comgr and loaded. `getJitAlgo` returns its first solution for K=512 and its second for K=256, and each runs through `hipblasLtMatmul` and `hipblaslt_ext::Gemm` with D checked against a host reference; a problem neither solution solves is not supported |
| `jit-disabled` | The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library |

## Test arguments

`hipblaslt-jit-bundle-writer` takes the `data` directory and an output
directory. `hipblaslt-jit-builder-test` takes the output directory, the HIP
kernel and a scratch directory. `hipblaslt-jit-end-to-end-test` takes the output directory
and replays the `plain-pair` of device 0's architecture. It creates the replay
backend with `jit::replay::createBackend` from `hipblaslt-jit-replay.hpp`, so
generation runs no generator, and solves FP16 problems with M=256, N=128 and
K=512 or 256. `hipblaslt-jit-loader-test` takes the output directory and a
scratch directory, and checks the same problems with the bundles of the same
architecture. `hipblaslt-jit-source-bundle-test` takes a scratch directory.
