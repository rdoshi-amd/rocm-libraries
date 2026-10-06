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
need device 0, which must be one of the build's `GPU_TARGETS`. Each test
empties its own directory under `clients/tests/jit/scratch` in the build
directory before it runs.

The CTest tests are:

- `jit-cpu`: `jit-source-bundle`, `jit-builder`, `jit-component` and
  `jit-code-object`. A build with `HIPBLASLT_ENABLE_JIT=OFF` has
  `jit-source-bundle` and `jit-disabled`.
- `jit-gpu`: `jit-code-object-gpu` and `jit-bundle-freshness`, which reads
  library entries; TensileLite queries the current device when it reads one.
  When `GPU_TARGETS` include gfx950 it also has `jit-loader`, and in a build
  with `HIPBLASLT_JIT_TESTING=ON` also `jit-end-to-end`,
  `jit-end-to-end-splitk`, `jit-failure`, `jit-api-splitk`, `jit-api-streamk`,
  `jit-api-amax`, `jit-api-alpha-zero` and `jit-replay-backend`.

A build with `HIPBLASLT_ENABLE_YAML=ON` has no `jit-bundle-freshness`,
`jit-loader` or test that replays a bundle, because the committed library
entries are MsgPack.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | The source bundle reader: relative paths, symbolic links that escape the bundle, size limits, and a library entry that is missing, empty or not named `library/TensileLibrary.dat` |
| `jit-builder` | The comgr builder building the committed bundle's assembly and a HIP helper unit for the bundle's target, without a GPU; the code object defines both kernels and has the builder's code-object version |
| `jit-component` | `Jit` over fake stages that break the stage contracts: a missing backend, builder or loader is rejected; a zero count generates nothing; solutions beyond the count are not built; failures, exceptions and a load without a bundle are reported at the stage `Jit` was running; allocation failures propagate; no scratch directory is left behind |
| `jit-code-object` | comgr assembly, HIP helper compilation and linking for gfx950, build options, concurrent builds, and the status and log of each kind of failed build, without a GPU; with `--bundle`, the same for the committed split-K bundle |
| `jit-code-object-gpu` | The same code objects built for device 0, loaded and launched, with their results checked; the split-K bundle only in a build for gfx950 |
| `jit-bundle-freshness` | Each committed bundle's layout and code-object versions against this tree, its library entry read by the host library, and its build; a copy whose manifest has another layout version must be reported stale |
| `jit-loader` | The committed bundle built with comgr for device 0 and loaded through the Tensile loader; its library selects its solution for the FP16 GEMM it was generated for and nothing for a transposed A. Launches no kernel |
| `jit-end-to-end`, `jit-end-to-end-splitk` | The `plain` or `splitk` bundle replayed, built with comgr and loaded, then run through `hipblasLtMatmul` and `hipblaslt_ext::Gemm` with D checked against a host reference; a problem the bundle does not solve is not supported |
| `jit-failure` | The replay backend, comgr builder and TensileLite loader through `Jit`: count, order and excluded kernels, a workspace limit failing support, generation and build faults that keep their log, a code object for another XNACK setting failing to load, damaged bundles rejected with their message, and a split-K solution without its helper kernels rejected before `hipblasLtMatmul` or `Gemm::initialize` writes D or the workspace |
| `jit-api-splitk`, `jit-api-streamk`, `jit-api-amax` | The bundle of that name replayed and run through `hipblasLtMatmul` and `hipblaslt_ext::Gemm`: copied algorithms, forged tokens and indices rejected, the workspace rules, repeated runs with changed inputs, a second solution beside the first, and a rejected reinitialization that keeps the prepared solution; D, and the amax output, checked against a host reference |
| `jit-api-alpha-zero` | Alpha=0 with null A and B and a nonzero K still computes beta*C and the amax output through both APIs |
| `jit-replay-backend` | The `splitk` bundle replayed through Jit and run through `hipblasLtMatmul` and `hipblaslt_ext::Gemm`: the request owns its scalars, copied algorithms and changed pointers and scalars reuse the solution, too little workspace, a forged token and a nonzero index are rejected before submission, a wrong device is rejected before generation, non-GEMM and mismatched requests are not supported, the record fault records each request, and a loaded bundle stays alive while an algorithm refers to it |
| `jit-disabled` | The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library |

The `GemmPointerCheck` tests in `hipblaslt-test` check that `Gemm::setProblem`
rejects a null A or B when alpha is nonzero, also with K=0, and accepts them
when alpha is zero, in builds with and without JIT.

## Test arguments

`hipblaslt-jit-end-to-end-test` takes a source bundle directory; CTest passes
`data/gfx950/plain` and `data/gfx950/splitk`. It creates the replay backend
with `jit::replay::createBackend` from `hipblaslt-jit-replay.hpp`, so
generation runs no generator, and solves an FP16 problem with M=256, N=128 and
K=512. `hipblaslt-jit-failure-test` takes the `plain` and `splitk` bundles and
a scratch directory, which it also uses as the temporary directory, so that it
can find the scratch directories that `Jit` keeps. It damages copies of the
bundles there, and selects the replay backend's generation and build faults
with `replay::Options::fault`.
`hipblaslt-jit-builder-test` and `hipblaslt-jit-loader-test` take the `plain`
bundle and a scratch directory; the loader test uses the same problem.
`hipblaslt-jit-bundle-freshness-test` takes the `data` directory and a scratch
directory. `hipblaslt-jit-component-test` and
`hipblaslt-jit-source-bundle-test` take a scratch directory.

`hipblaslt-jit-api-test --replay BUNDLE` runs the API checks on the solution
the replay backend replays from `BUNDLE`. `--m`, `--n`, `--k`, `--trans-b`,
`--amax` and `--alpha-zero` shape the problem, `--workspace-fallback 1` expects
a run with too little workspace to succeed, as Stream-K's does, and
`--second-replay` names the bundle of the second solution. CTest runs the
Stream-K case with `TENSILE_PERSISTENT_FIXED_GRID=16` and
`TENSILE_PERSISTENT_DYNAMIC_GRID=0`.

`hipblaslt-jit-replay-backend-test` takes the `splitk` bundle. It also selects
two more replay faults: the record fault, which appends each generation
request to the file that `replay::Options::record` names and fails the
generation, and the trap fault, which aborts the process when a generation
starts, so a wrong device must be rejected before generation.

## Code-object tests

`hipblaslt-jit-code-object-test` compiles the comgr code-object builder
directly. `--out` names its results directory, and either `--target` selects a
compile-only run for that target ID or `--gpu` also loads and runs the results
on device 0. `--ffm` runs the GPU part on the simulator that
`HSA_MODEL_TOPOLOGY` and `HSA_MODEL_LIB` select; simulator runs are manual, and
CTest does not run `--ffm`. `--bundle` adds the checks for a TensileLite source
bundle, whose target the device must match. `--only` selects tests by name.
