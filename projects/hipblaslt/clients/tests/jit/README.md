# Validate the JIT implementation

The JIT tests check the comgr code-object builder, the `HIPBLASLT_JIT_DEBUG`
lines, the source bundle reader, the TensileLite loader and, through the replay
backend, the internal entry points that run a JIT solution with the GEMM APIs,
and `HIPBLASLT_JIT` through the public heuristic queries and `hipblaslt-bench`.
The JIT headers are not installed. The tests include them from
`library/src/amd_detail`. They build the gfx950 source bundles committed in
[`data`](data/README.md), so they need no generator. The Python tests run with
the interpreter CMake finds, which needs `msgpack`.

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

`HIPBLASLT_JIT_TESTING=ON` makes the test backend the process's JIT backend:
heuristic queries rank candidates with Origami and replay, through the replay
backend, the bundles that `HIPBLASLT_JIT_TEST_REPLAY` lists, separated by `:`
(`;` on Windows). `HIPBLASLT_JIT_TEST_FAULT` set to `generate`, `build`,
`record` or `trap` injects the replay backend's fault, and `record` appends
each request to the file that `HIPBLASLT_JIT_TEST_RECORD` names. The CTest
tests are:

- `jit-cpu`: `jit-source-bundle`, `jit-builder`, `jit-component`, `jit-debug`
  and `jit-code-object`. A build with `HIPBLASLT_ENABLE_JIT=OFF` has
  `jit-source-bundle` and `jit-disabled`.
- `jit-gpu`: `jit-code-object-gpu`, `jit-library`, `jit-library-concurrency`
  and `jit-bundle-freshness`, which read library entries; TensileLite queries
  the current device when it reads one. When `GPU_TARGETS` include gfx950 it
  also has `jit-loader`, and in a build with `HIPBLASLT_JIT_TESTING=ON` also
  `jit-end-to-end`, `jit-end-to-end-splitk`, `jit-failure`, `jit-api-splitk`,
  `jit-api-streamk`, `jit-api-amax`, `jit-api-alpha-zero`,
  `jit-replay-backend`, `jit-replay-backend-library`, `jit-heuristic-<route>`
  for each heuristic route in the table below, run through the test backend
  replaying the `rank-1`, `rank-2` and `splitk` bundles, and with
  `HIPBLASLT_ENABLE_CLIENT=ON` `jit-bench-smoke`. A build with
  `HIPBLASLT_ENABLE_JIT=OFF` has `jit-heuristic-jit-off`, and with
  `HIPBLASLT_ENABLE_CLIENT=ON` `jit-bench-smoke-jit-off`.

A build with `HIPBLASLT_ENABLE_YAML=ON` has no `jit-library`,
`jit-library-concurrency`, `jit-bundle-freshness`, `jit-loader` or test that
replays a bundle, because the library entries are MsgPack.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | The source bundle reader: relative paths, symbolic links that escape the bundle, size limits, and a library entry that is missing, empty or not named `library/TensileLibrary.dat` |
| `jit-builder` | The comgr builder building the committed bundle's assembly and a HIP helper unit for the bundle's target, without a GPU; the code object defines both kernels and has the builder's code-object version |
| `jit-component` | `Jit` over fake stages that break the stage contracts: a missing backend, builder or loader is rejected; a zero count generates nothing; solutions beyond the count are not built; failures, exceptions and a load without a bundle are reported at the stage `Jit` was running; a store that throws or returns too few indices fails at publish, and the solutions load instead; a backend that consumes predictions without a predictor and knowledge is rejected; candidates of a contract the backend does not transport are dropped; a failed or empty prediction stops before generation; allocation failures propagate; no scratch directory is left behind; the catalog knowledge's seeds for each architecture; and, with `HIPBLASLT_JIT_DEBUG=all`, the order of the generation events and the outcome and failure stage of each solution when some solutions of a generation fail to build or are unsupported |
| `jit-debug` | The `HIPBLASLT_JIT_DEBUG` line writer, without a GPU: value parsing and its warning, JSON escaping and truncation, the line size cap, per-process file names, lines from several threads and processes intact in one file, and rate limiting with aggregate lines |
| `jit-code-object` | comgr assembly, HIP helper compilation and linking for gfx950, build options, the time of each comgr action without a change to the output, concurrent builds, and the status and log of each kind of failed build, without a GPU; with `--bundle`, the same for the committed split-K bundle |
| `jit-code-object-gpu` | The same code objects built for device 0, loaded and launched, with their results checked; the split-K bundle only in a build for gfx950 |
| `jit-bundle-freshness` | Each committed bundle's layout and code-object versions against this tree, its library entry read by the host library, and its build; a copy whose manifest has another layout version must be reported stale |
| `jit-loader` | The committed bundle built with comgr for device 0 and loaded through the Tensile loader; its library selects its solution for the FP16 GEMM it was generated for and nothing for a transposed A. Launches no kernel |
| `jit-end-to-end`, `jit-end-to-end-splitk` | The `plain` or `splitk` bundle replayed, built with comgr and loaded, then run through `hipblasLtMatmul` and `hipblaslt_ext::Gemm` with D checked against a host reference; a problem the bundle does not solve is not supported |
| `jit-failure` | The replay backend, comgr builder and TensileLite loader through `Jit`: count, order and excluded kernels, a workspace limit failing support, generation and build faults that keep their log, a code object for another XNACK setting failing to load, damaged bundles rejected with their message, and a split-K solution without its helper kernels rejected before `hipblasLtMatmul` or `Gemm::initialize` writes D or the workspace |
| `jit-api-splitk`, `jit-api-streamk`, `jit-api-amax` | The bundle of that name replayed and run through `hipblasLtMatmul` and `hipblaslt_ext::Gemm`: copied algorithms, forged tokens and indices rejected, the workspace rules, repeated runs with changed inputs, a second solution beside the first, and a rejected reinitialization that keeps the prepared solution; D, and the amax output, checked against a host reference |
| `jit-api-alpha-zero` | Alpha=0 with null A and B and a nonzero K still computes beta*C and the amax output through both APIs |
| `jit-replay-backend` | The `splitk` bundle replayed through Jit and run through `hipblasLtMatmul` and `hipblaslt_ext::Gemm`: the request owns its scalars, copied algorithms and changed pointers and scalars reuse the solution, too little workspace, a forged token and a nonzero index are rejected before submission, a wrong device is rejected before generation, non-GEMM and mismatched requests are not supported, the record fault records each request, only a backend with a modeled contract replays after an Origami prediction, under the composed version, an unmodeled contract is rejected, and a loaded bundle stays alive while an algorithm refers to it |
| `jit-replay-backend-library` | `getLibraryAlgos` publishes the replayed solution into a fresh JIT solution library and returns a reserved index, which `getAlgosFromIndex` and `hipblasLtMatmul` run with checked numerics. A query for two solutions generates only for the shortfall and skips the published kernel. A second process then runs that index before any lookup, and `getLibraryAlgos` finds it there with a backend that aborts the process if it generates |
| `jit-library` | The JIT solution library: cache-key fields and compiler-environment filtering; rejected group- or other-writable, linked and non-directory roots; the stock TensileLite loader reading a published library; exact-size matching with the solution predicates still applied; deduplication, hash collisions, order, count and excluded kernels; mismatched and tampered keys ignored and left untouched; index allocation up to `INT32_MAX` and exhaustion; a publisher killed after each publication step; readers reloading after another instance publishes; and a fused GEMM and all-to-all problem rejected by lookup, publication and the ProblemType key without touching the library, even beside a plain solution of the same sizes |
| `jit-library-concurrency` | Eight processes publish shared and private entries into one library while another process looks them up: shared entries get one index, private ones unique indices with no gaps, and every reader snapshot loads |
| `jit-disabled` | The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library |
| `jit-heuristic-fallback-c`, `jit-heuristic-fallback-cpp` | `HIPBLASLT_JIT=1` with an empty device library: the C or C++ heuristic query returns only JIT indices for one and three requested solutions, each checked through `hipblasLtMatmul` or `Gemm`, publishes them, and reports any shortfall as a warning |
| `jit-heuristic-forced` | `HIPBLASLT_JIT=2` with an empty and with the build's device library: both queries return only JIT indices with checked numerics |
| `jit-heuristic-cache-hit` | A second process whose backend fails if it generates gets the first process's published index from both queries, with no JIT report; a third process with `HIPBLASLT_JIT=0` resolves that index through `getAlgosFromIndex` and runs it with checked numerics |
| `jit-heuristic-distinct` | With one solution already published, a request for three in mode 2 returns that solution first and two new ones, three distinct kernels in all, with no JIT report |
| `jit-heuristic-unsupported` | A problem the predictor cannot rank (K=0) in modes 1 and 2: exactly one `hipblaslt error: JIT predict failed` line naming the reason across two handles, two queries and both APIs; the queries return no results with the status the mode defines, and nothing is published |
| `jit-heuristic-concurrent` | In modes 1 and 2, four processes of four threads each start the same query through a file barrier: every query returns the same two distinct solutions with checked numerics and no JIT report, and the library holds exactly those two entries with the allocator just past them |
| `jit-heuristic-report` | In modes 1 and 2, a backend that fails to configure and one that fails to generate each print exactly one `hipblaslt error: JIT` line across two handles, two queries and both APIs; the queries return no results with the status the mode defines, and the log named in the report is kept |
| `jit-heuristic-debug-timing` | `HIPBLASLT_JIT_DEBUG=timing` in mode 1: one `process` and one `setup` line, no progress lines, a `generation` line whose stage times add up, one `solution` line per published solution with its HIP compile times, and `query` lines whose `from` counts add up to the returned count; the results equal a run without the variable. A second process gets cache hits and no `generation` line, mode 2 queries take every result from JIT, and of two threads that start together the one that waits names the generation it waited for |
| `jit-heuristic-debug-progress` | `HIPBLASLT_JIT_DEBUG=progress` in mode 1: query, lookup, generation, build and publish events in order, with no timing lines or durations |
| `jit-heuristic-debug-off` | `HIPBLASLT_JIT_DEBUG` unset, empty or `0`: no lines and the same results; `0` and an unknown name each print one warning and leave the names they accompany in effect; with `HIPBLASLT_JIT=0`, any value leaves the output unchanged |
| `jit-heuristic-debug-file` | `HIPBLASLT_JIT_DEBUG_FILE` with `%i` writes one owner-only file per process and nothing to stderr; two processes sharing one file leave every line intact; a file that cannot be opened prints one warning and the lines go to stderr |
| `jit-heuristic-jit-off` | In a build without JIT, `HIPBLASLT_JIT=1` prints one warning across two handles and three queries, leaves the results unchanged, and creates no JIT solution library |
| `jit-bench-smoke` | `hipblaslt-bench` with `HIPBLASLT_JIT=2` and an empty device library runs and verifies one replayed JIT solution with no JIT report and publishes it; a second run whose backend fails if it generates runs the same kernel from the library |
| `jit-bench-smoke-jit-off` | `hipblaslt-bench` in a build without JIT prints one warning for `HIPBLASLT_JIT=2` and creates no JIT solution library |

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
directory. `hipblaslt-jit-component-test`, `hipblaslt-jit-debug-test` and
`hipblaslt-jit-source-bundle-test` take a scratch directory; the component and
debug tests start their child processes themselves.

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
starts, so a wrong device must be rejected before generation. With
`--library` after the bundle it runs the `jit-replay-backend-library` checks
instead, and starts its second process itself. That mode empties
`HIPBLASLT_JIT_LIBRARY_PATH` first and refuses to run unless it is set, so that
it never publishes into the default library.

## JIT solution library tests

`hipblaslt-jit-library-test` compiles the JIT solution library directly. It
needs a GPU, because TensileLite queries the current device when it reads a
library entry. It takes a split-K source bundle, `data/gfx950/splitk` in CTest,
whose library entry it publishes under several kernel names, and a scratch
directory for the libraries it creates; it ignores
`HIPBLASLT_JIT_LIBRARY_PATH`. Adding `--writers N --per-writer M` runs the
multi-process check instead: N writer processes each publish M entries shared
by all writers and M of their own, while one reader process looks them up.

## Heuristic tests

`hipblaslt-jit-heuristic-test` uses only the public API, so it is built with
and without JIT. It queries `hipblasLtMatmulAlgoGetHeuristic` and
`GemmInstance::algoGetHeuristic` for an FP16 GEMM with FP32 accumulation
(M=256, N=128, K=512 by default), prints one JSON line per query with the
status, the returned solution indices, their workspace sizes and kernel names,
and then runs every returned algorithm and compares the output with a CPU
reference. `--api c|cpp|both|none`, `--requested`, `--m`, `--n`, `--k`,
`--handles`, `--queries` and `--workspace` shape the queries,
`--from-index i,j,...` resolves indices through
`hipblaslt_ext::getAlgosFromIndex` and runs them, and `--no-run` skips
execution. `--threads N` runs the whole sequence in N threads, each with its
own handles; with `--barrier DIR` each thread claims a `ready-<n>` file in
`DIR` after creating its first handle and waits for `DIR/go` before its first
query, so that threads in several processes start together.

`test_heuristic.py <binary> <route> <output> --replay BUNDLE` empties
`<output>`, then runs the binary for one route in fresh processes, each with
its own JIT solution library, temporary and cache directories under
`<output>`, and an empty `HIPBLASLT_TENSILE_LIBPATH` unless the route uses the
build's device library. `--replay` is repeated once per bundle; the `jit-off` route takes
none. The build's JIT backend must be the test backend: the routes replay
those bundles, and its record fault stands in for a backend that fails if it
generates.

`test_bench_smoke.py <hipblaslt-bench> <output> --replay BUNDLE` runs
`hipblaslt-bench` the same way, and `--jit-off` instead checks a build without
JIT.

## Code-object tests

`hipblaslt-jit-code-object-test` compiles the comgr code-object builder
directly. `--out` names its results directory, and either `--target` selects a
compile-only run for that target ID or `--gpu` also loads and runs the results
on device 0. `--ffm` runs the GPU part on the simulator that
`HSA_MODEL_TOPOLOGY` and `HSA_MODEL_LIB` select; simulator runs are manual, and
CTest does not run `--ffm`. `--bundle` adds the checks for a TensileLite source
bundle, whose target the device must match. `--only` selects tests by name.
