# hipBLASLt just-in-time (JIT) GEMM generation

This source guide describes just-in-time (JIT) kernel generation for general
matrix multiplication (GEMM) in hipBLASLt. It is written for hipBLASLt and
TensileLite contributors and integration developers, and it is maintained under
the existing @ROCm/hipblaslt-reviewers and @ROCm/hipblaslt-docs-reviewers rules
in [.github/CODEOWNERS](../../.github/CODEOWNERS). It is not a released
application programming interface (API) or support statement. Release-document
integration is undecided.

It describes what the code implements today. "Implemented" means present in the
source, not released or approved as product naming. A JIT backend is the
generator that turns a request into kernel sources; this page describes the
parts of hipBLASLt around any backend. The
[JIT test guide](clients/tests/jit/README.md) covers building and running the
tests.

## Summary

hipBLASLt runs a GEMM with a kernel from its pre-tuned library, so a problem
that the library serves poorly, or not at all, has no better kernel available.
JIT generation produces kernels for a problem when they are needed. The `Jit`
component defines the stages of that generation and the order in which they
run. Its interfaces, a comgr code-object builder and a TensileLite solution
loader are implemented. Internal entry points let the JIT test binaries
generate a GEMM solution through a backend and run it with `hipblasLtMatmul`
or `hipblaslt_ext::Gemm`. The only backend is a test backend that replays
pre-generated source bundles. No generator backend is implemented yet,
generated solutions live only in the process that built them, and no public
API reaches JIT.

## Current behavior

### Entry points

The internal entry points, in `library/src/amd_detail/hipblaslt-jit.hpp`,
generate through `Jit`, keep generated algorithms in one process-local
registry, and return an algorithm for the existing C and C++ GEMM execution
APIs:

- A backend factory, such as `jit::replay::createBackend`, returns a `Backend`
  handle that owns a `Jit` configured with that backend.
- `jit::makeGemmRequest` captures existing GEMM descriptors and host scalars.
- `jit::getJitAlgo` compiles on the selected device and returns an owned
  `Solution`.
- `jit::getGemmAlgo` adapts the solution to the algorithm that
  `hipblasLtMatmul` and `Gemm` accept.

The header is internal, as are `hipblaslt-jit-replay.hpp` and
`hipblaslt-jit-gemm-internal.hpp`: they are not installed and
`hipblaslt-ext.hpp` does not include them. `libhipblaslt.so` exports four
functions and one type from them with `HIPBLASLT_EXPORT` for the JIT test
binaries, which link against the shared library: `jit::makeGemmRequest`,
`jit::getJitAlgo`, `jit::getGemmAlgo` and the `jit::detail::GemmRequest`
request type, and `jit::replay::createBackend`. No installed header declares
them, and they are not a supported API.

The backend's configuration belongs to the options of its factory. The
application owns its buffers and workspace. The request owns descriptor values
and host scalars; it does not take ownership of device pointers. Compilation
and support checks finish before graphics processing unit (GPU) work is
submitted; call the entry points before stream capture. GEMM is the
implemented operation.

Internally, the GEMM request reuses `RocblasltContractionProblem` with owned
scalar values. The generic `Solution` and private `CompiledSolution` retain
the `Jit`, device target, request, workspace and bundle lifetime around the
existing GEMM support and execution machinery. A matmul algorithm is an
adaptation token, not a general owning executable object.

### Components

`Jit`, in `library/src/amd_detail/hipblaslt-jit-component.{hpp,cpp}`, runs one
request for one device target through these stages. Each stage is an
interface, so that each implementation can be replaced and tested on its own.

| Stage | Interface | Contract |
| --- | --- | --- |
| Generate | `Backend::generate(GenerationRequest, std::vector<GeneratedSolution>&)` | Returns entries holding up to `GenerationRequest::count` solutions, best first, and builds and loads nothing. Each `GeneratedSolution` holds a TensileLite library entry for one processor with one or more solutions, the names of their main kernels, and the source units to build. `NotSupported` means the request is outside the backend's domain. |
| Build | `CodeObjectBuilder::build(GeneratedSolution, BuildRequest, BuiltSolution&)` | Builds an entry's units into one code object. `GeneratedSolution` has no code-object field; only `BuiltSolution` adds the code object. |
| Support | `SolutionLoader::support` | Evaluates the predicates and workspace of the entry's solutions for the request, returns the local indices of those that support it, best first, and loads no code. |
| Load | `SolutionLoader::load` | Loads the code objects once and returns a process-local executable `KernelBundle` for each supported solution. |

`Jit::generate` is the only code that sequences these stages. It asks the
backend for at most the requested count of solutions in a private scratch
directory, forwarding the workspace limit and the kernels the caller already
has. It then builds each entry and checks the support of its solutions until
the count of supported solutions is reached, and loads them. A failure in one
entry's build, support or load skips that entry and keeps the rest.

Each failure is recorded with its stage (configure, generate, build, support
or load) in `Jit::Outcome::failures`, in the order it happened.
The scratch directory is created under the temporary directory, removed when
the call succeeds, and kept after a failure that left files in it. `Jit`
requires a backend, a builder and a loader. One `Jit` can generate from
several threads at once.

`OperationRequest` names only its kind, so `Jit` sees no GEMM. The interfaces
are for compiled-in implementations and do not establish a stable external
plugin application binary interface (ABI).

The implementations are:

- Code-object builder: `makeComgrBuilder()`; see
  [building generated sources](#building-generated-sources).
- Loader: `makeTensileLoader()` reads the entry with the TensileLite loader (see
  [loading a built solution](#loading-a-built-solution)), checks the support
  and workspace of each solution with TensileLite's predicates, in index order,
  and returns a process-local `TensileGemmBundle` for each supported solution.
  The bundles of one entry share its library and adapter. It is in
  `rocblaslt/src/tensile_host.cpp`, next to the GEMM problem translation that
  the support check reuses.
- Replay backend: `hipblaslt-jit-replay-backend.cpp` replays a list of source
  bundles without a generator. A generation returns, in list order, the
  bundles with a solution whose predicates accept the device and problem, until
  they hold the requested count of such solutions. It skips a bundle when every
  such solution's kernel is excluded; the comgr builder still builds them.
  It implements `Backend`, so `Jit` builds and loads its entries with the same
  builder and loader as any other backend. Tests reach it through
  `jit::replay::createBackend` in `hipblaslt-jit-replay.hpp`, and every JIT
  build compiles it.

### Build

`HIPBLASLT_ENABLE_JIT` is disabled by default. A disabled build compiles and
exports no JIT entry points. The enabled build requires the host library, ROCm
and ROCm's `amd_comgr` CMake package, which only a JIT build links. comgr
compiles HIP sources against the host's C and C++ standard library headers,
so those must be installed where JIT runs. From the repository root:

```bash
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
```

The [JIT test guide](clients/tests/jit/README.md) lists the test targets and the
validation commands.

### Algorithm lifetime and failures

The returned heuristic result contains the required workspace size. Supply that
workspace and follow the same handle, stream and workspace sharing rules as
`hipblasLtMatmul` and `Gemm` calls using prebuilt algorithms. The algorithm
carries its solution's local index, resolves to its bundle's library and
adapter, and then runs through the same launch path as a prebuilt algorithm,
including its synchronization storage. Registry synchronization
protects algorithm lookup; it does not protect application buffers or make
simultaneous calls on one `Gemm` object safe.

Copies of an algorithm remain usable on its generating device within the same
process, and its modules are retained until process exit. Reuse within one
program invocation needs no recompilation. The opaque algorithm bytes that
`getJitAlgo` and `getGemmAlgo` return are not a library index:
`hipblaslt_ext::getIndexFromAlgo` returns -1 for them, and a different program
invocation cannot use them. Save the bundle manifests for reproduction.

An empty GEMM output (M=0 or N=0) returns `HIPBLAS_STATUS_NOT_SUPPORTED` from
the request factory without compilation. K=0 can use a solution that
implements beta*C. The backend owns its datatype, instruction and scale-layout
restrictions. The library propagates support failures, including a mismatch
between the supplied physical MX scale layout and the compiled solution. Jit
never benchmarks generated solutions. A failed build names the retained
`comgr.log` in its message.

### Building generated sources

Generators emit assembly or HIP source plus metadata only; they do not assemble,
link or bundle. `makeComgrBuilder()` returns the `CodeObjectBuilder` that builds
code objects in process through AMD comgr (`hipblaslt-jit-builder.cpp` and
`hipblaslt-jit-code-object.cpp`). It uses three comgr actions:

- Assembly: `AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE`. A unit's
  `.amdgcn_target` directive must name the device's processor and only
  features the device has; it is then rewritten to the device's full target
  ID. Assembly units that declare different wavefront sizes are rejected.
- HIP source: `AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE` with
  `--rocm-path` and a content-derived `-cuid`, so HIP objects link together.
  The ROCm path is `HIP_PATH` when set, otherwise the prefix of the loaded HIP
  runtime.
- Link: `AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE` joins the assembly
  and HIP relocatables into one code object per solution, with
  `-Xlinker --build-id=sha1`.

The generator and the builder use the same code-object version, which
`GenerationRequest::codeObjectVersion` and `BuildRequest::codeObjectVersion`
carry (4 by default). The output is a
raw, uncompressed executable code object. comgr cannot bundle or compress it,
and `hipModuleLoadData` accepts raw executable and linkable format (ELF)
objects. A generator needs no offload bundler.

After linking, the builder reads the code object's metadata and checks that its
instruction set architecture (ISA) is the device's and that it defines every
main kernel of the entry's solutions. Its message
carries the first error line of the comgr log, and the builder appends the full
log to `comgr.log` in `BuildRequest::scratch` and names that file in the
message.

### Loading a built solution

`hipblaslt-jit-loader.{hpp,cpp}` turns a built TensileLite entry into a loaded
library of its solutions:

- `readTensileSourceBundle` reads a source bundle into a `TensileSource`: a
  `GeneratedSolution` (the library entry, the main kernel names of its
  solutions, each `sources/*.s` as an assembly unit, and each HIP source with
  the bundle's headers as a HIP unit) and the TensileLite library parsed from
  that entry. A caller that needs the library keeps the one
  `readTensileSourceBundle` returned.
- `parseTensileBundle` reads a built solution's entry into a TensileLite
  `MasterSolutionLibrary` for the device's hardware. It requires local
  solutions 0 to N-1, each naming one of the built kernels and every built
  kernel named by one of them, and loads no code.
- `loadTensileBundle` loads the built code object into a new
  `SolutionAdapter` and resolves every main kernel.

The entry's solutions are listed best first, each with its own `Problem` row.
Their hardware and problem predicates decide which problems each one serves,
so `findBestSolution` on the loaded library selects the first solution that
serves a problem, and nothing for a problem none of them was generated for.

### Source bundle format

A source bundle is a generated library entry stored as a directory.
`source_bundle::readSourceBundle`, in `hipblaslt-jit-source-bundle.hpp`, reads
one by directory convention and requires at least one kernel source:

| Path | Contents |
| --- | --- |
| `library/TensileLibrary.dat` | The library entry, stored as uncompressed MsgPack |
| `sources/*.s` | Kernel assembly |
| `sources/*.hip`, `sources/*.cpp` | HIP kernel sources |
| Other files in `sources/` | Headers that the HIP sources include |
| `manifest.json` | A JavaScript Object Notation (JSON) provenance record that hipBLASLt does not read |

Artifact paths must be relative and stay inside the bundle, including through
symbolic links, and `sources/` may hold only regular files. The reader bounds
the file count (1024), each file (64 MiB) and the sources in total (256 MiB).

The JIT tests use one kernel, generated for gfx90a, gfx942 and gfx950, in
`clients/tests/jit/data`, of which only the assembly is committed. The assembly
records the kernel-argument and persistent-loop argument layout versions of the
generator that wrote it. The `jit-bundles` test writes source bundles from it
for each architecture, each with the library entry and manifest built from the
assembly and a description of each solution: `plain` with one solution and
`plain-pair` with two. The tests that need a GPU run the bundles of its
architecture.
[The data README](clients/tests/jit/data/README.md) gives the command that
generated the assembly.
