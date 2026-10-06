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
parts of hipBLASLt that build and load what a backend generates. The
[JIT test guide](clients/tests/jit/README.md) covers building and running the
tests.

## Summary

hipBLASLt runs a GEMM with a kernel from its pre-tuned library, so a problem
that the library serves poorly, or not at all, has no better kernel available.
JIT generation produces kernels for a problem when they are needed. This page
describes the parts that turn generated kernel sources into a loaded
TensileLite solution: a comgr code-object builder, a source bundle reader and a
TensileLite loader. The library does not call them yet; the JIT tests build and
load a pre-generated source bundle with them. No generator backend is
implemented, and no API reaches JIT.

## Current behavior

The interfaces are in `library/src/amd_detail/hipblaslt-jit-component.hpp`. A
`GeneratedSolution` holds a one-solution TensileLite library entry, its main
kernel name, and the source units to build. `CodeObjectBuilder::build` builds
those units into the code objects of a `BuiltSolution`; `GeneratedSolution` has
no code-object field. The interfaces are for compiled-in implementations and do
not establish a stable external plugin application binary interface (ABI).

### Build

`HIPBLASLT_ENABLE_JIT` is disabled by default. A disabled build compiles no
JIT code. The enabled build requires the host library, ROCm and ROCm's
`amd_comgr` CMake package, which only a JIT build links. comgr compiles helper
sources against the host's C and C++ standard library headers, so those must be
installed where JIT runs. From the repository root:

```bash
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
```

The [JIT test guide](clients/tests/jit/README.md) lists the test targets and
the validation commands.

### Building generated sources

Generators emit assembly or HIP source plus metadata only; they do not assemble,
link or bundle. `makeComgrBuilder()` returns the `CodeObjectBuilder` that builds
code objects in process through AMD comgr (`hipblaslt-jit-builder.cpp` and
`hipblaslt-jit-code-object.cpp`). It uses three comgr actions:

- Assembly: `AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE`. A unit's
  `.amdgcn_target` directive must name the device's processor and only
  features the device has; it is then rewritten to the device's full target
  ID. Assembly units that declare different wavefront sizes are rejected.
- HIP helper source: `AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE` with
  `--rocm-path` and a content-derived `-cuid`, so helper objects link together.
  The ROCm path is `HIP_PATH` when set, otherwise the prefix of the loaded HIP
  runtime.
- Link: `AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE` joins the main kernel
  and helper relocatables into one code object per solution, with
  `-Xlinker --build-id=sha1`.

The generator and the builder use the same code-object version, which
`BuildRequest::codeObjectVersion` carries (4 by default). The output is a
raw, uncompressed executable code object. comgr cannot bundle or compress it,
and `hipModuleLoadData` accepts raw executable and linkable format (ELF)
objects. A generator needs no offload bundler.

After linking, the builder reads the code object's metadata and checks that its
instruction set architecture (ISA) is the device's and that it defines the
solution's main kernel. A build failure has the build stage. Its message
carries the first error line of the comgr log, and the builder appends the full
log to `comgr.log` in `BuildRequest::scratch` and names that file in the
message.

comgr's own on-disk cache (`~/.cache/comgr`) keeps its default: hipBLASLt never
sets `AMD_COMGR_CACHE`. That cache holds the results of comgr actions for every
comgr user in the process, such as hipRTC, and comgr reads its setting once per
process.

### Loading a built solution

`hipblaslt-jit-loader.{hpp,cpp}` turns a built TensileLite solution into a
loaded one-solution library:

- `readTensileSourceBundle` reads a source bundle into a `GeneratedSolution`:
  the library entry, its main kernel name, each `sources/*.s` as a main
  assembly unit, and `sources/Kernels.cpp` with its headers as a helper unit.
- `parseTensileBundle` reads a built solution's entry into a TensileLite
  `MasterSolutionLibrary` for the device's hardware. It requires exactly one
  local solution, index 0, named after the built kernel, and loads no code.
- `loadTensileBundle` loads the main and helper code objects into a new
  `SolutionAdapter` and resolves the main kernel.

The entry's own hardware and problem predicates decide which problems the
solution serves, so `findBestSolution` on the loaded library selects it only
for the problems it was generated for.

### Source bundle format

A source bundle is a generated solution stored as a directory.
`source_bundle::readSourceBundle`, in `hipblaslt-jit-source-bundle.hpp`, reads
one by directory convention:

| Path | Contents |
| --- | --- |
| `library/TensileLibrary.dat` | The one-solution library entry, stored as uncompressed MsgPack |
| `sources/*.s` | The main kernel assembly; at least one |
| `sources/Kernels.cpp` | The helper kernels, when the solution needs helpers |
| Other files in `sources/` | Headers that the helper source includes |
| `manifest.json` | A JavaScript Object Notation (JSON) provenance record that hipBLASLt does not read |

Artifact paths must be relative and stay inside the bundle, including through
symbolic links, and `sources/` may hold only regular files. The reader bounds
the file count (1024), each file (64 MiB) and the sources in total (256 MiB).

The JIT tests use a gfx950 source bundle committed in `clients/tests/jit/data`.
Its manifest records the kernel-argument and persistent-loop argument layout
versions of the generator that wrote it, and
[its README](clients/tests/jit/data/README.md) gives the command that
generated it.
