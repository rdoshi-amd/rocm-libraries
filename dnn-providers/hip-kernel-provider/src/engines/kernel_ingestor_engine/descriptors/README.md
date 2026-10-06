# Shipped descriptors

This root holds the descriptors the provider **ships**. Its sibling `test_descriptors/`
stages into the build tree for the unit and integration binaries and is installed only
under `HIPKERNELPROVIDER_ENABLE_TESTS`. It holds one bundle,
`rocKE/gfx950_attention_dense/` (stored in DVC, see "rocKE bundles live in DVC" below), whose KDP declares gfx950 only, so production packaging
runs for a build whose GPU targets include gfx950 and is dormant for every other build
unless the cache variable below is pointed elsewhere.

## Authoring a bundle

```
descriptors/<producer>/<bundle>/
```

A bundle is authored under the producer that builds its kernels. The subpath is a
convention: `kernel_source.kind` selects the behaviour. `rocKE/` is spelled with
that capitalization because the packer preserves an authored subpath verbatim into the
staged and installed trees. A bundle sits one level under its producer, so every
descriptor lands in a child of its shard root and the archive can be written at the root.

Nothing here is registered in CMake: the packer walks this root recursively, so adding a
bundle is dropping files in a folder. Kernel-source *embedding* is a separate mechanism,
required only for `kernel_source.kind == "embedded_source"`.

A `hip` or `rocke` bundle here is compiled at pack time, one comgr invocation per variant,
on every build that has this root wired — CI included. An `hsaco` bundle is packed as-is:
no comgr and no hipcc run. Trim an authored variant set to a covering subset before it
lands, and register the symbols its UKDs name in a native pack, or the loader refuses the
engine at provider load and every lowered kernel is wasted build time.

The packer does not check an `hsaco` object's format or target processor. Every
`hsaco` UKD must list its `arch`(es) (non-empty; a generic-target object lists every arch
it runs on). `hkp_pack` rejects one without, because an unrestricted `hsaco` UKD would ship
the same bytes into every shard, and they fail at module load on the wrong device.

## Root selection and dormancy

`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` is a `CACHE PATH` defaulting to this
directory; a consumer needing a different root overrides that variable rather than adding
CMake. Production wiring is gated on at least one non-hidden `*.kdp.json` here, since a
KDP is what architecture pruning consumes. With none, packaging stays dormant and any
stale product tree is removed — not an error. A KDP that *is* present but prunes on every
architecture is a hard failure for a root the build NAMED, and dormancy for this root
reached as the built-in default.

Do not delete this README: git tracks no empty directory, and the cache variable's
set-but-not-a-directory check is fatal.

## Relationship to the examples tree

`descriptor-packaging/examples/descriptors/` is a **test fixture** tree. Bundles are
authored and proved there against the packaging suite, then relocated onto this root once
a native pack registers the symbols their UKDs name. The Linux superbuild CI lane
overrides `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` to that fixture tree, so that lane
packs the fixtures and never this root.

## rocKE bundles live in DVC

Bundles under `rocKE/` are not stored in git. Each bundle folder is one DVC output:

```
rocKE/<bundle>.dvc      tracked in git: md5 of the folder, file count, remote: ingestor
rocKE/<bundle>/         not in git (ignored): the uncompressed descriptor files
```

The blobs live in the `ingestor` remote (`s3://therock-dvc/rocm-libraries/hipdnn/ingestor`,
anonymous read, declared in `.dvc/config`). Each file is stored as-is under its md5.

**Fetch.** TheRock's `build_tools/fetch_sources.py` pulls every `*.dvc` pointer in
`rocm-libraries`, so a source fetch populates the bundle folder. By hand:

```
dvc pull -r ingestor dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/rocKE/<bundle>.dvc
```

**Build behavior.** Keyed on the existing `HIPKERNELPROVIDER_ENABLE_ROCKE`:

- `ON`: configure fails if a bundle folder is missing or holds fewer files than its
  pointer's `nfiles`. The error names the `dvc pull` command. Only the root the build
  packs from is checked, so a build that sets `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`
  elsewhere does not need these bundles.
- `OFF`: rocKE descriptors are not consumed and DVC is not needed.

**Add or change a bundle.** Author the folder, then:

```
cd .../descriptors/rocKE
dvc add <bundle>
printf '  remote: ingestor\n' >> <bundle>.dvc   # add under the single outs entry
dvc push -r ingestor                            # needs S3 write access
git add <bundle>.dvc .gitignore
```

Commit the `.dvc` file in the same change as the push. A changed file changes the folder
md5, so the pointer must be updated whenever the folder content changes.
