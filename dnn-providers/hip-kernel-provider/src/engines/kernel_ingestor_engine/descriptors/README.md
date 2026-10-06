# Shipped descriptors

This root holds the descriptors the provider **ships**. Its sibling `test_descriptors/`
stages into the build tree for the unit and integration binaries and is installed only
under `HIPKERNELPROVIDER_ENABLE_TESTS`. It holds one bundle,
`rocKE/gfx950_attention_dense/` (its KDP is stored in DVC, see "Large rocKE files live in DVC" below), whose KDP declares gfx950 only, so production packaging
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

## Large rocKE files live in DVC

Only the large files of a rocKE bundle are stored in DVC: the KDPs (`*.kdp.json`) and any
kernel object (`*.co`, `*.hsaco`). The other descriptors (KMD, UDD, UED, UHD, UMD) are small,
human-readable JSON and stay in git, so their changes show in review. Each DVC file has a
pointer beside it:

```
rocKE/<bundle>/<name>.kdp.json.dvc   in git: md5, size, remote: ingestor
rocKE/<bundle>/<name>.kdp.json       not in git (ignored): the file itself
```

The blobs live in the `ingestor` remote (`s3://therock-dvc/rocm-libraries/hipdnn/ingestor`,
anonymous read, declared in `.dvc/config`). Each file is stored as-is under its md5, with
no compression. The packer reads only `*.json`, so the `.dvc` and `.gitignore` files beside
the descriptors are ignored.

**Fetch.** TheRock's `build_tools/fetch_sources.py` pulls every `*.dvc` pointer in
`rocm-libraries`, so a source fetch populates these files. By hand:

```
dvc pull -r ingestor dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/rocKE/<bundle>/<name>.kdp.json.dvc
```

**Build behavior.** Keyed on the existing `HIPKERNELPROVIDER_ENABLE_ROCKE`:

- `ON`: configure fails if a DVC file is missing or its size differs from its pointer. The
  error names the `dvc pull` command. Only the root the build packs from is checked, so a
  build that sets `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` elsewhere does not need these
  files.
- `OFF`: rocKE descriptors are not consumed and DVC is not needed.

**Add or change a large file.**

```
cd .../descriptors/rocKE/<bundle>
dvc add <name>.kdp.json
printf '  remote: ingestor\n' >> <name>.kdp.json.dvc   # new pointers only; dvc add keeps it later
dvc push -r ingestor                                   # needs S3 write access, see below
git add <name>.kdp.json.dvc .gitignore
```

Commit the `.dvc` file in the same change as the push. Any change to the file changes its
md5, so the pointer must be updated with it. The pointer diff shows only the new md5 and
size, not the content change; describe the change in the commit message.

**Pushing needs a signed request.** The committed `.dvc/config` sets
`allow_anonymous_login = true` on `ingestor` so that CI and `fetch_sources` can pull
without credentials. With that setting DVC sends unsigned requests, ignores your AWS
credentials, and every write fails with `AccessDenied`. Override it locally; the override
goes in `.dvc/config.local`, which git ignores, so the committed value stays `true`:

```
dvc remote modify --local ingestor allow_anonymous_login false
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... dvc push -r ingestor
```

Pass the credentials as environment variables for the one command. Do not write them into
any config file. The identity needs `s3:PutObject` and `s3:ListBucket` on
`s3://therock-dvc/rocm-libraries/hipdnn/ingestor/`. `dvc push -r ingestor` uploads every
object in your local cache, not only this bundle, so do not pull other remotes' data
into the same worktree before pushing.
