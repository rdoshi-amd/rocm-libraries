"""The `hsaco` producer: a pre-built code object, checked in beside its descriptor.

The other producers make their code object at pack time -- `hip` runs hipcc over
a source file, `rocke` imports a builder and compiles through comgr. An `hsaco`
UKD names an object that already exists, for a kernel built ahead of time by a
toolchain this packer does not run. Producing it is therefore resolving and
checking the file, and everything downstream -- the archive, the TOC key, the
SHA256 the runtime verifies, the signature read out of the object -- is the same
path the compiling producers take.
"""

import hashlib
from pathlib import Path

from .errors import HkpPackError
from .variant import _hash_payload


def hsaco_variant_key(rel_file):
    """Stable key for a pre-built object: its root-relative path.

    The path is the identity, as it is for a hip source. Content is not hashed
    in: the key names the TOC entry, and an object regenerated in place keeps
    its entry while its recorded SHA256 changes with it. `rel_file` is the
    root-relative path from hip_source_relpath, so two folders that each hold a
    `kernel.hsaco` key apart.
    """
    return _hash_payload(Path(rel_file).stem, {"file": rel_file})


def resolve_hsaco_input(source_root, rel_dir, file, expected_sha256=None):
    """The checked-in object a UKD names, resolved and verified.

    Resolves **relative to the descriptor that named it** --
    `source_root / rel_dir / file` -- with no root-relative fallback, and refuses
    a path that leaves the root, on the same terms as a hip source.

    `expected_sha256` is optional in the authored form. When given, a mismatch is
    a hard error here rather than a load failure on a machine that cannot say
    why: it means the object changed without its descriptor being regenerated.
    """
    root = Path(source_root).resolve()
    path = (root / rel_dir / file).resolve()
    if not path.is_relative_to(root):
        raise HkpPackError(
            f"code object escapes the source root: {file} "
            f"(from {Path(rel_dir).as_posix()}, resolved to {path})"
        )
    if not path.is_file():
        raise HkpPackError(
            f"code object not found: {file} (looked for {path}, "
            f"resolved relative to descriptor folder {Path(rel_dir).as_posix()})"
        )
    if expected_sha256 is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise HkpPackError(
                f"code object {file} hashes to {actual}, but its descriptor "
                f"records {expected_sha256}; regenerate the descriptor with the "
                "object"
            )
    return path
