# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The AOT instance table: one row per kernel object this provider ships.

This module is the single source of truth for *what gets built*. The generator
compiles from it, the packer keys its table of contents off it, and the
descriptor emitter turns each row into a kernelDescriptor with its ``.kmd.json``
metadata and its ``priority``. Adding an instance is a row here, not an edit in
three places that can disagree.

**Two tiers per op.** A specialized instance bakes a knob in and matches that
value exactly; a generic instance leaves the knob to runtime, covers everything,
and is slower. The selector picks the specialist when one applies and falls back
to the generic otherwise, so the declared surface has no holes. Which tier wins
a shape is expressed purely as data -- ``priority`` is the ingestor's tie-break
when no heuristic model is loaded -- so no ranking code is written for it.

For RMSNorm the specialized knob is the row width ``N``: FlyDSL's vectorized
128-bit path caches tiles in a Python list of registers, which only exists under
compile-time loop unrolling, so it genuinely cannot take a runtime ``N``. The
generic tier is the vendored runtime-``N`` variant (``kernels_src`` modification
1), which reads ``N`` from the input tensor descriptor and predicates the tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ranking is `score desc -> priority desc -> kernelId asc`, and with no
# heuristic model every score ties, so priority is what orders the catalog.
# Two bands, far enough apart to insert tiers between them later.
PRIORITY_SPECIALIZED = 100
PRIORITY_GENERIC = 10

# FlyDSL dtype spellings, as `build_*_module` takes them.
DTYPES = ("bf16", "f16")


@dataclass(frozen=True)
class Instance:
    """One compiled kernel object and everything the pipeline needs about it."""

    op: str
    dtype: str
    priority: int
    # Baked knob values, verbatim into `.kmd.json`. A knob whose value is None is
    # the runtime/generic tier for that axis -- the matcher reads null as "any".
    knobs: dict = field(default_factory=dict)

    @property
    def is_generic(self) -> bool:
        return any(value is None for value in self.knobs.values())

    @property
    def name(self) -> str:
        """Stable instance name: the object's filename stem and TOC key leaf."""
        parts = [self.op, self.dtype]
        for key, value in self.knobs.items():
            if key == "dtype":
                continue
            parts.append(f"{_abbrev(key)}{'generic' if value is None else value}")
        return "_".join(parts)

    def to_record(self) -> dict:
        return {
            "name": self.name,
            "op": self.op,
            "dtype": self.dtype,
            "priority": self.priority,
            "tier": "generic" if self.is_generic else "specialized",
            "knobs": dict(self.knobs),
        }


_ABBREV = {
    "N": "n",
    "block_threads": "bt",
    "head_dim": "d",
    "block_m": "bm",
    "block_n": "bn",
}


def _abbrev(key: str) -> str:
    return _ABBREV.get(key, key.lower())


# --- RMSNorm ----------------------------------------------------------------

# Below this width `build_rmsnorm_module` does not specialize the kernel above --
# it returns `_build_rmsnorm_large_m_small_n_module`, a *different* kernel with a
# different ABI (an extra `m_in` kernarg) and a different launch geometry (grid
# divided by BLOCK_M rather than one block per row). Mirrors SMALL_N_THRESHOLD at
# kernels_src/kernels/norm/rmsnorm_kernel.py:73; the generator asserts the two
# still agree, because a drift here is the kind that produces a correct-looking
# object dispatched with the wrong grid.
#
# That path is worth having -- it is upstream's large-M/small-N specialization --
# but it is a second launch record, so it lands as its own instance family rather
# than hiding inside this one. Until then small N is served by the generic tier.
RMSNORM_SMALL_N_THRESHOLD = 2048

# The specialized widths: the hidden sizes that dominate the models this release
# targets, not a sweep. 3072/4096/5120/8192 are Llama-family, 3584 is Qwen2-7B.
# Every other N -- and every N a future model introduces -- is served by the
# generic tier, which is the point of having one. Widening this list is cheap; it
# is not a coverage fix.
#
# 2048 was in the original plan and is deliberately absent: `N <= threshold` is
# the small-N branch above, so asking for it here would not produce a
# specialization of this kernel at all.
RMSNORM_SPECIALIZED_N = (3072, 3584, 4096, 5120, 8192)

# The kernel's `BLOCK_THREADS`. Single-valued today: it is a launch-geometry
# knob, not a correctness one, so the second value only earns its instances once
# benchmarking says it wins somewhere. Kept in the knob dict so descriptors and
# the native matcher already carry the axis when it grows.
RMSNORM_BLOCK_THREADS = (256,)


def rmsnorm_instances() -> list[Instance]:
    """The RMSNorm instance table: generic tier first, then the specialists."""
    instances: list[Instance] = []
    for block_threads in RMSNORM_BLOCK_THREADS:
        for dtype in DTYPES:
            instances.append(
                Instance(
                    op="rmsnorm",
                    dtype=dtype,
                    priority=PRIORITY_GENERIC,
                    knobs={"N": None, "block_threads": block_threads},
                )
            )
        for dtype in DTYPES:
            for n in RMSNORM_SPECIALIZED_N:
                instances.append(
                    Instance(
                        op="rmsnorm",
                        dtype=dtype,
                        priority=PRIORITY_SPECIALIZED,
                        knobs={"N": n, "block_threads": block_threads},
                    )
                )
    _assert_unique(instances)
    return instances


# --- SDPA forward ------------------------------------------------------------

# The vendored kernel (kernels_src/kernels/attention/flash_attn_func_gfx1151.py)
# bakes only head_dim, the causal flag, dtype and its tile. Batch, both sequence
# lengths, both head counts (GQA/MQA), every stride, the softmax scale and the
# causal offset are runtime arguments, so each row below serves every shape and
# layout of its (dtype, head_dim, causal) class. That is what keeps this table
# small: the axes it enumerates are model-determined, never request-determined.
#
# head_dim: 64 and 128 cover the large majority of current LLM and diffusion
# attention; 96 is the Phi-family width. The kernel builds for any multiple of
# 32 from 64 up, but 256 does not fit: its O accumulators and register-resident
# Q operands alone fill the 256-VGPR ceiling and the object spills, so it needs
# a schedule that stages Q through LDS (COVERAGE.md, what is missing).
SDPA_HEAD_DIMS = (64, 96, 128)

# Two variants of one kernel rather than a runtime flag: causal changes the KV
# loop bound and skips fully-masked tiles, and the non-causal variant carries a
# V prefetch across iterations that the causal one drops for register pressure.
SDPA_CAUSAL = (0, 1)

# The kernel's tile: upstream's RDNA4 defaults, unchanged. Launch geometry, not
# correctness -- kept in the knob dict so a second tile is a row, and so the
# dispatcher reads the grid's block_m from the selected object rather than
# restating it.
SDPA_BLOCK_M = 128
SDPA_BLOCK_N = 32


def sdpa_instances() -> list[Instance]:
    """The SDPA-forward instance table: one row per (dtype, head_dim, causal)."""
    instances: list[Instance] = []
    for dtype in DTYPES:
        for head_dim in SDPA_HEAD_DIMS:
            for causal in SDPA_CAUSAL:
                instances.append(
                    Instance(
                        op="sdpa",
                        dtype=dtype,
                        priority=PRIORITY_SPECIALIZED,
                        knobs={
                            "head_dim": head_dim,
                            "causal": causal,
                            "block_m": SDPA_BLOCK_M,
                            "block_n": SDPA_BLOCK_N,
                        },
                    )
                )
    _assert_unique(instances)
    return instances


def _assert_unique(instances: list[Instance]) -> None:
    """Instance names are filenames and TOC keys; a collision silently drops one."""
    seen: dict[str, Instance] = {}
    for instance in instances:
        if instance.name in seen:
            raise ValueError(
                f"duplicate instance name {instance.name!r}: {seen[instance.name]} "
                f"and {instance}. Names are derived from the knob dict, so two rows "
                "differing only in an axis absent from the name collide."
            )
        seen[instance.name] = instance


OPS = {"rmsnorm": rmsnorm_instances, "sdpa": sdpa_instances}


def instances_for(op: str) -> list[Instance]:
    if op not in OPS:
        raise KeyError(f"unknown op {op!r}; known: {', '.join(sorted(OPS))}")
    return OPS[op]()
