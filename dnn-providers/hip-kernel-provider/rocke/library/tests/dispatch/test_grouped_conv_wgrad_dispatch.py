# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support + grid tests for grouped wgrad dispatch.

CPU-only (no GPU / no comgr): asserts that the grouped-convolution dispatcher
admits grouped backward-weight requests, and that the launch grid it derives
matches the kernel's block_id_z contract --

    grid = (ceil(wg_N / tile_n), ceil(wg_M / tile_m), groups * split_k)

with the per-group dims wg_M = kpg, wg_N = spatial * cpg. This is the same grid
the GPU correctness test (platform tests ``test_conv_wgrad_correctness.py``)
launches and validates numerically, so a match here proves the dispatch path
launches a correct grid.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

from dispatch.grouped_convolution import (
    ConvGroupedRequest,
    _block,
    _problem,
    conv_grouped_candidates,
    dispatch_conv_grouped,
)


def _wgrad(arch="gfx942", **kw):
    base = dict(
        N=2,
        C=64,
        K=64,
        Hi=14,
        Wi=14,
        Y=3,
        X=3,
        pad_h=1,
        pad_w=1,
        arch=arch,
        direction="wgrad",
        # force default epilogue (vec_size_c=1) so grouped isn't rejected for
        # cshuffle; grouped wgrad supports only the direct-store epilogue.
        vec_size_c=1,
    )
    base.update(kw)
    return ConvGroupedRequest(**base)


def _expected_grid(req, spec):
    # Mirror dispatch._wgrad_grid: per-group tiling on x/y, and z = groups *
    # split_k with the group riding block_id_z alongside the K-slice. split_k
    # == -1 is the auto sentinel; resolve it via the same CK formula the grid
    # uses so this stays an independent re-derivation of the wiring.
    p = _problem(req)
    spatial = (p.Z if p.is_3d else 1) * p.Y * p.X
    kpg = p.K // p.groups
    cpg = p.C // p.groups
    wg_M = kpg
    wg_N = spatial * cpg
    gx = math.ceil(wg_N / spec.tile_n)
    gy = math.ceil(wg_M / spec.tile_m)
    split_k = spec.split_k
    if split_k == -1:
        from rocke.helpers.split_k import select_split_k_wgrad

        split_k = select_split_k_wgrad(
            wg_M=wg_M,
            wg_N=wg_N,
            wg_K=p.N * p.Ho * p.Wo * (p.Do if p.is_3d else 1),
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            arch=spec.arch,
            groups=p.groups,
            block_size=_block(spec)[0],
        ).split_k
    return (gx, gy, p.groups * split_k)


class TestGroupedWgradDispatch(unittest.TestCase):
    # ---- admittance + grid ---------------------------------------------------

    def test_grouped_admitted_grid_per_group(self):
        # groups=4: grid-per-group. The group rides block_id_z alongside the
        # K-slice, so z = groups*split_k (split_k auto-resolved, >= 1).
        for arch in ("gfx942", "gfx950"):
            r = dispatch_conv_grouped(_wgrad(arch, G=4))
            self.assertEqual(r.spec.direction, "wgrad")
            self.assertEqual(r.spec.epilogue, "default")
            self.assertEqual(r.grid[2] % 4, 0, "z must be a multiple of groups")
            self.assertGreaterEqual(r.grid[2] // 4, 1, "split_k >= 1 per group")
            self.assertEqual(r.grid, _expected_grid(r.request, r.spec))

    def test_grouped_cshuffle_admitted(self):
        # A grouped request whose vec derives a cshuffle epilogue is admitted:
        # grouping is orthogonal to the epilogue (the staged store threads the
        # per-group k_out fold).
        for arch in ("gfx942", "gfx950"):
            r = dispatch_conv_grouped(_wgrad(arch, G=4, vec_size_c=8))
            self.assertEqual(r.spec.direction, "wgrad")
            self.assertEqual(r.spec.epilogue, "cshuffle")
            self.assertEqual(r.grid, _expected_grid(r.request, r.spec))

    def test_gfx1250_grouped_admitted_wmma(self):
        # gfx1250 (wave32 WMMA 16x16x32): grouped grid-per-group, split_k forced
        # to 1 (WMMA has no split_k), direct-store epilogue.
        r = dispatch_conv_grouped(_wgrad("gfx1250", G=4))
        self.assertEqual(r.spec.direction, "wgrad")
        self.assertEqual(r.spec.epilogue, "default")
        self.assertEqual(r.spec.split_k, 1, "WMMA wgrad must use split_k=1")
        self.assertEqual(r.grid[2], 4, "z must be one index per group")
        self.assertEqual(r.grid, _expected_grid(r.request, r.spec))

    def test_ungrouped_grid_unchanged(self):
        # groups=1: the grid reduces to the pre-grouped (gx, gy, split_k) form:
        # gx/gy from the DENSE dims (wg_M=K, wg_N=spatial*C) and z the
        # auto-resolved split_k (>=1).
        req = _wgrad("gfx942", G=1, vec_size_c=None)
        r = dispatch_conv_grouped(req)
        gx = math.ceil(req.Y * req.X * req.C / r.spec.tile_n)
        gy = math.ceil(req.K / r.spec.tile_m)
        self.assertEqual(r.grid[0], gx)
        self.assertEqual(r.grid[1], gy)
        self.assertGreaterEqual(r.grid[2], 1)

    # ---- candidate admittance ------------------------------------------------

    def test_candidate_admits_grouped(self):
        # candidate-level admittance mirrors the dispatch result for a valid
        # grouped request.
        cands = {c.name: c for c in conv_grouped_candidates("wgrad")}
        self.assertTrue(any("gfx942" in n for n in cands))
        c = next(c for n, c in cands.items() if "gfx942" in n)
        ok, why = c.admits(_wgrad("gfx942", G=4))
        self.assertTrue(ok, why)


class TestGroupedConvDirectionSurface(unittest.TestCase):
    """The grouped-conv directions handled here are reachable from one module.

    ``dispatch.grouped_convolution`` covers forward, backward-weight (wgrad) and
    backward-data (dgrad); all three share a single ``ConvGroupedRequest``
    import surface.
    """

    def test_each_direction_returns_candidates(self):
        for direction in ("fwd", "wgrad", "dgrad"):
            self.assertGreater(len(conv_grouped_candidates(direction)), 0, direction)


class TestTwoStageSelection(unittest.TestCase):
    """two_stage is chosen by atomic availability, not by a caller flag."""

    def test_odd_wg_N_takes_the_scratch_path(self):
        # wg_N = Y*X*cpg odd => the packed <2 x bf16> atomic pair is not
        # dword-aligned, so split-K can only go through the f32 scratch.
        r = dispatch_conv_grouped(_wgrad("gfx950", C=3, K=24, Y=3, X=3, dtype="bf16"))
        ws = r.spec.to_wgrad_spec(_problem(r.request))
        if ws.split_k > 1:
            self.assertTrue(
                ws.two_stage,
                "odd wg_N with a 16-bit dW must resolve to the two-stage path",
            )

    def test_even_wg_N_stays_on_the_packed_atomic(self):
        # wg_N even => the packed atomic can address dW directly and no
        # scratch/second launch is needed.
        r = dispatch_conv_grouped(_wgrad("gfx950", C=64, K=64, Y=3, X=3, dtype="bf16"))
        ws = r.spec.to_wgrad_spec(_problem(r.request))
        self.assertFalse(
            ws.two_stage, "an even wg_N should reach split-K via packed atomics"
        )

    def test_split_k_1_needs_no_two_stage(self):
        r = dispatch_conv_grouped(_wgrad("gfx1250"))
        ws = r.spec.to_wgrad_spec(_problem(r.request))
        self.assertEqual(ws.split_k, 1, "gfx1250 always uses split_k=1")
        self.assertFalse(ws.two_stage, "split_k=1 needs no two_stage")


class TestTwoStageGridShape(unittest.TestCase):
    """Stage 1 and Stage 2 grid shapes for the two-stage deterministic path."""

    def test_stage1_grid_z_is_groups_times_split_k(self):
        # Stage 1 grid z encodes both group and split-K slice:
        #   z = groups * split_k
        for arch in ("gfx942", "gfx950"):
            r = dispatch_conv_grouped(_wgrad(arch, G=4))
            groups = r.request.G
            split_k = r.grid[2] // groups
            self.assertEqual(r.grid[2], groups * split_k)

    def test_stage2_grid_z_is_groups(self):
        # Stage 2 (workspace-reduce) uses grid z = groups: block_id_z is the
        # group index, one CTA per group covering wg_M x wg_N output elements.
        from kernels.common.conv_wgrad_workspace_reduce import (
            WgradReduceSpec,
            wgrad_reduce_grid,
        )

        for arch in ("gfx942", "gfx950"):
            r = dispatch_conv_grouped(_wgrad(arch, G=4))
            ws = r.spec.to_wgrad_spec(_problem(r.request))
            s2_spec = WgradReduceSpec(
                problem=ws.problem,
                dtype_d=ws.data.dtype_d,
                groups=r.request.G,
            )
            grid = wgrad_reduce_grid(s2_spec)
            self.assertEqual(grid[2], r.request.G, "Stage 2 grid z must equal groups")


def _dgrad(arch="gfx950", **kw):
    base = dict(
        N=2,
        C=64,
        K=64,
        Hi=14,
        Wi=14,
        Y=3,
        X=3,
        pad_h=1,
        pad_w=1,
        arch=arch,
        direction="dgrad",
    )
    base.update(kw)
    return ConvGroupedRequest(**base)


class TestGroupedDgradDispatch(unittest.TestCase):
    """Selection + grid + K-outer policy for the gfx950 dgrad candidate.

    The grid contract differs from wgrad's: dgrad's M-tile count is not a closed
    form over the problem dims, because stride > 1 splits the convolution into
    ``y_tilde * x_tilde`` sub-GEMMs of differing sizes. The x extent is the
    cumulative tile count of the last sub-GEMM, and the group rides ``blockIdx.y``
    rather than sharing z with the K-slice.
    """

    def _select(self, req):
        # These tests pin the igemm candidate's own contract. Grouped stride-1
        # requests are also admitted by the higher-priority direct-MFMA
        # candidate (covered by TestGroupedDirectDgradDispatch), so look the
        # igemm one up by name rather than expecting it to be the only taker.
        cands = [
            c
            for c in conv_grouped_candidates("dgrad")
            if c.name == "implicit_gemm_conv_dgrad" and c.admits(req)[0]
        ]
        self.assertEqual(len(cands), 1, f"igemm dgrad must admit {req}: {cands}")
        return cands[0], cands[0].select_spec(req)

    def test_admitted_and_grid_matches_sub_gemm_tiling(self):
        req = _dgrad()
        cand, spec = self._select(req)
        p = _problem(req)
        # Independent re-derivation: stride 1 is a single sub-GEMM, so the flat
        # tile count is just the M/N tiling of that one GEMM.
        gemm_m = p.N * p.Hi * p.Wi
        expected_x = math.ceil(gemm_m / spec.tile_m) * math.ceil(
            (p.C // p.groups) / spec.tile_n
        )
        self.assertEqual(cand.grid(spec, req), (expected_x, 1, 1))

    def test_strided_grid_uses_tilde_decomposition(self):
        # stride 2 gives y_tilde = x_tilde = 2: four sub-GEMMs over a quarter of
        # the rows each. The flat tile count must therefore differ from the
        # stride-1 count rather than reusing a single-GEMM formula.
        strided = _dgrad(stride_h=2, stride_w=2)
        cand, spec = self._select(strided)
        gx_strided = cand.grid(spec, strided)[0]

        plain = _dgrad()
        cand1, spec1 = self._select(plain)
        gx_plain = cand1.grid(spec1, plain)[0]

        self.assertNotEqual(gx_strided, gx_plain)
        self.assertGreater(gx_strided, 0)

    def test_group_rides_block_id_y(self):
        req = _dgrad(C=64, K=64, G=4)
        cand, spec = self._select(req)
        self.assertEqual(cand.grid(spec, req)[1], 4)

    def test_k_outer_selected_on_even_channel_run(self):
        _cand, spec = self._select(_dgrad())
        self.assertTrue(spec.lds_k_outer)
        self.assertEqual(spec.direction, "dgrad")

    def test_k_outer_declined_on_odd_channel_run(self):
        # cpg = 48 / 16 = 3. The B load width collapses to 1, axis_b is already
        # "col", and there is no transpose-on-store left to remove -- K-outer
        # would only add the read-side cost. The predicate must decline.
        _cand, spec = self._select(_dgrad(C=48, G=16))
        self.assertFalse(spec.lds_k_outer)

    def test_dgrad_candidate_rejects_other_directions(self):
        cand = conv_grouped_candidates("dgrad")[0]
        for direction in ("fwd", "wgrad"):
            ok, why = cand.admits(_dgrad(direction=direction))
            self.assertFalse(ok, direction)
            self.assertIn("dgrad", why)

    def test_epilogue_follows_store_vector_width(self):
        # dX's last dim is C, so a wide store vector needs cshuffle's LDS
        # staging; the direct-store 'default' path writes scalars. Pinning
        # 'default' unconditionally is silently valid -- vector_size_c is left
        # unset, so the validator rule never fires -- and costs store bandwidth
        # on every non-grouped shape. Derive it instead.
        _cand, wide = self._select(_dgrad(C=128, K=128, Hi=32, Wi=32))
        self.assertEqual(wide.epilogue, "cshuffle")

        # cpg = 3: no legal width > 1, so the scalar direct store is correct.
        _cand, narrow = self._select(_dgrad(C=48, G=16))
        self.assertEqual(narrow.epilogue, "default")

    def test_vec_size_c_uses_the_dgrad_formula(self):
        # Each direction has its own default_vector_sizes and they are not
        # interchangeable: dgrad's takes the per-group runs (cpg, kpg), so a
        # fallthrough to the forward formula sizes off the wrong extent once
        # groups > 1.
        from dispatch.grouped_convolution import _vec_size_c
        from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

        req = _dgrad(C=48, G=16)
        p = _problem(req)
        _va, _vb, expected = DgradConvSpec.default_vector_sizes(
            p.cpg, p.kpg, req.dtype.lower()
        )
        self.assertEqual(_vec_size_c(req), expected)

    def test_spec_round_trips_to_instance_spec(self):
        req = _dgrad()
        _cand, spec = self._select(req)
        inst = spec.to_dgrad_spec(_problem(req))
        # The dispatcher's K-outer decision must survive into the instance spec;
        # a spec that silently reverts to M-outer would still run and still be
        # correct, so nothing else would catch it.
        self.assertEqual(inst.lds_k_outer, spec.lds_k_outer)
        self.assertEqual(inst.tile_m, spec.tile_m)
        self.assertEqual(inst.warp_tile_n, spec.warp_tile_mn)
        inst.validate()


class TestGfx950DgradTileTable(unittest.TestCase):
    """Shape-keyed tile selection of the gfx950 dgrad candidate.

    The table (``_gfx950_dgrad_tile``) serves every problem the dY halo pick
    does not take; stride-1 same-size multi-tap problems with kpg % 64 == 0
    are checked against the table directly, since dispatch gives them the
    halo pick (see TestGfx950DgradHaloPick).
    """

    def _tile(self, **kw):
        req = _dgrad(dtype="bf16", **kw)
        r = dispatch_conv_grouped(req)
        s = r.spec
        inst = s.to_dgrad_spec(_problem(req))
        ok, why = _dgrad_valid(inst)
        self.assertTrue(ok, why)
        return (s.tile_m, s.tile_n, s.tile_k, s.warp_m, s.warp_n, s.warp_tile_mn), inst

    def _table(self, **kw):
        from dispatch.grouped_convolution import _gfx950_dgrad_tile

        return _gfx950_dgrad_tile(_dgrad(dtype="bf16", **kw))[:6]

    def test_large_grid_takes_the_128x128_tile(self):
        # N=8 C=640 K=768 32x32: 64 M tiles x 5 N tiles = 320 workgroups.
        self.assertEqual(
            self._table(N=8, C=640, K=768, Hi=32, Wi=32), (128, 128, 64, 2, 2, 16)
        )
        # Outside the halo pick dispatch takes the table: kpg % 64 != 0 (the
        # flat folded loop), and a size-changing pad (the tap-outer loop).
        tile, inst = self._tile(N=8, C=640, K=800, Hi=32, Wi=32)
        self.assertEqual(tile, (128, 128, 64, 2, 2, 16))
        self.assertEqual(inst.dy_halo, 0)
        tile, inst = self._tile(N=8, C=640, K=768, Hi=32, Wi=32, pad_h=0, pad_w=0)
        self.assertEqual(tile, (128, 128, 64, 2, 2, 16))
        self.assertTrue(inst.uses_tap_outer_k)
        self.assertEqual(inst.dy_halo, 0)

    def test_one_workgroup_per_cu_keeps_the_64x64_tile(self):
        # N=8 C=512 K=768 32x32: 64 M tiles x 4 N tiles = 256 workgroups,
        # one per CU: below the 128x128 floor.
        self.assertEqual(
            self._table(N=8, C=512, K=768, Hi=32, Wi=32), (64, 64, 64, 2, 2, 32)
        )

    def test_small_grid_keeps_the_64x64_tile(self):
        # N=4 C=1024 16x16: the 128x128 tile would leave 64 workgroups.
        tile, _ = self._tile(N=4, C=1024, K=1024, Hi=16, Wi=16)
        self.assertEqual(tile, (64, 64, 64, 2, 2, 32))

    def test_few_channels_keep_the_64x64_tile(self):
        # cpg = 64 < 128: a 128-wide N tile would be half empty.
        self.assertEqual(
            self._table(N=16, C=64, K=192, Hi=40, Wi=40), (64, 64, 64, 2, 2, 32)
        )

    def test_kpg_odd_multiple_of_32_keeps_the_default_tile(self):
        # kpg % 64 == 32: tile_k 64 straddles taps (flat folded loop), but
        # tile_k 32 only won with cache-resident operands, so the default is
        # kept for every such problem -- few or many groups, short or long
        # reductions, small and large grids.
        for kw in (
            {"N": 1, "C": 64, "K": 96, "Hi": 13, "Wi": 17, "G": 1, "Y": 3},
            {"N": 1, "C": 9600, "K": 9600, "Hi": 8, "Wi": 8, "G": 300, "Y": 7},
            {"N": 4, "C": 8192, "K": 4096, "Hi": 7, "Wi": 7, "G": 128, "Y": 3},
            {"N": 8, "C": 512, "K": 768, "Hi": 7, "Wi": 7, "G": 8, "Y": 7},
            {"N": 1, "C": 9600, "K": 28800, "Hi": 20, "Wi": 20, "G": 300, "Y": 7},
            {"N": 8, "C": 12800, "K": 19200, "Hi": 9, "Wi": 9, "G": 200, "Y": 3},
            {"N": 4, "C": 1024, "K": 2560, "Hi": 8, "Wi": 8, "G": 16, "Y": 1},
        ):
            with self.subTest(**kw):
                y = kw.pop("Y")
                tile, inst = self._tile(Y=y, X=y, pad_h=y // 2, pad_w=y // 2, **kw)
                self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
                if y > 1:
                    self.assertFalse(inst.uses_tap_outer_k)
                    self.assertTrue(inst.folds_sub_gemm_record)

    def test_small_pointwise_keeps_the_default_tile(self):
        # Ungrouped 1x1 below the 128x128 grid bound: the unchanged 64x64
        # default (a wider tile halves an already small grid), and the
        # runtime record (no fold, so no constant trip count to unroll).
        for kw in (
            {"N": 1, "C": 256, "K": 256, "Hi": 14, "Wi": 14},
            {"N": 1, "C": 512, "K": 512, "Hi": 7, "Wi": 7},
            {"N": 8, "C": 512, "K": 1024, "Hi": 28, "Wi": 28},
            {"N": 2, "C": 64, "K": 128, "Hi": 28, "Wi": 28},
        ):
            with self.subTest(**kw):
                tile, inst = self._tile(Y=1, X=1, pad_h=0, pad_w=0, **kw)
                self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
                self.assertFalse(inst.folds_sub_gemm_record)
                self.assertFalse(inst.uses_tap_outer_k)

    def test_large_pointwise_takes_the_128x128_tile(self):
        # 392 M tiles x 2 N tiles: the grid still fills the device.
        tile, inst = self._tile(
            N=64, C=256, K=512, Hi=28, Wi=28, Y=1, X=1, pad_h=0, pad_w=0
        )
        self.assertEqual(tile, (128, 128, 64, 2, 2, 16))
        self.assertFalse(inst.folds_sub_gemm_record)
        # Three tile_k steps (kpg 129..192) is the shortest K loop that takes it.
        tile, _ = self._tile(
            N=8, C=256, K=160, Hi=56, Wi=56, Y=1, X=1, pad_h=0, pad_w=0
        )
        self.assertEqual(tile, (128, 128, 64, 2, 2, 16))

    def test_short_k_pointwise_keeps_the_64x64_tile_on_large_grids(self):
        # Ungrouped 1x1 with a K loop of one or two tile_k steps (kpg <= 128):
        # the 64x64 tile even when the 128x128 grid is far above the floor
        # (784-3136 workgroups here).
        for kw in (
            {"N": 8, "C": 512, "K": 64, "Hi": 56, "Wi": 56},
            {"N": 8, "C": 1024, "K": 64, "Hi": 56, "Wi": 56},
            {"N": 8, "C": 256, "K": 32, "Hi": 56, "Wi": 56},
            {"N": 8, "C": 512, "K": 128, "Hi": 56, "Wi": 56},
            {"N": 32, "C": 512, "K": 96, "Hi": 56, "Wi": 56},
        ):
            with self.subTest(**kw):
                tile, inst = self._tile(Y=1, X=1, pad_h=0, pad_w=0, **kw)
                self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
                self.assertFalse(inst.folds_sub_gemm_record)

    def test_wide_c_pointwise_needs_eight_workgroups_per_cu(self):
        # Ungrouped 1x1 with cpg > 768 and a K loop of at most eight tile_k
        # steps (kpg <= 512) keeps the 64x64 tile below 2048 128x128
        # workgroups (8 per CU on gfx950).
        for kw in (
            {"N": 64, "C": 1024, "K": 256, "Hi": 14, "Wi": 14},  # 784
            {"N": 128, "C": 1024, "K": 256, "Hi": 14, "Wi": 14},  # 1568
            {"N": 128, "C": 2048, "K": 512, "Hi": 7, "Wi": 7},  # 784
            {"N": 8, "C": 896, "K": 256, "Hi": 28, "Wi": 28},  # 343
        ):
            with self.subTest(**kw):
                tile, _ = self._tile(Y=1, X=1, pad_h=0, pad_w=0, **kw)
                self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
        # From 8 workgroups per CU up, with a longer K loop, or at cpg <= 768
        # the 1.25-per-CU floor alone applies.
        for kw in (
            {"N": 64, "C": 1024, "K": 256, "Hi": 28, "Wi": 28},  # 3136
            {"N": 32, "C": 2048, "K": 512, "Hi": 28, "Wi": 28},  # 3136
            {"N": 64, "C": 1024, "K": 1024, "Hi": 14, "Wi": 14},  # 16 steps
            {"N": 64, "C": 768, "K": 256, "Hi": 14, "Wi": 14},  # cpg 768
        ):
            with self.subTest(**kw):
                tile, _ = self._tile(Y=1, X=1, pad_h=0, pad_w=0, **kw)
                self.assertEqual(tile, (128, 128, 64, 2, 2, 16))

    def test_grouped_pointwise_folds_the_record(self):
        # Grouped 1x1 has no divide-free fast path, so it keeps the fold.
        tile, inst = self._tile(
            N=8, C=256, K=256, Hi=28, Wi=28, Y=1, X=1, pad_h=0, pad_w=0, G=2
        )
        self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
        self.assertTrue(inst.folds_sub_gemm_record)

    def test_strided_keeps_the_default_tile(self):
        tile, inst = self._tile(
            N=16, C=256, K=256, Hi=56, Wi=56, stride_h=2, stride_w=2
        )
        self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
        self.assertFalse(inst.folds_sub_gemm_record)


def _dgrad_valid(inst):
    from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

    return is_valid_dgrad_spec(inst, "gfx950")


class TestGfx950DgradHaloPick(unittest.TestCase):
    """dY halo reuse picks of the gfx950 dgrad candidate (_gfx950_dgrad_halo_pick).

    Stride-1 problems with a same-size output, a multi-tap filter and kpg % 64
    == 0 take the staged-halo loop (dy_halo=2) on a 64x64 (small grids),
    128x64 or 256x64 (large grids) tile; everything else keeps the tile table.
    """

    def _pick(self, dtype="bf16", **kw):
        req = _dgrad(dtype=dtype, **kw)
        s = dispatch_conv_grouped(req).spec
        inst = s.to_dgrad_spec(_problem(req))
        ok, why = _dgrad_valid(inst)
        self.assertTrue(ok, why)
        return s, inst

    def _assert_halo(self, s, inst, tile, setprio, kpad):
        self.assertEqual((s.tile_m, s.tile_n, s.warp_m, s.warp_n), tile)
        self.assertEqual(
            (s.dy_halo, s.dy_halo_setprio, s.dy_halo_kouter_pad), (2, setprio, kpad)
        )
        # The dispatch spec forwards every knob and names every one it sets.
        self.assertTrue(inst.uses_dy_halo)
        self.assertEqual(
            (
                inst.dy_halo,
                inst.dy_halo_setprio,
                inst.dy_halo_kouter_pad,
                inst.dy_halo_2d,
            ),
            (s.dy_halo, s.dy_halo_setprio, s.dy_halo_kouter_pad, s.dy_halo_2d),
        )
        name = s.kernel_name()
        self.assertIn("halo2", name)
        self.assertEqual("hprio" in name, setprio > 0)
        self.assertEqual("hkp" in name, kpad > 0)

    def test_small_grid_takes_the_64x64_tile(self):
        # 8x14x14: 25 M tiles of 64 x 4 N tiles; below 192 128x64 workgroups.
        s, inst = self._pick(N=8, C=256, K=256, Hi=14, Wi=14)
        self._assert_halo(s, inst, (64, 64, 2, 2), 0, 0)

    def test_mid_grid_takes_the_128x64_tile(self):
        # 8x16x16 C1280: 16 M tiles x 20 N tiles = 320 128x64 workgroups. The
        # 32-element B pad keeps three workgroups per CU in LDS, so it is set.
        s, inst = self._pick(N=8, C=1280, K=1280, Hi=16, Wi=16)
        self._assert_halo(s, inst, (128, 64, 4, 1), 1, 32)

    def test_b_pad_only_where_it_keeps_the_occupancy(self):
        # 8x32x32 C640: the pad would lift the 128x64 tile over 160 KB / 3,
        # dropping a workgroup per CU.
        s, inst = self._pick(N=8, C=640, K=1280, Hi=32, Wi=32)
        self._assert_halo(s, inst, (128, 64, 4, 1), 1, 0)

    def test_large_grid_takes_the_256x64_tile(self):
        # 128x60x60: 1800 256x64 workgroups.
        s, inst = self._pick(dtype="fp16", N=128, C=64, K=256, Hi=60, Wi=60)
        self._assert_halo(s, inst, (256, 64, 4, 1), 1, 32)

    def test_wide_halo_mid_grid_takes_the_256x64_tile(self):
        # 2x64x64 C640: the 128x64 tile's halo is a whole tile of extra rows
        # and the 256x64 tile keeps its LDS occupancy (two per CU).
        s, inst = self._pick(N=2, C=640, K=640, Hi=64, Wi=64)
        self._assert_halo(s, inst, (256, 64, 4, 1), 1, 0)

    def test_grouped_takes_the_halo(self):
        s, inst = self._pick(N=32, C=512, K=512, Hi=14, Wi=14, G=2)
        self._assert_halo(s, inst, (128, 64, 4, 1), 1, 32)

    def test_not_applied(self):
        # 1x1, stride 2, a size-changing pad, kpg % 64 != 0: tile table.
        for kw in (
            {"N": 32, "C": 256, "K": 1024, "Y": 1, "X": 1, "pad_h": 0, "pad_w": 0},
            {
                "N": 16,
                "C": 256,
                "K": 256,
                "Hi": 56,
                "Wi": 56,
                "stride_h": 2,
                "stride_w": 2,
            },
            {"N": 8, "C": 256, "K": 256, "Hi": 28, "Wi": 28, "pad_h": 0, "pad_w": 0},
            {"N": 1, "C": 64, "K": 96, "Hi": 13, "Wi": 17},
        ):
            with self.subTest(**kw):
                s, inst = self._pick(**kw)
                self.assertEqual(s.dy_halo, 0)
                self.assertFalse(inst.uses_dy_halo)
                self.assertNotIn("halo", s.kernel_name())

    def test_wide_images_fall_back(self):
        # A halo that leaves one workgroup per CU on a grid that needs more
        # than one per CU: the 256x64 tile falls back to 128x64 when that
        # keeps two per CU, otherwise to the tile table.
        s, _ = self._pick(N=2, C=512, K=512, Hi=96, Wi=96)
        self._assert_halo(s, _, (128, 64, 4, 1), 1, 32)
        for kw in (
            {"dtype": "fp16", "N": 2, "C": 128, "K": 128, "Hi": 192, "Wi": 192},
            {"N": 1, "C": 128, "K": 256, "Hi": 300, "Wi": 300},
        ):
            with self.subTest(**kw):
                s, inst = self._pick(**kw)
                self.assertEqual(s.dy_halo, 0)
        # A wide image on a grid that fits one workgroup per CU keeps it.
        s, inst = self._pick(N=1, C=64, K=64, Hi=224, Wi=224)
        self._assert_halo(s, inst, (256, 64, 4, 1), 1, 32)

    def test_exact_128x128_table_pick_is_kept_on_wide_images(self):
        # cpg 128 or 256 on 96..136-pixel rows: the halo pick would be the
        # 128x64 tile at two workgroups per CU, and the tile table's 128x128
        # tile covers the input channels exactly in one or two N tiles, so
        # dispatch keeps it.
        from dispatch.grouped_convolution import _gfx950_dgrad_tile

        for kw in (
            {"N": 4, "C": 128, "K": 128, "Hi": 128, "Wi": 128},
            {"dtype": "fp16", "N": 4, "C": 128, "K": 128, "Hi": 136, "Wi": 136},
            {"dtype": "fp16", "N": 8, "C": 128, "K": 128, "Hi": 120, "Wi": 120},
            {"N": 4, "C": 128, "K": 256, "Hi": 128, "Wi": 128},
            {"N": 4, "C": 256, "K": 256, "Hi": 128, "Wi": 128, "G": 2},
            {"N": 4, "C": 256, "K": 256, "Hi": 96, "Wi": 128},
            {"dtype": "fp16", "N": 4, "C": 256, "K": 128, "Hi": 96, "Wi": 128},
        ):
            with self.subTest(**kw):
                s, inst = self._pick(**kw)
                self.assertEqual(s.dy_halo, 0)
                self.assertFalse(inst.uses_dy_halo)
                table = _gfx950_dgrad_tile(_dgrad(**{"dtype": "bf16", **kw}))
                self.assertEqual(table[:2], (128, 128))
                self.assertEqual((s.tile_m, s.tile_n), (128, 128))
        # Partly empty table N tiles (cpg 192) or more than two of them (cpg
        # 384): the halo pick is taken.
        s, inst = self._pick(N=4, C=192, K=192, Hi=128, Wi=128)
        self._assert_halo(s, inst, (128, 64, 4, 1), 1, 0)
        s, inst = self._pick(dtype="fp16", N=4, C=384, K=384, Hi=96, Wi=96)
        self._assert_halo(s, inst, (128, 64, 4, 1), 1, 32)
        # cpg = 128 where the table keeps the 64x64 tile: the halo pick stays.
        s, inst = self._pick(N=2, C=128, K=128, Hi=128, Wi=128)
        self._assert_halo(s, inst, (128, 64, 4, 1), 1, 0)
        # cpg = 128 on the 256x64 tile (narrower image): the halo pick stays.
        s, inst = self._pick(N=16, C=128, K=128, Hi=56, Wi=56)
        self._assert_halo(s, inst, (256, 64, 4, 1), 1, 32)

    def test_more_than_81_taps_keep_the_tile_table(self):
        # 9x9 takes the halo; 11x11 and larger keep the tile table (the halo
        # loop unrolls every tap).
        s, inst = self._pick(N=2, C=64, K=64, Hi=32, Wi=32, Y=9, X=9, pad_h=4, pad_w=4)
        self.assertEqual(s.dy_halo, 2)
        for y, x in ((11, 11), (1, 83), (15, 15)):
            with self.subTest(Y=y, X=x):
                s, inst = self._pick(
                    N=2, C=64, K=64, Hi=32, Wi=32, Y=y, X=x, pad_h=y // 2, pad_w=x // 2
                )
                self.assertEqual(s.dy_halo, 0)
                self.assertFalse(inst.uses_dy_halo)

    def test_4x1_tiles_take_at_most_7x7_filters(self):
        # 9x9, 7x9 and 9x7 filters that would take a 4x1-wave tile (128x64)
        # keep the tile table (large weights lost there with cold caches);
        # 7x7 keeps the 128x64 halo tile and 9x9 keeps the 64x64 one.
        import dispatch.grouped_convolution as gc

        big = {"N": 16, "C": 4096, "K": 4096, "Hi": 3, "Wi": 30, "G": 64}
        for y, x in ((9, 9), (7, 9), (9, 7)):
            with self.subTest(Y=y, X=x):
                kw = {**big, "Y": y, "X": x, "pad_h": y // 2, "pad_w": x // 2}
                self._assert_table(kw)
                saved = gc._GFX950_DGRAD_HALO_4X1_MAX_TAPS
                gc._GFX950_DGRAD_HALO_4X1_MAX_TAPS = y * x
                try:
                    tile, _knobs = gc._gfx950_dgrad_halo_pick(_dgrad(**kw))
                finally:
                    gc._GFX950_DGRAD_HALO_4X1_MAX_TAPS = saved
                self.assertEqual(tile[3:5], (4, 1))
        kw = {"N": 8, "C": 2048, "K": 2048, "Hi": 7, "Wi": 30, "G": 16}
        s, inst = self._pick(Y=7, X=7, pad_h=3, pad_w=3, **kw)
        self._assert_halo(s, inst, (128, 64, 4, 1), 1, 32)
        s, inst = self._pick(N=2, Hi=32, Wi=32, Y=9, X=9, pad_h=4, pad_w=4)
        self._assert_halo(s, inst, (64, 64, 2, 2), 0, 0)

    def test_non_square_filters_take_the_halo(self):
        # One-dimensional filters (one row or one column) keep the tile table.
        for y, x in ((7, 1), (3, 1), (1, 7), (1, 3), (1, 5)):
            with self.subTest(Y=y, X=x):
                kw = {"N": 8, "C": 256, "K": 256, "Hi": 14, "Wi": 14}
                self._assert_table(
                    {"Y": y, "X": x, "pad_h": y // 2, "pad_w": x // 2, **kw}
                )
        # 1x3 on a wide image (the 256x64 tile lost there) too.
        self._assert_table(
            {"N": 16, "C": 128, "K": 128, "Hi": 136, "Wi": 136, "Y": 1, "X": 3,
             "pad_h": 0, "pad_w": 1, "dtype": "fp16"}
        )  # fmt: skip
        for y, x in ((3, 5), (5, 3)):
            with self.subTest(Y=y, X=x):
                s, inst = self._pick(
                    N=8,
                    C=256,
                    K=256,
                    Hi=14,
                    Wi=14,
                    Y=y,
                    X=x,
                    pad_h=y // 2,
                    pad_w=x // 2,
                )
                self._assert_halo(s, inst, (64, 64, 2, 2), 0, 0)

    def test_small_grid_256x64_needs_a_tall_halo(self):
        # A grid of at most one 256x64 workgroup per CU: the 256x64 tile only
        # when the per-tap halo spans two or more 128x64 tiles, else 128x64.
        for kw in (
            {"N": 8, "C": 64, "K": 64},
            {"N": 4, "C": 128, "K": 128},
        ):
            with self.subTest(**kw):
                s, inst = self._pick(dtype="fp16", Hi=64, Wi=64, **kw)
                self.assertEqual((s.tile_m, s.tile_n), (128, 64))
                self.assertEqual(s.dy_halo, 2)
        # 3x3 on 200-pixel rows (402 halo rows): the 256x64 tile amortizes
        # the halo better.
        for kw in (
            {"N": 1, "C": 128, "K": 128, "Hi": 128, "G": 2},
            {"dtype": "fp16", "N": 2, "C": 256, "K": 128, "Hi": 32, "G": 2},
        ):
            with self.subTest(**kw):
                s, inst = self._pick(Wi=200, **kw)
                self._assert_halo(s, inst, (256, 64, 4, 1), 1, 32)

    def test_low_dy_reuse_keeps_the_tile_table(self):
        # dY reuse = taps * tile_m / (tile_m + (Y-1)*Wo + (X-1)). A 3x3 filter
        # on a very wide image stages more dY rows than the tap-outer loop
        # gathers: the 64x64 tile below 1.0 on more than one workgroup per CU
        # keeps the tile table.
        self._assert_table(
            {"dtype": "fp16", "N": 4, "C": 64, "K": 64, "Hi": 16, "Wi": 600}
        )
        # Square filters keep plenty of reuse on ordinary images.
        s, inst = self._pick(dtype="fp16", N=4, C=128, K=128, Hi=96, Wi=96)
        self.assertEqual(s.dy_halo, 2)
        s, inst = self._pick(N=4, C=64, K=64, Hi=48, Wi=80)
        self._assert_halo(s, inst, (64, 64, 2, 2), 0, 0)

    def _assert_table(self, kw):
        from dispatch.grouped_convolution import _gfx950_dgrad_tile

        s, inst = self._pick(**kw)
        self.assertEqual(s.dy_halo, 0)
        self.assertFalse(inst.uses_dy_halo)
        self.assertNotIn("halo", s.kernel_name())
        table = _gfx950_dgrad_tile(_dgrad(**{"dtype": "bf16", **kw}))
        self.assertEqual((s.tile_m, s.tile_n), table[:2])

    def test_small_grid_vertical_filters_keep_the_tile_table(self):
        # Vertical-only filters whose 256x64 pick has at most one workgroup
        # per CU keep the tile table (not the 128x64 tile), whatever the dY
        # reuse or grid size.
        for kw in (
            {"N": 1, "C": 512, "K": 128, "Hi": 32, "Wi": 160, "Y": 3, "G": 2},
            {"dtype": "fp16", "N": 4, "C": 128, "K": 256, "Hi": 32, "Wi": 152, "Y": 3},
            {"N": 4, "C": 256, "K": 64, "Hi": 16, "Wi": 176, "Y": 5},
            {"N": 4, "C": 128, "K": 256, "Hi": 32, "Wi": 144, "Y": 5},
            {"dtype": "fp16", "N": 1, "C": 192, "K": 192, "Hi": 112, "Wi": 112, "Y": 5},
            {"N": 2, "C": 64, "K": 64, "Hi": 168, "Wi": 168, "Y": 5},
            {"dtype": "fp16", "N": 8, "C": 128, "K": 128, "Hi": 64, "Wi": 64, "Y": 7},
            {"dtype": "fp16", "N": 2, "C": 128, "K": 128, "Hi": 112, "Wi": 104, "Y": 7},
        ):
            with self.subTest(**kw):
                self._assert_table({"X": 1, "pad_h": kw["Y"] // 2, "pad_w": 0, **kw})

    def test_small_grid_64x64_tile_reuse_floor(self):
        # On grids of at most one workgroup per CU the 64x64 halo tile needs a
        # dY reuse of _GFX950_DGRAD_HALO_MIN_REUSE_2X2_SMALL_GRID: 3x3 on
        # 400- and 420-pixel rows (0.67, 0.63) keeps the tile table, on 320-
        # and 256-pixel rows (0.82, 1.0) takes the halo.
        for kw in (
            {"N": 1, "C": 128, "K": 128, "Hi": 20, "Wi": 400},
            {"N": 1, "C": 64, "K": 64, "Hi": 8, "Wi": 420},
        ):
            with self.subTest(**kw):
                self._assert_table({"dtype": "fp16", **kw})
        for kw in (
            {"dtype": "fp16", "N": 1, "C": 64, "K": 64, "Hi": 16, "Wi": 320},
            {"N": 1, "C": 64, "K": 256, "Hi": 24, "Wi": 256},
        ):
            with self.subTest(**kw):
                s, inst = self._pick(**kw)
                self._assert_halo(s, inst, (64, 64, 2, 2), 0, 0)

    def test_pick_is_memoized_per_request(self):
        from dispatch.grouped_convolution import _gfx950_dgrad_pick

        req = _dgrad(dtype="bf16", N=8, C=256, K=256, Hi=14, Wi=14)
        tile, knobs = _gfx950_dgrad_pick(req)
        knobs["dy_halo"] = 0  # the caller's copy; the cached pick is unchanged
        self.assertEqual(_gfx950_dgrad_pick(req), (tile, {"dy_halo": 2}))


class TestConvGroupedSpecHaloFields(unittest.TestCase):
    """The dY halo fields of ConvGroupedSpec: dgrad only, hashed only when set."""

    def _spec(self, direction, **kw):
        from dispatch.grouped_convolution import ConvGroupedSpec

        return ConvGroupedSpec(
            direction=direction,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_mn=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            dtype="bf16",
            arch="gfx950",
            **kw,
        )

    def test_rejected_outside_dgrad(self):
        for direction in ("fwd", "wgrad"):
            for knob, val in (
                ("dy_halo", 2),
                ("dy_halo_2d", True),
                ("dy_halo_setprio", 1),
                ("dy_halo_kouter_pad", 32),
            ):
                with (
                    self.subTest(direction=direction, knob=knob),
                    self.assertRaisesRegex(ValueError, f"{knob} only apply"),
                ):
                    self._spec(direction, **{knob: val})
            self._spec(direction)  # all off: accepted
        self._spec("dgrad", dy_halo=2, dy_halo_setprio=1)

    def test_spec_hash_ignores_unset_halo_fields(self):
        from dataclasses import asdict

        from dispatch.grouped_convolution import _DGRAD_HALO_FIELDS, _kernel_id

        req = _dgrad(dtype="bf16")
        cand = conv_grouped_candidates("dgrad")[0]
        spec = self._spec("dgrad")
        legacy = {k: v for k, v in asdict(spec).items() if k not in _DGRAD_HALO_FIELDS}
        from rocke.dispatch.core import stable_json_hash

        self.assertEqual(
            _kernel_id(req, cand, spec).spec_hash, stable_json_hash(legacy, n=16)
        )
        halo = self._spec("dgrad", dy_halo=2)
        self.assertNotEqual(
            _kernel_id(req, cand, halo).spec_hash, _kernel_id(req, cand, spec).spec_hash
        )


class TestGroupedDirectDgradDispatch(unittest.TestCase):
    """Selection, launch plan and grid for the gfx950 direct-MFMA dgrad.

    Grouped stride-1 dgrad with cpg/kpg multiples of 4 up to 32 and filters up
    to 7x7, outside the measured igemm-win corners, must route to the
    direct-MFMA candidate: one kernel that reads W with flipped,
    channel-swapped addressing (the batched 4x4x4 kernel for cpg = kpg = 4),
    or the pre-pass pipeline past the fused form's register budget;
    everything else must keep the igemm candidate.
    """

    _DIRECT = "direct_mfma_conv_dgrad"
    _IGEMM = "implicit_gemm_conv_dgrad"

    @staticmethod
    def _grouped(**kw):
        base = {
            "N": 8,
            "C": 128,
            "K": 128,
            "Hi": 14,
            "Wi": 14,
            "G": 32,
            "dtype": "bf16",
        }
        base.update(kw)
        return _dgrad(**base)

    def _pick(self, req):
        r = dispatch_conv_grouped(req)
        return r.candidate.name, r

    # ---- routing: in-region requests take the direct pipeline -------------

    def test_target_shapes_route_to_direct_with_table_knobs(self):
        # (N, C, K, H, W, G) -> (variant, block_q, block_groups, block_h,
        #                         fold_k32, rule, waves_per_eu)
        cases = {
            (128, 128, 128, 56, 56, 32): ("4c", 4, 16, 0, False, "cpg_kpg_4", 0),
            (128, 512, 512, 14, 14, 32): ("generic", 16, 2, 0, False, "chan_16_31", 4),
            (128, 512, 512, 14, 14, 16): ("generic", 16, 1, 0, True, "chan_ge_32", 0),
        }
        for (n, c, k, h, w, g), want in cases.items():
            with self.subTest(shape=(n, c, k, h, w, g)):
                name, r = self._pick(self._grouped(N=n, C=c, K=k, Hi=h, Wi=w, G=g))
                self.assertEqual(name, self._DIRECT)
                s = r.spec
                self.assertEqual(
                    (
                        s.variant,
                        s.block_q,
                        s.block_groups,
                        s.block_h,
                        s.fold_k32,
                        s.rule_id,
                        s.waves_per_eu,
                    ),
                    want,
                )
                # Single kernel: weights transformed in the prologue, staged
                # through LDS with transpose reads on gfx950.
                self.assertEqual((s.fused_weights, s.weights_lds), (True, True))
                self.assertEqual(
                    (s.waves_q, s.waves_k, s.runtime_k_loop), (1, 1, False)
                )

    def test_both_candidates_admit_and_direct_outranks(self):
        req = self._grouped()
        names = [c.name for c in conv_grouped_candidates("dgrad") if c.admits(req)[0]]
        self.assertEqual(names, [self._DIRECT, self._IGEMM])

    # ---- routing: out-of-region requests keep igemm ------------------------

    def test_out_of_region_requests_keep_igemm(self):
        cases = {
            "dense": {"C": 64, "K": 64, "G": 1},
            "stride2": {"stride_h": 2, "stride_w": 2},
            "dilation2": {"dilation_h": 2, "dilation_w": 2, "pad_h": 2, "pad_w": 2},
            "cpg64": {"C": 2048, "K": 2048, "G": 32},
            "cpg6_not_vec4": {"C": 192, "K": 128, "G": 32},
            "kpg6_not_vec4": {"C": 128, "K": 192, "G": 32},
            "nonsquare_filter": {"Y": 3, "X": 1, "pad_w": 0},
            "asym_pad": {"pad_h": 1, "pad_w": 0},
            "pad_beyond_filter": {"pad_h": 3, "pad_w": 3},
            "pad0_not_same": {"pad_h": 0, "pad_w": 0},
            "depthwise": {"C": 64, "K": 64, "G": 64},
            "filter9x9": {"N": 128, "Y": 9, "X": 9, "pad_h": 4, "pad_w": 4},
            "filter11x11": {"N": 128, "Y": 11, "X": 11, "pad_h": 5, "pad_w": 5},
        }
        for label, kw in cases.items():
            with self.subTest(case=label):
                req = self._grouped(**kw)
                direct = next(
                    c
                    for c in conv_grouped_candidates("dgrad")
                    if c.name == self._DIRECT
                )
                ok, why = direct.admits(req)
                self.assertFalse(ok, f"{label}: direct must decline ({why})")
                if label != "depthwise":
                    name, _ = self._pick(req)
                    self.assertEqual(name, self._IGEMM, label)

    # ---- measured policy: corners where igemm wins keep igemm --------------

    @staticmethod
    def _req_from(shape):
        n, c, k, h, w, y, g, dt = shape
        return _dgrad(
            N=n,
            C=c,
            K=k,
            Hi=h,
            Wi=w,
            Y=y,
            X=y,
            pad_h=y // 2,
            pad_w=y // 2,
            G=g,
            dtype=dt,
        )

    def test_policy_declines_where_igemm_measured_faster(self):
        # Same-session measurements of the selected direct spec against the
        # igemm candidate put each of these well on the igemm side.
        # (N, C, K, H, W, Y, G, dtype)
        from dispatch.grouped_convolution import _DIRECT_DGRAD_POLICY_PREFIX

        cases = {
            "tiny_2x2_cpg32_G2": (16384, 64, 64, 2, 2, 3, 2, "bf16"),
            "tiny_3x3_cpg32_G2": (8192, 64, 64, 3, 3, 3, 2, "bf16"),
            "tiny_4x4_cpg32_G8": (1024, 256, 256, 4, 4, 3, 8, "bf16"),
            "tiny_4x4_cpg16_G2_fp16": (4096, 32, 32, 4, 4, 3, 2, "fp16"),
            "tiny_4x4_1x1_cpg32_G4": (2048, 128, 128, 4, 4, 1, 4, "bf16"),
            "W3_column_cpg8_kpg4": (68, 64, 32, 31, 3, 3, 8, "bf16"),
            "W5_cpg32_G4": (2048, 128, 128, 5, 5, 3, 4, "bf16"),
            "prepass_7x7_cpg24_kpg24_5x5img": (581, 288, 288, 5, 5, 7, 12, "bf16"),
            "prepass_7x7_cpg32_kpg24": (105, 128, 96, 22, 18, 7, 4, "bf16"),
            "prepass_7x7_cpg32_kpg4_W24": (210, 64, 8, 20, 24, 7, 2, "bf16"),
            "prepass_7x7_cpg32_kpg4_W24_G4": (105, 128, 16, 20, 24, 7, 4, "bf16"),
            "prepass_7x7_cpg32_kpg8_14x14": (384, 128, 32, 14, 14, 7, 4, "bf16"),
            "1x1_cpg32_kpg32_G8": (64, 256, 256, 28, 28, 1, 8, "bf16"),
            "1x1_cpg8_kpg32_G16": (128, 128, 512, 14, 14, 1, 16, "bf16"),
            "5x5_cpg24_kpg4_8x8_lowfill": (440, 192, 32, 8, 8, 5, 8, "bf16"),
            # Pre-pass form whose weight transpose is not amortized (many
            # groups, 7x7 or 5x5 weights, 10x10 images, few images), or
            # whose main kernel loses outright (8x8, partial K atom).
            "prepass_unamortized_G200_N1": (1, 6400, 3200, 10, 10, 7, 200, "fp16"),
            "prepass_unamortized_G300_N1": (1, 9600, 9600, 10, 10, 7, 300, "fp16"),
            "prepass_unamortized_G300_N4": (4, 9600, 4800, 10, 10, 7, 300, "fp16"),
            "prepass_unamortized_G200_bf16": (1, 5600, 3200, 10, 10, 7, 200, "bf16"),
            "prepass_unamortized_5x5_G300": (4, 9600, 9600, 10, 10, 5, 300, "fp16"),
            "prepass_main_loses_8x8_kpg20": (42, 1980, 1980, 8, 8, 7, 99, "bf16"),
            # Narrow reduction (kpg 8 under cpg 20: half-filled K atoms) on a
            # short, narrow image.
            "prepass_main_loses_narrow_5x8": (153, 5060, 2024, 5, 8, 7, 253, "bf16"),
        }
        direct = next(
            c for c in conv_grouped_candidates("dgrad") if c.name == self._DIRECT
        )
        for label, shape in cases.items():
            with self.subTest(case=label):
                req = self._req_from(shape)
                ok, why = direct.admits(req)
                self.assertFalse(ok, label)
                self.assertIn(_DIRECT_DGRAD_POLICY_PREFIX, why, label)
                self.assertEqual(self._pick(req)[0], self._IGEMM, label)

    def test_prepass_decline_names_the_cost_model(self):
        # The pre-pass decline quotes the predicted cost ratio, and the ratio
        # falls as the same weights are amortized over more images.
        from dispatch.grouped_convolution import (
            _DIRECT_DGRAD_PREPASS_MAX_COST_RATIO,
            _direct_dgrad_policy_errors,
            _direct_dgrad_prepass_cost_ratio,
            _direct_dgrad_problem,
            _select_direct_dgrad_spec,
        )

        req = self._req_from((1, 6400, 3200, 10, 10, 7, 200, "fp16"))
        spec = _select_direct_dgrad_spec(req)
        self.assertFalse(spec.fused_weights)
        errors = _direct_dgrad_policy_errors(req, spec)
        self.assertEqual(len(errors), 1)
        self.assertIn("not amortized", errors[0])
        ratios = []
        for n in (1, 4, 16):
            r = self._req_from((n, 6400, 3200, 16, 16, 7, 200, "fp16"))
            ratios.append(
                _direct_dgrad_prepass_cost_ratio(
                    _direct_dgrad_problem(r), _select_direct_dgrad_spec(r)
                )
            )
        self.assertEqual(ratios, sorted(ratios, reverse=True))
        self.assertGreater(ratios[0], _DIRECT_DGRAD_PREPASS_MAX_COST_RATIO)
        self.assertLess(ratios[-1], _DIRECT_DGRAD_PREPASS_MAX_COST_RATIO)

    def test_policy_admits_where_direct_measured_faster(self):
        # Measured direct wins over the igemm candidate, across the classes
        # the policy separates (fused single kernel, 4c row, pre-pass fallback
        # with full K atoms, 1x1 narrow reductions, few groups).
        cases = {
            "S1_cpg4_G32": (128, 128, 128, 56, 56, 3, 32, "bf16"),
            "S2_cpg16_G32": (128, 512, 512, 14, 14, 3, 32, "bf16"),
            "S4_cpg32_G16": (128, 512, 512, 14, 14, 3, 16, "bf16"),
            "prepass_7x7_cpg32_kpg32": (16, 256, 256, 28, 28, 7, 8, "fp16"),
            "5x5_cpg32_kpg32_28x28": (32, 128, 128, 28, 28, 5, 4, "bf16"),
            "prepass_7x7_cpg4_kpg32_G12": (193, 48, 384, 20, 20, 7, 12, "fp16"),
            "prepass_7x7_cpg28_kpg32_G64": (8, 1792, 2048, 16, 16, 7, 64, "bf16"),
            "prepass_7x7_cpg32_kpg16_G200_16x16": (
                4,
                6400,
                3200,
                16,
                16,
                7,
                200,
                "fp16",
            ),
            "3x3_cpg32_kpg8_G2_16x16": (128, 64, 16, 16, 16, 3, 2, "bf16"),
            "3x3_cpg32_kpg8_G7_12x12": (96, 224, 56, 12, 12, 3, 7, "bf16"),
            "1x1_cpg32_kpg8_G2": (144, 64, 16, 14, 14, 1, 2, "bf16"),
            "1x1_cpg24_kpg24_G2": (141, 48, 48, 56, 56, 1, 2, "fp16"),
            "3x3_cpg32_kpg8_21x21_G32": (8, 1024, 256, 21, 21, 3, 32, "bf16"),
            "7x7_cpg16_kpg16_G2": (128, 32, 32, 14, 14, 7, 2, "bf16"),
        }
        for label, shape in cases.items():
            with self.subTest(case=label):
                self.assertEqual(
                    self._pick(self._req_from(shape))[0], self._DIRECT, label
                )

    def test_4c_row_needs_its_grid_floor(self):
        from dispatch.grouped_convolution import _DIRECT_DGRAD_4C_MIN_GRID

        # S1: (56/4) * (32/16) * 128 workgroups -> the batched 4x4x4 kernel.
        _name, r = self._pick(self._grouped(N=128, Hi=56, Wi=56))
        s = r.spec
        self.assertEqual((s.variant, s.block_q, s.block_groups), ("4c", 4, 16))
        self.assertTrue(s.fused_weights)
        self.assertEqual(r.block, (64, 1, 1))
        self.assertEqual(r.grid, (14, 2, 128))
        # One image: 28 workgroups, below the floor -> the generic kernel.
        name, r = self._pick(self._grouped(N=1, Hi=56, Wi=56))
        self.assertEqual(name, self._DIRECT)
        self.assertEqual(r.spec.variant, "generic")
        self.assertLess(14 * 2 * 1, _DIRECT_DGRAD_4C_MIN_GRID)
        # 4c needs groups % 16 == 0 and a 1x1/3x3 filter.
        _name, r = self._pick(self._grouped(N=256, C=32, K=32, G=8))
        self.assertEqual(r.spec.variant, "generic")

    def test_4c_row_stage_rows_policy(self):
        import dispatch.grouped_convolution as gc

        # (N, H, Y) at C = K = 128, G = 32 (grid = ceil(W/4) * 2 * N)
        # -> (variant, stage_rows)
        cases = {
            "S1": ((128, 56, 3), ("4c", True)),
            "1x1_h14_above_floor": ((64, 14, 1), ("4c", True)),
            "1x1_h7_above_floor": ((128, 7, 1), ("4c", False)),
            "1x1_h14_below_floor": ((16, 14, 1), ("generic", False)),
            "3x3_h8_grid96": ((24, 8, 3), ("4c", True)),
            "3x3_h8_grid64": ((16, 8, 3), ("generic", False)),
            "3x3_h8_grid48": ((12, 8, 3), ("generic", False)),
            "3x3_h14_grid192": ((24, 14, 3), ("generic", False)),
            "3x3_h14_grid64": ((8, 14, 3), ("generic", False)),
            "3x3_h9_grid288": ((48, 9, 3), ("generic", False)),
            "3x3_h20_grid160": ((16, 20, 3), ("generic", False)),
        }
        for label, ((n, hw, y), want) in cases.items():
            with self.subTest(case=label):
                req = self._grouped(
                    N=n, Hi=hw, Wi=hw, Y=y, X=y, pad_h=y // 2, pad_w=y // 2
                )
                name, r = self._pick(req)
                self.assertEqual(name, self._DIRECT, label)
                s = r.spec
                self.assertEqual((s.variant, s.stage_rows), want, label)
                if s.stage_rows:
                    self.assertTrue(s.kernel_name().endswith("_sr"))
                    main = r.spec.to_fprop_spec(gc._direct_dgrad_problem(req))
                    self.assertTrue(main.stage_rows)
                    self.assertEqual(main.waves_q, 1)
                    self.assertTrue(main.kernel_name().endswith("_fwl_sr1"))
                    self.assertEqual(r.block, (64, 1, 1))
        # 3x3 above the floor: images shorter than 4 rows keep the direct-load
        # 4c kernel (exactly the knob-off pick); from 4 rows on they take the
        # staged kernel. 1x1 keeps its own 8-row edge.
        # (N, C, H, W, G, Y, dtype) -> stage_rows
        self.assertEqual(gc._DIRECT_DGRAD_4C_STAGED_MIN_H_3X3, 4)
        self.assertEqual(gc._DIRECT_DGRAD_4C_STAGED_MIN_H_1X1, 8)
        for (n, c, h, w, g, y, dt), want in (
            ((80, 256, 3, 1, 64, 3, "fp16"), False),
            ((19, 1024, 3, 1, 256, 3, "bf16"), False),
            ((256, 128, 3, 2, 32, 3, "fp16"), False),
            ((80, 256, 4, 1, 64, 3, "fp16"), True),
            ((256, 128, 4, 2, 32, 3, "fp16"), True),
            ((80, 256, 4, 1, 64, 1, "fp16"), False),
            ((256, 128, 7, 2, 32, 1, "fp16"), False),
            ((256, 128, 8, 2, 32, 1, "fp16"), True),
        ):
            with self.subTest(N=n, C=c, H=h, W=w, G=g, Y=y):
                grid = -(-w // 4) * (g // 16) * n
                self.assertGreaterEqual(grid, gc._DIRECT_DGRAD_4C_MIN_GRID)
                req = self._grouped(
                    N=n, C=c, K=c, Hi=h, Wi=w, G=g, Y=y, X=y,
                    pad_h=y // 2, pad_w=y // 2, dtype=dt,
                )  # fmt: skip
                name, r = self._pick(req)
                self.assertEqual(name, self._DIRECT)
                self.assertEqual((r.spec.variant, r.spec.stage_rows), ("4c", want))
                if not want:
                    saved = gc._DIRECT_DGRAD_4C_STAGE_ROWS
                    gc._DIRECT_DGRAD_4C_STAGE_ROWS = False
                    try:
                        self.assertEqual(self._pick(req)[1].spec, r.spec)
                    finally:
                        gc._DIRECT_DGRAD_4C_STAGE_ROWS = saved
        # Knob off: the PR-branch pick (direct-load 4c above the floor only).
        saved = gc._DIRECT_DGRAD_4C_STAGE_ROWS
        gc._DIRECT_DGRAD_4C_STAGE_ROWS = False
        try:
            s = self._pick(self._grouped(N=128, Hi=56, Wi=56))[1].spec
            self.assertEqual((s.variant, s.stage_rows), ("4c", False))
            s = self._pick(self._grouped(N=16, Hi=8, Wi=8))[1].spec
            self.assertEqual(s.variant, "generic")
        finally:
            gc._DIRECT_DGRAD_4C_STAGE_ROWS = saved

    def test_4c_stage_rows_keeps_previous_pick_below_floor(self):
        # 3x3 cpg = kpg = 4 shapes below the 4c grid floor where the row-staged
        # 4c kernel lost to the previous pick with cold caches: images taller
        # than the generic kernel's H tile (grids 192-256) keep the generic
        # kernel with 4-row H tiles, and 5-row images on a 64-workgroup grid
        # keep igemm. (N, C, H, W, G, dtype) -> (candidate, variant, block_h)
        from dispatch.grouped_convolution import (
            _DIRECT_DGRAD_4C_MIN_GRID,
            _DIRECT_DGRAD_4C_STAGED_SHORT_MIN_GRID,
        )

        generic = (self._DIRECT, "generic", 4)
        igemm = (self._IGEMM, None, None)
        cases = {
            "h16_g128_grid256": ((8, 512, 16, 16, 128, "bf16"), generic),
            "h16_w9_g16_grid192": ((64, 64, 16, 9, 16, "fp16"), generic),
            "h14_g48_grid192": ((16, 192, 14, 14, 48, "fp16"), generic),
            "h13_g192_grid192": ((4, 768, 13, 13, 192, "bf16"), generic),
            "h15_g64_grid256": ((16, 256, 15, 15, 64, "fp16"), generic),
            "h12_g128_grid192": ((8, 512, 12, 12, 128, "bf16"), generic),
            "h13_g16_grid256": ((64, 64, 13, 13, 16, "fp16"), generic),
            "h12_w21_g32_grid192": ((16, 128, 12, 21, 32, "bf16"), generic),
            "h5_g64_grid64": ((8, 256, 5, 5, 64, "fp16"), igemm),
            "h5_g32_grid64": ((16, 128, 5, 5, 32, "fp16"), igemm),
            "h5_g128_grid64": ((4, 512, 5, 5, 128, "fp16"), igemm),
        }
        for label, ((n, c, h, w, g, dt), want) in cases.items():
            with self.subTest(case=label):
                grid = -(-w // 4) * (g // 16) * n
                self.assertLess(grid, _DIRECT_DGRAD_4C_MIN_GRID, label)
                req = self._grouped(N=n, C=c, K=c, Hi=h, Wi=w, G=g, dtype=dt)
                name, r = self._pick(req)
                s = r.spec
                got = (
                    (name, s.variant, s.block_h)
                    if name == self._DIRECT
                    else (name, None, None)
                )
                self.assertEqual(got, want, label)
        # Images of 1 to 3 columns keep the previous pick below the floor
        # (igemm here); from 4 columns on they take the staged kernel.
        from dispatch.grouped_convolution import _DIRECT_DGRAD_4C_STAGED_SHORT_MIN_W

        self.assertEqual(_DIRECT_DGRAD_4C_STAGED_SHORT_MIN_W, 4)
        for n, c, w, dt in ((32, 192, 1, "bf16"), (32, 192, 3, "bf16"),
                            (96, 64, 3, "fp16"), (32, 256, 3, "bf16")):  # fmt: skip
            with self.subTest(N=n, C=c, W=w):
                req = self._grouped(N=n, C=c, K=c, Hi=8, Wi=w, G=c // 4, dtype=dt)
                self.assertEqual(self._pick(req)[0], self._IGEMM)
        for n, c, dt in ((32, 192, "bf16"), (96, 64, "fp16")):
            with self.subTest(N=n, C=c, W=4):
                req = self._grouped(N=n, C=c, K=c, Hi=8, Wi=4, G=c // 4, dtype=dt)
                name, r = self._pick(req)
                self.assertEqual(name, self._DIRECT)
                self.assertEqual((r.spec.variant, r.spec.stage_rows), ("4c", True))
        # The same 5-row images at the staged floor take the staged kernel.
        req = self._grouped(N=8, C=512, K=512, Hi=5, Wi=5, G=128)
        self.assertEqual(-(-5 // 4) * 8 * 8, 128)
        self.assertGreaterEqual(128, _DIRECT_DGRAD_4C_STAGED_SHORT_MIN_GRID)
        s = self._pick(req)[1].spec
        self.assertEqual((s.variant, s.stage_rows), ("4c", True))
        # Images of 1 to 4 rows keep the previous pick below the floor; from
        # 5 rows on they take the staged kernel.
        import dispatch.grouped_convolution as gc
        from dispatch.grouped_convolution import _DIRECT_DGRAD_4C_STAGED_SHORT_MIN_H

        self.assertEqual(_DIRECT_DGRAD_4C_STAGED_SHORT_MIN_H, 5)
        saved = gc._DIRECT_DGRAD_4C_STAGE_ROWS
        for n, c, h, w, dt in ((16, 192, 4, 5, "bf16"), (12, 256, 4, 5, "fp16"),
                               (24, 256, 2, 4, "bf16"), (16, 256, 1, 8, "fp16"),
                               (16, 192, 5, 5, "bf16"), (12, 256, 8, 5, "fp16")):  # fmt: skip
            with self.subTest(N=n, C=c, H=h, W=w):
                req = self._grouped(N=n, C=c, K=c, Hi=h, Wi=w, G=c // 4, dtype=dt)
                grid = -(-w // 4) * (c // 64) * n
                self.assertGreaterEqual(grid, _DIRECT_DGRAD_4C_STAGED_SHORT_MIN_GRID)
                self.assertLess(grid, _DIRECT_DGRAD_4C_MIN_GRID)
                name, r = self._pick(req)
                if h >= 5:
                    self.assertEqual(name, self._DIRECT)
                    self.assertEqual((r.spec.variant, r.spec.stage_rows), ("4c", True))
                    continue
                gc._DIRECT_DGRAD_4C_STAGE_ROWS = False
                try:
                    prev = self._pick(req)
                finally:
                    gc._DIRECT_DGRAD_4C_STAGE_ROWS = saved
                self.assertEqual(name, prev[0])
                self.assertEqual(r.spec, prev[1].spec)

    def test_generic_row_stream_knobs_policy(self):
        import dispatch.grouped_convolution as gc

        # (N, C, K, H, W, G, Y) -> (prefetch_rows, lds_only_sync, waves_m,
        #                           lds_pad, stage_out, xcd_tiles)
        cases = {
            # 16-channel groups: the whole stack.
            "S2": ((128, 512, 512, 14, 14, 32, 3), (2, True, 1, 8, True, True)),
            # 4-channel output tiles: no 16-wide stage_out, padded rows.
            "cpg4": ((8, 128, 128, 28, 28, 32, 3), (2, True, 1, 8, False, True)),
            # A 48-byte staged column (odd 16-byte count): no pad; 12 output
            # channels: no 16-wide stage_out.
            "cpg12_7x7": ((8, 384, 384, 16, 16, 32, 7), (2, True, 1, 0, False, True)),
        }
        # Output channels that split into two 16-wide tile halves (waves_m = 2
        # valid) keep the previous pick: no row-stream knobs by default. So do
        # grids with fewer than 8 image row bands (no XCD remap): small
        # batches of unsplit images, here 2, 3 and 4 images of 56 rows and 2
        # of 12 rows; groups with cpg != kpg; and 32-column strips
        # (block_q 32), here 8-, 12- and 20-channel groups.
        for label, (n, c, k, hw, g, y) in {
            "bq32_cpg12_7x7": (8, 384, 384, 24, 32, 7),
            "bq32_cpg12_7x7_G384": (2, 4608, 4608, 18, 384, 7),
            "bq32_cpg8_7x7": (1, 1024, 1024, 160, 128, 7),
            "bq32_cpg20": (4, 2560, 2560, 18, 128, 3),
            "cpg16_kpg4_7x7": (8, 512, 128, 24, 32, 7),
            "cpg20_kpg8": (1, 10240, 4096, 28, 512, 3),
            "cpg20_kpg16": (12, 1280, 1024, 24, 64, 3),
            "cpg24_kpg16": (2, 1152, 768, 64, 48, 3),
            "S4": (128, 512, 512, 14, 16, 3),
            "cpg32_7x7img": (64, 1024, 1024, 7, 32, 3),
            "cpg32_14x14img_N32": (32, 1024, 1024, 14, 32, 3),
            "cpg32_kpg16_N1_G256": (1, 8192, 4096, 56, 256, 3),
            "rows_N4_G192": (4, 3072, 3072, 56, 192, 3),
            "rows_N3_G256": (3, 4096, 4096, 56, 256, 3),
            "rows_N2_G512": (2, 4096, 4096, 56, 512, 3),
            "rows_5x5_G300": (2, 7200, 4800, 12, 300, 5),
        }.items():
            with self.subTest(case=label):
                req = self._grouped(
                    N=n, C=c, K=k, Hi=hw, Wi=hw, G=g, Y=y, X=y,
                    pad_h=y // 2, pad_w=y // 2,
                )  # fmt: skip
                name, r = self._pick(req)
                self.assertEqual(name, self._DIRECT, label)
                s = r.spec
                self.assertEqual((s.variant, s.fused_weights), ("generic", True))
                if label.startswith("bq32"):
                    self.assertEqual(s.block_q, 32, label)
                self.assertEqual(
                    (s.prefetch_rows, s.lds_only_sync, s.waves_m, s.lds_pad),
                    (0, False, 1, 0),
                )
                self.assertEqual((s.stage_out, s.xcd_tiles), (False, False))
                p = gc._direct_dgrad_problem(req)
                stream = replace(s, prefetch_rows=2, lds_only_sync=True)
                split = replace(stream, waves_m=2)
                xcd = replace(stream, xcd_tiles=True)
                self.assertTrue(
                    p.cpg != p.kpg
                    or s.block_q != gc._DIRECT_DGRAD_BLOCK_Q
                    or gc._direct_dgrad_main_is_valid(split, p)[0]
                    or not gc._direct_dgrad_main_is_valid(xcd, p)[0]
                )
        for label, ((n, c, k, hw, _w, g, y), want) in cases.items():
            with self.subTest(case=label):
                req = self._grouped(
                    N=n, C=c, K=k, Hi=hw, Wi=_w, G=g, Y=y, X=y,
                    pad_h=y // 2, pad_w=y // 2,
                )  # fmt: skip
                name, r = self._pick(req)
                self.assertEqual(name, self._DIRECT, label)
                s = r.spec
                self.assertEqual(s.variant, "generic")
                self.assertTrue(s.fused_weights)
                got = (
                    s.prefetch_rows,
                    s.lds_only_sync,
                    s.waves_m,
                    s.lds_pad,
                    s.stage_out,
                    s.xcd_tiles,
                )
                self.assertEqual(got, want, label)
                p = gc._direct_dgrad_problem(req)
                main = s.to_fprop_spec(p)
                self.assertEqual(
                    (
                        main.prefetch_rows,
                        main.lds_only_sync,
                        main.waves_m,
                        main.lds_pad,
                        main.stage_out,
                        main.xcd_tiles,
                    ),
                    want,
                )
                self.assertTrue(gc._direct_dgrad_main_is_valid(s, p)[0])
                self.assertEqual(r.block, (main.threads_per_block, 1, 1))
                # The pick never needs more launch rounds than the spec
                # without the LDS-adding knobs.
                bare = replace(s, lds_pad=0, stage_out=False)
                self.assertLessEqual(
                    gc._direct_dgrad_lds_rounds(s, p),
                    gc._direct_dgrad_lds_rounds(bare, p),
                )
        # Opt-in waves_m = 2 stack (_DIRECT_DGRAD_STREAM_SPLIT_M): 32-channel
        # groups split over two waves. On S4 the pad and the staged output
        # tile would push 2048 single-round workgroups into a second round,
        # so the rounds guard drops them; they stay where the round count
        # does not change.
        saved = gc._DIRECT_DGRAD_STREAM_SPLIT_M
        gc._DIRECT_DGRAD_STREAM_SPLIT_M = True
        try:
            for (n, c, hw, g), want in (
                ((128, 512, 14, 16), (2, True, 2, 0, False, True)),
                ((64, 1024, 7, 32), (2, True, 2, 0, False, True)),
                ((32, 1024, 14, 32), (2, True, 2, 8, True, True)),
            ):
                req = self._grouped(N=n, C=c, K=c, Hi=hw, Wi=hw, G=g)
                s = self._pick(req)[1].spec
                got = (
                    s.prefetch_rows,
                    s.lds_only_sync,
                    s.waves_m,
                    s.lds_pad,
                    s.stage_out,
                    s.xcd_tiles,
                )
                self.assertEqual(got, want, (n, c, hw, g))
            # The rounds model: S4 fits one round without the pad / staged
            # output tile and needs two with them.
            req = self._grouped(N=128, C=512, K=512, Hi=14, Wi=14, G=16)
            s = self._pick(req)[1].spec
            p = gc._direct_dgrad_problem(req)
            self.assertEqual(gc._direct_dgrad_lds_rounds(s, p), 1)
            full = replace(s, lds_pad=8, stage_out=True)
            self.assertTrue(gc._direct_dgrad_main_is_valid(full, p)[0])
            self.assertEqual(gc._direct_dgrad_lds_rounds(full, p), 2)
        finally:
            gc._DIRECT_DGRAD_STREAM_SPLIT_M = saved
        # The pre-pass pipeline (fused weights over budget) is unchanged.
        r = dispatch_conv_grouped(self._req_from((16, 256, 256, 28, 28, 7, 8, "fp16")))
        self.assertFalse(r.spec.fused_weights)
        self.assertEqual(
            (r.spec.prefetch_rows, r.spec.lds_only_sync, r.spec.waves_m),
            (0, False, 1),
        )
        # Knob off: the PR-branch pick.
        saved = gc._DIRECT_DGRAD_STREAM_KNOBS
        gc._DIRECT_DGRAD_STREAM_KNOBS = False
        try:
            s = self._pick(self._grouped(N=128, C=512, K=512, Hi=14, Wi=14))[1].spec
            self.assertEqual(
                (s.prefetch_rows, s.lds_only_sync, s.waves_m, s.lds_pad),
                (0, False, 1, 0),
            )
            self.assertEqual((s.stage_out, s.xcd_tiles), (False, False))
            self.assertNotIn("_pf2", s.kernel_name())
        finally:
            gc._DIRECT_DGRAD_STREAM_KNOBS = saved

    def test_generic_row_stream_box_edges(self):
        import dispatch.grouped_convolution as gc

        self.assertEqual(gc._DIRECT_DGRAD_STREAM_UNTILED_MAX_WO, 64)
        self.assertFalse(hasattr(gc, "_DIRECT_DGRAD_STREAM_UNTILED_CPG16_MAX_WO"))
        self.assertEqual(gc._DIRECT_DGRAD_STREAM_TILED_1X1_MIN_CPG, 12)
        # (N, cpg, H, W, G, Y) -> (block_h, stack taken)
        cases = {
            # Untiled, non-power-of-two groups: up to 64 output columns.
            "cpg20_untiled_W64": ((32, 20, 16, 64, 48, 1), (0, True)),
            "cpg20_untiled_W80": ((32, 20, 16, 80, 48, 1), (0, False)),
            "cpg28_untiled_3x3_W64": ((32, 28, 8, 64, 48, 3), (0, True)),
            "cpg28_untiled_3x3_W96": ((32, 28, 8, 96, 48, 3), (0, False)),
            "cpg12_untiled_W64": ((32, 12, 32, 64, 48, 1), (0, True)),
            "cpg12_untiled_W256": ((32, 12, 32, 256, 48, 1), (0, False)),
            # 8-channel groups: same cap.
            "cpg8_untiled_W64": ((32, 8, 32, 64, 48, 1), (0, True)),
            "cpg8_untiled_3x3_W80": ((32, 8, 32, 80, 48, 3), (0, False)),
            "cpg8_untiled_W192": ((32, 8, 32, 192, 48, 1), (0, False)),
            # Untiled 16-channel groups: the same 64-column cap (wider ones lost
            # at 128 columns without the LDS pad and on 1-row 1x1 images).
            "cpg16_untiled_W64": ((32, 16, 32, 64, 48, 1), (0, True)),
            "cpg16_untiled_W65": ((32, 16, 32, 65, 48, 1), (0, False)),
            "cpg16_untiled_3x3_W128": ((16, 16, 16, 128, 256, 3), (0, False)),
            "cpg16_untiled_1row_W176": ((128, 16, 1, 176, 192, 1), (0, False)),
            "cpg16_untiled_W192": ((32, 16, 32, 192, 48, 1), (0, False)),
            # H-tiled 1x1: from 12 channels per group; 3x3 any group.
            "cpg8_tiled_1x1": ((32, 8, 14, 64, 16, 1), (8, False)),
            "cpg12_tiled_1x1": ((4, 12, 32, 64, 16, 1), (4, True)),
            "cpg8_tiled_3x3": ((4, 8, 32, 64, 16, 3), (4, True)),
        }
        for label, ((n, cpg, h, w, g, y), (want_bh, want_stack)) in cases.items():
            with self.subTest(case=label):
                req = self._grouped(
                    N=n, C=cpg * g, K=cpg * g, Hi=h, Wi=w, G=g, Y=y, X=y,
                    pad_h=y // 2, pad_w=y // 2,
                )  # fmt: skip
                name, r = self._pick(req)
                self.assertEqual(name, self._DIRECT, label)
                s = r.spec
                self.assertEqual((s.variant, s.fused_weights), ("generic", True))
                self.assertEqual((s.block_q, s.block_h), (16, want_bh), label)
                p = gc._direct_dgrad_problem(req)
                self.assertEqual(
                    gc._direct_dgrad_stream_box_admits(s, p), want_stack, label
                )
                stack = (s.prefetch_rows, s.lds_only_sync, s.xcd_tiles)
                self.assertEqual(
                    stack, (2, True, True) if want_stack else (0, False, False), label
                )
                if not want_stack:
                    self.assertEqual((s.lds_pad, s.stage_out, s.waves_m), (0, False, 1))
                    # Outside the box: exactly the spec without the stack.
                    saved = gc._DIRECT_DGRAD_STREAM_KNOBS
                    gc._DIRECT_DGRAD_STREAM_KNOBS = False
                    try:
                        self.assertEqual(self._pick(req)[1].spec, s, label)
                    finally:
                        gc._DIRECT_DGRAD_STREAM_KNOBS = saved

    def test_vec_size_c_is_ignored_and_reported(self):
        r = dispatch_conv_grouped(self._grouped(vec_size_c=8))
        self.assertEqual(r.candidate.name, self._DIRECT)
        self.assertTrue(any("vec_size_c=8 ignored" in e for e in r.explanation))

    def test_other_arches_never_see_direct(self):
        direct = next(
            c for c in conv_grouped_candidates("dgrad") if c.name == self._DIRECT
        )
        for arch in ("gfx942", "gfx1250"):
            ok, why = direct.admits(self._grouped(arch=arch))
            self.assertFalse(ok)
            self.assertIn("capability", why)

    # ---- spatial policy -----------------------------------------------------

    def test_block_h_policy(self):
        # Small grid (few wave columns) -> tile H; tiny H -> never tile;
        # large filter -> tile; very tall image -> tile even when the grid is big.
        def bh(**kw):
            return self._pick(self._grouped(**kw))[1].spec.block_h

        # wave columns = ceil(W/16) * G * N; target 3072 (2048 when H <= 16).
        self.assertEqual(bh(N=1, C=128, K=128, Hi=56, Wi=56), 4)  # 128 * 7 rows
        self.assertEqual(bh(N=4, C=128, K=128, Hi=56, Wi=56), 8)  # 512 * 7
        self.assertEqual(bh(N=128, C=128, K=128, Hi=56, Wi=56), 0)  # 16384
        self.assertEqual(bh(N=32, C=512, K=512, Hi=14, Wi=14), 8)  # 1024 * 2
        self.assertEqual(bh(N=16, C=512, K=512, Hi=14, Wi=14), 4)  # 512 * 2
        self.assertEqual(bh(N=64, C=512, K=512, Hi=14, Wi=14), 0)  # 2048
        self.assertEqual(bh(N=128, Hi=7, Wi=7), 0)
        self.assertEqual(bh(N=128, Hi=8, Wi=8), 0)
        self.assertEqual(bh(N=128, Hi=14, Wi=14, Y=5, X=5, pad_h=2, pad_w=2), 4)
        # (cpg 8: the cpg = kpg = 4 shapes above 300 workgroups take the 4c
        # row, which streams whole columns.)
        self.assertEqual(bh(N=128, C=256, K=256, Hi=96, Wi=96), 8)

    def test_short_image_block_h_policy(self):
        # H <= 16: whole image unless tiling buys enough extra waves to pay
        # for the near-empty last tile and the re-loaded halo rows.
        # (N, C, K, H, W, G) -> block_h
        cases = {
            "10x10_G8_cpg32_kpg8": ((128, 256, 64, 10, 10, 8), 0),
            "9x9_G4_cpg32": ((200, 128, 128, 9, 9, 4), 0),
            "9x9_G8_cpg16": ((200, 128, 128, 9, 9, 8), 0),
            "10x10_G4_cpg32_N256": ((256, 128, 128, 10, 10, 4), 0),
            "9x9_G4_cpg32_kpg16": ((200, 128, 64, 9, 9, 4), 0),
            "12x12_G8_cpg32_N128": ((128, 256, 256, 12, 12, 8), 0),
            "16x16_G4_cpg32_kpg8": ((128, 128, 32, 16, 16, 4), 8),
            "14x14_G2_cpg32_N128": ((128, 64, 64, 14, 14, 2), 4),
            "14x14_G4_cpg16_kpg8": ((128, 64, 32, 14, 14, 4), 4),
            "16x16_G2_cpg32_N128": ((128, 64, 64, 16, 16, 2), 4),
        }
        for label, ((n, c, k, h, w, g), want) in cases.items():
            with self.subTest(case=label):
                name, r = self._pick(self._grouped(N=n, C=c, K=k, Hi=h, Wi=w, G=g))
                self.assertEqual(name, self._DIRECT, label)
                self.assertEqual(r.spec.block_h, want, label)

    def test_block_q_policy(self):
        # 32-wide strips need H tiling, no extra W padding, enough channels
        # per wave (>= 16, or >= 8 with a 5x5/7x7) and >= 768 waves after
        # halving the strip count.
        def bq(**kw):
            return self._pick(self._grouped(**kw))[1].spec.block_q

        self.assertEqual(bq(N=16, C=1024, K=128, Hi=28, Wi=28), 32)  # cpg 32
        self.assertEqual(bq(N=128, C=512, K=512, Hi=14, Wi=14), 16)  # untiled
        self.assertEqual(bq(N=8, C=256, K=256, Hi=28, Wi=28), 16)  # cpg 8
        self.assertEqual(
            bq(N=8, C=256, K=256, Hi=28, Wi=28, Y=7, X=7, pad_h=3, pad_w=3), 32
        )
        self.assertEqual(
            bq(N=32, C=512, K=512, Hi=40, Wi=40), 16
        )  # pads 40 to 64, not 48
        # G=3: one 32-wide strip per row tile leaves 3 * 32 * 7 = 672 waves.
        self.assertEqual(bq(N=32, C=72, K=72, Hi=28, Wi=28, G=3), 16)

    def test_big_filter_halves_block_groups(self):
        def bg(**kw):
            return self._pick(self._grouped(**kw))[1].spec.block_groups

        big = {"Y": 5, "X": 5, "pad_h": 2, "pad_w": 2}
        self.assertEqual(bg(N=32, C=256, K=256, Hi=28, Wi=28), 4)  # cpg 8
        self.assertEqual(bg(N=32, C=256, K=256, Hi=28, Wi=28, **big), 2)
        self.assertEqual(bg(N=32, C=512, K=512, Hi=28, Wi=28, **big), 1)  # cpg 16

    def test_block_groups_divides_groups(self):
        # kpg <= 8 wants block_groups=4; G=2 forces it down to 2.
        # N=128 keeps both inside the admitted region.
        _name, r = self._pick(self._grouped(N=128, C=8, K=8, G=2))
        self.assertEqual(r.spec.block_groups, 2)
        _name, r = self._pick(self._grouped(N=128, C=24, K=24, G=3))
        self.assertEqual(r.spec.block_groups, 1)

    # ---- launch plan / grid contract ---------------------------------------

    def test_grid_and_block_match_the_launch_plan(self):
        for kw in (
            {},
            {"N": 1, "Hi": 56, "Wi": 56},
            {"N": 32, "C": 512, "K": 512, "G": 16},
        ):
            with self.subTest(kw=kw):
                req = self._grouped(**kw)
                _name, r = self._pick(req)
                plan = r.spec.launch_plan(req)
                self.assertEqual(r.grid, plan.main.grid)
                self.assertEqual(r.block, plan.main.block)
                p = _problem(req)
                # Independent re-derivation of the main grid.
                h_tiles = math.ceil(p.Hi / r.spec.block_h) if r.spec.block_h else 1
                self.assertEqual(
                    r.grid,
                    (
                        math.ceil(p.Wo / r.spec.block_q),
                        p.groups // r.spec.block_groups,
                        p.N * h_tiles,
                    ),
                )
                self.assertEqual(
                    r.block, (r.spec.block_groups * r.spec.waves_m * 64, 1, 1)
                )

    def test_plan_fused_path_is_one_kernel_without_workspace(self):
        req = self._grouped()
        r = dispatch_conv_grouped(req)
        plan = r.spec.launch_plan(req)
        self.assertEqual([s.role for s in plan.stages], ["main"])
        self.assertEqual(plan.workspace_bytes, 0)
        main = plan.main
        self.assertEqual((main.a, main.b, main.d), ("dY", "W", "dX"))
        # The main spec is the transposed problem: channels swapped.
        p = _problem(req)
        fp = main.spec.problem
        self.assertEqual((fp.cpg, fp.kpg), (p.kpg, p.cpg))
        self.assertTrue(main.spec.dgrad_fused_weights)
        self.assertTrue(any("pipeline=main" in e for e in r.explanation))

    def test_plan_falls_back_to_the_pre_pass_over_the_register_budget(self):
        # 7x7 with cpg = kpg = 32: the fused weight fragments exceed their
        # register budget, so the transpose pre-pass pipeline runs instead.
        req = self._req_from((16, 256, 256, 28, 28, 7, 8, "fp16"))
        r = dispatch_conv_grouped(req)
        self.assertEqual(r.candidate.name, self._DIRECT)
        self.assertFalse(r.spec.fused_weights)
        plan = r.spec.launch_plan(req)
        self.assertEqual([s.role for s in plan.stages], ["transpose", "main"])
        p = _problem(req)
        self.assertEqual(plan.workspace_bytes, p.C * p.Y * p.X * p.kpg * 2)
        self.assertEqual(plan.main.b, "ws_wt")
        self.assertTrue(any("pipeline=transpose+main" in e for e in r.explanation))

    def test_kernel_name_separates_knobs(self):
        from dataclasses import replace

        spec = dispatch_conv_grouped(self._grouped()).spec
        names = {
            spec.kernel_name(),
            replace(spec, block_h=spec.block_h + 8).kernel_name(),
            replace(spec, block_groups=1).kernel_name(),
            replace(spec, fold_k32=True).kernel_name(),
            replace(spec, runtime_k_loop=True).kernel_name(),
            replace(spec, fused_weights=False, weights_lds=False).kernel_name(),
            replace(spec, weights_lds=False).kernel_name(),
            replace(spec, waves_per_eu=spec.waves_per_eu + 2).kernel_name(),
            replace(spec, variant="4c").kernel_name(),
            replace(spec, prefetch_rows=3).kernel_name(),
            replace(spec, lds_only_sync=not spec.lds_only_sync).kernel_name(),
            replace(spec, waves_m=spec.waves_m + 1).kernel_name(),
            replace(spec, lds_pad=spec.lds_pad + 8).kernel_name(),
            replace(spec, stage_out=not spec.stage_out).kernel_name(),
            replace(spec, xcd_tiles=not spec.xcd_tiles).kernel_name(),
        }
        self.assertEqual(len(names), 15)

    def test_rule_table_has_catch_all_last(self):
        from dispatch.grouped_convolution import GFX950_DIRECT_DGRAD_RULES

        self.assertTrue(GFX950_DIRECT_DGRAD_RULES[-1].applies(4, 28))
        for rule in GFX950_DIRECT_DGRAD_RULES:
            self.assertIn(rule.block_groups, (1, 2, 4, 8, 16))


class TestGfx1250WgradKOuterReachable(unittest.TestCase):
    """The gfx1250 wgrad candidate must actually enable the K-outer layout.

    ``WgradConvSpec.default_lds_k_outer`` returns True for every fp16/bf16
    gfx1250 wgrad request (wave32, 16x16 atom edge), but the candidate used to
    never ask -- so the headline transpose-read path was unreachable through
    library dispatch and was exercised only by the sweep driver and the
    direct-build tests.
    """

    def _spec(self, dtype="fp16"):
        return dispatch_conv_grouped(_wgrad("gfx1250", G=4, dtype=dtype)).spec

    def test_dispatch_spec_enables_k_outer(self):
        for dtype in ("fp16", "bf16"):
            self.assertTrue(
                self._spec(dtype).lds_k_outer,
                f"gfx1250 wgrad dispatch must enable lds_k_outer for {dtype}",
            )

    def test_decision_survives_into_the_instance_spec(self):
        r = dispatch_conv_grouped(_wgrad("gfx1250", G=4))
        inst = r.spec.to_wgrad_spec(_problem(r.request))
        self.assertTrue(inst.lds_k_outer)
        inst.validate()

    def test_agrees_with_the_selection_policy(self):
        # Dispatch must not hand-roll the gate; it must match the one policy
        # function the sweep driver also calls.
        from rocke.core.arch import ArchTarget
        from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

        spec = self._spec()
        self.assertEqual(
            spec.lds_k_outer,
            WgradConvSpec.default_lds_k_outer(
                arch="gfx1250",
                dtype_a="fp16",
                dtype_b="fp16",
                warp_tile_m=spec.warp_tile_mn,
                warp_tile_n=spec.warp_tile_mn,
                wave_size=ArchTarget.from_gfx("gfx1250").wave_size,
            ),
        )


class TestGroupedSpecKernelNameDistinguishesBody(unittest.TestCase):
    """Dispatch kernel names must separate specs that emit different bodies.

    This is the layer whose names key the host-side compile cache, so two specs
    that lower differently sharing one name is a cache-collision bug, not a
    cosmetic one.
    """

    def test_k_outer_changes_the_name(self):
        from dispatch.grouped_convolution import ConvGroupedSpec

        base = dispatch_conv_grouped(_wgrad("gfx950", G=4)).spec
        from dataclasses import replace

        on = replace(base, lds_k_outer=True)
        off = replace(base, lds_k_outer=False)
        self.assertNotEqual(
            on.kernel_name(),
            off.kernel_name(),
            "lds_k_outer changes the LDS tile shape and operand fetch",
        )
        self.assertIn("kouter", on.kernel_name())
        assert isinstance(base, ConvGroupedSpec)

    def test_ws_replicas_changes_the_instance_name(self):
        from dataclasses import replace as _replace

        r = dispatch_conv_grouped(_wgrad("gfx950", C=3, K=24, Y=3, X=3, dtype="bf16"))
        ws = r.spec.to_wgrad_spec(_problem(r.request))
        self.assertNotEqual(
            ws.kernel_name(),
            _replace(ws, ws_replicas=ws.ws_replicas + 1).kernel_name(),
            "ws_replicas changes the scratch addressing, so it must reach the "
            "name the compile cache keys on",
        )


if __name__ == "__main__":
    unittest.main()
