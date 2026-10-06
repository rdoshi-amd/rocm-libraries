# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dispatch of depthwise (cpg = kpg = 1) dgrad to the windowed direct kernel.

Host-only: which candidate wins, which requests fall back, and the invariants
of the knob heuristic (block_w, waves, channel packing, dot2, H split), and
of the Toeplitz MFMA form: its admission box (7x7 'same' filters, C a
multiple of 64, 7-16 rows, 7-8 / 13-16 / 19-112 columns, size caps), pinned
at its edges, and its knob rule (w_fold, waves, H split).
"""

from __future__ import annotations

import math
import unittest

from dispatch.grouped_convolution import (
    ConvDepthwiseDgradSpec,
    ConvGroupedRequest,
    _dw_dgrad_mfma_admits,
    _dw_dgrad_mfma_spec,
    _dw_dgrad_valu_spec,
    conv_grouped_candidates,
    dispatch_conv_grouped,
)
from kernels.common.conv_direct_grouped import (
    DW_DGRAD_MFMA_CH,
    DW_DGRAD_MFMA_MAX_UNROLL,
    DirectDepthwiseDgradWindowedSpec,
    is_valid_depthwise_dgrad_win_spec,
)

_DW = "direct_depthwise_dgrad_win"
_IGEMM = "implicit_gemm_conv_dgrad"


def _dw(C=512, Hi=14, Wi=14, Y=7, X=7, pad=3, N=128, dtype="bf16", **kw):
    base = {
        "N": N,
        "C": C,
        "K": C,
        "G": C,
        "Hi": Hi,
        "Wi": Wi,
        "Y": Y,
        "X": X,
        "pad_h": pad,
        "pad_w": pad,
        "dtype": dtype,
        "arch": "gfx950",
        "direction": "dgrad",
    }
    base.update(kw)
    return ConvGroupedRequest(**base)


# Depthwise layers of common mobile / ConvNeXt-style networks plus odd sizes.
_COHORT = (
    _dw(),
    _dw(Hi=16, Wi=16),
    _dw(C=1536, Hi=12, Wi=12, Y=3, X=3, pad=1, dtype="fp16"),
    _dw(N=32, C=128, Hi=56, Wi=56),
    _dw(N=32, C=256, Hi=28, Wi=28),
    _dw(C=768, Hi=7, Wi=7),
    _dw(N=64, C=240, Hi=28, Wi=28, Y=5, X=5, pad=2, dtype="fp16"),
    _dw(N=64, C=144, Hi=56, Wi=56, Y=3, X=3, pad=1, dtype="fp16"),
    _dw(N=32, C=32, Hi=112, Wi=112, Y=3, X=3, pad=1, dtype="fp16"),
    _dw(N=1, C=66, Hi=13, Wi=17, Y=5, X=5, pad=2),
    _dw(N=3, C=130, Hi=15, Wi=7, Y=3, X=3, pad=0, dtype="fp16"),
)


class TestDepthwiseDgradDispatch(unittest.TestCase):
    def test_depthwise_requests_take_the_windowed_kernel(self):
        for req in _COHORT:
            with self.subTest(req=req):
                res = dispatch_conv_grouped(req)
                self.assertEqual(res.candidate.name, _DW)
                self.assertIsInstance(res.spec, ConvDepthwiseDgradSpec)
                inst = res.spec.instance
                self.assertEqual(res.grid, inst.grid())
                self.assertEqual(res.block, (inst.threads_per_block, 1, 1))
                self.assertTrue(is_valid_depthwise_dgrad_win_spec(inst, req.arch)[0])
                self.assertEqual(res.spec.kernel_name(), inst.kernel_name())
                # A non-sentinel field round-trips into the instance problem.
                self.assertEqual(inst.problem.PAD, req.pad_h)
                self.assertEqual(inst.problem.dtype, req.dtype)

    def test_target_shape_picks(self):
        s6 = dispatch_conv_grouped(_dw()).spec.instance
        self.assertEqual(
            (s6.mfma, s6.w_fold, s6.block_waves, s6.h_tiles, s6.prefetch_rows),
            (True, 2, 8, 1, 2),
        )
        s8 = dispatch_conv_grouped(_dw(Hi=16, Wi=16)).spec.instance
        self.assertEqual((s8.mfma, s8.w_fold, s8.h_tiles), (True, 2, 1))
        s10 = dispatch_conv_grouped(_COHORT[2]).spec.instance
        self.assertEqual(
            (s10.mfma, s10.block_w, s10.ch_per_lane, s10.dot2, s10.h_tiles),
            (False, 12, 2, False, 1),
        )

    def test_mfma_admission_box_edges(self):
        # Inside the box: the MFMA form. Corners and edges of every axis.
        inside = (
            {},  # S6
            {"Hi": 16, "Wi": 16},  # S8
            {"N": 1, "C": 64, "Hi": 7, "Wi": 7},
            {"N": 256, "C": 64, "Hi": 7, "Wi": 8},
            {"N": 256, "C": 2048, "Hi": 7, "Wi": 13},
            {"N": 256, "C": 2048, "Hi": 16, "Wi": 16},
            {"N": 1, "C": 2048, "Hi": 16, "Wi": 19},
            {"N": 64, "C": 1024, "Hi": 12, "Wi": 27, "dtype": "fp16"},
            {"N": 32, "C": 192, "Hi": 9, "Wi": 56},
            {"N": 4, "C": 64, "Hi": 16, "Wi": 112},
            # dY exactly at the 512 MiB cap.
            {"N": 256, "C": 2048, "Hi": 16, "Wi": 32},
        )
        for kw in inside:
            req = _dw(**kw)
            with self.subTest(kw=kw):
                self.assertTrue(_dw_dgrad_mfma_admits(req))
                inst = dispatch_conv_grouped(req).spec.instance
                self.assertTrue(inst.mfma)
                self.assertEqual(inst, _dw_dgrad_mfma_spec(req))
        # Just outside: the VALU pick, unchanged.
        outside = (
            {"Hi": 14, "Wi": 6},
            {"Hi": 14, "Wi": 9},
            {"Hi": 14, "Wi": 12},
            {"Hi": 14, "Wi": 17},
            {"Hi": 14, "Wi": 18},
            {"N": 4, "C": 64, "Hi": 16, "Wi": 113},
            {"Hi": 6, "Wi": 14},
            {"Hi": 17, "Wi": 14},
            {"N": 8, "Hi": 56, "Wi": 56},
            {"N": 257, "C": 64},
            {"C": 2112},
            {"C": 96},
            {"C": 1000},
            # dY just over the 512 MiB cap.
            {"N": 256, "C": 2048, "Hi": 16, "Wi": 33},
            {"Y": 5, "X": 5, "pad": 2},
            {"Y": 3, "X": 3, "pad": 1},
            {"Y": 9, "X": 9, "pad": 4},
            {"Y": 7, "X": 5, "pad": 3},
            {"Y": 5, "X": 7, "pad": 2},
            {"Y": 7, "X": 9, "pad": 3},
            {"pad": 2},
            {"arch": "gfx942"},
        )
        for kw in outside:
            req = _dw(**kw)
            with self.subTest(kw=kw):
                self.assertFalse(_dw_dgrad_mfma_admits(req))
                if req.arch != "gfx950" or req.pad_h != req.Y // 2:
                    continue  # rejected by, or valid only off, the gfx950 gate
                inst = dispatch_conv_grouped(req).spec.instance
                self.assertFalse(inst.mfma)
                self.assertEqual(inst, _dw_dgrad_valu_spec(req))

    def test_mfma_knob_rule(self):
        # The knob rule alone (_dw_dgrad_mfma_spec), on shapes in and out of
        # the admission box.
        def pick(**kw):
            return _dw_dgrad_mfma_spec(_dw(**kw))

        # w_fold: 4 images per tile up to 8 columns, 2 up to 16, 1 up to 32.
        for W, fold in ((5, 4), (7, 4), (8, 4), (9, 2), (14, 2), (16, 2), (17, 1)):
            self.assertEqual(pick(Hi=W, Wi=W).w_fold, fold, W)
        for W, fold in ((24, 1), (32, 1)):
            self.assertEqual(pick(N=8, Hi=W, Wi=W).w_fold, fold, W)
        # Wider images: fewest padded tile columns plus the tile_w + KW - 1
        # column window every real image reads per column tile, so W just
        # past 32 avoids a half-empty second 32-column tile.
        for W, X, fold in (
            (33, 7, 4),
            (36, 7, 4),
            (40, 7, 4),
            (44, 7, 2),
            (48, 7, 2),
            (56, 7, 1),
            (64, 7, 1),
            (80, 7, 2),
            (96, 7, 1),
            (112, 7, 2),
            (36, 9, 2),
            (40, 9, 2),
            (40, 11, 2),
            (56, 9, 1),
        ):
            with self.subTest(W=W, X=X):
                inst = pick(N=4, C=256, Hi=W, Wi=W, Y=X, X=X, pad=X // 2)
                self.assertTrue(inst.mfma)
                self.assertEqual(inst.w_fold, fold)
        # Empty image slots count too: one image keeps one image per tile,
        # two take 16-column tiles, many take the 8-column fit.
        for N, fold in ((1, 1), (2, 2), (6, 2), (16, 4)):
            with self.subTest(N=N):
                inst = pick(N=N, C=1024, Hi=40, Wi=40)
                self.assertEqual(inst.w_fold, fold)
        # Large grids keep 8 waves and the whole image up to 16 rows.
        big = pick(N=256, C=1024, Hi=16, Wi=16)
        self.assertEqual((big.block_waves, big.h_tiles), (8, 1))
        # Tall images start from balanced chunks of about 14 rows.
        tall = pick(N=128, C=256, Hi=56, Wi=56)
        self.assertEqual((tall.block_waves, tall.rows_per_block), (8, 14))
        self.assertEqual(pick(N=128, C=256, Hi=20, Wi=20).rows_per_block, 10)
        # Small grids drop to 4 waves first, then to smaller chunks.
        small = pick(N=2, C=256, Hi=14, Wi=14)
        self.assertEqual((small.block_waves, small.rows_per_block), (4, 4))
        # Fewer than 64 channels: one wave per 8 channels.
        self.assertEqual(pick(N=64, C=24, Hi=14, Wi=14).block_waves, 3)
        # Filters whose fragments leave room for one 8-wave block per CU
        # take 4-wave blocks where the 8-wave grid ends in a partial round
        # over the 256 CUs, on large grids too; the others keep 8 waves.
        for Y, X, waves in (
            (7, 7, 8),
            (5, 9, 8),
            (8, 7, 8),
            (7, 11, 4),
            (9, 9, 4),
            (11, 11, 4),
        ):
            with self.subTest(Y=Y, X=X):
                inst = pick(N=24, C=512, Hi=36, Wi=36, Y=Y, X=X, pad=X // 2)
                self.assertTrue(inst.mfma)
                self.assertEqual(inst.block_waves, waves)
        for N, waves in ((32, 8), (48, 4), (64, 8), (72, 4)):
            with self.subTest(N=N):
                # 8-wave grids of 256, 384, 512 and 576 blocks.
                inst = pick(N=N, C=1024, Hi=16, Wi=16, Y=9, X=9, pad=4)
                self.assertEqual(inst.block_waves, waves)

    def test_heuristic_invariants(self):
        for req in _COHORT:
            inst: DirectDepthwiseDgradWindowedSpec = dispatch_conv_grouped(
                req
            ).spec.instance
            with self.subTest(req=req):
                self.assertEqual(inst.mfma, _dw_dgrad_mfma_admits(req))
                if inst.mfma:
                    self.assertEqual((req.Y, req.X), (7, 7))
                    self.assertEqual(req.C % DW_DGRAD_MFMA_CH, 0)
                    self.assertLessEqual(inst.block_waves, 8)
                    self.assertLessEqual(
                        inst.block_waves * DW_DGRAD_MFMA_CH,
                        req.C + DW_DGRAD_MFMA_CH - 1,
                    )
                    self.assertLessEqual(
                        inst.unrolled_mfmas(), DW_DGRAD_MFMA_MAX_UNROLL
                    )
                    if req.Hi <= 16 and inst.h_tiles > 1:
                        # Only split to fill a small grid.
                        gx, gy, gz = inst.grid()
                        self.assertLess(gx * gy * gz // inst.h_tiles, 256)
                    continue
                W = req.Wi
                if W <= 16:
                    self.assertEqual(inst.block_w, W)
                else:
                    # Tiles of about 8 columns; never more tiles than needed.
                    self.assertLessEqual(inst.block_w, 8)
                    tiles = math.ceil(W / inst.block_w)
                    self.assertEqual(tiles, math.ceil(W / 8))
                self.assertEqual(inst.dot2, req.X >= 5)
                self.assertEqual(inst, _dw_dgrad_valu_spec(req))
                if inst.dot2:
                    self.assertEqual(inst.ch_per_lane, 1)
                self.assertLessEqual(inst.block_waves, 4)
                # Never more lanes than channels by a whole wave.
                self.assertLess(
                    inst.block_waves * 64 * inst.ch_per_lane,
                    req.C + 64 * inst.ch_per_lane,
                )
                if inst.h_tiles > 1:
                    self.assertGreaterEqual(inst.rows_per_block, 2 * req.Y)
                self.assertLessEqual(inst.unrolled_fmas(), 1 << 14)

    def test_block_w_divides_w_when_possible(self):
        for W, bw in ((28, 7), (56, 8), (112, 8), (14, 14), (12, 12), (7, 7)):
            inst = dispatch_conv_grouped(
                _dw(N=8, C=64, Hi=8, Wi=W, Y=3, X=3, pad=1)
            ).spec.instance
            self.assertEqual(inst.block_w, bw, W)
            self.assertEqual(W % inst.block_w, 0, W)

    def test_small_grid_splits_h(self):
        inst = dispatch_conv_grouped(_dw(N=32, C=128, Hi=56, Wi=56)).spec.instance
        self.assertGreater(inst.h_tiles, 1)
        inst = dispatch_conv_grouped(_dw()).spec.instance
        self.assertEqual(inst.h_tiles, 1)

    def test_large_filter_balances_rows_and_block_w(self):
        # Past one filter height of rows the unroll budget shrinks the larger
        # of rows and block_w, so neither collapses to 1 while the other is
        # still wide (each block would re-read a whole filter-height halo).
        for req in (
            _dw(N=1, C=128, Hi=64, Wi=64, Y=31, X=31, pad=15, dtype="fp16"),
            _dw(N=1, C=64, Hi=64, Wi=64, Y=33, X=33, pad=16, dtype="fp16"),
            _dw(N=32, C=512, Hi=28, Wi=28, Y=31, X=31, pad=15),
        ):
            inst = dispatch_conv_grouped(req).spec.instance
            with self.subTest(req=req):
                self.assertFalse(inst.mfma)
                self.assertLessEqual(inst.unrolled_fmas(), 1 << 14)
                self.assertGreater(inst.rows_per_block, 1)
                bw, rows = inst.block_w, inst.rows_per_block
                self.assertLessEqual(max(bw, rows), 2 * min(bw, rows))
                self.assertTrue(is_valid_depthwise_dgrad_win_spec(inst, req.arch)[0])

    def test_fallback_and_rejections(self):
        # Requests the windowed kernel declines: no other candidate covers
        # depthwise dgrad, so these raise with the windowed reason in the text.
        for kw, needle in (
            ({"stride_h": 2, "stride_w": 2}, "stride 1"),
            ({"dilation_h": 2, "dilation_w": 2}, "dilation"),
            ({"pad_w": 2}, "pad_h == pad_w"),
            ({"arch": "gfx942"}, "gfx942"),
            ({"N": 4096, "Hi": 112, "Wi": 112, "C": 64}, "sentinel"),
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError) as cm:
                    dispatch_conv_grouped(_dw(**kw))
                self.assertIn(needle, str(cm.exception))
        # Grouped (cpg > 1) dgrad is not this candidate's (the grouped direct
        # or the igemm candidate takes it, depending on the shape).
        res = dispatch_conv_grouped(_dw(C=512, G=128, K=512))
        self.assertNotEqual(res.candidate.name, _DW)
        cand = next(c for c in conv_grouped_candidates("dgrad") if c.name == _DW)
        self.assertFalse(cand.admits(_dw(C=512, G=128, K=512))[0])

    def test_mfma_grid_past_launch_limit_keeps_valu_form(self):
        # Very large batches of 8 channels: the fold-1 MFMA grid would pass
        # the 65535 y/z launch limit, so the knob rule declines (outside the
        # admission box anyway) and dispatch keeps the VALU form (whose grid
        # fits) instead of rejecting the request.
        for req in (
            _dw(N=16384, C=8, Hi=56, Wi=56, dtype="fp16"),
            _dw(N=32768, C=8, Hi=17, Wi=17),
        ):
            with self.subTest(N=req.N, Hi=req.Hi):
                self.assertIsNone(_dw_dgrad_mfma_spec(req))
                res = dispatch_conv_grouped(req)
                self.assertEqual(res.candidate.name, _DW)
                inst = res.spec.instance
                self.assertFalse(inst.mfma)
                self.assertLessEqual(max(res.grid[1:]), 65535)

    def test_candidate_rejects_other_directions(self):
        cand = next(c for c in conv_grouped_candidates("dgrad") if c.name == _DW)
        for direction in ("fwd", "wgrad"):
            ok, why = cand.admits(_dw(direction=direction))
            self.assertFalse(ok)
            self.assertIn("dgrad", why)

    def test_candidate_build_matches_selection(self):
        req = _dw(N=2, C=64, Hi=9, Wi=13, Y=5, X=5, pad=2)
        res = dispatch_conv_grouped(req)
        kernel = res.candidate.built(res.spec, req.arch)
        self.assertEqual(kernel.name, res.spec.kernel_name())


if __name__ == "__main__":
    unittest.main()
