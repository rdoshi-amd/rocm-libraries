// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CoexecPatterns.hpp"

#include <algorithm>
#include <cstdlib>

#include "stinkytofu/transforms/asm/CoexecSimRepairPass.hpp"

namespace stinkytofu {
namespace coexec {

MoveSpace::MoveSpace(Inputs inputs) : in_(std::move(inputs)) {}

bool MoveSpace::dependsOn(int later, int earlier) const {
    std::vector<char> seen(size(), 0);
    std::vector<int> pending{earlier};
    seen[earlier] = 1;
    while (!pending.empty()) {
        const int id = pending.back();
        pending.pop_back();
        for (int s : in_.succs[id]) {
            if (s == later) return true;
            if (!seen[s]) {
                seen[s] = 1;
                pending.push_back(s);
            }
        }
    }
    return false;
}

std::vector<std::pair<int, int>> MoveSpace::legalSlots(const std::vector<int>& order,
                                                       const std::vector<int>& pos,
                                                       const std::vector<int>& unit) const {
    std::vector<std::pair<int, int>> slots;
    const int seg = in_.segment[unit.front()];
    if (seg < 0) return slots;
    auto inUnit = [&](int id) { return std::find(unit.begin(), unit.end(), id) != unit.end(); };

    // The unit must land after its latest dependence and before its earliest dependent.
    int lo = -1, hi = static_cast<int>(order.size());
    for (int u : unit) {
        for (int p : in_.preds[u])
            if (!inUnit(p)) lo = std::max(lo, pos[p]);
        for (int s : in_.succs[u])
            if (!inUnit(s)) hi = std::min(hi, pos[s]);
    }

    // A slot is "before order[p]": inside the segment, or at its end when order[p] is the
    // first instruction past it.
    int slot = 0, window = 0;
    bool previousInSegment = false;
    for (int p = 0; p < static_cast<int>(order.size()); ++p) {
        const int id = order[p];
        if (inUnit(id)) continue;
        const bool here = in_.segment[id] == seg;
        if (((here && !in_.noInsertBefore[id]) || (!here && previousInSegment)) && p > lo &&
            p <= hi)
            slots.push_back({slot, window});
        previousInSegment = here;
        if (in_.wmma[id]) ++window;
        ++slot;
    }
    if (previousInSegment) slots.push_back({slot, window});
    return slots;
}

std::vector<int> MoveSpace::apply(const std::vector<int>& order, const Move& move) const {
    std::vector<int> out;
    out.reserve(order.size());
    for (int id : order)
        if (std::find(move.unit.begin(), move.unit.end(), id) == move.unit.end()) out.push_back(id);
    out.insert(out.begin() + move.slot, move.unit.begin(), move.unit.end());
    return out;
}

bool MoveSpace::respects(const std::vector<int>& order) const {
    const int n = size();
    if (static_cast<int>(order.size()) != n) return false;
    std::vector<int> pos(n, -1);
    for (int p = 0; p < n; ++p) {
        const int id = order[p];
        if (id < 0 || id >= n || pos[id] >= 0) return false;
        pos[id] = p;
        // Segments keep their spans, so every position stays in the segment it was in.
        if (in_.segment[id] != in_.segment[p]) return false;
    }
    int lastFixed = -1;
    for (int id : order) {
        if (in_.filler[id]) continue;
        if (id < lastFixed) return false;
        lastFixed = id;
    }
    for (int id = 0; id < n; ++id)
        for (int s : in_.succs[id])
            if (pos[id] >= pos[s]) return false;
    return true;
}

namespace {

enum SlotEnd : unsigned { kFirstSlot = 1, kLastSlot = 2, kBothSlots = 3 };

using Slots = std::vector<std::pair<int, int>>;

// Positions [begin, end) of window w: after v_wmma w-1, through v_wmma w.
std::pair<int, int> windowPositions(const PatternContext& ctx, int w) {
    const SimResult& sim = ctx.sim;
    const int begin = w == 0 ? 0 : sim.windows[w - 1].wmmaPos + 1;
    const int end =
        w < sim.numWmma ? sim.windows[w].wmmaPos + 1 : static_cast<int>(ctx.order.size());
    return {begin, end};
}

void addSlots(const std::vector<int>& unit, const Slots& slots, int window, unsigned end,
              unsigned pattern, std::vector<Move>& out) {
    int first = -1, last = -1;
    for (const auto& [slot, w] : slots) {
        if (w != window) continue;
        if (first < 0) first = slot;
        last = slot;
    }
    if (first < 0) return;
    if (end & kFirstSlot) out.push_back({unit, first, window, pattern});
    if ((end & kLastSlot) && !((end & kFirstSlot) && last == first))
        out.push_back({unit, last, window, pattern});
}

bool holdsScalarReadingValu(const PatternContext& ctx, int w) {
    if (w < 0 || w > ctx.sim.numWmma) return false;
    const auto [begin, end] = windowPositions(ctx, w);
    for (int p = begin; p < end; ++p)
        if (ctx.model.facts(ctx.order[p]).scalarSrcValu) return true;
    return false;
}

bool holdsFiller(const PatternContext& ctx, int w) {
    if (w < 0 || w > ctx.sim.numWmma) return false;
    const auto [begin, end] = windowPositions(ctx, w);
    for (int p = begin; p < end; ++p)
        if (ctx.space.isFiller(ctx.order[p])) return true;
    return false;
}

// The window's own issue time, bank switches and waits included, is more than the pipe
// time plus the lead the wave brings in, and the pipe idles.
bool isOverfull(const PatternContext& ctx, int w) {
    const WindowTiming& window = ctx.sim.windows[w];
    return window.idle > 0 && window.ownCycles > window.room;
}

// P1. Every scalar writer in the shadow of a held VALU moves out of it: one the VALU
// depends on moves earlier, into a window without a scalar-reading VALU; one it does not
// depend on may also move after it. Failing both, the VALU itself moves later.
void proposeScalarShadow(const PatternContext& ctx, int w, std::vector<Move>& out) {
    const auto [begin, end] = windowPositions(ctx, w);
    for (int p = begin; p < end; ++p) {
        const InstTiming& t = ctx.sim.inst[p];
        if (t.idleStall <= 0 || t.stallCause != Cause::ScalarInterlock) continue;
        const int valu = ctx.order[p];
        for (int q = p - 1; q >= 0; --q) {
            if (ctx.sim.inst[q].issue + ctx.timing.saluScalarToValu <= t.reach) break;
            const int writer = ctx.order[q];
            if (!ctx.space.isFiller(writer) || !ctx.model.facts(writer).scalarWrite) continue;
            const std::vector<int>& unit = ctx.space.unitOf(writer);
            const Slots slots = ctx.space.legalSlots(ctx.order, ctx.pos, unit);
            const int from = ctx.sim.inst[q].window;
            for (int k = from - 1; k >= std::max(0, from - 3); --k)
                if (!holdsScalarReadingValu(ctx, k))
                    addSlots(unit, slots, k, kLastSlot, kCoexecScalarShadow, out);
            addSlots(unit, slots, from, kFirstSlot, kCoexecScalarShadow, out);
            if (!ctx.space.dependsOn(valu, writer)) {
                addSlots(unit, slots, w, kLastSlot, kCoexecScalarShadow, out);
                for (int k = w + 1; k <= w + 2; ++k)
                    addSlots(unit, slots, k, kFirstSlot, kCoexecScalarShadow, out);
            }
        }
        if (!ctx.space.isFiller(valu)) continue;
        const std::vector<int>& unit = ctx.space.unitOf(valu);
        const Slots slots = ctx.space.legalSlots(ctx.order, ctx.pos, unit);
        for (int k = w + 1; k <= w + 3; ++k)
            addSlots(unit, slots, k, kBothSlots, kCoexecScalarShadow, out);
    }
}

// P2. Keep the VCC producer early, at the start of its window or in the one before, and
// put its reader late in the next window or one further.
void proposeVccPair(const PatternContext& ctx, int w, std::vector<Move>& out) {
    const auto [begin, end] = windowPositions(ctx, w);
    for (int p = begin; p < end; ++p) {
        const InstTiming& t = ctx.sim.inst[p];
        if (t.idleStall <= 0 || t.stallCause != Cause::VccInterlock) continue;
        const int reader = ctx.order[p];
        for (int q = p - 1; q >= 0; --q) {
            const int producer = ctx.order[q];
            if (!ctx.model.facts(producer).vccWrite) continue;
            if (ctx.space.isFiller(producer)) {
                const std::vector<int>& unit = ctx.space.unitOf(producer);
                const Slots slots = ctx.space.legalSlots(ctx.order, ctx.pos, unit);
                const int from = ctx.sim.inst[q].window;
                addSlots(unit, slots, from, kFirstSlot, kCoexecVccPair, out);
                addSlots(unit, slots, from - 1, kLastSlot, kCoexecVccPair, out);
                addSlots(unit, slots, from - 2, kLastSlot, kCoexecVccPair, out);
            }
            break;
        }
        if (!ctx.space.isFiller(reader)) continue;
        const std::vector<int>& unit = ctx.space.unitOf(reader);
        const Slots slots = ctx.space.legalSlots(ctx.order, ctx.pos, unit);
        addSlots(unit, slots, w + 1, kBothSlots, kCoexecVccPair, out);
        addSlots(unit, slots, w + 2, kLastSlot, kCoexecVccPair, out);
    }
}

// P3. The fillers with the most room move to the nearest windows that have spare time,
// preferring windows the wave enters furthest ahead.
void proposeOverfull(const PatternContext& ctx, int w, std::vector<Move>& out) {
    if (!isOverfull(ctx, w)) return;
    struct Candidate {
        int id;
        int room;  // windows its legal range spans
        Slots slots;
    };
    std::vector<Candidate> candidates;
    std::vector<int> seen;
    const auto [begin, end] = windowPositions(ctx, w);
    for (int p = begin; p < end; ++p) {
        const int id = ctx.order[p];
        if (!ctx.space.isFiller(id)) continue;
        const std::vector<int>& unit = ctx.space.unitOf(id);
        if (std::find(seen.begin(), seen.end(), unit.front()) != seen.end()) continue;
        seen.push_back(unit.front());
        Slots slots = ctx.space.legalSlots(ctx.order, ctx.pos, unit);
        std::vector<int> windows;
        for (const auto& slot : slots)
            if (windows.empty() || windows.back() != slot.second) windows.push_back(slot.second);
        candidates.push_back({unit.front(), static_cast<int>(windows.size()), std::move(slots)});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.room != b.room ? a.room > b.room : a.id < b.id;
    });
    // The window's last fillers shift into the next windows together, and its first ones
    // into the window before: a dependent chain moves only as a whole, and the windows
    // after it pass work on the same way in later rounds.
    std::vector<int> fillers;
    for (int p = begin; p < end; ++p)
        if (ctx.space.isFiller(ctx.order[p])) fillers.push_back(ctx.order[p]);
    auto blockOf = [&](auto first, auto last) {
        std::vector<int> unit;
        for (auto it = first; it != last; ++it)
            for (int member : ctx.space.unitOf(*it))
                if (std::find(unit.begin(), unit.end(), member) == unit.end())
                    unit.push_back(member);
        std::sort(unit.begin(), unit.end(), [&](int a, int b) { return ctx.pos[a] < ctx.pos[b]; });
        return unit;
    };
    const int shifts = std::min<int>(3, static_cast<int>(fillers.size()));
    for (int count = 2; count <= shifts; ++count) {
        const std::vector<int> tail = blockOf(fillers.end() - count, fillers.end());
        const Slots tailSlots = ctx.space.legalSlots(ctx.order, ctx.pos, tail);
        addSlots(tail, tailSlots, w + 1, kBothSlots, kCoexecOverfull, out);
        addSlots(tail, tailSlots, w + 2, kFirstSlot, kCoexecOverfull, out);
        const std::vector<int> head = blockOf(fillers.begin(), fillers.begin() + count);
        addSlots(head, ctx.space.legalSlots(ctx.order, ctx.pos, head), w - 1, kLastSlot,
                 kCoexecOverfull, out);
    }

    if (candidates.size() > 3) candidates.resize(3);
    for (const Candidate& c : candidates) {
        std::vector<int> targets;
        for (const auto& [slot, k] : c.slots) {
            (void)slot;
            if (k == w || std::abs(k - w) > 3 || k >= ctx.sim.numWmma) continue;
            if (ctx.sim.windows[k].lead <= 0) continue;
            if (targets.empty() || targets.back() != k) targets.push_back(k);
        }
        std::stable_sort(targets.begin(), targets.end(), [&](int a, int b) {
            const int da = std::abs(a - w), db = std::abs(b - w);
            if (da != db) return da < db;
            if (ctx.sim.windows[a].leadIn != ctx.sim.windows[b].leadIn)
                return ctx.sim.windows[a].leadIn > ctx.sim.windows[b].leadIn;
            return a < b;
        });
        if (targets.size() > 4) targets.resize(4);
        const std::vector<int>& unit = ctx.space.unitOf(c.id);
        for (int k : targets) addSlots(unit, c.slots, k, kBothSlots, kCoexecOverfull, out);
    }
}

// Any filler of the window or the one before it, by one to three windows.
void proposeFallback(const PatternContext& ctx, int w, std::vector<Move>& out) {
    for (int src : {w, w - 1}) {
        if (src < 0) continue;
        std::vector<int> seen;
        const auto [begin, end] = windowPositions(ctx, src);
        for (int p = begin; p < end; ++p) {
            const int id = ctx.order[p];
            if (!ctx.space.isFiller(id)) continue;
            const std::vector<int>& unit = ctx.space.unitOf(id);
            if (std::find(seen.begin(), seen.end(), unit.front()) != seen.end()) continue;
            seen.push_back(unit.front());
            const Slots slots = ctx.space.legalSlots(ctx.order, ctx.pos, unit);
            for (int d : {1, -1, 2, -2, 3, -3})
                addSlots(unit, slots, src + d, kBothSlots, kCoexecFallback, out);
        }
    }
}

}  // namespace

unsigned explainingPatterns(const PatternContext& ctx, int window, unsigned mask) {
    if (ctx.sim.windows[window].idle <= 0) return 0;
    unsigned found = 0;
    const auto [begin, end] = windowPositions(ctx, window);
    for (int p = begin; p < end; ++p) {
        const InstTiming& t = ctx.sim.inst[p];
        if (t.idleStall > 0 && t.stallCause == Cause::ScalarInterlock) found |= kCoexecScalarShadow;
        if (t.idleStall > 0 && t.stallCause == Cause::VccInterlock) found |= kCoexecVccPair;
    }
    if (holdsFiller(ctx, window) && isOverfull(ctx, window)) found |= kCoexecOverfull;
    if (holdsFiller(ctx, window) || holdsFiller(ctx, window - 1)) found |= kCoexecFallback;
    return found & mask;
}

std::vector<Move> proposeMoves(const PatternContext& ctx, int window, unsigned mask) {
    std::vector<Move> moves;
    const unsigned explained = explainingPatterns(ctx, window, mask);
    if (explained & kCoexecScalarShadow) proposeScalarShadow(ctx, window, moves);
    if (explained & kCoexecVccPair) proposeVccPair(ctx, window, moves);
    if (explained & kCoexecOverfull) proposeOverfull(ctx, window, moves);
    if (explained & kCoexecFallback) proposeFallback(ctx, window, moves);
    return moves;
}

const char* patternName(unsigned pattern) {
    switch (pattern) {
        case kCoexecScalarShadow:
            return "P1 scalar shadow";
        case kCoexecVccPair:
            return "P2 vcc pair";
        case kCoexecOverfull:
            return "P3 overfull window";
        case kCoexecFallback:
            return "fallback";
        default:
            return "?";
    }
}

}  // namespace coexec
}  // namespace stinkytofu
