"""Quad-path tile traversals and the keyed QuadPath scrambling permutation.

The four path families generalise the zig-zag traversal of QUAD-PATH (ICICC 2024):
    0 = zig-zag (JPEG diagonal order), 1 = boustrophedon (snake),
    2 = clockwise inward spiral,       3 = Hilbert curve (square power-of-two tiles only).
Each tile is additionally transformed by one of the 8 elements of the dihedral group D4
(the 90-degree rotations of the 2024 scheme plus reflections).

A scrambling permutation is returned as a gather index G (int64, length H*W): Y.flat = X.flat[G].
"""
import math

import numpy as np
from numba import njit

PHI_INV = (math.sqrt(5.0) - 1.0) / 2.0  # 0.6180339887...


@njit(cache=True)
def _hilbert_d2xy(n, d):
    """Map Hilbert index d to (x, y) on an n x n grid (n a power of two)."""
    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


@njit(cache=True)
def _path(ptype, h, w, oi, oj):
    """Write the visiting order of path family `ptype` on an h x w grid into oi/oj (length h*w)."""
    k = 0
    if ptype == 0:  # zig-zag over anti-diagonals, alternating direction (JPEG order)
        for s in range(h + w - 1):
            if s % 2 == 0:
                i = min(s, h - 1)
                while i >= 0 and s - i < w:
                    oi[k] = i
                    oj[k] = s - i
                    k += 1
                    i -= 1
            else:
                i = max(0, s - w + 1)
                while i <= min(s, h - 1):
                    oi[k] = i
                    oj[k] = s - i
                    k += 1
                    i += 1
    elif ptype == 1:  # boustrophedon
        for i in range(h):
            if i % 2 == 0:
                for j in range(w):
                    oi[k] = i
                    oj[k] = j
                    k += 1
            else:
                for j in range(w - 1, -1, -1):
                    oi[k] = i
                    oj[k] = j
                    k += 1
    elif ptype == 2:  # clockwise spiral from the top-left corner inwards
        top = 0
        bot = h - 1
        lef = 0
        rig = w - 1
        while top <= bot and lef <= rig:
            for j in range(lef, rig + 1):
                oi[k] = top
                oj[k] = j
                k += 1
            for i in range(top + 1, bot + 1):
                oi[k] = i
                oj[k] = rig
                k += 1
            if top < bot:
                for j in range(rig - 1, lef - 1, -1):
                    oi[k] = bot
                    oj[k] = j
                    k += 1
            if lef < rig:
                for i in range(bot - 1, top, -1):
                    oi[k] = i
                    oj[k] = lef
                    k += 1
            top += 1
            bot -= 1
            lef += 1
            rig -= 1
    else:  # Hilbert (caller guarantees h == w == 2^m)
        for d in range(h * w):
            x, y = _hilbert_d2xy(h, d)
            oi[k] = y
            oj[k] = x
            k += 1


@njit(cache=True)
def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


@njit(cache=True)
def tile_order(ptype, g, h, w, oi, oj):
    """Visiting order of tile (h x w) for path family ptype and dihedral element g in {0..7}.

    g bit0 = transpose, bit1 = flip rows, bit2 = flip columns. A transpose is realised by generating the
    path on the w x h grid and swapping coordinates, so every rectangle admits all 8 variants.
    """
    swap = g & 1
    if ptype == 3 and not (h == w and _is_pow2(h)):
        ptype = ptype % 3  # Hilbert needs a square power-of-two tile; fall back to zig-zag
    if swap:
        _path(ptype, w, h, oj, oi)
    else:
        _path(ptype, h, w, oi, oj)
    if (g >> 1) & 1:
        for k in range(h * w):
            oi[k] = h - 1 - oi[k]
    if (g >> 2) & 1:
        for k in range(h * w):
            oj[k] = w - 1 - oj[k]


@njit(cache=True)
def fisher_yates(n, words):
    """Keyed uniform permutation of 0..n-1 from 64-bit keystream words (len >= n).

    j = w mod (i+1); the statistical distance from a uniform permutation is below n^2 / 2^65
    (2^-49 for the 256-entry S-boxes, below 2^-37 for the tile permutations of images up to 2048 x 2048).
    """
    p = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = words[i] % np.uint64(i + 1)
        t = p[i]
        p[i] = p[j]
        p[j] = t
    return p


@njit(cache=True)
def quadpath_sequence(H, W, B, tile_perm, sel):
    """Flat plaintext indices in QuadPath reading order.

    Tiles (B x B, edge tiles truncated) are visited in the keyed order tile_perm; inside tile t the
    path family is sel[t] & 3 and the dihedral element is (sel[t] >> 2) & 7.
    """
    nw = (W + B - 1) // B
    seq = np.empty(H * W, dtype=np.int64)
    oi = np.empty(B * B, dtype=np.int64)
    oj = np.empty(B * B, dtype=np.int64)
    k = 0
    for q in range(tile_perm.size):
        t = tile_perm[q]
        a = t // nw
        b = t % nw
        r0 = a * B
        c0 = b * B
        h = min(B, H - r0)
        w = min(B, W - c0)
        tile_order(sel[t] & 3, (sel[t] >> 2) & 7, h, w, oi, oj)
        for m in range(h * w):
            seq[k] = (r0 + oi[m]) * W + (c0 + oj[m])
            k += 1
    return seq


def golden_multiplier(L, delta):
    """Interleaver multiplier near L / phi, coprime with L; delta (keyed, small) perturbs it."""
    a = int(round(L * PHI_INV)) + int(delta)
    a = max(1, a % L) if L > 1 else 1
    while math.gcd(a, L) != 1:
        a += 1
    return a


def quadpath_gather(H, W, B, tile_perm, sel, a, b):
    """Gather index G of the full QuadPath scrambler: Y.flat[o] = X.flat[G[o]].

    Reading order seq (tile paths), then the golden interleaver o -> (a*o + b) mod L, which places
    raster-adjacent output pixels ~L/phi apart in the reading sequence.
    """
    return _interleave(quadpath_sequence(H, W, B, tile_perm, sel), a, b)


@njit(cache=True)
def _interleave(seq, a, b):
    L = seq.size
    G = np.empty(L, dtype=np.int64)
    t = b % L
    for o in range(L):
        G[o] = seq[t]
        t += a
        if t >= L:
            t -= L
    return G


def quadpath2024_gather(H, W, rounds=4):
    """Gather index of the original keyless QUAD-PATH transform (ICICC 2024).

    One path = zig-zag read of the matrix into a 1-D array, row-major refill to a 2-D matrix of the same
    shape, then a 90-degree anticlockwise rotation; four paths are applied in sequence.
    Returns (G, Hout, Wout) with Y.flat = X.flat[G] and Y of shape (Hout, Wout).
    """
    idx = np.arange(H * W, dtype=np.int64).reshape(H, W)
    for _ in range(rounds):
        h, w = idx.shape
        oi = np.empty(h * w, dtype=np.int64)
        oj = np.empty(h * w, dtype=np.int64)
        _path(0, h, w, oi, oj)
        idx = np.rot90(idx[oi, oj].reshape(h, w), 1)
    return np.ascontiguousarray(idx).ravel(), idx.shape[0], idx.shape[1]
