"""QuadPath-AE: nonce-based authenticated image encryption with keyed quad-path scrambling.

Encryption of an image P (H x W x C, uint8) under a 256-bit master key K and a fresh 128-bit nonce N:

  1. Key schedule   (K_e || K_m) = HKDF-Expand-SHA256(K, info = "QPAE-v1" || header || N)
                    every random quantity is drawn from ChaCha20(K_e) with a distinct 96-bit purpose label.
  2. R rounds of
       a. QuadPath scrambling of every channel plane (keyed tile order, path family, D4 element,
          golden interleaver); planes are scrambled independently;
       b. channel interleaving into one byte sequence x (pixel-major, channel-minor);
       c. three chained diffusion passes, forward, backward, forward (FBF2), each with a two-byte window:
            forward   y_i = S[(x_i + y_{i-1} + T[y_{i-2}]) mod 256] XOR k_i
            backward  y_i = S[(x_i + y_{i+1} + T[y_{i+2}]) mod 256] XOR k_i
          with keyed S-boxes S and T, a fresh keystream k per pass and keyed initial values.
  3. Encrypt-then-MAC: one 32-bit HMAC-SHA256 tag per T x T ciphertext tile (localisation) and a
     128-bit HMAC-SHA256 tag over header || nonce || ciphertext || tile tags (integrity).

The cipher image has exactly the size and type of the plain image. Decryption is non-recursive:
x_i = S^-1[y_i XOR k_i] - y_{i-1} - T[y_{i-2}] only reads received values, so it vectorises and a
corrupted ciphertext byte affects at most seven plaintext bytes per round before inverse scrambling.
"""
import hashlib
import hmac
import os
import struct
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from numba import njit

from .paths import fisher_yates, golden_multiplier, quadpath2024_gather, quadpath_gather

MAGIC = b"QPAE"
VERSION = 1
NONCE_BYTES = 16
GTAG_BYTES = 16
TTAG_BYTES = 4


# --------------------------------------------------------------------------- key schedule
def header_bytes(H, W, C, R, B, T):
    return MAGIC + struct.pack(">BIIBBBB", VERSION, H, W, C, R, B, T)


def derive_keys(key, nonce, hdr):
    """(K_e || K_m) = HKDF-Expand(K, "QPAE-v1" || header || nonce, 64).

    The master key is a uniformly random 256-bit key, so the extract step of HKDF is skipped (RFC 5869,
    Sec. 3.3) and HKDF-Expand is used as a PRF keyed by K on the public input (header, nonce).
    """
    if len(key) != 32 or len(nonce) != NONCE_BYTES:
        raise ValueError("key must be 32 bytes and nonce 16 bytes")
    okm = HKDFExpand(algorithm=hashes.SHA256(), length=64, info=b"QPAE-v1" + hdr + nonce).derive(key)
    return okm[:32], okm[32:]


def keystream(k_enc, label, nbytes):
    """ChaCha20 keystream for one purpose. label: 12-byte domain separator; block counter starts at 0."""
    nonce16 = b"\x00\x00\x00\x00" + label
    enc = Cipher(algorithms.ChaCha20(k_enc, nonce16), mode=None).encryptor()
    return enc.update(bytes(nbytes))


def _label(kind, rnd, ch):
    return struct.pack(">4sBB6x", kind, rnd, ch)


@dataclass
class RoundKeys:
    gathers: list  # per channel: gather index (int64, H*W)
    sbox: np.ndarray
    sinv: np.ndarray
    tbox: np.ndarray  # second keyed permutation, applied to y_{i-2} (window w = 2)
    ks: list  # one keystream per diffusion pass
    ivs: list  # per pass: (y_{-1}, y_{-2}) initial values


SCRAMBLERS = ("quadpath", "tiles_only", "random", "qp2024", "none")


def round_keys(k_enc, H, W, C, B, rnd, scrambler="quadpath"):
    """Round key material. scrambler selects the permutation layer (ablations):
    quadpath = tile paths + golden interleaver (proposed); tiles_only = tile paths without interleaver;
    random = uniformly random full permutation; qp2024 = keyless QUAD-PATH (2024); none = identity."""
    L = H * W
    nh, nw = (H + B - 1) // B, (W + B - 1) // B
    T = nh * nw
    gathers = []
    for ch in range(C):
        if scrambler in ("quadpath", "tiles_only"):
            tw = np.frombuffer(keystream(k_enc, _label(b"TPRM", rnd, ch), 8 * T), dtype="<u8")
            tile_perm = fisher_yates(T, tw)
            sel = np.frombuffer(keystream(k_enc, _label(b"TSEL", rnd, ch), T), dtype=np.uint8).astype(np.int64)
            if scrambler == "quadpath":
                prm = np.frombuffer(keystream(k_enc, _label(b"ILVR", rnd, ch), 16), dtype="<u8")
                a = golden_multiplier(L, int(prm[0] % np.uint64(1024)))
                b = int(prm[1] % np.uint64(L))
            else:
                a, b = 1, 0
            gathers.append(quadpath_gather(H, W, B, tile_perm, sel, a, b))
        elif scrambler == "random":
            pw = np.frombuffer(keystream(k_enc, _label(b"FPRM", rnd, ch), 8 * L), dtype="<u8")
            gathers.append(fisher_yates(L, pw))
        elif scrambler == "qp2024":
            g, h2, w2 = quadpath2024_gather(H, W)
            if (h2, w2) != (H, W):
                raise ValueError("QUAD-PATH 2024 permutation changes the shape of non-square images")
            gathers.append(g)
        elif scrambler == "none":
            gathers.append(np.arange(L, dtype=np.int64))
        else:
            raise ValueError(scrambler)
    sw = np.frombuffer(keystream(k_enc, _label(b"SBOX", rnd, 0), 8 * 256), dtype="<u8")
    sbox = fisher_yates(256, sw).astype(np.uint8)
    sinv = np.empty(256, dtype=np.uint8)
    sinv[sbox] = np.arange(256, dtype=np.uint8)
    tw = np.frombuffer(keystream(k_enc, _label(b"TBOX", rnd, 0), 8 * 256), dtype="<u8")
    tbox = fisher_yates(256, tw).astype(np.uint8)
    n = L * C
    ks, ivs = [], []
    for p in range(3):
        raw = np.frombuffer(keystream(k_enc, _label(b"DIFF", rnd, p), n + 2), dtype=np.uint8)
        ks.append(raw[:n])
        ivs.append((int(raw[n]), int(raw[n + 1])))
    return RoundKeys(gathers, sbox, sinv, tbox, ks, ivs)


# --------------------------------------------------------------------------- diffusion kernels
# One pass with chaining window w (1 or 2):
#   forward   y_i = S[(x_i + y_{i-1} + w2 * T[y_{i-2}]) mod 256] XOR k_i      (w2 = 1 if w == 2 else 0)
#   backward  y_i = S[(x_i + y_{i+1} + w2 * T[y_{i+2}]) mod 256] XOR k_i
# A difference entering a pass where all earlier inputs agree can never die (the update is injective in
# y_{i-1}, and for w = 2 two consecutive equal outputs would force equality further back). Where both the
# input and the chain already differ, it dies only after w consecutive output collisions (prob. ~256^-w).
@njit(cache=True)
def _pass_fwd(x, S, T, k, iv1, iv2, w):
    n = x.size
    y = np.empty(n, dtype=np.uint8)
    p1 = iv1
    p2 = iv2
    for i in range(n):
        s = np.int64(x[i]) + p1
        if w == 2:
            s += np.int64(T[p2])
        v = S[s & 255] ^ k[i]
        y[i] = v
        p2 = p1
        p1 = np.int64(v)
    return y


@njit(cache=True)
def _pass_bwd(x, S, T, k, iv1, iv2, w):
    n = x.size
    y = np.empty(n, dtype=np.uint8)
    p1 = iv1
    p2 = iv2
    for i in range(n - 1, -1, -1):
        s = np.int64(x[i]) + p1
        if w == 2:
            s += np.int64(T[p2])
        v = S[s & 255] ^ k[i]
        y[i] = v
        p2 = p1
        p1 = np.int64(v)
    return y


def _shifted(y, d, iv1, iv2):
    """z_i = y_{i-d} (d = 1, 2, forward) or y_{i+|d|} (d = -1, -2, backward), using the initial values
    y_{-1} = iv1, y_{-2} = iv2 (forward) or y_n = iv1, y_{n+1} = iv2 (backward)."""
    y16 = y.astype(np.int16)
    n = y16.size
    if d == 1:
        return np.concatenate([np.array([iv1], np.int16), y16])[:n]
    if d == 2:
        return np.concatenate([np.array([iv2, iv1], np.int16), y16])[:n]
    if d == -1:
        return np.concatenate([y16[1:], np.array([iv1], np.int16)])[-n:]
    return np.concatenate([y16[2:], np.array([iv1, iv2], np.int16)])[-n:]


@njit(cache=True)
def _inv_kernel(y, Sinv, T, k, iv1, iv2, w, forward):
    """Inverse of one pass. Each output reads only received bytes (y_i, y_{i-+1}, y_{i-+2})."""
    n = y.size
    x = np.empty(n, dtype=np.uint8)
    for i in range(n):
        if forward:
            p1 = np.int64(y[i - 1]) if i >= 1 else np.int64(iv1)
            p2 = np.int64(y[i - 2]) if i >= 2 else (np.int64(iv1) if i == 1 else np.int64(iv2))
        else:
            p1 = np.int64(y[i + 1]) if i + 1 < n else np.int64(iv1)
            p2 = np.int64(y[i + 2]) if i + 2 < n else (np.int64(iv1) if i + 2 == n else np.int64(iv2))
        v = np.int64(Sinv[y[i] ^ k[i]]) - p1
        if w == 2:
            v -= np.int64(T[p2])
        x[i] = v & 255
    return x


def _inv_pass(y, rk, p, forward, w):
    iv1, iv2 = rk.ivs[p]
    return _inv_kernel(y, rk.sinv, rk.tbox, rk.ks[p], iv1, iv2, w, forward)


# Diffusion schedules (ablations): F = forward only; FB = forward + backward; FBF = forward, backward,
# forward. Suffix 1/2 = chaining window. With w = 1, FB leaves a deterministic trail (bytes before the first
# difference always differ) and FB1/FBF1 lose the avalanche with probability ~1/256. FB2 and FBF2 both pass
# the avalanche tests (exp_ablation.py); FBF2 (proposed) adds a third keyed nonlinear layer, so that under
# nonce reuse the backward-pass outputs (whose differences S[a] XOR S[b] do not depend on the keystream)
# are never exposed directly.
DEFAULT_DIFFUSION = "FBF2"


def _parse_sched(name):
    if name[-1] in "12":
        return name[:-1], int(name[-1])
    return name, 1


def diffuse(x, rk, schedule=DEFAULT_DIFFUSION):
    sched, w = _parse_sched(schedule)
    for p, d in enumerate(sched):
        fn = _pass_fwd if d == "F" else _pass_bwd
        x = fn(x, rk.sbox, rk.tbox, rk.ks[p], rk.ivs[p][0], rk.ivs[p][1], w)
    return x


def undiffuse(y, rk, schedule=DEFAULT_DIFFUSION):
    sched, w = _parse_sched(schedule)
    for p in range(len(sched) - 1, -1, -1):
        y = _inv_pass(y, rk, p, sched[p] == "F", w)
    return y


def propagate_errors(bad, schedule=DEFAULT_DIFFUSION):
    """Bytes of the diffusion input that a set of corrupted diffusion-output bytes can affect."""
    sched, w = _parse_sched(schedule)
    for p in range(len(sched) - 1, -1, -1):
        b = bad.copy()
        for d in range(1, w + 1):
            if sched[p] == "F":
                b[d:] |= bad[:-d]
            else:
                b[:-d] |= bad[d:]
        bad = b
    return bad


# --------------------------------------------------------------------------- core transform
def _as3(img):
    img = np.asarray(img, dtype=np.uint8)
    return (img[:, :, None] if img.ndim == 2 else img), img.ndim == 2


def core_encrypt(P, k_enc, R=1, B=16, scrambler="quadpath", diffusion=DEFAULT_DIFFUSION):
    """Keyed transform without tags. diffusion: FB2 (proposed) | F1 | FB1 | FBF1 | FBF2 | additive | none."""
    X, gray = _as3(P)
    H, W, C = X.shape
    for r in range(R):
        rk = round_keys(k_enc, H, W, C, B, r, scrambler)
        planes = [X[:, :, c].ravel()[rk.gathers[c]] for c in range(C)]
        X = np.stack(planes, axis=1).reshape(H, W, C)
        x = np.ascontiguousarray(X).ravel()
        if diffusion == "none":
            c = x ^ rk.ks[0]
        elif diffusion == "additive":
            c = _diffuse_additive(x, rk.ks[0], rk.ks[1], rk.ivs[0][0], rk.ivs[1][0])
        else:
            c = diffuse(x, rk, diffusion)
        X = c.reshape(H, W, C)
    return X[:, :, 0] if gray else X


def core_decrypt(Cimg, k_enc, R=1, B=16, scrambler="quadpath", diffusion=DEFAULT_DIFFUSION):
    X, gray = _as3(Cimg)
    H, W, C = X.shape
    for r in range(R - 1, -1, -1):
        rk = round_keys(k_enc, H, W, C, B, r, scrambler)
        x = undiffuse(np.ascontiguousarray(X).ravel(), rk, diffusion)
        Y = x.reshape(H, W, C)
        out = np.empty_like(Y)
        for c in range(C):
            plane = np.empty(H * W, dtype=np.uint8)
            plane[rk.gathers[c]] = Y[:, :, c].ravel()
            out[:, :, c] = plane.reshape(H, W)
        X = out
    return X[:, :, 0] if gray else X


def error_mask(bad, k_enc, H, W, C, R=1, B=16, scrambler="quadpath", diffusion=DEFAULT_DIFFUSION):
    """Plaintext bytes that can be affected by the corrupted ciphertext bytes `bad` (H x W x C bool).

    Exact propagation: FBF2 undiffusion maps an error at i to {i-2, ..., i+4}; inverse scrambling moves it.
    """
    M = bad.reshape(H, W, C).copy()
    for r in range(R - 1, -1, -1):
        rk = round_keys(k_enc, H, W, C, B, r, scrambler)
        Y = propagate_errors(M.ravel(), diffusion).reshape(H, W, C)
        out = np.empty_like(Y)
        for c in range(C):
            plane = np.empty(H * W, dtype=bool)
            plane[rk.gathers[c]] = Y[:, :, c].ravel()
            out[:, :, c] = plane.reshape(H, W)
        M = out
    return M


@njit(cache=True)
def _diffuse_additive(x, k1, k2, iv1, iv2):
    """Ablation: classical linear chaining y_i = x_i + y_{i-1} + k_i (forward, then backward), no S-box."""
    n = x.size
    u = np.empty(n, dtype=np.uint8)
    prev = iv1
    for i in range(n):
        v = (np.int64(x[i]) + prev + k1[i]) & 255
        u[i] = v
        prev = v
    c = np.empty(n, dtype=np.uint8)
    nxt = iv2
    for i in range(n - 1, -1, -1):
        v = (np.int64(u[i]) + nxt + k2[i]) & 255
        c[i] = v
        nxt = v
    return c


# --------------------------------------------------------------------------- authentication
def tile_tags(k_mac, hdr, nonce, Cimg, T):
    X, _ = _as3(Cimg)
    H, W, _ = X.shape
    nh, nw = (H + T - 1) // T, (W + T - 1) // T
    out = bytearray()
    for a in range(nh):
        for b in range(nw):
            blk = np.ascontiguousarray(X[a * T:(a + 1) * T, b * T:(b + 1) * T]).tobytes()
            msg = b"TILE" + hdr + nonce + struct.pack(">I", a * nw + b) + blk
            out += hmac.new(k_mac, msg, hashlib.sha256).digest()[:TTAG_BYTES]
    return bytes(out)


def global_tag(k_mac, hdr, nonce, cbytes, ttags):
    return hmac.new(k_mac, b"GLOB" + hdr + nonce + cbytes + ttags, hashlib.sha256).digest()[:GTAG_BYTES]


@dataclass
class Sealed:
    header: bytes
    nonce: bytes
    cipher: np.ndarray  # same shape/dtype as the plain image
    ttags: bytes
    gtag: bytes

    def to_bytes(self):
        return self.header + self.nonce + self.cipher.tobytes() + self.ttags + self.gtag

    @property
    def overhead_bytes(self):
        return len(self.header) + len(self.nonce) + len(self.ttags) + len(self.gtag)


def encrypt(P, key, nonce=None, R=1, B=16, T=32):
    P = np.asarray(P, dtype=np.uint8)
    H, W = P.shape[:2]
    C = 1 if P.ndim == 2 else P.shape[2]
    nonce = os.urandom(NONCE_BYTES) if nonce is None else nonce
    hdr = header_bytes(H, W, C, R, B, T)
    k_enc, k_mac = derive_keys(key, nonce, hdr)
    Cimg = core_encrypt(P, k_enc, R, B)
    tt = tile_tags(k_mac, hdr, nonce, Cimg, T)
    return Sealed(hdr, nonce, Cimg, tt, global_tag(k_mac, hdr, nonce, Cimg.tobytes(), tt))


def _parse(hdr):
    _, ver, H, W, C, R, B, T = struct.unpack(">4sBIIBBBB", hdr)
    return H, W, C, R, B, T


class AuthenticationError(Exception):
    pass


def decrypt(sealed, key):
    """Strict mode: verify the 128-bit tag first; reject on any modification."""
    H, W, C, R, B, T = _parse(sealed.header)
    k_enc, k_mac = derive_keys(key, sealed.nonce, sealed.header)
    tag = global_tag(k_mac, sealed.header, sealed.nonce, sealed.cipher.tobytes(), sealed.ttags)
    if not hmac.compare_digest(tag, sealed.gtag):
        raise AuthenticationError("global tag mismatch")
    return core_decrypt(sealed.cipher, k_enc, R, B)


def decrypt_resilient(sealed, key, known_bad=None):
    """Loss-tolerant mode for unreliable channels.

    Bytes listed in known_bad (e.g. lost packets) are marked individually. A tile whose 32-bit tag fails
    although it contains no known loss is marked as a whole (unexplained modification). Tiles that contain
    known losses cannot be verified; their remaining bytes are kept and counted as unverified.
    Returns (plain, unreliable_mask H x W x C, failed_tile_count, global_ok); failed_tile_count counts
    the tiles marked because of an unexplained tag failure.
    """
    H, W, C, R, B, T = _parse(sealed.header)
    k_enc, k_mac = derive_keys(key, sealed.nonce, sealed.header)
    known = np.zeros((H, W, C), dtype=bool) if known_bad is None else known_bad.reshape(H, W, C)
    bad = known.copy()
    expect = tile_tags(k_mac, sealed.header, sealed.nonce, sealed.cipher, T)
    nw = (W + T - 1) // T
    failed = 0
    for t in range(len(expect) // TTAG_BYTES):
        if expect[4 * t:4 * t + 4] != sealed.ttags[4 * t:4 * t + 4]:
            a, b = divmod(t, nw)
            region = (slice(a * T, (a + 1) * T), slice(b * T, (b + 1) * T))
            if known[region].any():
                continue  # failure explained by known losses; the rest of the tile is unverified
            bad[region] = True
            failed += 1
    g_ok = hmac.compare_digest(
        global_tag(k_mac, sealed.header, sealed.nonce, sealed.cipher.tobytes(), sealed.ttags), sealed.gtag)
    plain = core_decrypt(sealed.cipher, k_enc, R, B)
    mask = error_mask(bad, k_enc, H, W, C, R, B)
    if C == 1:
        mask = mask[:, :, 0]
    return plain, mask, failed, g_ok
