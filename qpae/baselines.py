"""Baseline image ciphers re-implemented under one interface.

  qp2024   : the keyless QUAD-PATH transform of Rizvi et al. (ICICC 2024), four zig-zag/rotation paths.
  logpd    : a representative chaos-based permutation-diffusion cipher in the architecture of Fridrich (1998):
             logistic-map sort permutation followed by forward and backward modular chaining diffusion
             driven by the same logistic orbit; the key is (x0, mu); deterministic (no nonce).
  aes_ctr  : AES-256 in counter mode (confidentiality only), 128-bit initial counter block as nonce.
  aes_gcm  : AES-256-GCM (NIST SP 800-38D), 96-bit nonce, 128-bit tag.
  chacha_poly : ChaCha20-Poly1305 (RFC 8439), 96-bit nonce, 128-bit tag.
  pctr     : ablation "QuadPath scrambling + AES-256-CTR whitening" (dispersion without diffusion).

Every cipher keeps the image shape for the image-domain part; AEAD tags are returned separately.
"""
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from numba import njit

from . import cipher as qp
from .paths import quadpath2024_gather


# ------------------------------------------------------------------ QUAD-PATH (2024)
def qp2024_encrypt(P):
    P = np.asarray(P, dtype=np.uint8)
    H, W = P.shape[:2]
    G, h2, w2 = quadpath2024_gather(H, W)
    if P.ndim == 2:
        return P.ravel()[G].reshape(h2, w2)
    return np.stack([P[:, :, c].ravel()[G].reshape(h2, w2) for c in range(P.shape[2])], axis=2)


def qp2024_decrypt(Cimg, H, W):
    G, h2, w2 = quadpath2024_gather(H, W)
    Cimg = np.asarray(Cimg, dtype=np.uint8)
    planes = [Cimg] if Cimg.ndim == 2 else [Cimg[:, :, c] for c in range(Cimg.shape[2])]
    out = []
    for pl in planes:
        x = np.empty(H * W, dtype=np.uint8)
        x[G] = pl.ravel()
        out.append(x.reshape(H, W))
    return out[0] if Cimg.ndim == 2 else np.stack(out, axis=2)


def qp2024_order(H, W):
    """Order of the QUAD-PATH permutation: applying the 'encryption' this many times returns the image."""
    from math import gcd
    G, h2, w2 = quadpath2024_gather(H, W)
    if (h2, w2) != (H, W):
        return None
    seen = np.zeros(G.size, dtype=bool)
    order = 1
    for s in range(G.size):
        if seen[s]:
            continue
        L = 0
        j = s
        while not seen[j]:
            seen[j] = True
            j = G[j]
            L += 1
        order = order * L // gcd(order, L)
    return order


# ------------------------------------------------------------------ logistic permutation-diffusion
@njit(cache=True)
def _logistic(x0, mu, n, skip=1000):
    x = x0
    for _ in range(skip):
        x = mu * x * (1.0 - x)
    out = np.empty(n)
    for i in range(n):
        x = mu * x * (1.0 - x)
        out[i] = x
    return out


@njit(cache=True)
def _cbc2(x, k):
    n = x.size
    u = np.empty(n, dtype=np.uint8)
    prev = 0
    for i in range(n):
        v = (np.int64(x[i]) + k[i] + prev) & 255
        u[i] = v
        prev = v
    c = np.empty(n, dtype=np.uint8)
    nxt = 0
    for i in range(n - 1, -1, -1):
        v = (np.int64(u[i]) + k[n - 1 - i] + nxt) & 255
        c[i] = v
        nxt = v
    return c


def _logpd_keys(key, n):
    x0 = 0.1 + 0.8 * (int.from_bytes(key[:8], "big") / 2.0 ** 64)
    mu = 3.99 + 0.009999 * (int.from_bytes(key[8:16], "big") / 2.0 ** 64)
    seq = _logistic(x0, mu, 2 * n)
    perm = np.argsort(seq[:n], kind="stable")
    ks = (np.floor(seq[n:] * 1e14) % 256).astype(np.int64)
    return perm, ks


def logpd_encrypt(P, key):
    P = np.asarray(P, dtype=np.uint8)
    x = P.ravel()
    perm, ks = _logpd_keys(key, x.size)
    return _cbc2(x[perm], ks).reshape(P.shape)


def logpd_decrypt(Cimg, key):
    c = np.asarray(Cimg, dtype=np.int64).ravel()
    n = c.size
    perm, ks = _logpd_keys(key, n)
    nxt = np.empty(n, dtype=np.int64)
    nxt[:-1] = c[1:]
    nxt[-1] = 0
    u = (c - ks[::-1] - nxt) & 255
    prv = np.empty(n, dtype=np.int64)
    prv[1:] = u[:-1]
    prv[0] = 0
    y = (u - ks - prv) & 255
    x = np.empty(n, dtype=np.uint8)
    x[perm] = y
    return x.reshape(np.asarray(Cimg).shape)


# ------------------------------------------------------------------ standard ciphers
def aes_ctr(P, key, nonce16):
    P = np.asarray(P, dtype=np.uint8)
    e = Cipher(algorithms.AES(key), modes.CTR(nonce16)).encryptor()
    return np.frombuffer(e.update(P.tobytes()) + e.finalize(), dtype=np.uint8).reshape(P.shape)


aes_ctr_decrypt = aes_ctr  # CTR decryption is the same keystream XOR


def aes_gcm_encrypt(P, key, nonce12):
    P = np.asarray(P, dtype=np.uint8)
    out = AESGCM(key).encrypt(nonce12, P.tobytes(), None)
    return np.frombuffer(out[:-16], dtype=np.uint8).reshape(P.shape), out[-16:]


def aes_gcm_decrypt(Cimg, tag, key, nonce12):
    """Raises cryptography.exceptions.InvalidTag on any modification."""
    pt = AESGCM(key).decrypt(nonce12, np.asarray(Cimg, dtype=np.uint8).tobytes() + tag, None)
    return np.frombuffer(pt, dtype=np.uint8).reshape(np.asarray(Cimg).shape)


def chacha_poly_encrypt(P, key, nonce12):
    P = np.asarray(P, dtype=np.uint8)
    out = ChaCha20Poly1305(key).encrypt(nonce12, P.tobytes(), None)
    return np.frombuffer(out[:-16], dtype=np.uint8).reshape(P.shape), out[-16:]


def chacha_poly_decrypt(Cimg, tag, key, nonce12):
    pt = ChaCha20Poly1305(key).decrypt(nonce12, np.asarray(Cimg, dtype=np.uint8).tobytes() + tag, None)
    return np.frombuffer(pt, dtype=np.uint8).reshape(np.asarray(Cimg).shape)


# ------------------------------------------------------------------ ablation: scrambling + CTR whitening
def pctr_encrypt(P, k_enc, B=16):
    X, gray = qp._as3(P)
    H, W, C = X.shape
    rk = qp.round_keys(k_enc, H, W, C, B, 0)
    Y = np.stack([X[:, :, c].ravel()[rk.gathers[c]] for c in range(C)], axis=1).reshape(H, W, C)
    Y = (Y.ravel() ^ rk.ks[0]).reshape(H, W, C)
    return Y[:, :, 0] if gray else Y


def pctr_decrypt(Cimg, k_enc, B=16):
    X, gray = qp._as3(Cimg)
    H, W, C = X.shape
    rk = qp.round_keys(k_enc, H, W, C, B, 0)
    Y = (X.ravel() ^ rk.ks[0]).reshape(H, W, C)
    out = np.empty_like(Y)
    for c in range(C):
        pl = np.empty(H * W, dtype=np.uint8)
        pl[rk.gathers[c]] = Y[:, :, c].ravel()
        out[:, :, c] = pl.reshape(H, W)
    return out[:, :, 0] if gray else out


def pctr_error_mask(bad, k_enc, H, W, C, B=16):
    rk = qp.round_keys(k_enc, H, W, C, B, 0)
    Y = bad.reshape(H, W, C)
    out = np.empty_like(Y)
    for c in range(C):
        pl = np.empty(H * W, dtype=bool)
        pl[rk.gathers[c]] = Y[:, :, c].ravel()
        out[:, :, c] = pl.reshape(H, W)
    return out
