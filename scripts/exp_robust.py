"""Robustness of decryption to corrupted cipher images, and tamper localisation.

Channels
  occlusion : a centred region of the cipher image (1/64 ... 1/2 of the area) is set to 0
  packet    : cipher bytes serialised in raster order into 1024-byte packets; packets lost i.i.d. (erasures)
  noise     : salt-and-pepper noise on cipher bytes (density d)
Receivers
  QuadPath-AE (tags)   : resilient decryption; corrupted bytes located by the 32-bit tile tags (occlusion) or
                         known erasures (packet); exact error propagation; harmonic inpainting of the mask
  QuadPath-AE (blind)  : plain decryption followed by a 3 x 3 median filter
  AES-256-CTR (oracle) : the receiver is given the exact corrupted region; harmonic inpainting
  P-CTR (oracle)       : QuadPath scrambling + CTR, exact mask; harmonic inpainting
  LOG-PD (blind)       : decryption + 3 x 3 median filter
  AES-GCM / ChaCha20-Poly1305 : the tag check rejects the whole image (no output)

  python scripts/exp_robust.py
"""
import os
import time

import numpy as np
import pandas as pd
from common import KODAK, MASTER_KEY, RES, key_for, nonce_for, rng_for
from skimage import data as skd

from qpae import baselines as bl
from qpae import cipher as cp
from qpae import data
from qpae import metrics as mt
from qpae import recovery as rc

OCCL = {"1/64": (1 / 8, 1 / 8), "1/16": (1 / 4, 1 / 4), "1/8": (1 / 4, 1 / 2), "1/4": (1 / 2, 1 / 2),
        "1/2": (1 / 2, 1.0)}
PACKET_LOSS = [0.01, 0.02, 0.05, 0.10, 0.20]
NOISE = [0.001, 0.005, 0.01, 0.05]
PKT = 1024


def occl_mask(shape, fh, fw):
    H, W = shape[:2]
    h, w = int(round(H * fh)), int(round(W * fw))
    r0, c0 = (H - h) // 2, (W - w) // 2
    m = np.zeros(shape, dtype=bool)
    m[r0:r0 + h, c0:c0 + w] = True
    return m


def packet_mask(shape, p, rng):
    n = int(np.prod(shape))
    npk = (n + PKT - 1) // PKT
    lost = rng.random(npk) < p
    return np.repeat(lost, PKT)[:n].reshape(shape)


def score(P, X):
    return mt.psnr(P, X), mt.ssim(P, X)


def evaluate(name, P, channel, level, bad, values, rng):
    """bad: corrupted cipher bytes (bool, P.shape); values: replacement byte values (array, P.shape)."""
    rows = []
    H, W = P.shape[:2]
    C = 1 if P.ndim == 2 else P.shape[2]
    nonce = nonce_for("robust", name)
    sealed = cp.encrypt(P, MASTER_KEY, nonce)
    k_enc, _ = cp.derive_keys(MASTER_KEY, nonce, sealed.header)
    Cb = np.where(bad, values, sealed.cipher).astype(np.uint8)
    corrupted = cp.Sealed(sealed.header, sealed.nonce, Cb, sealed.ttags, sealed.gtag)
    known = bad.reshape(H, W, C) if channel == "packet" else None
    t = time.time()
    plain, mask, failed, gok = cp.decrypt_resilient(corrupted, MASTER_KEY, known_bad=known)
    if channel != "noise":
        rec = rc.harmonic_inpaint(plain, mask)
        ps, ss = score(P, rec)
        rows.append(dict(scheme="QuadPath-AE (tags)", psnr=ps, ssim=ss, mask_frac=float(mask.mean()),
                         isolation=rc.isolation_ratio(mask), largest_hole=rc.largest_hole(mask),
                         failed_tiles=failed, seconds=time.time() - t))
    ps, ss = score(P, plain)
    rows.append(dict(scheme="QuadPath-AE (raw)", psnr=ps, ssim=ss))
    ps, ss = score(P, rc.median3(plain))
    rows.append(dict(scheme="QuadPath-AE (blind median)", psnr=ps, ssim=ss))
    # AES-256-CTR
    akey = key_for("aes")
    ca = bl.aes_ctr(P, akey, nonce)
    da = bl.aes_ctr(np.where(bad, values, ca).astype(np.uint8), akey, nonce)
    ps, ss = score(P, da)
    rows.append(dict(scheme="AES-256-CTR (raw)", psnr=ps, ssim=ss))
    if channel != "noise":
        ps, ss = score(P, rc.harmonic_inpaint(da, bad))
        rows.append(dict(scheme="AES-256-CTR (oracle mask)", psnr=ps, ssim=ss, mask_frac=float(bad.mean()),
                         isolation=rc.isolation_ratio(bad), largest_hole=rc.largest_hole(bad)))
        # P-CTR
        cpc = bl.pctr_encrypt(P, k_enc)
        dpc = bl.pctr_decrypt(np.where(bad, values, cpc).astype(np.uint8), k_enc)
        mpc = bl.pctr_error_mask(bad.reshape(H, W, C), k_enc, H, W, C).reshape(P.shape)
        ps, ss = score(P, rc.harmonic_inpaint(dpc, mpc))
        rows.append(dict(scheme="P-CTR (oracle mask)", psnr=ps, ssim=ss, mask_frac=float(mpc.mean())))
    else:
        ps, ss = score(P, rc.median3(da))
        rows.append(dict(scheme="AES-256-CTR (blind median)", psnr=ps, ssim=ss))
    # LOG-PD
    lk = key_for("logpd")
    cl = bl.logpd_encrypt(P, lk)
    dl = bl.logpd_decrypt(np.where(bad, values, cl).astype(np.uint8), lk)
    ps, ss = score(P, dl)
    rows.append(dict(scheme="LOG-PD (raw)", psnr=ps, ssim=ss))
    ps, ss = score(P, rc.median3(dl))
    rows.append(dict(scheme="LOG-PD (blind median)", psnr=ps, ssim=ss))
    rows.append(dict(scheme="AES-256-GCM / ChaCha20-Poly1305", psnr=np.nan, ssim=np.nan, rejected=True))
    rows.append(dict(scheme="QuadPath-AE (strict)", psnr=np.nan, ssim=np.nan, rejected=not gok))
    for r in rows:
        r.update(image=name, group=data.group_of(name), channel=channel, level=level,
                 corrupted_frac=float(bad.mean()))
    return rows


def tamper(name, P, events=100):
    """Random modifications of the cipher image: strict rejection and tile-level localisation."""
    rng = rng_for("tamper", name)
    nonce = nonce_for("tamper", name)
    s = cp.encrypt(P, MASTER_KEY, nonce)
    H, W, T = P.shape[0], P.shape[1], 32
    nw = (W + T - 1) // T
    rows = []
    for e in range(events):
        kind = ["bitflip", "patch16", "copymove64"][e % 3]
        Cb = s.cipher.copy()
        if kind == "bitflip":
            idx = tuple(int(rng.integers(0, d)) for d in Cb.shape)
            Cb[idx] ^= np.uint8(1 << int(rng.integers(0, 8)))
        elif kind == "patch16":
            r, c = int(rng.integers(0, H - 16)), int(rng.integers(0, W - 16))
            Cb[r:r + 16, c:c + 16] = rng.integers(0, 256, Cb[r:r + 16, c:c + 16].shape, dtype=np.uint8)
        else:
            r, c = int(rng.integers(0, H - 64)), int(rng.integers(0, W - 64))
            r2, c2 = int(rng.integers(0, H - 64)), int(rng.integers(0, W - 64))
            Cb[r:r + 64, c:c + 64] = s.cipher[r2:r2 + 64, c2:c2 + 64]
        diff = (Cb != s.cipher)
        if diff.ndim == 3:
            diff = diff.any(axis=2)
        truth = set()
        for (y, x) in zip(*np.nonzero(diff)):
            truth.add((y // T) * nw + x // T)
        sb = cp.Sealed(s.header, s.nonce, Cb, s.ttags, s.gtag)
        try:
            cp.decrypt(sb, MASTER_KEY)
            rejected = False
        except cp.AuthenticationError:
            rejected = True
        k_enc, k_mac = cp.derive_keys(MASTER_KEY, nonce, s.header)
        exp = cp.tile_tags(k_mac, s.header, s.nonce, Cb, T)
        flagged = {t for t in range(len(exp) // 4) if exp[4 * t:4 * t + 4] != s.ttags[4 * t:4 * t + 4]}
        rows.append(dict(image=name, event=e, kind=kind, modified=bool(diff.any()), rejected=rejected,
                         true_tiles=len(truth), flagged_tiles=len(flagged), detected=len(truth & flagged),
                         false_flags=len(flagged - truth)))
    return rows


if __name__ == "__main__":
    imgs = {"Cameraman": skd.camera(), "Moon": skd.moon(), "Astronaut": skd.astronaut(),
            "IHC": skd.immunohistochemistry()}
    imgs.update(data.kodak(KODAK))
    rows, trows = [], []
    t0 = time.time()
    for name, P in imgs.items():
        t = time.time()
        rng = rng_for("robust", name)
        for lv, (fh, fw) in OCCL.items():
            rows += evaluate(name, P, "occlusion", lv, occl_mask(P.shape, fh, fw), np.zeros(P.shape, np.uint8), rng)
        for p in PACKET_LOSS:
            rows += evaluate(name, P, "packet", p, packet_mask(P.shape, p, rng), np.zeros(P.shape, np.uint8), rng)
        for d in NOISE:
            bad = rng.random(P.shape) < d
            vals = np.where(rng.random(P.shape) < 0.5, 0, 255).astype(np.uint8)
            rows += evaluate(name, P, "noise", d, bad, vals, rng)
        trows += tamper(name, P)
        print(f"{name:10s} {time.time() - t:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "robust_per_image.csv"), index=False)
    pd.DataFrame(trows).to_csv(os.path.join(RES, "tamper_events.csv"), index=False)
    print(df.groupby(["channel", "level", "scheme"])["psnr"].mean().unstack().round(2).to_string())
    print(f"done in {time.time() - t0:.0f}s")
