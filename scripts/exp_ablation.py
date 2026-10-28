"""Ablations: diffusion schedule, scrambling layer, tile size B and number of rounds R.

  python scripts/exp_ablation.py [--trials 30]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from common import KODAK, MASTER_KEY, RES, nonce_for, rng_for
from skimage import data as skd

from qpae import cipher as cp
from qpae import data
from qpae import metrics as mt
from qpae import recovery as rc


def test_images():
    k = data.kodak(KODAK)
    return {"Cameraman": skd.camera(), "Moon": skd.moon(), "Astronaut": skd.astronaut(),
            "IHC": skd.immunohistochemistry(), "kodim05": k["kodim05"], "kodim23": k["kodim23"]}


def kenc(P, label, R=1, B=16):
    H, W = P.shape[:2]
    C = 1 if P.ndim == 2 else P.shape[2]
    return cp.derive_keys(MASTER_KEY, nonce_for("abl", label), cp.header_bytes(H, W, C, R, B, 32))[0]


def one_pixel(P, rng):
    Q = P.copy()
    idx = tuple(int(rng.integers(0, d)) for d in P.shape)
    Q[idx] = (int(Q[idx]) + 1) % 256
    return Q


def diffusion_ablation(imgs, trials):
    rows = []
    for name, P in imgs.items():
        ke = kenc(P, name)
        n = P.size
        lo = mt.npcr_critical(n)
        ul, uh = mt.uaci_critical(n)
        P2 = None
        for other, Q in imgs.items():
            if other != name and Q.shape == P.shape:
                P2 = Q
                break
        for sched in ("F1", "FB1", "FBF1", "FB2", "FBF2", "additive", "none"):
            rng = rng_for("abl-diff", name)
            c0 = cp.core_encrypt(P, ke, diffusion=sched)
            npc, uac, tds = [], [], []
            for _ in range(trials):
                c1 = cp.core_encrypt(one_pixel(P, rng), ke, diffusion=sched)
                npcr, uaci = mt.npcr_uaci(c0, c1)
                npc.append(npcr)
                uac.append(uaci)
                d = ((c0.astype(np.int16) - c1.astype(np.int16)) % 256).astype(np.uint8).ravel()
                tds.append(np.bincount(d, minlength=256)[1:].max() / d.size)  # most frequent non-zero difference
            npc, uac = np.array(npc), np.array(uac)
            # error expansion of one corrupted cipher byte (exact, from the propagation rule)
            if sched not in ("additive", "none"):
                bad = np.zeros(P.size, dtype=bool)
                bad[P.size // 2] = True
                expansion = int(cp.propagate_errors(bad, sched).sum())
            else:
                expansion = {"none": 1, "additive": None}[sched]
            row = dict(image=name, schedule=sched, npcr_mean=npc.mean(), npcr_sd=npc.std(), npcr_min=npc.min(),
                       lost_avalanche=int((npc < 99.0).sum()), uaci_mean=uac.mean(), uaci_sd=uac.std(),
                       uaci_min=uac.min(), uaci_max=uac.max(),
                       pass_rate=float(((npc > lo) & (uac > ul) & (uac < uh)).mean()), error_expansion=expansion,
                       onepixel_top_diff_share=float(np.mean(tds)))
            if P2 is not None:
                c1, c2 = cp.core_encrypt(P, ke, diffusion=sched), cp.core_encrypt(P2, ke, diffusion=sched)
                d = ((c1.astype(np.int16) - c2.astype(np.int16)) % 256).astype(np.uint8).ravel()
                row["misuse_top_diff_share"] = float(np.bincount(d, minlength=256).max() / d.size)
                row["misuse_rho"] = float(np.corrcoef((c1 ^ c2).ravel().astype(float),
                                                      (P ^ P2).ravel().astype(float))[0, 1])
            t = time.perf_counter()
            for _ in range(5):
                cp.core_encrypt(P, ke, diffusion=sched)
            row["enc_ms"] = (time.perf_counter() - t) / 5 * 1e3
            rows.append(row)
    return rows


def avalanche_study(trials=4000):
    """Many single-pixel changes on a 256 x 256 crop: NPCR/UACI moments and lost-avalanche events."""
    P = np.ascontiguousarray(skd.camera()[128:384, 128:384])
    ke = kenc(P, "avalanche")
    n = P.size
    lo = mt.npcr_critical(n)
    ul, uh = mt.uaci_critical(n)
    rows = []
    for sched in ("FB1", "FBF1", "FB2", "FBF2", "additive"):
        rng = rng_for("avalanche", sched)
        c0 = cp.core_encrypt(P, ke, diffusion=sched)
        r = np.array([mt.npcr_uaci(c0, cp.core_encrypt(one_pixel(P, rng), ke, diffusion=sched))
                      for _ in range(trials)])
        rows.append(dict(schedule=sched, trials=trials, npcr_mean=r[:, 0].mean(), npcr_sd=r[:, 0].std(),
                         uaci_mean=r[:, 1].mean(), uaci_sd=r[:, 1].std(), npcr_pass=(r[:, 0] > lo).mean(),
                         uaci_pass=((r[:, 1] > ul) & (r[:, 1] < uh)).mean(), lost_avalanche=int((r[:, 0] < 99).sum())))
    rng = rng_for("avalanche", "random")
    r = np.array([mt.npcr_uaci(rng.integers(0, 256, P.shape), rng.integers(0, 256, P.shape)) for _ in range(trials)])
    rows.append(dict(schedule="independent random images", trials=trials, npcr_mean=r[:, 0].mean(),
                     npcr_sd=r[:, 0].std(), uaci_mean=r[:, 1].mean(), uaci_sd=r[:, 1].std(),
                     npcr_pass=(r[:, 0] > lo).mean(), uaci_pass=((r[:, 1] > ul) & (r[:, 1] < uh)).mean(),
                     lost_avalanche=int((r[:, 0] < 99).sum())))
    return rows


def occlusion_recovery(P, ke, R, B, scrambler, frac_hw):
    H, W = P.shape[:2]
    C = 1 if P.ndim == 2 else P.shape[2]
    fh, fw = frac_hw
    h, w = int(round(H * fh)), int(round(W * fw))
    bad = np.zeros((H, W, C), dtype=bool)
    bad[(H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w] = True
    c = cp.core_encrypt(P, ke, R, B, scrambler)
    cb = c.copy()
    cb[bad.reshape(c.shape)] = 0
    d = cp.core_decrypt(cb, ke, R, B, scrambler)
    m = cp.error_mask(bad, ke, H, W, C, R, B, scrambler).reshape(P.shape)
    rec = rc.harmonic_inpaint(d, m)
    return dict(psnr=mt.psnr(P, rec), ssim=mt.ssim(P, rec), isolation=rc.isolation_ratio(m),
                largest_hole=rc.largest_hole(m), mask_frac=float(m.mean()))


def scrambler_ablation(imgs):
    rows = []
    for name, P in imgs.items():
        H, W = P.shape[:2]
        C = 1 if P.ndim == 2 else P.shape[2]
        for scr in cp.SCRAMBLERS:
            if scr == "qp2024" and H != W:
                continue
            ke = kenc(P, name)
            t = time.perf_counter()
            for _ in range(5):
                cp.round_keys(ke, H, W, C, 16, 0, scr)
            tk = (time.perf_counter() - t) / 5 * 1e3
            for lv, fr in (("1/16", (0.25, 0.25)), ("1/4", (0.5, 0.5))):
                r = occlusion_recovery(P, ke, 1, 16, scr, fr)
                rows.append(dict(image=name, scrambler=scr, occlusion=lv, round_key_ms=tk, **r))
    return rows


def tile_ablation(imgs):
    rows = []
    for name, P in imgs.items():
        for B in (4, 8, 16, 32, 64):
            ke = kenc(P, name, B=B)
            for lv, fr in (("1/16", (0.25, 0.25)), ("1/4", (0.5, 0.5))):
                r = occlusion_recovery(P, ke, 1, B, "quadpath", fr)
                rows.append(dict(image=name, B=B, occlusion=lv, **r))
    return rows


def rounds_ablation(imgs):
    rows = []
    for name, P in imgs.items():
        for R in (1, 2):
            ke = kenc(P, name, R=R)
            t = time.perf_counter()
            for _ in range(5):
                cp.core_encrypt(P, ke, R)
            te = (time.perf_counter() - t) / 5 * 1e3
            c = cp.core_encrypt(P, ke, R)
            cb = c.copy()
            cb.flat[c.size // 3] ^= 1
            exp_ = int((cp.core_decrypt(cb, ke, R) != P).sum())
            for lv, fr in (("1/16", (0.25, 0.25)), ("1/4", (0.5, 0.5))):
                r = occlusion_recovery(P, ke, R, 16, "quadpath", fr)
                rows.append(dict(image=name, R=R, occlusion=lv, enc_ms=te, single_byte_expansion=exp_, **r))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    imgs = test_images()
    t0 = time.time()
    pd.DataFrame(avalanche_study()).to_csv(os.path.join(RES, "ablation_avalanche.csv"), index=False)
    print("avalanche", time.time() - t0, flush=True)
    pd.DataFrame(diffusion_ablation(imgs, a.trials)).to_csv(os.path.join(RES, "ablation_diffusion.csv"), index=False)
    print("diffusion", time.time() - t0, flush=True)
    pd.DataFrame(scrambler_ablation(imgs)).to_csv(os.path.join(RES, "ablation_scrambler.csv"), index=False)
    print("scrambler", time.time() - t0, flush=True)
    pd.DataFrame(tile_ablation(imgs)).to_csv(os.path.join(RES, "ablation_tile.csv"), index=False)
    print("tile", time.time() - t0, flush=True)
    pd.DataFrame(rounds_ablation(imgs)).to_csv(os.path.join(RES, "ablation_rounds.csv"), index=False)
    print("rounds", time.time() - t0, flush=True)
