"""Validation: (1) NIST SP 800-22 implementation vs. the Appendix-B reference P-values for data.e;
(2) Monte-Carlo check of the NPCR/UACI and local-entropy critical values with truly random images;
(3) exhaustive round-trip / error-mask unit tests of QuadPath-AE.

  python scripts/exp_validate.py
"""
import os
import time

import numpy as np
import pandas as pd
from common import RES, key_for, rng_for, save_json

from qpae import cipher as cp
from qpae import metrics as mt
from qpae import nist
from qpae import paths


def nist_e():
    e = nist.e_bits()
    r = nist.run_all(e)
    rows = []
    for test, refs in nist.E_REFERENCE.items():
        for idx, ref in refs:
            ours = r[test][idx]
            rows.append(dict(test=test, subtest=idx, reference=ref, ours=round(ours, 6),
                             abs_diff=abs(round(ours, 6) - ref)))
    nu = nist._overlap_counts(e, 9, 1032, 968).tolist()
    return pd.DataFrame(rows), nu


def null_distributions(trials=2000, shape=(256, 256)):
    """NPCR/UACI of independent uniform images and local entropy of uniform images."""
    rng = rng_for("null")
    n = shape[0] * shape[1]
    np_, ua, le = [], [], []
    for _ in range(trials):
        a = rng.integers(0, 256, shape, dtype=np.uint8)
        b = rng.integers(0, 256, shape, dtype=np.uint8)
        x, y = mt.npcr_uaci(a, b)
        np_.append(x)
        ua.append(y)
    np_, ua = np.array(np_), np.array(ua)
    lo = mt.npcr_critical(n)
    ul, uh = mt.uaci_critical(n)
    return dict(
        trials=trials, shape=list(shape),
        npcr_mean=np_.mean(), npcr_theory=255 / 256 * 100, npcr_crit_05=lo, npcr_reject_rate=(np_ < lo).mean(),
        uaci_mean=ua.mean(), uaci_theory=257 / 768 * 100, uaci_interval_05=[ul, uh],
        uaci_reject_rate=((ua < ul) | (ua > uh)).mean(),
    )


def local_entropy_mc(trials=100_000, k=30, tb=1936):
    """Null distribution of the k-block mean entropy for uniformly random pixels (blocks drawn directly)."""
    rng = rng_for("local-entropy")
    vals = np.empty(trials)
    for t in range(trials):
        blk = rng.integers(0, 256, (k, tb), dtype=np.uint8)
        h = np.zeros(k)
        for i in range(k):
            c = np.bincount(blk[i], minlength=256)
            p = c[c > 0] / tb
            h[i] = -(p * np.log2(p)).sum()
        vals[t] = h.mean()
    el, eh = mt.LOCAL_ENTROPY_CRIT_PUBLISHED[0.001]
    # analytic: plug-in entropy ~ log2 K - chi2_{K-1} / (2 T_B ln 2), so sd(mean) ~ sqrt(2 (K-1)) / (2 T_B ln2 sqrt(k))
    sd_analytic = np.sqrt(2 * 255) / (2 * tb * np.log(2) * np.sqrt(k))
    return dict(trials=trials, mean=vals.mean(), sd=vals.std(ddof=1), sd_analytic=sd_analytic,
                interval_05=[float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
                interval_001=[float(np.quantile(vals, 0.0005)), float(np.quantile(vals, 0.9995))],
                published_interval_001=[el, eh], published_mean=mt.LOCAL_ENTROPY_MEAN,
                reject_rate_published_001=float(((vals < el) | (vals > eh)).mean()))


def unit_tests():
    rng = rng_for("unit")
    n_paths = 0
    for h in range(1, 33):
        for w in range(1, 33):
            for pt in range(4):
                for g in range(8):
                    oi = np.empty(h * w, np.int64)
                    oj = np.empty(h * w, np.int64)
                    paths.tile_order(pt, g, h, w, oi, oj)
                    assert np.unique(oi * w + oj).size == h * w
                    n_paths += 1
    n_rt = 0
    shapes = [(1, 1), (7, 3), (16, 16), (31, 17), (64, 48, 3), (100, 131, 3), (256, 256), (300, 451, 3)]
    for shape in shapes:
        P = rng.integers(0, 256, shape, dtype=np.uint8)
        for R in (1, 2):
            for B in (4, 8, 16, 32):
                for scr in cp.SCRAMBLERS:
                    if scr == "qp2024" and shape[0] != shape[1]:
                        continue
                    ke = key_for("unit", shape, R, B, scr)
                    c = cp.core_encrypt(P, ke, R, B, scr)
                    assert np.array_equal(cp.core_decrypt(c, ke, R, B, scr), P)
                    bad = rng.random(P.shape) < 0.02
                    cb = c.copy()
                    cb[bad] ^= 0xA5
                    H, W = P.shape[:2]
                    C = 1 if P.ndim == 2 else P.shape[2]
                    M = cp.error_mask(bad.reshape(H, W, C), ke, H, W, C, R, B, scr).reshape(P.shape)
                    assert ((cp.core_decrypt(cb, ke, R, B, scr) != P) & ~M).sum() == 0
                    n_rt += 1
    return dict(path_orders_checked=n_paths, roundtrip_and_mask_cases=n_rt)


if __name__ == "__main__":
    t = time.time()
    df, nu = nist_e()
    df.to_csv(os.path.join(RES, "validation_nist_e.csv"), index=False)
    print(df.to_string(index=False))
    out = dict(nist_e_max_abs_diff=float(df.abs_diff.max()), overlapping_template_nu=nu,
               null=null_distributions(), local_entropy_mc=local_entropy_mc(), unit=unit_tests(),
               seconds=time.time() - t)
    save_json(out, "validation.json")
    print(out)
