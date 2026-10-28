"""Statistical, differential and key-sensitivity analysis on the full image set.

Outputs
  results/stats_per_image.csv        entropy, local entropy, chi-square, adjacent correlations (plain + 4 ciphers)
  results/differential_per_image.csv NPCR/UACI for single-pixel changes under a FIXED nonce (nonce misuse)
  results/sensitivity_per_image.csv  key / nonce sensitivity and non-determinism under fresh nonces

  python scripts/exp_security.py [--trials 50]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from common import MASTER_KEY, RES, images, key_for, nonce_for, rng_for

from qpae import baselines as bl
from qpae import cipher as cp
from qpae import data
from qpae import metrics as mt


def ours(P, nonce, key=MASTER_KEY):
    return cp.encrypt(P, key, nonce).cipher


def flip_bit(b, i):
    a = bytearray(b)
    a[i // 8] ^= 1 << (i % 8)
    return bytes(a)


SCHEMES = {
    "QuadPath-AE": lambda P, name: ours(P, nonce_for("stats", name)),
    "QUAD-PATH'24": lambda P, name: bl.qp2024_encrypt(P),
    "LOG-PD": lambda P, name: bl.logpd_encrypt(P, key_for("logpd")),
    "AES-256-CTR": lambda P, name: bl.aes_ctr(P, key_for("aes"), nonce_for("stats-aes", name)),
}


def stats_rows(name, P):
    rows = []
    r = mt.cipher_stats(P)
    rows.append(dict(image=name, group=data.group_of(name), scheme="Plain", **r))
    for s, f in SCHEMES.items():
        C = f(P, name)
        rows.append(dict(image=name, group=data.group_of(name), scheme=s, **mt.cipher_stats(C)))
    return rows


def one_pixel_change(P, rng):
    Q = P.copy()
    i = rng.integers(0, P.shape[0])
    j = rng.integers(0, P.shape[1])
    if P.ndim == 2:
        Q[i, j] = (int(Q[i, j]) + 1) % 256
    else:
        c = rng.integers(0, P.shape[2])
        Q[i, j, c] = (int(Q[i, j, c]) + 1) % 256
    return Q


def differential_rows(name, P, trials):
    rng = rng_for("diff", name)
    n = P.size
    lo = mt.npcr_critical(n)
    ul, uh = mt.uaci_critical(n)
    nonce = nonce_for("diff", name)
    fixed = {
        "QuadPath-AE": lambda X: ours(X, nonce),
        "QUAD-PATH'24": bl.qp2024_encrypt,
        "LOG-PD": lambda X: bl.logpd_encrypt(X, key_for("logpd")),
        "AES-256-CTR": lambda X: bl.aes_ctr(X, key_for("aes"), nonce),
    }
    base = {s: f(P) for s, f in fixed.items()}
    rows = []
    for t in range(trials):
        Q = one_pixel_change(P, rng)
        for s, f in fixed.items():
            npcr, uaci = mt.npcr_uaci(base[s], f(Q))
            rows.append(dict(image=name, group=data.group_of(name), scheme=s, trial=t, npcr=npcr, uaci=uaci,
                             npcr_pass=npcr > lo, uaci_pass=ul < uaci < uh, npcr_crit=lo, uaci_lo=ul, uaci_hi=uh))
    return rows


def sensitivity_rows(name, P, trials):
    rng = rng_for("sens", name)
    nonce = nonce_for("sens", name)
    s0 = cp.encrypt(P, MASTER_KEY, nonce)
    rows = []
    for t in range(trials):
        kb = int(rng.integers(0, 256))
        k2 = flip_bit(MASTER_KEY, kb)
        c_k2 = cp.encrypt(P, k2, nonce).cipher
        e_npcr, e_uaci = mt.npcr_uaci(s0.cipher, c_k2)
        ke2, _ = cp.derive_keys(k2, nonce, s0.header)
        wrong = cp.core_decrypt(s0.cipher, ke2)
        d_npcr, d_uaci = mt.npcr_uaci(P, wrong)
        nb = int(rng.integers(0, 128))
        c_n2 = cp.encrypt(P, MASTER_KEY, flip_bit(nonce, nb)).cipher
        n_npcr, n_uaci = mt.npcr_uaci(s0.cipher, c_n2)
        fresh = cp.encrypt(P, MASTER_KEY, nonce_for("fresh", name, t)).cipher
        f_npcr, f_uaci = mt.npcr_uaci(s0.cipher, fresh)
        rows.append(dict(image=name, group=data.group_of(name), trial=t, key_bit=kb, nonce_bit=nb,
                         enc_key_npcr=e_npcr, enc_key_uaci=e_uaci, dec_wrongkey_npcr=d_npcr,
                         dec_wrongkey_uaci=d_uaci, dec_wrongkey_psnr=mt.psnr(P, wrong),
                         enc_nonce_npcr=n_npcr, enc_nonce_uaci=n_uaci, fresh_nonce_npcr=f_npcr,
                         fresh_nonce_uaci=f_uaci))
    # baselines: determinism (same image encrypted twice) and LOG-PD key sensitivity (one bit of x0)
    k = key_for("logpd")
    c1 = bl.logpd_encrypt(P, k)
    c2 = bl.logpd_encrypt(P, flip_bit(k, 63))  # least significant key bit of x0 (perturbation 2^-64 * 0.8)
    rows.append(dict(image=name, group=data.group_of(name), trial=-1, scheme="LOG-PD",
                     enc_key_npcr=mt.npcr_uaci(c1, c2)[0], fresh_nonce_npcr=0.0))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--sens-trials", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    st, df, se = [], [], []
    for name, P in images().items():
        t = time.time()
        st += stats_rows(name, P)
        if P.std() > 0:  # constant images are only used for the statistical tests
            df += differential_rows(name, P, a.trials)
        se += sensitivity_rows(name, P, a.sens_trials)
        print(f"{name:12s} {P.shape} {time.time() - t:.1f}s", flush=True)
    pd.DataFrame(st).to_csv(os.path.join(RES, "stats_per_image.csv"), index=False)
    pd.DataFrame(df).to_csv(os.path.join(RES, "differential_per_image.csv"), index=False)
    pd.DataFrame(se).to_csv(os.path.join(RES, "sensitivity_per_image.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")
