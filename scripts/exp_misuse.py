"""Nonce misuse: two different images encrypted under the same key AND the same nonce.

For a stream cipher (AES-CTR, ChaCha20) C1 XOR C2 = P1 XOR P2, a two-time pad that shows both images.
Metrics per pair: rho = corr(C1 XOR C2, P1 XOR P2), entropy and chi-square of C1 XOR C2, NPCR/UACI(C1, C2),
and the share of the most frequent byte difference C1 - C2 (ideal 1/256 = 0.39 %).

  python scripts/exp_misuse.py
"""
import os

import numpy as np
import pandas as pd
from common import KODAK, MASTER_KEY, RES, key_for, nonce_for
from skimage import data as skd

from qpae import baselines as bl
from qpae import cipher as cp
from qpae import data
from qpae import metrics as mt


def schemes(nonce):
    hdr_cache = {}

    def ke_for(P):
        H, W = P.shape[:2]
        C = 1 if P.ndim == 2 else P.shape[2]
        k = (H, W, C)
        if k not in hdr_cache:
            hdr_cache[k] = cp.derive_keys(MASTER_KEY, nonce, cp.header_bytes(H, W, C, 1, 16, 32))[0]
        return hdr_cache[k]

    akey = key_for("aes")
    return {
        "QuadPath-AE": lambda P: cp.encrypt(P, MASTER_KEY, nonce).cipher,
        "QP-AE (additive chaining)": lambda P: cp.core_encrypt(P, ke_for(P), diffusion="additive"),
        "QP-AE (FB1 diffusion)": lambda P: cp.core_encrypt(P, ke_for(P), diffusion="FB1"),
        "P-CTR (scrambling + CTR)": lambda P: bl.pctr_encrypt(P, ke_for(P)),
        "AES-256-CTR": lambda P: bl.aes_ctr(P, akey, nonce),
        "ChaCha20-Poly1305": lambda P: bl.chacha_poly_encrypt(P, akey, nonce[:12])[0],
        "LOG-PD": lambda P: bl.logpd_encrypt(P, key_for("logpd")),
    }


def pair_metrics(P1, P2, C1, C2):
    x = (C1 ^ C2).ravel()
    p = (P1 ^ P2).ravel().astype(np.float64)
    rho = float(np.corrcoef(x.astype(np.float64), p)[0, 1]) if x.std() > 0 else 1.0
    d = ((C1.astype(np.int16) - C2.astype(np.int16)) % 256).astype(np.uint8).ravel()
    npcr, uaci = mt.npcr_uaci(C1, C2)
    return dict(rho_xor=rho, entropy_xor=mt.entropy(x), chi2_xor=mt.chi_square(x)[0],
                npcr=npcr, uaci=uaci, top_diff_share=float(np.bincount(d, minlength=256).max() / d.size),
                entropy_plain_xor=mt.entropy((P1 ^ P2).ravel()))


if __name__ == "__main__":
    kod = data.kodak(KODAK)
    names = sorted(kod)
    by_shape = {}
    for n in names:
        by_shape.setdefault(kod[n].shape, []).append(n)
    pairs = []
    for shp, lst in by_shape.items():
        pairs += [(lst[i], lst[i + 1]) for i in range(0, len(lst) - 1, 2)]
    imgs = dict(kod)
    imgs["Cameraman"], imgs["Moon"] = skd.camera(), skd.moon()
    imgs["Astronaut"], imgs["IHC"] = skd.astronaut(), skd.immunohistochemistry()
    pairs += [("Cameraman", "Moon"), ("Astronaut", "IHC")]
    rows = []
    vis = {}
    for a, b in pairs:
        nonce = nonce_for("misuse", a, b)
        for s, f in schemes(nonce).items():
            C1, C2 = f(imgs[a]), f(imgs[b])
            rows.append(dict(pair=f"{a}/{b}", scheme=s, **pair_metrics(imgs[a], imgs[b], C1, C2)))
            if (a, b) == ("Cameraman", "Moon"):
                vis[s] = C1 ^ C2
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "misuse_per_pair.csv"), index=False)
    np.savez_compressed(os.path.join(RES, "misuse_visual.npz"), P1=imgs["Cameraman"], P2=imgs["Moon"],
                        **{k.replace(" ", "_"): v for k, v in vis.items()})
    print(df.groupby("scheme")[["rho_xor", "entropy_xor", "npcr", "uaci", "top_diff_share"]].mean().to_string())
