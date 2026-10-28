"""Security analysis of the original QUAD-PATH transform (Rizvi, Singh and Kumar, ICICC 2024).

QUAD-PATH is a fixed, key-free permutation: (i) anyone can invert it; (ii) the cipher histogram equals the
plain histogram; (iii) a single-pixel change moves a single pixel; (iv) the permutation has a finite order,
so iterating the 'encryption' returns the plaintext. This script quantifies (i)-(iv).

  python scripts/exp_qp2024.py
"""
import os

import numpy as np
import pandas as pd
from common import RES, images, save_json

from qpae import baselines as bl
from qpae import metrics as mt
from qpae.paths import quadpath2024_gather


def cycle_stats(G):
    seen = np.zeros(G.size, dtype=bool)
    lens = []
    for s in range(G.size):
        if seen[s]:
            continue
        L, j = 0, s
        while not seen[j]:
            seen[j] = True
            j = G[j]
            L += 1
        lens.append(L)
    lens = np.array(lens)
    from math import gcd
    order = 1
    for L in np.unique(lens):
        order = order * int(L) // gcd(order, int(L))
    return dict(cycles=int(lens.size), fixed_points=int((lens == 1).sum()), longest_cycle=int(lens.max()),
                order=str(order), log10_order=float(np.log10(float(order))) if order < 10 ** 300 else len(str(order)) - 1)


def mean_displacement(H, W):
    G, _, _ = quadpath2024_gather(H, W)
    src_r, src_c = np.divmod(G, W)
    dst_r, dst_c = np.divmod(np.arange(H * W), W)
    return float(np.mean(np.abs(src_r - dst_r) + np.abs(src_c - dst_c)))


if __name__ == "__main__":
    perm_rows = []
    for H, W in [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (512, 768), (768, 512)]:
        G, _, _ = quadpath2024_gather(H, W)
        perm_rows.append(dict(H=H, W=W, mean_manhattan_displacement=mean_displacement(H, W), **cycle_stats(G)))
        print(perm_rows[-1], flush=True)
    pd.DataFrame(perm_rows).to_csv(os.path.join(RES, "qp2024_permutation.csv"), index=False)

    rows = []
    for name, P in images().items():
        H, W = P.shape[:2]
        C = bl.qp2024_encrypt(P)
        D = bl.qp2024_decrypt(C, H, W)  # no key is needed
        Q = P.copy()
        Q.flat[P.size // 2] = (int(Q.flat[P.size // 2]) + 1) % 256
        npcr, uaci = mt.npcr_uaci(C, bl.qp2024_encrypt(Q))
        hist_equal = bool(np.array_equal(np.bincount(P.ravel(), minlength=256), np.bincount(C.ravel(), minlength=256)))
        sp, sc = mt.cipher_stats(P), mt.cipher_stats(C)
        rows.append(dict(image=name, keyless_decryption_exact=bool(np.array_equal(D, P)), histogram_identical=hist_equal,
                         entropy_plain=sp["entropy"], entropy_cipher=sc["entropy"], corr_h_plain=sp["corr_h"],
                         corr_h_cipher=sc["corr_h"], corr_v_plain=sp["corr_v"], corr_v_cipher=sc["corr_v"],
                         npcr_one_pixel=npcr, uaci_one_pixel=uaci))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "qp2024_images.csv"), index=False)
    print(df.to_string(index=False))
    save_json(dict(all_keyless_exact=bool(df.keyless_decryption_exact.all()),
                   all_hist_identical=bool(df.histogram_identical.all())), "qp2024_summary.json")
