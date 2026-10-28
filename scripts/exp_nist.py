"""NIST SP 800-22 on cipher images: 100 sequences of 10^6 bits per scheme and plaintext source.

Sources: (a) the 24 Kodak images, each encrypted under its own fresh nonce, concatenated;
         (b) a constant all-black 512 x 512 image encrypted repeatedly under fresh nonces (chosen-plaintext
             worst case). Baselines on (a): LOG-PD and QUAD-PATH'24.

  python scripts/exp_nist.py [--seqs 100]
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from common import KODAK, MASTER_KEY, RES, key_for, nonce_for

from qpae import baselines as bl
from qpae import cipher as cp
from qpae import data, nist

NBITS = 1_000_000


def bitstream(chunks, need_bytes):
    buf = bytearray()
    for c in chunks:
        buf += np.ascontiguousarray(c).tobytes()
        if len(buf) >= need_bytes:
            break
    if len(buf) < need_bytes:
        raise RuntimeError("not enough cipher bytes")
    return np.unpackbits(np.frombuffer(bytes(buf[:need_bytes]), dtype=np.uint8))


def kodak_chunks(scheme):
    imgs = data.kodak(KODAK)
    while True:  # cycle over the set with new nonces if more bits are needed
        for rep in range(1000):
            for name, P in imgs.items():
                if scheme == "QuadPath-AE":
                    yield cp.encrypt(P, MASTER_KEY, nonce_for("nist", name, rep)).cipher
                elif scheme == "LOG-PD":
                    yield bl.logpd_encrypt(P, key_for("logpd", rep))
                elif scheme == "QUAD-PATH'24":
                    yield bl.qp2024_encrypt(P)


def black_chunks():
    P = np.zeros((512, 512), dtype=np.uint8)
    i = 0
    while True:
        yield cp.encrypt(P, MASTER_KEY, nonce_for("nist-black", i)).cipher
        i += 1


def _run(seq):
    return nist.run_all(seq)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=100)
    a = ap.parse_args()
    need = a.seqs * NBITS // 8
    jobs = {
        "QuadPath-AE | Kodak-24": lambda: bitstream(kodak_chunks("QuadPath-AE"), need),
        "QuadPath-AE | all-black": lambda: bitstream(black_chunks(), need),
        "LOG-PD | Kodak-24": lambda: bitstream(kodak_chunks("LOG-PD"), need),
        "QUAD-PATH'24 | Kodak-24": lambda: bitstream(kodak_chunks("QUAD-PATH'24"), need),
    }
    rows = []
    for label, make in jobs.items():
        t = time.time()
        bits = make()
        seqs = [bits[i * NBITS:(i + 1) * NBITS] for i in range(a.seqs)]
        with ProcessPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(_run, seqs))
        for r in nist.summarize(res):
            scheme, source = label.split(" | ")
            rows.append(dict(scheme=scheme, source=source, **r))
        print(f"{label}: {time.time() - t:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "nist_summary.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
