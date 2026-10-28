"""Calibration of the NIST SP 800-22 decision rule with a reference generator.

The AES-256-CTR keystream (OpenSSL) is used as a stand-in for a truly random source: 10 independent batches of
100 sequences of 10^6 bits. For each batch we record how many of the 15 tests pass and how many of the 148
non-overlapping-template sub-tests fail; this shows how many sub-test failures a random source produces.

  python scripts/exp_nist_null.py [--batches 10]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from common import RES, key_for
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from qpae import nist

NBITS = 1_000_000


def keystream_bits(batch, nbytes):
    enc = Cipher(algorithms.AES(key_for("nist-null", batch)), modes.CTR(bytes(16))).encryptor()
    return np.unpackbits(np.frombuffer(enc.update(bytes(nbytes)), dtype=np.uint8))


def _run(seq):
    return nist.run_all(seq)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--seqs", type=int, default=100)
    a = ap.parse_args()
    rows, sub = [], []
    for b in range(a.batches):
        bits = keystream_bits(b, a.seqs * NBITS // 8)
        seqs = [bits[i * NBITS:(i + 1) * NBITS] for i in range(a.seqs)]
        with ProcessPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(_run, seqs))
        summ = nist.summarize(res)
        for r in summ:
            sub.append(dict(batch=b, **r))
        nt = [r for r in summ if r["test"] == "NonOverlappingTemplate"][0]
        rows.append(dict(batch=b, tests_passed=int(sum(bool(r.get("passed")) for r in summ)),
                         nonoverlap_subtests_failed=int(nt["subtests"] - nt["subtests_passed"])))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "nist_null_batches.csv"), index=False)
    pd.DataFrame(sub).to_csv(os.path.join(RES, "nist_null_summary.csv"), index=False)
    print(pd.DataFrame(rows).describe())
