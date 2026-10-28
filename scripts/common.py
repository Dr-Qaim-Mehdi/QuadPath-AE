"""Shared set-up for all experiments: reproducible keys/nonces, image set, output folders."""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qpae import data  # noqa: E402

RES = os.path.join(ROOT, "results")
KODAK = os.path.join(ROOT, "data", "kodak")
os.makedirs(RES, exist_ok=True)

# Experiments use a fixed master key and nonces derived from labels so every number is reproducible.
# (A deployment draws the nonce from os.urandom for every image.)
MASTER_KEY = hashlib.sha256(b"QuadPath-AE experiments, master key, 2026").digest()


def nonce_for(*labels):
    return hashlib.sha256(b"nonce|" + "|".join(map(str, labels)).encode()).digest()[:16]


def key_for(*labels):
    return hashlib.sha256(b"key|" + "|".join(map(str, labels)).encode()).digest()


def rng_for(*labels):
    seed = int.from_bytes(hashlib.sha256(b"rng|" + "|".join(map(str, labels)).encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def images(include_kodak=True):
    return data.standard_set(KODAK, include_kodak)


def save_json(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)
