"""Known-answer test vectors for QuadPath-AE v1 (results/kat.json) and a check against them.

  python scripts/make_kat.py          # (re)generate
  python scripts/make_kat.py --check  # verify the current implementation against results/kat.json
"""
import hashlib
import json
import os
import sys

import numpy as np
from common import RES

from qpae import cipher as cp

KEY = bytes(range(32))
NONCE = bytes(range(16))


def vectors():
    rng = np.random.default_rng(2026)
    cases = {
        "gray_4x5_ramp": np.arange(20, dtype=np.uint8).reshape(4, 5),
        "rgb_8x8_random": rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
        "gray_64x48_zeros": np.zeros((64, 48), dtype=np.uint8),
        "rgb_37x53_random": rng.integers(0, 256, (37, 53, 3), dtype=np.uint8),
    }
    out = {}
    for name, P in cases.items():
        s = cp.encrypt(P, KEY, NONCE)
        assert np.array_equal(cp.decrypt(s, KEY), P)
        out[name] = dict(shape=list(P.shape), plain_sha256=hashlib.sha256(P.tobytes()).hexdigest(),
                         header_hex=s.header.hex(), cipher_sha256=hashlib.sha256(s.cipher.tobytes()).hexdigest(),
                         cipher_hex=s.cipher.tobytes().hex() if P.size <= 192 else None,
                         tile_tags_hex=s.ttags.hex(), global_tag_hex=s.gtag.hex(),
                         sealed_sha256=hashlib.sha256(s.to_bytes()).hexdigest())
    return dict(version=1, key_hex=KEY.hex(), nonce_hex=NONCE.hex(), params=dict(R=1, B=16, Ba=32, diffusion="FBF2"),
                vectors=out)


if __name__ == "__main__":
    path = os.path.join(RES, "kat.json")
    v = vectors()
    if "--check" in sys.argv:
        ref = json.load(open(path))
        ok = ref == json.loads(json.dumps(v))
        print("KAT", "OK" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)
    json.dump(v, open(path, "w"), indent=1)
    print("wrote", path)
