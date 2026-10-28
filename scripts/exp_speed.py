"""Single-thread run time of encryption and decryption (median of repeated runs after warm-up).

  python scripts/exp_speed.py [--reps 15]
"""
import argparse
import os
import platform
import time

import numpy as np
import pandas as pd
from common import MASTER_KEY, RES, key_for, nonce_for, rng_for, save_json

from qpae import baselines as bl
from qpae import cipher as cp


def timeit(fn, reps):
    fn()
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def breakdown(P, reps):
    H, W = P.shape[:2]
    C = 1 if P.ndim == 2 else P.shape[2]
    nonce = nonce_for("speed")
    hdr = cp.header_bytes(H, W, C, 1, 16, 32)
    ke, km = cp.derive_keys(MASTER_KEY, nonce, hdr)
    rk = cp.round_keys(ke, H, W, C, 16, 0)
    x = np.ascontiguousarray(P).ravel()
    X = P if P.ndim == 3 else P[:, :, None]
    Cc = cp.core_encrypt(P, ke)
    tt = cp.tile_tags(km, hdr, nonce, Cc, 32)
    return {
        "key derivation (HKDF)": timeit(lambda: cp.derive_keys(MASTER_KEY, nonce, hdr), reps),
        "round keys (ChaCha20 keystreams, permutations, S-box)": timeit(lambda: cp.round_keys(ke, H, W, C, 16, 0), reps),
        "QuadPath scrambling (gather)": timeit(
            lambda: np.stack([X[:, :, c].ravel()[rk.gathers[c]] for c in range(C)], axis=1), reps),
        "FBF diffusion": timeit(lambda: cp.diffuse(x, rk), reps),
        "FBF undiffusion": timeit(lambda: cp.undiffuse(x, rk), reps),
        "tile tags (HMAC-SHA256)": timeit(lambda: cp.tile_tags(km, hdr, nonce, Cc, 32), reps),
        "global tag (HMAC-SHA256)": timeit(lambda: cp.global_tag(km, hdr, nonce, Cc.tobytes(), tt), reps),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()
    rng = rng_for("speed")
    key = key_for("aes")
    n16, n12 = nonce_for("s16"), nonce_for("s12")[:12]
    rows = []
    for side in (256, 512, 1024, 2048):
        for C in (1, 3):
            shape = (side, side) if C == 1 else (side, side, 3)
            P = rng.integers(0, 256, shape, dtype=np.uint8)
            s = cp.encrypt(P, MASTER_KEY, n16)
            cl = bl.logpd_encrypt(P, key)
            cq = bl.qp2024_encrypt(P)
            cg, tg = bl.aes_gcm_encrypt(P, key, n12)
            cc, tc = bl.chacha_poly_encrypt(P, key, n12)
            jobs = {
                ("QuadPath-AE", "enc"): lambda: cp.encrypt(P, MASTER_KEY, n16),
                ("QuadPath-AE", "dec"): lambda: cp.decrypt(s, MASTER_KEY),
                ("LOG-PD", "enc"): lambda: bl.logpd_encrypt(P, key),
                ("LOG-PD", "dec"): lambda: bl.logpd_decrypt(cl, key),
                ("QUAD-PATH'24", "enc"): lambda: bl.qp2024_encrypt(P),
                ("QUAD-PATH'24", "dec"): lambda: bl.qp2024_decrypt(cq, side, side),
                ("AES-256-CTR", "enc"): lambda: bl.aes_ctr(P, key, n16),
                ("AES-256-GCM", "enc"): lambda: bl.aes_gcm_encrypt(P, key, n12),
                ("AES-256-GCM", "dec"): lambda: bl.aes_gcm_decrypt(cg, tg, key, n12),
                ("ChaCha20-Poly1305", "enc"): lambda: bl.chacha_poly_encrypt(P, key, n12),
                ("ChaCha20-Poly1305", "dec"): lambda: bl.chacha_poly_decrypt(cc, tc, key, n12),
            }
            for (scheme, op), fn in jobs.items():
                reps = a.reps if side <= 1024 else max(5, a.reps // 3)
                t = timeit(fn, reps)
                rows.append(dict(scheme=scheme, op=op, side=side, channels=C, bytes=P.nbytes, seconds=t,
                                 ms=t * 1e3, MBps=P.nbytes / t / 1e6))
            print(side, C, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "speed.csv"), index=False)
    P = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    bd = {k: v * 1e3 for k, v in breakdown(P, a.reps).items()}
    env = dict(machine=platform.machine(), processor=platform.processor(), python=platform.python_version(),
               platform=platform.platform(), numpy=np.__version__)
    try:
        import cryptography
        import numba
        env.update(numba=numba.__version__, cryptography=cryptography.__version__)
    except ImportError:
        pass
    try:  # macOS power state: battery + Low Power Mode lowers the CPU clock substantially
        import subprocess
        pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
        bt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        env["low_power_mode"] = any(ln.split()[-1] == "1" for ln in pm.splitlines() if "lowpowermode" in ln)
        env["power_source"] = "battery" if "Battery Power" in bt else "AC"
    except (OSError, IndexError):
        pass
    save_json(dict(breakdown_ms_512x512x3=bd, env=env), "speed_breakdown.json")
    print(df.pivot_table(index=["scheme", "op"], columns=["side", "channels"], values="MBps").round(1).to_string())
    print(bd)
