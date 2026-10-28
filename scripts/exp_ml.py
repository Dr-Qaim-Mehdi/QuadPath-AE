"""Deep-learning security evaluation: can a neural network extract knowledge from cipher images?

E1  Neural distinguisher (ciphertext only): CNN separating 64 x 64 cipher patches from uniformly random patches.
    A secure cipher keeps the test accuracy at 50%.
E2  Neural source identification (ciphertext only): CNN predicting which of the 24 Kodak images a 32 x 32 x 3 cipher
    patch was cut from; trained and tested on disjoint regions of the cipher images. Chance level is 1/24.
E3  Neural plaintext recovery (known plaintext, nonce misuse): bit-plane MLP trained on 60 000 pairs of 16 x 16 plain /
    cipher patches that share one key and one nonce (so one fixed "equivalent key" exists), tested on patches of
    unseen images. Pixels are decoded as the expected value under the predicted bit probabilities, so an attacker
    that learns nothing scores at the level of the mean-patch predictor (floor_psnr).

Outputs: results/ml_distinguisher.csv, results/ml_source_id.csv, results/ml_recovery.csv, results/ml_recovery_examples.npz

  python scripts/exp_ml.py [--quick]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import KODAK, MASTER_KEY, RES, key_for, nonce_for, rng_for, save_json
from skimage import data as skd
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

from qpae import baselines as bl
from qpae import cipher as cp
from qpae import data

DEV = torch.device("cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def gray_u8(img):
    return img if img.ndim == 2 else (rgb2gray(img) * 255).round().astype(np.uint8)


# ------------------------------------------------------------------------------------------- ciphers
def encrypt_image(scheme, P, label):
    """Normal-use encryption of a whole image (fresh nonce where the scheme has one)."""
    if scheme == "Plain":
        return P
    if scheme == "QUAD-PATH'24":
        return bl.qp2024_encrypt(P)
    if scheme == "LOG-PD":
        return bl.logpd_encrypt(P, key_for("logpd"))
    if scheme == "AES-256-CTR":
        return bl.aes_ctr(P, key_for("aes"), nonce_for("ml", label))
    if scheme == "P-CTR":
        H, W = P.shape[:2]
        C = 1 if P.ndim == 2 else P.shape[2]
        ke = cp.derive_keys(MASTER_KEY, nonce_for("ml-pctr", label), cp.header_bytes(H, W, C, 1, 16, 32))[0]
        return bl.pctr_encrypt(P, ke)
    if scheme == "QuadPath-AE":
        return cp.encrypt(P, MASTER_KEY, nonce_for("ml", label)).cipher
    raise ValueError(scheme)


class FixedKey:
    """Encryption of many small patches under ONE key and ONE nonce (nonce misuse / equivalent-key setting)."""

    def __init__(self, scheme, h, w):
        self.scheme = scheme
        n = nonce_for("ml-misuse")
        self.ke = cp.derive_keys(MASTER_KEY, n, cp.header_bytes(h, w, 1, 1, 16, 32))[0]
        self.akey, self.nonce = key_for("aes"), n

    def __call__(self, P, i):
        s = self.scheme
        if s == "QUAD-PATH'24":
            return bl.qp2024_encrypt(P)
        if s == "AES-256-CTR (nonce reuse)":
            return bl.aes_ctr(P, self.akey, self.nonce)
        if s == "P-CTR (nonce reuse)":
            return bl.pctr_encrypt(P, self.ke)
        if s == "LOG-PD":
            return bl.logpd_encrypt(P, key_for("logpd"))
        if s == "QuadPath-AE (nonce reuse)":
            return cp.core_encrypt(P, self.ke)
        if s == "QuadPath-AE (fresh nonces)":
            h, w = P.shape
            ke = cp.derive_keys(MASTER_KEY, nonce_for("ml-fresh", i), cp.header_bytes(h, w, 1, 1, 16, 32))[0]
            return cp.core_encrypt(P, ke)
        raise ValueError(s)


# ------------------------------------------------------------------------------------------- models
class SmallCNN(nn.Module):
    def __init__(self, cin, nout, width=32):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())

        w = width
        self.f = nn.Sequential(block(cin, w), nn.MaxPool2d(2), block(w, 2 * w), nn.MaxPool2d(2), block(2 * w, 4 * w))
        self.head = nn.Linear(8 * w, nout)  # global average and global standard deviation of the last block

    def forward(self, x):
        z = self.f(x)
        feat = torch.cat([z.mean(dim=(2, 3)), z.std(dim=(2, 3))], dim=1)
        return self.head(feat)


def fit_classifier(model, Xtr, ytr, Xva, yva, epochs, lr=1e-3, bs=256, binary=False):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx].to(DEV), ytr[idx].to(DEV)
            out = model(xb)
            loss = lossf(out.squeeze(1), yb.float()) if binary else lossf(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate(model, Xva, yva, binary)
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, X, y, binary=False, bs=1024):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs].to(DEV)).cpu()
        pred = (out.squeeze(1) > 0).long() if binary else out.argmax(1)
        correct += int((pred == y[i:i + bs]).sum())
    return correct / len(X)


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def to_tensor(patches):
    x = torch.tensor(np.asarray(patches), dtype=torch.float32) / 255.0
    if x.ndim == 3:
        x = x[:, None]
    else:
        x = x.permute(0, 3, 1, 2)
    return (x - 0.5) / 0.29


# ------------------------------------------------------------------------------------------- E1
def patches_from(img, size, n, rng):
    H, W = img.shape[:2]
    ys = rng.integers(0, H - size + 1, n)
    xs = rng.integers(0, W - size + 1, n)
    return [img[y:y + size, x:x + size] for y, x in zip(ys, xs)]


def e1_distinguisher(schemes, kod, extra, quick):
    names = sorted(kod)
    train_imgs = {k: kod[k] for k in names[:18]} | extra
    test_imgs = {k: kod[k] for k in names[18:]}
    ntr, nte = (4000, 1000) if quick else (20000, 5000)
    rows = []
    for s in schemes:
        set_seed(1)
        rng = rng_for("e1", s)
        pos_tr, pos_te = [], []
        for split, imgs, n, out in (("tr", train_imgs, ntr, pos_tr), ("te", test_imgs, nte, pos_te)):
            per = n // len(imgs) + 1
            for name, P in imgs.items():
                C = encrypt_image(s, P, name)
                if C.ndim == 3:  # use the first channel of colour cipher images
                    C = np.ascontiguousarray(C[:, :, 0])
                out += patches_from(C, 64, per, rng)
            del out[n:]
        neg_tr = rng.integers(0, 256, (ntr, 64, 64), dtype=np.uint8)
        neg_te = rng.integers(0, 256, (nte, 64, 64), dtype=np.uint8)
        X = to_tensor(np.concatenate([pos_tr, neg_tr]))
        y = torch.tensor([1] * ntr + [0] * ntr)
        Xte = to_tensor(np.concatenate([pos_te, neg_te]))
        yte = torch.tensor([1] * nte + [0] * nte)
        nva = len(X) // 10
        perm = torch.randperm(len(X))
        va, tr = perm[:nva], perm[nva:]
        t = time.time()
        m = fit_classifier(SmallCNN(1, 1), X[tr], y[tr], X[va], y[va], epochs=4 if quick else 8, binary=True)
        acc = evaluate(m, Xte, yte, binary=True)
        lo, hi = wilson(acc, len(Xte))
        rows.append(dict(scheme=s, train=len(tr), test=len(Xte), accuracy=acc, ci_lo=lo, ci_hi=hi,
                         seconds=time.time() - t))
        print("E1", rows[-1], flush=True)
    return rows


# ------------------------------------------------------------------------------------------- E2
def e2_source_id(schemes, kod, quick):
    names = sorted(kod)
    rows = []
    for s in schemes:
        set_seed(2)
        Xtr, ytr, Xte, yte = [], [], [], []
        for lab, name in enumerate(names):
            P = kod[name]
            if P.shape[0] > P.shape[1]:  # portrait images: use the transposed orientation for a common grid
                P = np.ascontiguousarray(P.transpose(1, 0, 2))
            C = encrypt_image(s, P, name)
            H, W = C.shape[:2]
            cols = W // 32
            split = int(round(cols * 0.7))
            for r in range(H // 32):
                for c in range(cols):
                    tile = C[32 * r:32 * r + 32, 32 * c:32 * c + 32]
                    (Xtr if c < split else Xte).append(tile)
                    (ytr if c < split else yte).append(lab)
        X, y = to_tensor(np.array(Xtr)), torch.tensor(ytr)
        Xt, yt = to_tensor(np.array(Xte)), torch.tensor(yte)
        perm = torch.randperm(len(X))
        nva = len(X) // 10
        t = time.time()
        m = fit_classifier(SmallCNN(3, len(names)), X[perm[nva:]], y[perm[nva:]], X[perm[:nva]], y[perm[:nva]],
                           epochs=8 if quick else 30)
        acc = evaluate(m, Xt, yt)
        lo, hi = wilson(acc, len(Xt))
        rows.append(dict(scheme=s, classes=len(names), train=len(X) - nva, test=len(Xt), accuracy=acc, ci_lo=lo,
                         ci_hi=hi, chance=1 / len(names), seconds=time.time() - t))
        print("E2", rows[-1], flush=True)
    return rows


# ------------------------------------------------------------------------------------------- E3
class BitMLP(nn.Module):
    def __init__(self, n, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n))

    def forward(self, x):
        return self.net(x)


def to_bits(P):
    return np.unpackbits(np.asarray(P, dtype=np.uint8).reshape(len(P), -1, 1), axis=2).reshape(len(P), -1)


def from_bits(b, shape):
    return np.packbits(b.reshape(len(b), -1, 8).astype(np.uint8), axis=2).reshape((len(b),) + shape)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def e3_recovery(schemes, kod, extra, quick):
    names = sorted(kod)
    gray_tr = [gray_u8(kod[k]) for k in names[:18]] + [gray_u8(v) for v in extra.values()]
    gray_te = [gray_u8(kod[k]) for k in names[18:]]
    ntr, nte = (12000, 1000) if quick else (60000, 5000)
    rng = rng_for("e3")
    Ptr = np.array(sum((patches_from(g, 16, ntr // len(gray_tr) + 1, rng) for g in gray_tr), [])[:ntr])
    Pte = np.array(sum((patches_from(g, 16, nte // len(gray_te) + 1, rng) for g in gray_te), [])[:nte])
    mean_patch = Ptr.mean(axis=0)
    floor_psnr = float(np.mean([psnr(p, mean_patch) for p in Pte]))
    rows, examples = [], {"plain": Pte[:8]}
    for s in schemes:
        set_seed(3)
        enc = FixedKey(s, 16, 16)
        t = time.time()
        Ctr = np.array([enc(p, i) for i, p in enumerate(Ptr)])
        Cte = np.array([enc(p, ntr + i) for i, p in enumerate(Pte)])
        Xtr = torch.tensor(to_bits(Ctr), dtype=torch.float32)
        Ytr = torch.tensor(to_bits(Ptr), dtype=torch.float32)
        Xte = torch.tensor(to_bits(Cte), dtype=torch.float32)
        model = BitMLP(Xtr.shape[1], 1024 if quick else 2048).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        epochs = 6 if quick else 30
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 256):
                idx = perm[i:i + 256]
                loss = lossf(model(Xtr[idx].to(DEV)), Ytr[idx].to(DEV))
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            prob = torch.cat([torch.sigmoid(model(Xte[i:i + 1024].to(DEV))).cpu()
                              for i in range(0, len(Xte), 1024)]).numpy()
        hard = (prob > 0.5).astype(np.uint8)
        bit_acc = float((hard == to_bits(Pte)).mean())
        exact = float(np.mean([np.array_equal(a, b) for a, b in zip(Pte, from_bits(hard, (16, 16)))]))
        # soft decoding: expected pixel value under the predicted bit probabilities (MSB first)
        weights = 2.0 ** np.arange(7, -1, -1)
        rec = np.clip(np.round((prob.reshape(len(prob), -1, 8) * weights).sum(axis=2)), 0, 255).astype(np.uint8)
        rec = rec.reshape(len(prob), 16, 16)
        ps = [psnr(a, b) for a, b in zip(Pte, rec)]
        ss = [structural_similarity(a, b, data_range=255, win_size=7) for a, b in zip(Pte, rec)]
        rows.append(dict(scheme=s, train_pairs=ntr, test_patches=nte, bit_accuracy=bit_acc,
                         psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), exact=exact,
                         floor_psnr=floor_psnr, seconds=time.time() - t))
        examples[s] = rec[:8]
        print("E3", rows[-1], flush=True)
    return rows, examples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    kod = data.kodak(KODAK)
    extra = {"Cameraman": skd.camera(), "Moon": skd.moon(), "Astronaut": skd.astronaut(), "Coffee": skd.coffee(),
             "Brick": skd.brick(), "Gravel": skd.gravel()}
    t0 = time.time()
    e1 = e1_distinguisher(["QUAD-PATH'24", "LOG-PD", "AES-256-CTR", "P-CTR", "QuadPath-AE"], kod, extra, a.quick)
    pd.DataFrame(e1).to_csv(os.path.join(RES, "ml_distinguisher.csv"), index=False)
    e2 = e2_source_id(["Plain", "QUAD-PATH'24", "LOG-PD", "AES-256-CTR", "QuadPath-AE"], kod, a.quick)
    pd.DataFrame(e2).to_csv(os.path.join(RES, "ml_source_id.csv"), index=False)
    e3, ex = e3_recovery(["QUAD-PATH'24", "AES-256-CTR (nonce reuse)", "P-CTR (nonce reuse)", "LOG-PD",
                          "QuadPath-AE (nonce reuse)", "QuadPath-AE (fresh nonces)"], kod, extra, a.quick)
    pd.DataFrame(e3).to_csv(os.path.join(RES, "ml_recovery.csv"), index=False)
    np.savez_compressed(os.path.join(RES, "ml_recovery_examples.npz"), **{k.replace(" ", "_"): v for k, v in ex.items()})
    save_json(dict(device=str(DEV), torch=torch.__version__, quick=a.quick, seconds=time.time() - t0), "ml_env.json")
    print(f"done in {time.time() - t0:.0f}s")
