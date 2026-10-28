"""Statistical security metrics for cipher images, with the published critical values.

NPCR/UACI critical values follow Wu, Noonan and Agaian (2011); the local Shannon entropy test follows
Wu et al. (2013) with k = 30 non-overlapping blocks of T_B = 1936 pixels.
"""
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity


def _planes(img):
    img = np.asarray(img)
    return [img] if img.ndim == 2 else [img[:, :, c] for c in range(img.shape[2])]


def entropy(plane):
    h = np.bincount(np.asarray(plane, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def local_entropy(plane, k=30, tb=1936, rng=None):
    """Mean Shannon entropy of k random non-overlapping 44 x 44 blocks (T_B = 1936 pixels)."""
    rng = np.random.default_rng(0) if rng is None else rng
    s = int(round(np.sqrt(tb)))
    H, W = plane.shape
    gh, gw = H // s, W // s
    cells = rng.choice(gh * gw, size=min(k, gh * gw), replace=False)
    vals = [entropy(plane[(c // gw) * s:(c // gw + 1) * s, (c % gw) * s:(c % gw + 1) * s]) for c in cells]
    return float(np.mean(vals))


# Local entropy test of Wu et al. (2013), k = 30, T_B = 1936: ideal mean 7.902469317 and the alpha = 0.001
# acceptance interval as quoted in the literature. scripts/exp_validate.py re-derives the interval by
# Monte-Carlo simulation (results/validation.json, key local_entropy_mc).
LOCAL_ENTROPY_MEAN = 7.902469317
LOCAL_ENTROPY_CRIT_PUBLISHED = {0.001: (7.901515798, 7.903422936)}


def chi_square(plane):
    h = np.bincount(np.asarray(plane, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    e = h.sum() / 256.0
    chi = float(((h - e) ** 2 / e).sum())
    return chi, float(stats.chi2.sf(chi, 255))


CHI2_CRIT_05 = float(stats.chi2.ppf(0.95, 255))  # 293.2478


def adjacent_correlation(plane, direction):
    """Pearson correlation over all adjacent pixel pairs in one direction (h, v, d, a)."""
    x = np.asarray(plane, dtype=np.float64)
    if direction == "h":
        a, b = x[:, :-1], x[:, 1:]
    elif direction == "v":
        a, b = x[:-1, :], x[1:, :]
    elif direction == "d":
        a, b = x[:-1, :-1], x[1:, 1:]
    elif direction == "a":
        a, b = x[:-1, 1:], x[1:, :-1]
    else:
        raise ValueError(direction)
    a = a.ravel()
    b = b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def npcr_uaci(c1, c2):
    c1 = np.asarray(c1, dtype=np.int16)
    c2 = np.asarray(c2, dtype=np.int16)
    npcr = float((c1 != c2).mean() * 100.0)
    uaci = float((np.abs(c1 - c2) / 255.0).mean() * 100.0)
    return npcr, uaci


def npcr_critical(n_pixels, alpha=0.05, F=255):
    """Lower critical value N*_alpha of the NPCR randomness test (in %)."""
    z = stats.norm.ppf(1 - alpha)
    return float((F - z * np.sqrt(F / n_pixels)) / (F + 1) * 100.0)


def uaci_critical(n_pixels, alpha=0.05, F=255):
    """Two-sided acceptance interval (U*-, U*+) of the UACI randomness test (in %)."""
    mu = (F + 2) / (3 * F + 3)
    var = (F + 2) * (F ** 2 + 2 * F + 3) / (18 * (F + 1) ** 2 * n_pixels * F)
    z = stats.norm.ppf(1 - alpha / 2)
    return float((mu - z * np.sqrt(var)) * 100.0), float((mu + z * np.sqrt(var)) * 100.0)


def psnr(a, b, peak=255.0):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(peak ** 2 / mse))


def ssim(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 3:
        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    return float(structural_similarity(a, b, data_range=255))


def cipher_stats(img, rng_seed=0):
    """All per-image statistics averaged over the colour planes."""
    planes = _planes(img)
    rng = np.random.default_rng(rng_seed)
    out = {
        "entropy": np.mean([entropy(p) for p in planes]),
        "local_entropy": np.mean([local_entropy(p, rng=rng) for p in planes]),
        "chi2": np.mean([chi_square(p)[0] for p in planes]),
        "chi2_pass": all(chi_square(p)[0] < CHI2_CRIT_05 for p in planes),
    }
    for d in "hvda":
        out["corr_" + d] = np.mean([adjacent_correlation(p, d) for p in planes])
    return {k: float(v) if not isinstance(v, bool) else v for k, v in out.items()}
