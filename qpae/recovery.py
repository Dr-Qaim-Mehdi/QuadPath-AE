"""Recovery of decrypted images from corrupted cipher images.

harmonic_inpaint : exact discrete harmonic (Laplace) interpolation of the masked pixels with the known
                   pixels as Dirichlet data; one sparse solve per channel (parameter-free, deterministic).
median3          : blind 3 x 3 median filter (the usual post-filter in the image-encryption literature).
"""
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve


def _harmonic_plane(img, mask):
    H, W = img.shape
    out = img.astype(np.float64).copy()
    idx = -np.ones((H, W), dtype=np.int64)
    ys, xs = np.nonzero(mask)
    n = ys.size
    if n == 0:
        return img.copy()
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    deg = np.zeros(n)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        k = np.nonzero(inside)[0]
        deg[k] += 1
        nyk, nxk = ny[k], nx[k]
        unk = mask[nyk, nxk]
        rows.append(k[unk])
        cols.append(idx[nyk[unk], nxk[unk]])
        vals.append(-np.ones(int(unk.sum())))
        kk = k[~unk]
        np.add.at(rhs, kk, out[nyk[~unk], nxk[~unk]])
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(deg)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    sol = spsolve(A.tocsc(), rhs)
    out[ys, xs] = sol
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def harmonic_inpaint(img, mask):
    img = np.asarray(img, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    if img.ndim == 2:
        return _harmonic_plane(img, mask if mask.ndim == 2 else mask[:, :, 0])
    m3 = mask if mask.ndim == 3 else np.repeat(mask[:, :, None], img.shape[2], axis=2)
    return np.stack([_harmonic_plane(img[:, :, c], m3[:, :, c]) for c in range(img.shape[2])], axis=2)


def median3(img):
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 2:
        return ndimage.median_filter(img, size=3, mode="reflect")
    return np.stack([ndimage.median_filter(img[:, :, c], size=3, mode="reflect") for c in range(img.shape[2])],
                    axis=2)


def isolation_ratio(mask):
    """Fraction of masked pixels none of whose 8-neighbours is masked (1 = perfectly scattered)."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 3:
        m = m.any(axis=2)
    if m.sum() == 0:
        return float("nan")
    nb = ndimage.convolve(m.astype(np.int32), np.ones((3, 3), dtype=np.int32), mode="constant") - m
    return float((m & (nb == 0)).sum() / m.sum())


def largest_hole(mask):
    """Area (pixels) of the largest 8-connected masked component."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 3:
        m = m.any(axis=2)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return 0
    return int(np.bincount(lab.ravel())[1:].max())
