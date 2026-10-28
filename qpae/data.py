"""Test images. Everything ships with scikit-image except Kodak-24 (download: scripts/download_kodak.sh).

Lena is deliberately not used (IEEE and several Elsevier journals no longer accept it).
"""
import os

import numpy as np
from skimage import data as skd
from skimage import io

GRAY = {  # name -> loader; all 8-bit grayscale
    "Cameraman": lambda: skd.camera(),
    "Moon": lambda: skd.moon(),
    "Coins": lambda: skd.coins(),
    "Clock": lambda: skd.clock(),
    "Page": lambda: skd.page(),
    "Brick": lambda: skd.brick(),
    "Gravel": lambda: skd.gravel(),
}
COLOR = {
    "Astronaut": lambda: skd.astronaut(),
    "Coffee": lambda: skd.coffee(),
    "Chelsea": lambda: skd.chelsea(),
    "Rocket": lambda: skd.rocket(),
}
MEDICAL = {  # retina is 1411 x 1411; IHC is a 512 x 512 histology slide
    "Retina": lambda: skd.retina(),
    "IHC": lambda: skd.immunohistochemistry(),
}
SPECIAL = {
    "AllBlack": lambda: np.zeros((512, 512), dtype=np.uint8),
    "AllWhite": lambda: np.full((512, 512), 255, dtype=np.uint8),
}


def kodak(root):
    out = {}
    for i in range(1, 25):
        p = os.path.join(root, f"kodim{i:02d}.png")
        if os.path.exists(p):
            out[f"kodim{i:02d}"] = io.imread(p)[:, :, :3]
    return out


def standard_set(kodak_root=None, include_kodak=True):
    imgs = {}
    for grp in (GRAY, COLOR, MEDICAL, SPECIAL):
        for k, f in grp.items():
            imgs[k] = np.ascontiguousarray(f()).astype(np.uint8)
    if include_kodak and kodak_root:
        imgs.update(kodak(kodak_root))
    return imgs


def group_of(name):
    if name in GRAY:
        return "Gray"
    if name in COLOR:
        return "Color"
    if name in MEDICAL:
        return "Medical"
    if name in SPECIAL:
        return "Special"
    return "Kodak"
