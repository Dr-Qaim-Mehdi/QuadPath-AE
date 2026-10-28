"""All figures of the paper (PDF + 600-dpi PNG) from the library and results/*.csv.

  python scripts/make_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import KODAK, MASTER_KEY, RES, ROOT, key_for, nonce_for  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from skimage import data as skd  # noqa: E402

from qpae import baselines as bl  # noqa: E402
from qpae import cipher as cp  # noqa: E402
from qpae import data  # noqa: E402
from qpae import metrics as mt  # noqa: E402
from qpae import paths  # noqa: E402
from qpae import recovery as rc  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# Fixed entity -> color mapping (reference categorical palette, slots in order), used in every chart.
COL = {"QuadPath-AE": "#2a78d6", "AES-256-CTR": "#eb6834", "P-CTR": "#1baf7a", "LOG-PD": "#eda100",
       "QUAD-PATH'24": "#e87ba4", "AES-256-GCM": "#008300", "ChaCha20-Poly1305": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "savefig.dpi": 600, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
})
W1, W2 = 3.5, 7.2  # single / double column width (inches)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    print("wrote", name)


def show(ax, img, title=None):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def grid(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------------------------- Fig. 1
def fig_architecture():
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f3ef", ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=INK, linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK2))

    blue = "#dce9f8"
    # row 1: key schedule
    box(0.5, 34, 19, 8, "Master key $K$ (256 bit),\nfresh nonce $N$ (128 bit),\nheader (size, $R$, $B$, $T$)")
    box(24, 34, 14, 8, "HKDF-Expand\n(SHA-256)\n$(K_e \\| K_m)$", fc=blue)
    box(42, 34, 18, 8, "ChaCha20($K_e$):\none purpose-labeled\nkeystream per secret", fc=blue)
    arrow(19.5, 38, 24, 38)
    arrow(38, 38, 42, 38)
    # row 2: per-image secret parameters
    box(12, 21, 62, 7.5, "Per-image secrets: tile order $\\sigma$, path family $\\tau_t$ and D$_4$ element $g_t$ per tile, "
        "interleaver $(a, b)$,\nkeyed S-box $S$, pass keystreams $k^{(1)}, k^{(2)}, k^{(3)}$ and initial values", fc="#ffffff")
    arrow(51, 34, 51, 28.7)
    # row 3: data path
    box(1, 4, 11, 10, "Plain image\n$P$ ($H \\times W \\times C$)")
    box(15, 4, 18, 10, "QuadPath tile\nscrambling (per channel)\nzig-zag | snake | spiral |\nHilbert, $\\times$ D$_4$", fc=blue)
    box(36, 4, 14, 10, "Golden\ninterleaver\n$o \\mapsto (ao+b)$ mod $L$", fc=blue)
    box(53, 4, 18, 10, "FBF diffusion\nforward, backward,\nforward, S-box $S$", fc=blue)
    box(74, 4, 11, 10, "Cipher image\n$C$ (same size\nand type)")
    box(88, 4, 11, 10, "Tile tags +\n128-bit tag\nHMAC($K_m$)")
    for x0, x1 in ((12, 15), (33, 36), (50, 53), (71, 74), (85, 88)):
        arrow(x0, 9, x1, 9)
    for xc in (24, 43, 62):
        arrow(xc, 21, xc, 14.3)
    ax.plot([31, 31, 93.5], [33.7, 31.2, 31.2], color=INK2, lw=0.6, ls=(0, (2, 2)))
    arrow(93.5, 31.2, 93.5, 14.3)
    ax.text(94.3, 24, "$K_m$", fontsize=6.6, color=INK2)
    save(fig, "fig01_architecture")


# ----------------------------------------------------------------------------------------------- Fig. 2
def fig_paths():
    names = ["Zig-zag ($\\tau$=0)", "Snake ($\\tau$=1)", "Spiral ($\\tau$=2)", "Hilbert ($\\tau$=3)"]
    fig, axes = plt.subplots(2, 6, figsize=(W2, 2.75), gridspec_kw=dict(hspace=0.35, wspace=0.12))
    n = 8
    cmap = plt.get_cmap("Blues")

    def draw(ax, pt, g, title):
        oi = np.empty(n * n, np.int64)
        oj = np.empty(n * n, np.int64)
        paths.tile_order(pt, g, n, n, oi, oj)
        rank = np.empty((n, n))
        rank[oi, oj] = np.arange(n * n)
        ax.imshow(rank, cmap=cmap, vmin=-12, vmax=n * n, interpolation="nearest")
        ax.plot(oj, oi, color=INK, lw=0.7)
        ax.plot(oj[0], oi[0], "o", ms=3, color="#eb6834")
        ax.plot(oj[-1], oi[-1], "s", ms=3, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=6.6, pad=2)

    for k in range(4):
        draw(axes[0, k], k, 0, names[k])
    # the original QUAD-PATH step (zig-zag read, row refill, rotation) shown as a raster map
    idx = np.arange(n * n).reshape(n, n)
    oi = np.empty(n * n, np.int64)
    oj = np.empty(n * n, np.int64)
    paths._path(0, n, n, oi, oj)
    step = np.rot90(idx[oi, oj].reshape(n, n), 1)
    axes[0, 4].imshow(step, cmap=cmap, vmin=-12, vmax=n * n, interpolation="nearest")
    axes[0, 4].set_title("QUAD-PATH'24, 1 path", fontsize=6.6, pad=2)
    G, _, _ = paths.quadpath2024_gather(n, n)
    axes[0, 5].imshow(G.reshape(n, n), cmap=cmap, vmin=-12, vmax=n * n, interpolation="nearest")
    axes[0, 5].set_title("QUAD-PATH'24, 4 paths", fontsize=6.6, pad=2)
    for a in axes[0, 4:]:
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    for k, g in enumerate([0, 1, 2, 3, 4, 5]):
        draw(axes[1, k], 0, g, f"zig-zag, $g$={g}")
    fig.text(0.5, 0.49, "Row 2: six of the eight D$_4$ variants of the zig-zag path (transpose / row flip / column "
             "flip bits of $g$); color = visiting rank, circle = first, square = last pixel.",
             ha="center", fontsize=6.4, color=INK2)
    save(fig, "fig02_paths")


# ----------------------------------------------------------------------------------------------- Fig. 3
def fig_qp2024():
    P = skd.camera()
    Cq = bl.qp2024_encrypt(P)
    Cq_b = bl.qp2024_encrypt(np.zeros((256, 256), np.uint8))
    s = cp.encrypt(P, MASTER_KEY, nonce_for("fig3"))
    fig = plt.figure(figsize=(W2, 1.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1.45, 1, 1], wspace=0.42)
    show(fig.add_subplot(gs[0]), P, "(a) Plain")
    show(fig.add_subplot(gs[1]), Cq, "(b) QUAD-PATH'24 cipher")
    ax = fig.add_subplot(gs[2])
    hp = np.bincount(P.ravel(), minlength=256)
    hq = np.bincount(Cq.ravel(), minlength=256)
    ax.bar(np.arange(256), hp, width=1.0, color=COL["QUAD-PATH'24"], label="plain")
    ax.plot(np.arange(256), hq, color=INK, lw=0.6, label="QUAD-PATH'24 cipher")
    ax.set_title("(c) Histograms are identical", fontsize=7, pad=2)
    ax.set_xlabel("Gray level")
    ax.set_ylabel("Count (thousands)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}"))
    ax.legend(frameon=False, loc="upper center", fontsize=6, bbox_to_anchor=(0.5, 1.0))
    grid(ax)
    show(fig.add_subplot(gs[3]), bl.qp2024_decrypt(Cq, 512, 512), "(d) Decrypted without key")
    show(fig.add_subplot(gs[4]), s.cipher, "(e) QuadPath-AE cipher")
    save(fig, "fig03_qp2024")
    return Cq_b


# ----------------------------------------------------------------------------------------------- Fig. 4
def fig_visual():
    imgs = [("Cameraman", skd.camera()), ("Astronaut", skd.astronaut()), ("Retina", skd.retina()),
            ("IHC", skd.immunohistochemistry()), ("Page", skd.page()), ("All-black", np.zeros((512, 512), np.uint8))]
    fig, axes = plt.subplots(3, len(imgs), figsize=(W2, 3.75), gridspec_kw=dict(hspace=0.12, wspace=0.06))
    for k, (name, P) in enumerate(imgs):
        s = cp.encrypt(P, MASTER_KEY, nonce_for("fig4", name))
        D = cp.decrypt(s, MASTER_KEY)
        assert np.array_equal(D, P)
        show(axes[0, k], P, name)
        show(axes[1, k], s.cipher)
        show(axes[2, k], D)
    for r, lab in enumerate(["Plain", "Cipher", "Decrypted"]):
        axes[r, 0].set_ylabel(lab, fontsize=7)
        axes[r, 0].yaxis.set_visible(True)
        axes[r, 0].set_yticks([])
    save(fig, "fig04_visual")


# ----------------------------------------------------------------------------------------------- Fig. 5
def fig_hist():
    P = skd.astronaut()
    s = cp.encrypt(P, MASTER_KEY, nonce_for("fig5"))
    fig, axes = plt.subplots(2, 3, figsize=(W2, 2.6), sharex=True, gridspec_kw=dict(hspace=0.3, wspace=0.28))
    chan = ["R", "G", "B"]
    for c in range(3):
        for r, (X, lab) in enumerate([(P, "plain"), (s.cipher, "cipher")]):
            ax = axes[r, c]
            h = np.bincount(X[:, :, c].ravel(), minlength=256)
            ax.bar(np.arange(256), h, width=1.0, color=COL["QuadPath-AE"] if r else INK2)
            chi, pv = mt.chi_square(X[:, :, c])
            ax.set_title(f"{chan[c]} channel, {lab}" + (f"  ($\\chi^2$ = {chi:.1f}, p = {pv:.2f})" if r else ""),
                         fontsize=6.8, pad=2)
            grid(ax)
            if r:
                ax.set_xlabel("Intensity")
            if c == 0:
                ax.set_ylabel("Count")
    save(fig, "fig05_histograms")


# ----------------------------------------------------------------------------------------------- Fig. 6
def fig_corr():
    P = skd.camera()
    s = cp.encrypt(P, MASTER_KEY, nonce_for("fig6"))
    rng = np.random.default_rng(6)
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.85), gridspec_kw=dict(hspace=0.42, wspace=0.32))
    dirs = [("h", "Horizontal"), ("v", "Vertical"), ("d", "Diagonal")]
    for r, (X, lab, col) in enumerate([(P, "plain", INK2), (s.cipher, "cipher", COL["QuadPath-AE"])]):
        x = X.astype(int)
        pairs = {"h": (x[:, :-1], x[:, 1:]), "v": (x[:-1, :], x[1:, :]), "d": (x[:-1, :-1], x[1:, 1:])}
        for c, (d, name) in enumerate(dirs):
            a, b = pairs[d]
            idx = rng.choice(a.size, 5000, replace=False)
            ax = axes[r, c]
            ax.scatter(a.ravel()[idx], b.ravel()[idx], s=0.6, color=col, alpha=0.6, linewidths=0, rasterized=True)
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_aspect("equal")
            ax.set_title(f"{name}, {lab}: r = {mt.adjacent_correlation(X, d):.4f}", fontsize=6.8, pad=2)
            ax.set_xlabel("$x(i,j)$")
            if c == 0:
                ax.set_ylabel("adjacent pixel")
    save(fig, "fig06_correlation")


# ----------------------------------------------------------------------------------------------- Fig. 7
def fig_misuse():
    v = np.load(os.path.join(RES, "misuse_visual.npz"))
    df = pd.read_csv(os.path.join(RES, "misuse_per_pair.csv"))
    sel = df[df.pair == "Cameraman/Moon"].set_index("scheme")
    fig, axes = plt.subplots(1, 5, figsize=(W2, 1.75), gridspec_kw=dict(wspace=0.06))
    show(axes[0], v["P1"], "(a) $P_1$")
    show(axes[1], v["P2"], "(b) $P_2$")
    for ax, key, lab, nm in [(axes[2], "AES-256-CTR", "(c) AES-CTR", "AES-256-CTR"),
                             (axes[3], "P-CTR_(scrambling_+_CTR)", "(d) P-CTR", "P-CTR (scrambling + CTR)"),
                             (axes[4], "QuadPath-AE", "(e) QuadPath-AE", "QuadPath-AE")]:
        show(ax, v[key], f"{lab}: $C_1\\oplus C_2$\n$\\rho$={sel.loc[nm, 'rho_xor']:.3f}, H={sel.loc[nm, 'entropy_xor']:.3f}")
    save(fig, "fig07_misuse")


# ----------------------------------------------------------------------------------------------- Fig. 8
def fig_occlusion():
    P = skd.astronaut()
    H, W, C = P.shape
    rows = []
    for lv, (fh, fw) in (("1/16", (0.25, 0.25)), ("1/4", (0.5, 0.5))):
        h, w = int(H * fh), int(W * fw)
        bad = np.zeros(P.shape, bool)
        bad[(H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w] = True
        nonce = nonce_for("robust", "Astronaut")
        s = cp.encrypt(P, MASTER_KEY, nonce)
        Cb = np.where(bad, 0, s.cipher).astype(np.uint8)
        plain, mask, failed, gok = cp.decrypt_resilient(cp.Sealed(s.header, s.nonce, Cb, s.ttags, s.gtag), MASTER_KEY)
        rec = rc.harmonic_inpaint(plain, mask)
        akey = key_for("aes")
        da = bl.aes_ctr(np.where(bad, 0, bl.aes_ctr(P, akey, nonce)).astype(np.uint8), akey, nonce)
        ra = rc.harmonic_inpaint(da, bad)
        rows.append((lv, Cb, da, ra, plain, rec))
    fig, axes = plt.subplots(2, 5, figsize=(W2, 3.15), gridspec_kw=dict(hspace=0.22, wspace=0.05))
    for r, (lv, Cb, da, ra, plain, rec) in enumerate(rows):
        show(axes[r, 0], Cb, f"Occluded cipher ({lv})")
        show(axes[r, 1], da, f"AES-CTR decrypted\nPSNR {mt.psnr(P, da):.2f} dB")
        show(axes[r, 2], ra, f"AES-CTR + inpainting\nPSNR {mt.psnr(P, ra):.2f} dB")
        show(axes[r, 3], plain, f"QuadPath-AE decrypted\nPSNR {mt.psnr(P, plain):.2f} dB")
        show(axes[r, 4], rec, f"QuadPath-AE recovered\nPSNR {mt.psnr(P, rec):.2f} dB")
    save(fig, "fig08_occlusion")


# ----------------------------------------------------------------------------------------------- Fig. 9
def fig_robust_curves():
    df = pd.read_csv(os.path.join(RES, "robust_per_image.csv"))
    series = [("QuadPath-AE (tags)", "QuadPath-AE, tag-guided", COL["QuadPath-AE"], "o", "-"),
              ("QuadPath-AE (blind median)", "QuadPath-AE, blind median", COL["QuadPath-AE"], "o", "--"),
              ("AES-256-CTR (oracle mask)", "AES-CTR, oracle mask", COL["AES-256-CTR"], "s", "-"),
              ("LOG-PD (blind median)", "LOG-PD, blind median", COL["LOG-PD"], "^", "--")]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw=dict(wspace=0.32))
    for ax, ch, xlab, order in [(axes[0], "occlusion", "Occluded fraction of the cipher image",
                                 ["1/64", "1/16", "1/8", "1/4", "1/2"]),
                                (axes[1], "packet", "Packet-loss rate", [0.01, 0.02, 0.05, 0.1, 0.2]),
                                (axes[2], "noise", "Salt-and-pepper density on cipher", [0.001, 0.005, 0.01, 0.05])]:
        d = df[df.channel == ch].copy()
        d["level"] = d["level"].astype(str)
        lv = [str(o) for o in order]
        sers = series if ch != "noise" else [
            ("QuadPath-AE (blind median)", "QuadPath-AE, blind median", COL["QuadPath-AE"], "o", "--"),
            ("AES-256-CTR (blind median)", "AES-CTR, blind median", COL["AES-256-CTR"], "s", "--"),
            ("LOG-PD (blind median)", "LOG-PD, blind median", COL["LOG-PD"], "^", "--")]
        for key, lab, col, mk, ls in sers:
            m = d[d.scheme == key].groupby("level")["psnr"].mean().reindex(lv)
            ax.plot(range(len(lv)), m.values, ls, color=col, marker=mk, ms=3.5, lw=1.4, label=lab)
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([str(o) if ch == "occlusion" else f"{float(o):g}" for o in order])
        ax.set_xlabel(xlab)
        ax.set_ylabel("PSNR (dB), mean of 28 images")
        grid(ax)
        ax.set_title({"occlusion": "(a) Occlusion", "packet": "(b) Packet erasures", "noise": "(c) Impulse noise"}[ch],
                     fontsize=7.5, pad=3)
    h, lab = axes[0].get_legend_handles_labels()
    h2, lab2 = axes[2].get_legend_handles_labels()
    fig.legend(h + [h2[1]], lab + [lab2[1]], loc="upper center", ncol=5, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.07))
    save(fig, "fig09_robustness")


# ----------------------------------------------------------------------------------------------- Fig. 10
def fig_tamper():
    P = data.kodak(KODAK)["kodim23"]
    s = cp.encrypt(P, MASTER_KEY, nonce_for("fig10"))
    Cb = s.cipher.copy()
    Cb[200:264, 300:364] = s.cipher[40:104, 600:664]  # copy-move inside the cipher image
    Cb[400:416, 100:116] ^= 0x01  # bit flips in a 16 x 16 patch
    sb = cp.Sealed(s.header, s.nonce, Cb, s.ttags, s.gtag)
    try:
        cp.decrypt(sb, MASTER_KEY)
        strict = "accepted"
    except cp.AuthenticationError:
        strict = "rejected"
    plain, mask, failed, gok = cp.decrypt_resilient(sb, MASTER_KEY)
    rec = rc.harmonic_inpaint(plain, mask)
    k_enc, k_mac = cp.derive_keys(MASTER_KEY, s.nonce, s.header)
    exp = cp.tile_tags(k_mac, s.header, s.nonce, Cb, 32)
    nw = (P.shape[1] + 31) // 32
    fig, axes = plt.subplots(1, 4, figsize=(W2, 1.4), gridspec_kw=dict(wspace=0.05))
    show(axes[0], Cb, f"(a) Tampered cipher (strict: {strict})")
    show(axes[1], Cb, f"(b) {failed} failed tile tags")
    for t in range(len(exp) // 4):
        if exp[4 * t:4 * t + 4] != s.ttags[4 * t:4 * t + 4]:
            a, b = divmod(t, nw)
            axes[1].add_patch(Rectangle((b * 32 - 0.5, a * 32 - 0.5), 32, 32, fill=False, ec="#eb6834", lw=1.0))
    show(axes[2], plain, f"(c) Decrypted, PSNR {mt.psnr(P, plain):.2f} dB")
    show(axes[3], rec, f"(d) Recovered, PSNR {mt.psnr(P, rec):.2f} dB")
    save(fig, "fig10_tamper")


# ----------------------------------------------------------------------------------------------- Fig. 11
def fig_speed():
    df = pd.read_csv(os.path.join(RES, "speed.csv"))
    d = df[df.channels == 3]
    series = [("QuadPath-AE", "enc", "QuadPath-AE, encrypt", COL["QuadPath-AE"], "-"),
              ("QuadPath-AE", "dec", "QuadPath-AE, decrypt", COL["QuadPath-AE"], "--"),
              ("LOG-PD", "enc", "LOG-PD, encrypt", COL["LOG-PD"], "-"),
              ("AES-256-GCM", "enc", "AES-256-GCM, encrypt", COL["AES-256-GCM"], "-"),
              ("ChaCha20-Poly1305", "enc", "ChaCha20-Poly1305, encrypt", COL["ChaCha20-Poly1305"], "-")]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for sch, op, lab, col, ls in series:
        m = d[(d.scheme == sch) & (d.op == op)].sort_values("side")
        ax.plot(m.side, m.MBps, ls, color=col, marker="o", ms=3, lw=1.4, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([256, 512, 1024, 2048])
    ax.set_xticklabels(["256²", "512²", "1024²", "2048²"])
    ax.set_xlabel("Color image size (pixels)")
    ax.set_ylabel("Throughput (MB/s, single thread)")
    grid(ax)
    ax.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig11_speed")


# ----------------------------------------------------------------------------------------------- Fig. 12
def fig_diffusion_ablation():
    P = skd.astronaut()
    ke = cp.derive_keys(MASTER_KEY, nonce_for("fig12"), cp.header_bytes(512, 512, 3, 1, 16, 32))[0]
    rng = np.random.default_rng(12)
    n = P.size
    lo = mt.npcr_critical(n)
    ul, uh = mt.uaci_critical(n)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2), gridspec_kw=dict(wspace=0.3))
    names = {"FB1": "FB1 (two passes, window 1)", "FBF1": "FBF1 (three passes, window 1)",
             "FBF2": "FBF2 (proposed)"}
    cols = {"FB1": COL["AES-256-CTR"], "FBF1": COL["LOG-PD"], "FBF2": COL["QuadPath-AE"]}
    for sched in ("FB1", "FBF1", "FBF2"):
        c0 = cp.core_encrypt(P, ke, diffusion=sched)
        pos, npc, uac = [], [], []
        rk = cp.round_keys(ke, 512, 512, 3, 16, 0)  # scrambling does not depend on the diffusion schedule
        inv = [np.argsort(g) for g in rk.gathers]
        for _ in range(80):
            Q = P.copy()
            i, j, c = int(rng.integers(0, 512)), int(rng.integers(0, 512)), int(rng.integers(0, 3))
            Q[i, j, c] = (int(Q[i, j, c]) + 1) % 256
            a, b = mt.npcr_uaci(c0, cp.core_encrypt(Q, ke, diffusion=sched))
            pos.append((inv[c][i * 512 + j] * 3 + c) / n)  # relative position of the change in the diffusion order
            npc.append(a)
            uac.append(b)
        axes[0].scatter(pos, npc, s=7, color=cols[sched], label=names[sched], linewidths=0, zorder=3)
        axes[1].scatter(pos, uac, s=7, color=cols[sched], label=names[sched], linewidths=0, zorder=3)
    axes[0].axhline(lo, color=INK2, lw=0.6, ls=":")
    axes[0].set_ylim(99.55, 100.0)
    axes[0].set_ylabel("NPCR (%)")
    axes[0].set_title("(a) NPCR (dotted: 5% critical value)", fontsize=7, pad=2)
    axes[1].axhspan(ul, uh, color="#e3ecf8", lw=0, zorder=0)
    axes[1].set_ylim(33.3, 33.7)
    axes[1].set_ylabel("UACI (%)")
    axes[1].set_title("(b) UACI (shaded: 5% acceptance band)", fontsize=7, pad=2)
    for ax in axes:
        ax.set_xlabel("Relative position of the changed byte (diffusion order)")
        grid(ax)
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower left")
    save(fig, "fig12_diffusion")


# ----------------------------------------------------------------------------------------------- Fig. ML
def fig_ml():
    e1 = pd.read_csv(os.path.join(RES, "ml_distinguisher.csv"))
    e2 = pd.read_csv(os.path.join(RES, "ml_source_id.csv"))
    ex = np.load(os.path.join(RES, "ml_recovery_examples.npz"))
    fig = plt.figure(figsize=(W2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.55], wspace=0.42)
    col = {"QUAD-PATH'24": COL["QUAD-PATH'24"], "LOG-PD": COL["LOG-PD"], "AES-256-CTR": COL["AES-256-CTR"],
           "P-CTR": COL["P-CTR"], "QuadPath-AE": COL["QuadPath-AE"], "Plain": INK2}
    short = {"QUAD-PATH'24": "QP'24", "LOG-PD": "LOG-PD", "AES-256-CTR": "AES-CTR", "P-CTR": "P-CTR",
             "QuadPath-AE": "QP-AE", "Plain": "Plain"}
    for k, (d, ttl, chance) in enumerate([(e1, "(a) Distinguisher (E1)", 0.5),
                                          (e2, "(b) Source identification (E2)", e2.chance.iloc[0])]):
        ax = fig.add_subplot(gs[k])
        x = np.arange(len(d))
        ax.bar(x, 100 * d.accuracy, width=0.62, color=[col[s] for s in d.scheme])
        ax.errorbar(x, 100 * d.accuracy, yerr=[100 * (d.accuracy - d.ci_lo), 100 * (d.ci_hi - d.accuracy)],
                    fmt="none", ecolor=INK, lw=0.7, capsize=2)
        ax.axhline(100 * chance, color=INK2, lw=0.8, ls="--")
        for xi, v in zip(x, d.accuracy):
            ax.text(xi, 100 * v + 2, f"{100 * v:.1f}", ha="center", fontsize=6, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([short[s] for s in d.scheme], rotation=35, ha="right", fontsize=6.5)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(ttl, fontsize=7.5, pad=3)
        grid(ax)
    sub = gs[2].subgridspec(4, 4, hspace=0.08, wspace=0.05)
    rows = [("plain", "Plain"), ("QUAD-PATH'24", "QP'24"), ("AES-256-CTR_(nonce_reuse)", "AES-CTR"),
            ("QuadPath-AE_(nonce_reuse)", "QP-AE")]
    for r, (key, lab) in enumerate(rows):
        for c in range(4):
            ax = fig.add_subplot(sub[r, c])
            ax.imshow(ex[key][c], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(lab, fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
            if r == 0 and c == 0:
                ax.set_title("(c) Recovery under nonce reuse (E3)", fontsize=7.5, pad=3, loc="left")
    save(fig, "fig13_ml")


if __name__ == "__main__":
    fig_architecture()
    fig_paths()
    fig_qp2024()
    fig_visual()
    fig_hist()
    fig_corr()
    fig_misuse()
    fig_occlusion()
    fig_tamper()
    fig_diffusion_ablation()
    if os.path.exists(os.path.join(RES, "robust_per_image.csv")):
        fig_robust_curves()
    if os.path.exists(os.path.join(RES, "speed.csv")):
        fig_speed()
    if os.path.exists(os.path.join(RES, "ml_recovery_examples.npz")):
        fig_ml()
