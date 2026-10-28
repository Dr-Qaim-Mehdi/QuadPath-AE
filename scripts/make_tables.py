"""Paper tables (tables.json) and in-text numbers (numbers.json) from results/*.csv.

  python scripts/make_tables.py OUTDIR
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from common import RES

OUT = sys.argv[1] if len(sys.argv) > 1 else RES
R = lambda f: pd.read_csv(os.path.join(RES, f))  # noqa: E731
J = lambda f: json.load(open(os.path.join(RES, f)))  # noqa: E731
N, T = {}, {}


def f(x, d=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


# ----------------------------------------------------------------------------------------------- data
st = R("stats_per_image.csv")
df = R("differential_per_image.csv")
se = R("sensitivity_per_image.csv")
qpi = R("qp2024_images.csv")
qpp = R("qp2024_permutation.csv")
mis = R("misuse_per_pair.csv")
rob = R("robust_per_image.csv")
tam = R("tamper_events.csv")
spd = R("speed.csv")
sbd = J("speed_breakdown.json")
val = J("validation.json")
nis = R("nist_summary.csv")
avl = R("ablation_avalanche.csv")
adf = R("ablation_diffusion.csv")
asc = R("ablation_scrambler.csv")
atl = R("ablation_tile.csv")
arn = R("ablation_rounds.csv")
vne = R("validation_nist_e.csv")
nn = R("nist_null_batches.csv")

images = st[st.scheme == "Plain"]
N["n_images"] = len(images)
N["n_gray"] = int((images.group == "Gray").sum())
N["n_kodak"] = int((images.group == "Kodak").sum())
ours = st[st.scheme == "QuadPath-AE"]
N["ours_entropy_mean"] = f(ours.entropy.mean(), 4)
N["ours_entropy_min"] = f(ours.entropy.min(), 4)
corr_cols = ["corr_h", "corr_v", "corr_d", "corr_a"]
N["ours_corr_absmax"] = f(float(ours[corr_cols].abs().max().max()), 4)
N["ours_corr_absmean"] = f(float(ours[corr_cols].abs().mean().mean()), 4)
N["ours_chi2_mean"] = f(ours.chi2.mean(), 1)
N["ours_chi2_pass"] = int(ours.chi2_pass.sum())
N["chi2_crit"] = "293.25"
plain = st[st.scheme == "Plain"]
ng = int(images.group.isin(["Gray", "Special"]).sum())
nc = len(images) - ng
exp_pass = ng * 0.95 + nc * 0.95 ** 3
N["chi2_expected_pass"] = f"{exp_pass:.1f}"
N["chi2_expected_rate"] = f"{100 * exp_pass / len(images):.1f}"
N["plain_corr_mean"] = f(float(plain[["corr_h", "corr_v", "corr_d"]].mean().mean()), 4)

# local entropy intervals
le = val["local_entropy_mc"]
lo001, hi001 = le["interval_001"]
plo, phi = le["published_interval_001"]
N["le_mc_lo"], N["le_mc_hi"] = f(lo001, 6), f(hi001, 6)
N["le_pub_lo"], N["le_pub_hi"] = f(plo, 6), f(phi, 6)
N["le_mc_mean"] = f(le["mean"], 6)
N["le_mc_sd"] = f(le["sd"], 6)
N["le_sd_analytic"] = f(le["sd_analytic"], 6)
N["le_pub_reject"] = pct(le["reject_rate_published_001"], 1)
N["le_trials"] = f"{le['trials']:,}".replace(",", " ")

# NIST validation
N["nist_e_tests_matched"] = int((vne.abs_diff <= 1.5e-6).sum())
N["nist_e_tests_total"] = len(vne)
N["null_npcr_reject"] = pct(val["null"]["npcr_reject_rate"], 1)
N["null_uaci_reject"] = pct(val["null"]["uaci_reject_rate"], 1)
N["unit_cases"] = val["unit"]["roundtrip_and_mask_cases"]
N["unit_paths"] = f"{val['unit']['path_orders_checked']:,}".replace(",", " ")

# ----------------------------------------------------------------------------------------------- T: capabilities
Y, X, P_ = "Yes", "No", "Partly"
T["capabilities"] = dict(
    caption="Properties of the compared image-encryption approaches (requirements R1–R5 of @sec:qp24).",
    header=["Approach", "Secret key", "Fresh nonce", "Security proof", "Integrity", "Avalanche, fixed nonce",
            "Disperses burst errors", "Damage localization"],
    rows=[["QUAD-PATH'24 [@rizvi2024]", X, X, X, X, X, Y, X],
          ["Chaos-based permutation–diffusion (e.g. LOG-PD)", Y, X, X, X, P_, Y, X],
          ["AES-256-CTR", Y, Y, Y, X, X, X, X],
          ["AES-256-GCM, ChaCha20-Poly1305", Y, Y, Y, "Yes (all-or-nothing)", X, X, X],
          ["P-CTR (quad-path scrambling + CTR, ablation)", Y, Y, Y, X, X, Y, X],
          ["QuadPath-AE (this work)", Y, Y, "Yes (Thm. 1, 2)", Y, Y, Y, "Yes (tile tags)"]],
    align=["l"] + ["c"] * 7,
    note="“Partly”: chaos-based permutation–diffusion ciphers typically show full avalanche in forward–backward "
         "chaining, but linear (S-box-free) chaining leaves structured differences (@sec:diff).")

# ----------------------------------------------------------------------------------------------- T: qp24 permutation
rows = []
for _, r in qpp.iterrows():
    rows.append([f"{r.H} × {r.W}", int(r.cycles), int(r.fixed_points), f"{int(r.longest_cycle):,}".replace(",", " "),
                 f"{10 ** (r.log10_order - int(r.log10_order)):.2f} × 10^{int(r.log10_order)}^",
                 f(r.mean_manhattan_displacement, 1)])
T["qp24perm"] = dict(caption="Cycle structure of the QUAD-PATH (2024) permutation π. The order is the number "
                             "of times the encryption must be iterated to return the plain image.",
                     header=["Size", "Cycles", "Fixed points", "Longest cycle", "Order of π",
                             "Mean displacement (px)"], rows=rows, align=["l", "r", "r", "r", "r", "r"])
r512 = qpi[qpi.image == "Cameraman"].iloc[0]
N["qp24_npcr_512"] = f"{r512.npcr_one_pixel:.4f}"
N["qp24_all_keyless"] = "all" if qpi.keyless_decryption_exact.all() else "not all"
N["qp24_corr_absmax"] = f(float(qpi[["corr_h_cipher", "corr_v_cipher"]].abs().max().max()), 3)

# ----------------------------------------------------------------------------------------------- T: algorithm
T["algorithm"] = dict(kind="algorithm", caption="QuadPath-AE encryption and decryption (one round, R = 1).", lines=[
    "Input: image P (H×W×C), master key K (256 bit). Output: (hdr, N, C, tile tags, tag)",
    " 1  N ← 128 random bits;  hdr ← (version, H, W, C, R, B, Ba)",
    " 2  (Ke ‖ Km) ← HKDF-Expand(K, \"QPAE-v1\" ‖ hdr ‖ N, 64)",
    " 3  for each channel c:  σ ← FisherYates(T; ChaCha20(Ke, \"TPRM\",c));",
    "        (τt, gt) ← bytes of ChaCha20(Ke, \"TSEL\",c);  (δ, b) ← ChaCha20(Ke, \"ILVR\",c)",
    "        s ← concatenation over t in σ of tile t read along path τt under gt",
    "        Yc[o] ← s[(a·o + b) mod L],  a = round(L/φ)+δ adjusted to gcd(a, L) = 1",
    " 4  x ← interleave(Y1..YC);  S, Q ← keyed S-boxes;  k(1..3) ← ChaCha20(Ke, \"DIFF\",p)",
    " 5  x ← Fwd(x, k(1));  x ← Bwd(x, k(2));  C ← Fwd(x, k(3))          ▷ Eq. (3)",
    " 6  τt ← HMAC_Km(\"TILE\" ‖ hdr ‖ N ‖ t ‖ Ct)[0:32] for each Ba×Ba tile t",
    " 7  τ ← HMAC_Km(\"GLOB\" ‖ hdr ‖ N ‖ C ‖ τ1..m)[0:128];  return (hdr, N, C, τ1..m, τ)",
    "Strict decryption: recompute τ; if it differs, REJECT; otherwise invert steps 5, 4, 3 (Eq. (4)).",
    "Resilient decryption: E ← known lost bytes ∪ tiles whose τt fails without a known loss;",
    "    P' ← invert steps 5, 4, 3;  M ← inverse-scramble(spread(E, {−2..+4}));",
    "    return HarmonicInpaint(P', M), M, (τ verified?)"])

# ----------------------------------------------------------------------------------------------- T: test set
grp = images.groupby("group")
desc = {"Gray": "Cameraman, Moon, Coins, Clock, Page, Brick, Gravel", "Color": "Astronaut, Coffee, Chelsea, Rocket",
        "Medical": "Retina (fundus, 1411 × 1411 × 3), IHC (histology, 512 × 512 × 3)",
        "Special": "All-black and all-white (512 × 512)", "Kodak": "kodim01–kodim24 (768 × 512 × 3 or 512 × 768 × 3)"}
T["testset"] = dict(caption="Test images (39 in total). All except Kodak-24 [@franzen1999] ship with scikit-image "
                            "[@vanderwalt2014]; Lena is not used.",
                    header=["Group", "Images", "Count"],
                    rows=[[g, desc[g], int(len(d))] for g, d in [(k, grp.get_group(k)) for k in
                                                               ["Gray", "Color", "Medical", "Special", "Kodak"]]],
                    align=["l", "l", "r"])

# ----------------------------------------------------------------------------------------------- T: statistics
order = ["Plain", "QUAD-PATH'24", "LOG-PD", "AES-256-CTR", "QuadPath-AE"]
rows = []
for s in order:
    d = st[st.scheme == s]
    lep = ((d.local_entropy > lo001) & (d.local_entropy < hi001)).mean()
    lpp = ((d.local_entropy > plo) & (d.local_entropy < phi)).mean()
    rows.append([s, f(d.entropy.mean(), 4), f(d.local_entropy.mean(), 4), f"{pct(lep, 0)} / {pct(lpp, 0)}",
                 f(d.chi2.mean(), 1) if s not in ("Plain", "QUAD-PATH'24") else f"{d.chi2.mean():.2e}",
                 f"{int(d.chi2_pass.sum())}/{len(d)}"]
                + [f(float(np.nanmean(d[c])), 4) for c in corr_cols])
T["stats"] = dict(caption=f"Statistical properties averaged over the {len(images)} test images. Local entropy: "
                          "pass rate (%) for the Monte-Carlo α = 0.001 interval / the published interval. "
                          "χ²: mean statistic and number of images whose every channel is below the 5% critical value 293.25 "
                          f"(a uniform source passes {N['chi2_expected_rate']}% of these images). "
                          "r: adjacent-pixel correlation (horizontal, vertical, diagonal, anti-diagonal), all pairs. "
                          "Color images: means over the three channels.",
                  header=["Scheme", "Entropy", "Local entropy", "LE pass (%)", "χ²", "χ² pass", "r_h", "r_v", "r_d",
                          "r_a"], rows=rows, align=["l"] + ["r"] * 9)
for s, key in [("QuadPath-AE", "ours"), ("AES-256-CTR", "aes"), ("LOG-PD", "logpd")]:
    d = st[st.scheme == s]
    N[f"{key}_le_pass_mc"] = pct(((d.local_entropy > lo001) & (d.local_entropy < hi001)).mean(), 0)
    N[f"{key}_le_pass_pub"] = pct(((d.local_entropy > plo) & (d.local_entropy < phi)).mean(), 0)
    N[f"{key}_chi2_pass"] = int(d.chi2_pass.sum())
N["logpd_chi2_pass"] = int(st[st.scheme == "LOG-PD"].chi2_pass.sum())

# per-image table (supplement and main text subset)
rows = []
for _, r in ours.iterrows():
    if r.group == "Kodak":
        continue
    P = images[images.image == r.image].iloc[0]
    rows.append([r.image, f(P.entropy, 4), f(r.entropy, 4), f(r.local_entropy, 4), f(r.chi2, 1),
                 f(P.corr_h, 4), f(r.corr_h, 4), f(r.corr_v, 4), f(r.corr_d, 4)])
k = ours[ours.group == "Kodak"]
kp = images[images.group == "Kodak"]
rows.append(["Kodak-24 (mean)", f(kp.entropy.mean(), 4), f(k.entropy.mean(), 4), f(k.local_entropy.mean(), 4),
             f(k.chi2.mean(), 1), f(kp.corr_h.mean(), 4), f(k.corr_h.mean(), 4), f(k.corr_v.mean(), 4),
             f(k.corr_d.mean(), 4)])
T["stats_perimage"] = dict(caption="QuadPath-AE cipher images, per image (plain-image values for reference).",
                           header=["Image", "Entropy (plain)", "Entropy", "Local entropy", "χ²", "r_h (plain)", "r_h",
                                   "r_v", "r_d"], rows=rows, align=["l"] + ["r"] * 8)

# ----------------------------------------------------------------------------------------------- T: differential
rows = []
for s in ["QUAD-PATH'24", "AES-256-CTR", "LOG-PD", "QuadPath-AE"]:
    d = df[df.scheme == s]
    rows.append([s, f"{d.npcr.mean():.4f} ± {d.npcr.std():.4f}", f"{d.uaci.mean():.4f} ± {d.uaci.std():.4f}",
                 pct(d.npcr_pass.mean()), pct(d.uaci_pass.mean())])
dd = df[df.scheme == "QuadPath-AE"]
for g in ["Gray", "Color", "Medical", "Kodak"]:
    d = dd[dd.group == g]
    rows.append([f"  QuadPath-AE, {g}", f"{d.npcr.mean():.4f} ± {d.npcr.std():.4f}",
                 f"{d.uaci.mean():.4f} ± {d.uaci.std():.4f}", pct(d.npcr_pass.mean()), pct(d.uaci_pass.mean())])
ntr = int(len(dd))
T["differential"] = dict(caption=f"Plaintext sensitivity under a fixed key and nonce (nonce misuse): {ntr // len(dd.image.unique())} random "
                                 f"single-byte changes (±1) per image, {ntr} trials per scheme over "
                                 f"{len(dd.image.unique())} non-constant images. Pass: NPCR above N*_0.05 and UACI "
                                 "inside the 5% interval of Wu et al. [@wu2011]; a truly random cipher passes ≈95% of trials.",
                         header=["Scheme", "NPCR (%)", "UACI (%)", "NPCR pass (%)", "UACI pass (%)"], rows=rows,
                         align=["l", "r", "r", "r", "r"])
N["diff_trials"] = ntr
N["diff_per_image"] = ntr // len(dd.image.unique())
N["ours_npcr_mean"] = f(dd.npcr.mean(), 4)
N["ours_uaci_mean"] = f(dd.uaci.mean(), 4)
N["ours_npcr_pass"] = pct(dd.npcr_pass.mean())
N["ours_uaci_pass"] = pct(dd.uaci_pass.mean())
N["ours_npcr_min"] = f(dd.npcr.min(), 4)
lp = df[df.scheme == "LOG-PD"]
N["logpd_npcr_pass"] = pct(lp.npcr_pass.mean())
N["logpd_uaci_pass"] = pct(lp.uaci_pass.mean())

# avalanche study
names = {"FB1": "FB1: forward + backward, window 1", "FBF1": "FBF1: three passes, window 1",
         "FB2": "FB2: forward + backward, window 2", "FBF2": "FBF2: three passes, window 2 (proposed)",
         "additive": "Linear chaining without S-boxes (FB)", "independent random images": "Independent random images"}
rows = []
for _, r in avl.iterrows():
    rows.append([names[r.schedule], f"{r.npcr_mean:.4f} ± {r.npcr_sd:.4f}", f"{r.uaci_mean:.4f} ± {r.uaci_sd:.4f}",
                 pct(r.npcr_pass), pct(r.uaci_pass), int(r.lost_avalanche)])
T["avalanche"] = dict(caption=f"Avalanche study: {int(avl.trials.iloc[0])} single-byte changes of a 256 × 256 "
                              "crop of Cameraman under a fixed key and nonce, for each diffusion schedule. "
                              "Lost avalanche: NPCR below 99%.",
                      header=["Diffusion", "NPCR (%)", "UACI (%)", "NPCR pass (%)", "UACI pass (%)", "Lost avalanche"],
                      rows=rows, align=["l", "r", "r", "r", "r", "r"])
a = avl.set_index("schedule")
N["aval_trials"] = int(avl.trials.iloc[0])
N["aval_fb1_lost"] = int(a.loc["FB1", "lost_avalanche"])
N["aval_fbf1_lost"] = int(a.loc["FBF1", "lost_avalanche"])
N["aval_fbf2_lost"] = int(a.loc["FBF2", "lost_avalanche"])
N["aval_fb1_npcr"] = f(a.loc["FB1", "npcr_mean"], 4)
N["aval_fbf2_npcr"] = f"{a.loc['FBF2', 'npcr_mean']:.4f} ± {a.loc['FBF2', 'npcr_sd']:.4f}"
N["aval_rand_npcr"] = f"{a.loc['independent random images', 'npcr_mean']:.4f} ± " \
                      f"{a.loc['independent random images', 'npcr_sd']:.4f}"
N["aval_fbf2_uaci"] = f"{a.loc['FBF2', 'uaci_mean']:.4f} ± {a.loc['FBF2', 'uaci_sd']:.4f}"
N["aval_rand_uaci"] = f"{a.loc['independent random images', 'uaci_mean']:.4f} ± " \
                      f"{a.loc['independent random images', 'uaci_sd']:.4f}"
N["aval_add_uaci_pass"] = pct(a.loc["additive", "uaci_pass"])
N["aval_expected_lost"] = f"{a.loc['FB1', 'trials'] / 256:.1f}"

# ----------------------------------------------------------------------------------------------- T: sensitivity
s0 = se[se.trial >= 0]
rows = [["Encryption, key with one bit flipped", f(s0.enc_key_npcr.mean(), 4), f(s0.enc_key_uaci.mean(), 4), "–"],
        ["Encryption, nonce with one bit flipped", f(s0.enc_nonce_npcr.mean(), 4), f(s0.enc_nonce_uaci.mean(), 4), "–"],
        ["Same image, two fresh nonces", f(s0.fresh_nonce_npcr.mean(), 4), f(s0.fresh_nonce_uaci.mean(), 4), "–"],
        ["Decryption with a one-bit-wrong key (vs. plain image)", f(s0.dec_wrongkey_npcr.mean(), 4),
         f(s0.dec_wrongkey_uaci.mean(), 4), f(s0.dec_wrongkey_psnr.mean(), 2)]]
T["sensitivity"] = dict(caption=f"Key and nonce sensitivity of QuadPath-AE: means over {len(s0)} trials "
                                f"({len(s0) // len(s0.image.unique())} random bit positions per image, "
                                f"{len(s0.image.unique())} images). Ideal values: NPCR 99.6094%, UACI 33.4635%.",
                        header=["Test", "NPCR (%)", "UACI (%)", "PSNR (dB)"], rows=rows, align=["l", "r", "r", "r"])
N["sens_trials"] = len(s0)
N["sens_wrong_psnr"] = f(s0.dec_wrongkey_psnr.mean(), 2)
N["sens_key_npcr"] = f(s0.enc_key_npcr.mean(), 4)
N["sens_fresh_npcr"] = f(s0.fresh_nonce_npcr.mean(), 4)
lb = se[se.trial < 0]
N["logpd_lsb_npcr"] = f(lb.enc_key_npcr.mean(), 4)

# ----------------------------------------------------------------------------------------------- T: NIST
pretty = {"Frequency": "Frequency", "BlockFrequency": "Block frequency (M = 128)", "CumulativeSums": "Cumulative sums (2)",
          "Runs": "Runs", "LongestRun": "Longest run of ones", "Rank": "Binary matrix rank", "FFT": "Spectral (DFT)",
          "NonOverlappingTemplate": "Non-overlapping templates (148)", "OverlappingTemplate": "Overlapping template",
          "Universal": "Maurer's universal", "ApproximateEntropy": "Approximate entropy (m = 10)",
          "RandomExcursions": "Random excursions (8)", "RandomExcursionsVariant": "Random excursions variant (18)",
          "Serial": "Serial (m = 16; 2)", "LinearComplexity": "Linear complexity (M = 500)"}
srcs = [("QuadPath-AE", "Kodak-24"), ("QuadPath-AE", "all-black"), ("LOG-PD", "Kodak-24"), ("QUAD-PATH'24", "Kodak-24")]
rows = []
for test in pretty:
    row = [pretty[test]]
    for sch, src in srcs:
        d = nis[(nis.scheme == sch) & (nis.source == src) & (nis.test == test)]
        if d.empty or d.iloc[0].sequences == 0:
            row.append("n/a")
            continue
        r = d.iloc[0]
        tag = "" if r.passed else " ✗"
        row.append(f"{r.min_proportion:.2f} / {r.min_uniformity_p:.3f}{tag}")
    rows.append(row)
summ = []
for sch, src in srcs:
    d = nis[(nis.scheme == sch) & (nis.source == src)]
    summ.append(f"{int(d.passed.sum())}/15")
rows.append(["Tests passed"] + summ)
seqs = int(nis[(nis.scheme == "QuadPath-AE") & (nis.source == "Kodak-24") & (nis.test == "Frequency")].sequences.iloc[0])
thr = nis[(nis.scheme == "QuadPath-AE") & (nis.source == "Kodak-24") & (nis.test == "Frequency")].proportion_threshold.iloc[0]
T["nist"] = dict(caption=f"NIST SP 800-22 results for {seqs} sequences of 10^6^ bits per source: minimum proportion "
                         f"of passing sequences / minimum uniformity P-value over the sub-tests. Pass: proportion ≥ "
                         f"{thr:.4f} and uniformity P ≥ 0.0001 for every sub-test; ✗ marks a failure. Random excursion "
                         "tests apply only to sequences with at least 500 cycles. Under this strict rule a strong reference "
                         f"generator (AES-256-CTR keystream, {len(nn)} batches) passes {nn.tests_passed.min()}–"
                         f"{nn.tests_passed.max()} of the 15 tests, failing up to {nn.nonoverlap_subtests_failed.max()} "
                         "of the 148 non-overlapping-template sub-tests.",
                 header=["Test", "QuadPath-AE, Kodak-24", "QuadPath-AE, all-black", "LOG-PD, Kodak-24",
                         "QUAD-PATH'24, Kodak-24"], rows=rows, align=["l", "r", "r", "r", "r"])
N["nist_seqs"] = seqs
N["nist_ours_passed"] = int(nis[(nis.scheme == "QuadPath-AE") & (nis.source == "Kodak-24")].passed.sum())
N["nist_black_passed"] = int(nis[(nis.scheme == "QuadPath-AE") & (nis.source == "all-black")].passed.sum())
N["nist_qp24_passed"] = int(nis[(nis.scheme == "QUAD-PATH'24") & (nis.source == "Kodak-24")].passed.sum())
N["nist_logpd_passed"] = int(nis[(nis.scheme == "LOG-PD") & (nis.source == "Kodak-24")].passed.sum())
N["nist_threshold"] = f"{thr:.4f}"
N["nist_null_batches"] = len(nn)
N["nist_null_pass_mean"] = f(nn.tests_passed.mean(), 1)
N["nist_null_pass_min"] = int(nn.tests_passed.min())
N["nist_null_pass_max"] = int(nn.tests_passed.max())
N["nist_null_tmpl_mean"] = f(nn.nonoverlap_subtests_failed.mean(), 1)
N["nist_null_tmpl_max"] = int(nn.nonoverlap_subtests_failed.max())
tm = lambda sch, src: int(148 - nis[(nis.scheme == sch) & (nis.source == src) & (nis.test == "NonOverlappingTemplate")].subtests_passed.iloc[0])  # noqa: E731
N["nist_ours_tmpl_fail"] = tm("QuadPath-AE", "Kodak-24")
N["nist_black_tmpl_fail"] = tm("QuadPath-AE", "all-black")
N["nist_logpd_tmpl_fail"] = tm("LOG-PD", "Kodak-24")

# ----------------------------------------------------------------------------------------------- T: misuse
order_m = ["AES-256-CTR", "ChaCha20-Poly1305", "P-CTR (scrambling + CTR)", "LOG-PD", "QP-AE (additive chaining)",
           "QP-AE (FB1 diffusion)", "QuadPath-AE"]
g = mis.groupby("scheme")
rows = []
for s in order_m:
    d = g.get_group(s)
    rows.append([s, f(d.rho_xor.mean(), 4), f(d.entropy_xor.mean(), 4), f(d.npcr.mean(), 4), f(d.uaci.mean(), 4),
                 f(100 * d.top_diff_share.mean(), 3)])
T["misuse"] = dict(caption=f"Nonce misuse: {mis.pair.nunique()} pairs of different images encrypted under the same key and "
                           "nonce. ρ: correlation between C1 ⊕ C2 and P1 ⊕ P2; H: entropy of C1 ⊕ C2; top difference: "
                           "share of the most frequent byte difference C1 − C2 (ideal 0.391%). The mean entropy of "
                           f"P1 ⊕ P2 is {mis.entropy_plain_xor.mean():.4f} bits.",
                   header=["Scheme", "ρ", "H(C1 ⊕ C2)", "NPCR (%)", "UACI (%)", "Top difference (%)"], rows=rows,
                   align=["l", "r", "r", "r", "r", "r"])
N["misuse_pairs"] = mis.pair.nunique()
N["misuse_ours_rho"] = f(g.get_group("QuadPath-AE").rho_xor.mean(), 4)
N["misuse_ours_H"] = f(g.get_group("QuadPath-AE").entropy_xor.mean(), 4)
N["misuse_ctr_H"] = f(g.get_group("AES-256-CTR").entropy_xor.mean(), 4)

# ----------------------------------------------------------------------------------------------- T: robustness
rob["level"] = rob["level"].astype(str)
cols = [("QuadPath-AE (tags)", "QuadPath-AE, tags"), ("QuadPath-AE (blind median)", "QuadPath-AE, median"),
        ("AES-256-CTR (oracle mask)", "AES-CTR, oracle mask"), ("P-CTR (oracle mask)", "P-CTR, oracle mask"),
        ("LOG-PD (blind median)", "LOG-PD, median"), ("AES-256-CTR (blind median)", "AES-CTR, median")]
levels = [("occlusion", ["1/64", "1/16", "1/8", "1/4", "1/2"]), ("packet", ["0.01", "0.02", "0.05", "0.1", "0.2"]),
          ("noise", ["0.001", "0.005", "0.01", "0.05"])]
rows = []
for ch, lvs in levels:
    for lv in lvs:
        d = rob[(rob.channel == ch) & (rob.level == lv)]
        lab = f"Occlusion {lv}" if ch == "occlusion" else (
            f"Packet loss {float(lv):g}" if ch == "packet" else f"S&P noise {float(lv):g}")
        row = [lab]
        for key, _ in cols:
            e = d[d.scheme == key]
            row.append("–" if e.empty else f"{e.psnr.mean():.2f} / {e.ssim.mean():.3f}")
        rows.append(row)
T["robust"] = dict(caption=f"Recovery from corrupted cipher images: PSNR (dB) / SSIM of the recovered image, mean over "
                           f"{rob.image.nunique()} images (Cameraman, Moon, Astronaut, IHC, Kodak-24). Tags: resilient "
                           "decryption with tile-tag localization (occlusion) or known erasures (packets) and harmonic "
                           "inpainting of the exact mask. Oracle mask: the receiver is told the corrupted cipher bytes. "
                           "AES-GCM, ChaCha20-Poly1305 and strict-mode QuadPath-AE reject every corrupted image.",
                   header=["Corruption"] + [c[1] for c in cols], rows=rows, align=["l"] + ["r"] * len(cols))


def rm(ch, lv, s, m="psnr"):
    d = rob[(rob.channel == ch) & (rob.level == lv) & (rob.scheme == s)]
    return float(d[m].mean())


N["occl_q_ours"] = f(rm("occlusion", "1/4", "QuadPath-AE (tags)"), 2)
N["occl_q_aes"] = f(rm("occlusion", "1/4", "AES-256-CTR (oracle mask)"), 2)
N["occl_q_ours_int"] = f"{rm('occlusion', '1/4', 'QuadPath-AE (tags)'):.0f}"
N["occl_q_aes_int"] = f"{rm('occlusion', '1/4', 'AES-256-CTR (oracle mask)'):.0f}"
N["occl_16_ours"] = f(rm("occlusion", "1/16", "QuadPath-AE (tags)"), 2)
N["occl_16_aes"] = f(rm("occlusion", "1/16", "AES-256-CTR (oracle mask)"), 2)
N["occl_16_gain"] = f(rm("occlusion", "1/16", "QuadPath-AE (tags)") - rm("occlusion", "1/16", "AES-256-CTR (oracle mask)"), 1)
N["occl_q_gain"] = f(rm("occlusion", "1/4", "QuadPath-AE (tags)") - rm("occlusion", "1/4", "AES-256-CTR (oracle mask)"), 1)
N["occl_h_ours"] = f(rm("occlusion", "1/2", "QuadPath-AE (tags)"), 2)
N["occl_h_aes"] = f(rm("occlusion", "1/2", "AES-256-CTR (oracle mask)"), 2)
N["occl_16_pctr"] = f(rm("occlusion", "1/16", "P-CTR (oracle mask)"), 2)
N["occl_16_logpd"] = f(rm("occlusion", "1/16", "LOG-PD (blind median)"), 2)
N["occl_16_median"] = f(rm("occlusion", "1/16", "QuadPath-AE (blind median)"), 2)
N["pkt10_ours"] = f(rm("packet", "0.1", "QuadPath-AE (tags)"), 2)
N["pkt10_aes"] = f(rm("packet", "0.1", "AES-256-CTR (oracle mask)"), 2)
N["pkt20_ours"] = f(rm("packet", "0.2", "QuadPath-AE (tags)"), 2)
N["pkt20_aes"] = f(rm("packet", "0.2", "AES-256-CTR (oracle mask)"), 2)
N["noise01_ours_med"] = f(rm("noise", "0.01", "QuadPath-AE (blind median)"), 2)
N["noise01_aes_med"] = f(rm("noise", "0.01", "AES-256-CTR (blind median)"), 2)
N["noise01_aes_raw"] = f(rm("noise", "0.01", "AES-256-CTR (raw)"), 2)
N["noise01_ours_raw"] = f(rm("noise", "0.01", "QuadPath-AE (raw)"), 2)
N["noise01_logpd_med"] = f(rm("noise", "0.01", "LOG-PD (blind median)"), 2)
oc = rob[(rob.channel == "occlusion") & (rob.scheme == "QuadPath-AE (tags)")]
N["occl_mask_expansion"] = f(float((oc.mask_frac / oc.corrupted_frac).mean()), 2)
N["occl_inpaint_s"] = f(float(oc[oc.level == "1/4"].seconds.mean()), 2)
N["n_robust_images"] = rob.image.nunique()

# ----------------------------------------------------------------------------------------------- T: tamper
rows = []
for kind, lab in [("bitflip", "Single bit flipped"), ("patch16", "16 × 16 patch replaced"),
                  ("copymove64", "64 × 64 copy-move")]:
    d = tam[(tam.kind == kind) & tam.modified]
    rows.append([lab, len(d), pct(d.rejected.mean()), f"{d.detected.sum()}/{d.true_tiles.sum()}",
                 int(d.false_flags.sum()), f(d.flagged_tiles.mean(), 2)])
dall = tam[tam.modified]
rows.append(["All", len(dall), pct(dall.rejected.mean()), f"{dall.detected.sum()}/{dall.true_tiles.sum()}",
             int(dall.false_flags.sum()), f(dall.flagged_tiles.mean(), 2)])
T["tamper"] = dict(caption=f"Tamper detection and localization over {len(dall)} random modifications of cipher images "
                           f"({tam.image.nunique()} images). Rejected: strict mode (128-bit tag). Detected tiles: "
                           "modified 32 × 32 tiles whose tag failed.",
                   header=["Modification", "Events", "Rejected (%)", "Detected tiles", "False flags",
                           "Flagged tiles per event"], rows=rows, align=["l", "r", "r", "r", "r", "r"])
N["tamper_events"] = len(dall)
N["tamper_rejected"] = pct(dall.rejected.mean())

# ----------------------------------------------------------------------------------------------- T: speed
rows = []
for sch, op, lab in [("QuadPath-AE", "enc", "QuadPath-AE encryption"), ("QuadPath-AE", "dec", "QuadPath-AE decryption"),
                     ("LOG-PD", "enc", "LOG-PD encryption"), ("LOG-PD", "dec", "LOG-PD decryption"),
                     ("QUAD-PATH'24", "enc", "QUAD-PATH'24"), ("AES-256-CTR", "enc", "AES-256-CTR"),
                     ("AES-256-GCM", "enc", "AES-256-GCM encryption"),
                     ("ChaCha20-Poly1305", "enc", "ChaCha20-Poly1305 encryption")]:
    row = [lab]
    for side, C in [(512, 1), (512, 3), (1024, 3), (2048, 3)]:
        d = spd[(spd.scheme == sch) & (spd.op == op) & (spd.side == side) & (spd.channels == C)].iloc[0]
        row.append(f"{d.ms:.2f} ({d.MBps:.0f})")
    rows.append(row)
env = sbd["env"]
T["speed"] = dict(caption=f"Single-thread run time in ms (throughput in MB/s in parentheses), median of repeated "
                          f"runs on an Apple M4 ({env['machine']}), Python {env['python']}, NumPy {env['numpy']}, "
                          f"Numba {env.get('numba', '?')}, cryptography {env.get('cryptography', '?')} (OpenSSL). "
                          "QuadPath-AE times include key derivation and all tags."
                          + (" Measured on battery power with macOS Low Power Mode enabled, which lowers the CPU "
                             "clock; all schemes were timed under the same conditions." if env.get("low_power_mode")
                             else ""),
                  header=["Operation", "512 × 512", "512 × 512 × 3", "1024 × 1024 × 3", "2048 × 2048 × 3"],
                  rows=rows, align=["l", "r", "r", "r", "r"])


def sp(sch, op, side, C, m="MBps"):
    return float(spd[(spd.scheme == sch) & (spd.op == op) & (spd.side == side) & (spd.channels == C)][m].iloc[0])


N["enc_ms_512c"] = f(sp("QuadPath-AE", "enc", 512, 3, "ms"), 1)
N["dec_ms_512c"] = f(sp("QuadPath-AE", "dec", 512, 3, "ms"), 1)
N["enc_MBps_512c"] = f"{sp('QuadPath-AE', 'enc', 512, 3):.0f}"
N["dec_MBps_512c"] = f"{sp('QuadPath-AE', 'dec', 512, 3):.0f}"
N["enc_MBps_2048c"] = f"{sp('QuadPath-AE', 'enc', 2048, 3):.0f}"
N["gcm_MBps_512c"] = f"{sp('AES-256-GCM', 'enc', 512, 3):.0f}"
N["logpd_MBps_512c"] = f"{sp('LOG-PD', 'enc', 512, 3):.0f}"
N["gcm_ratio"] = f"{sp('AES-256-GCM', 'enc', 512, 3) / sp('QuadPath-AE', 'enc', 512, 3):.0f}"
N["logpd_ratio"] = f"{sp('QuadPath-AE', 'enc', 512, 3) / sp('LOG-PD', 'enc', 512, 3):.1f}"
bd = sbd["breakdown_ms_512x512x3"]
tot = sum(v for k, v in bd.items() if "undiffusion" not in k)
rows = [[k, f(v, 2), f(100 * v / tot, 0)] for k, v in bd.items() if "undiffusion" not in k]
rows.append(["Sum of encryption stages", f(tot, 2), "100"])
T["breakdown"] = dict(caption="Breakdown of QuadPath-AE encryption time for a 512 × 512 × 3 image (ms).",
                      header=["Stage", "Time (ms)", "Share (%)"], rows=rows, align=["l", "r", "r"])
N["diff_share"] = f(100 * bd["FBF diffusion"] / tot, 0)

# ----------------------------------------------------------------------------------------------- T: ablations
an = {"F1": "F1 (forward only)", "FB1": "FB1", "FBF1": "FBF1", "FB2": "FB2", "FBF2": "FBF2 (proposed)",
      "additive": "Linear chaining, no S-box", "none": "No diffusion (keystream XOR)"}
g = adf.groupby("schedule")
rows = []
for s in ["F1", "FB1", "FBF1", "FB2", "FBF2", "additive", "none"]:
    d = g.get_group(s)
    exp_ = d.error_expansion.iloc[0]
    rows.append([an[s], f(d.npcr_mean.mean(), 3), f(d.uaci_mean.mean(), 3), pct(d.pass_rate.mean(), 0),
                 int(d.lost_avalanche.sum()), f(100 * d.onepixel_top_diff_share.mean(), 2),
                 "–" if pd.isna(exp_) else int(exp_), f(d.enc_ms.mean(), 1)])
T["abl_diffusion"] = dict(caption="Ablation of the diffusion layer (six images, 30 single-byte changes each, fixed "
                                  "nonce). Top difference: share of the most frequent non-zero byte difference "
                                  "between the two cipher images (ideal 0.39%). Span: plain bytes affected by one "
                                  "corrupted cipher byte. Time: encryption of the image without tags.",
                          header=["Diffusion", "NPCR (%)", "UACI (%)", "Pass (%)", "Lost avalanche",
                                  "Top difference (%)", "Span (bytes)", "Time (ms)"], rows=rows,
                          align=["l"] + ["r"] * 7)
dl = g.get_group("additive")
N["abl_add_topdiff"] = f(100 * dl.onepixel_top_diff_share.mean(), 1)
N["abl_fb1_pass"] = pct(g.get_group("FB1").pass_rate.mean(), 0)
N["abl_fbf2_pass"] = pct(g.get_group("FBF2").pass_rate.mean(), 0)
N["abl_fb2_pass"] = pct(g.get_group("FB2").pass_rate.mean(), 0)
N["abl_fbf1_pass"] = pct(g.get_group("FBF1").pass_rate.mean(), 0)

sn = {"quadpath": "QuadPath (proposed)", "tiles_only": "Tile paths without interleaver",
      "random": "Uniform random permutation", "qp2024": "QUAD-PATH'24 permutation (keyless)",
      "none": "No scrambling"}
rows = []
for s in ["quadpath", "tiles_only", "random", "qp2024", "none"]:
    d = asc[asc.scrambler == s]
    a16, a4 = d[d.occlusion == "1/16"], d[d.occlusion == "1/4"]
    rows.append([sn[s], f(a16.psnr.mean(), 2), f(a4.psnr.mean(), 2), f(a16.isolation.mean(), 3),
                 f(a16.largest_hole.mean(), 0), f(d.round_key_ms.mean(), 1)])
for B in [4, 8, 32, 64]:
    d = atl[atl.B == B]
    a16, a4 = d[d.occlusion == "1/16"], d[d.occlusion == "1/4"]
    rows.append([f"QuadPath, tile size B = {B}", f(a16.psnr.mean(), 2), f(a4.psnr.mean(), 2), f(a16.isolation.mean(), 3),
                 f(a16.largest_hole.mean(), 0), "–"])
for Rr in [2]:
    d = arn[arn.R == Rr]
    a16, a4 = d[d.occlusion == "1/16"], d[d.occlusion == "1/4"]
    rows.append([f"QuadPath-AE, R = {Rr} rounds", f(a16.psnr.mean(), 2), f(a4.psnr.mean(), 2),
                 f(a16.isolation.mean(), 3), f(a16.largest_hole.mean(), 0), "–"])
T["abl_scrambler"] = dict(caption="Ablation of the scrambling layer, tile size and number of rounds (six images; "
                                  "centered occlusion of 1/16 and 1/4 of the cipher image, exact error mask, harmonic "
                                  "inpainting). Isolation: share of unreliable pixels without an unreliable 8-neighbor "
                                  "(1/16 occlusion). Hole: largest connected unreliable region (pixels). Key setup: "
                                  "time to derive the round keys of a 512 × 512 × 3 image.",
                          header=["Variant", "PSNR 1/16 (dB)", "PSNR 1/4 (dB)", "Isolation", "Largest hole",
                                  "Key setup (ms)"], rows=rows, align=["l"] + ["r"] * 5)
q = asc.groupby(["scrambler", "occlusion"]).psnr.mean()
N["abl_qp_16"] = f(q[("quadpath", "1/16")], 2)
N["abl_rand_16"] = f(q[("random", "1/16")], 2)
N["abl_qp24_16"] = f(q[("qp2024", "1/16")], 2)
N["abl_none_16"] = f(q[("none", "1/16")], 2)
N["abl_tiles_16"] = f(q[("tiles_only", "1/16")], 2)
iso = asc.groupby(["scrambler", "occlusion"]).isolation.mean()
N["abl_qp24_iso"] = f(iso[("qp2024", "1/16")], 2)
N["abl_qp_iso"] = f(iso[("quadpath", "1/16")], 2)
N["abl_r2_16"] = f(arn[(arn.R == 2) & (arn.occlusion == "1/16")].psnr.mean(), 2)
N["abl_r2_span"] = f"{arn[arn.R == 2].single_byte_expansion.mean():.0f}"
rk = asc.groupby("scrambler").round_key_ms.mean()
N["abl_qp_keyms"] = f(rk["quadpath"], 1)
N["abl_qp24_keyms"] = f(rk["qp2024"], 1)

# ----------------------------------------------------------------------------------------------- overheads
N["tag_overhead_gray"] = f(100 * 4 / 1024, 2)
N["tag_overhead_color"] = f(100 * 4 / 3072, 2)
N["fixed_overhead"] = 4 + 1 + 4 + 4 + 4 + 16 + 16  # magic, version, H, W, (C,R,B,T), nonce, tag

os.makedirs(OUT, exist_ok=True)
json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, ensure_ascii=False)
json.dump(T, open(os.path.join(OUT, "tables.json"), "w"), indent=1, ensure_ascii=False)
print(f"{len(N)} numbers, {len(T)} tables -> {OUT}")


# =============================================================================================== compact variants
# Tables used by the 12-15-page compact manuscript; the full tables above go to the Supplementary Material.
def keep_rows(t, pred, **kw):
    d = dict(t)
    d["rows"] = [r for r in t["rows"] if pred(r)]
    d.update(kw)
    return d


T["qp24perm_compact"] = keep_rows(T["qp24perm"], lambda r: r[0] in ("64 × 64", "256 × 256", "512 × 512", "1024 × 1024",
                                                                    "512 × 768"))
T["differential_compact"] = keep_rows(
    T["differential"], lambda r: not str(r[0]).startswith("  "),
    caption=f"Plaintext sensitivity under a fixed key and nonce (nonce misuse): {N['diff_trials']} single-byte changes "
            f"(±1) per scheme over the non-constant test images. Pass: NPCR above N*_0.05 and UACI inside the 5% "
            "interval of Wu et al. [@wu2011]; a truly random cipher passes about 95% of trials.")
T["misuse_compact"] = keep_rows(
    T["misuse"], lambda r: r[0] != "ChaCha20-Poly1305",
    caption=T["misuse"]["caption"] + " ChaCha20-Poly1305 gives the same values as AES-256-CTR.")

# merged diffusion table: large avalanche study (moments, lost avalanche) + six-image ablation (structure, span, time)
rows = []
for sched, lab in [("FB1", "FB1: forward + backward, window 1"), ("FBF1", "FBF1: three passes, window 1"),
                   ("FB2", "FB2: forward + backward, window 2"), ("FBF2", "FBF2: three passes, window 2 (proposed)"),
                   ("additive", "Linear chaining, no S-boxes"), ("independent random images", "Independent random images")]:
    r = a.loc[sched]
    if sched in adf.schedule.values:
        dd_ = adf[adf.schedule == sched]
        top = f(100 * dd_.onepixel_top_diff_share.mean(), 2)
        span = "–" if pd.isna(dd_.error_expansion.iloc[0]) else int(dd_.error_expansion.iloc[0])
        tm_ = f(dd_.enc_ms.mean(), 1)
    else:
        top, span, tm_ = "–", "–", "–"
    rows.append([lab, f"{r.npcr_mean:.4f} ± {r.npcr_sd:.4f}", f"{r.uaci_mean:.4f} ± {r.uaci_sd:.4f}",
                 int(r.lost_avalanche), top, span, tm_])
T["diffusion"] = dict(caption=f"Diffusion variants under a fixed key and nonce. NPCR, UACI and lost avalanche (NPCR "
                              f"below 99%): {N['aval_trials']} single-byte changes of a 256 × 256 crop of Cameraman. "
                              "Top difference: share of the most frequent non-zero byte difference between the two "
                              "cipher images (six images, ideal 0.39%). Span: plain bytes affected by one corrupted "
                              "cipher byte. Time: encryption of a 512 × 512 × 3 image without tags (ms).",
                      header=["Diffusion", "NPCR (%)", "UACI (%)", "Lost avalanche", "Top difference (%)", "Span",
                              "Time"], rows=rows, align=["l", "r", "r", "r", "r", "r", "r"])

rows = []
for sch, op, lab in [("QuadPath-AE", "enc", "QuadPath-AE encryption"), ("QuadPath-AE", "dec", "QuadPath-AE decryption"),
                     ("LOG-PD", "enc", "LOG-PD encryption"), ("AES-256-GCM", "enc", "AES-256-GCM encryption"),
                     ("ChaCha20-Poly1305", "enc", "ChaCha20-Poly1305 encryption")]:
    row = [lab]
    for side in (512, 2048):
        d_ = spd[(spd.scheme == sch) & (spd.op == op) & (spd.side == side) & (spd.channels == 3)].iloc[0]
        row.append(f"{d_.ms:.1f} ms ({d_.MBps:.0f} MB/s)")
    rows.append(row)
T["speed_compact"] = dict(caption="Single-thread run time (throughput) for color images on an Apple M4 laptop, median "
                                  "of repeated runs; QuadPath-AE includes key derivation and all tags."
                                  + (" Measured on battery power with macOS Low Power Mode enabled; all schemes under "
                                     "the same conditions." if env.get("low_power_mode") else ""),
                          header=["Operation", "512 × 512 × 3", "2048 × 2048 × 3"], rows=rows, align=["l", "r", "r"])

T["abl_scrambler_compact"] = keep_rows(
    T["abl_scrambler"], lambda r: "tile size" not in r[0],
    caption="Ablation of the scrambling layer and of the number of rounds (six images; centered occlusion of 1/16 and "
            "1/4 of the cipher image, exact error mask, harmonic inpainting). Isolation: share of unreliable pixels "
            "without an unreliable 8-neighbor (1/16). Hole: largest connected unreliable region (pixels). Key setup: "
            "round-key derivation for a 512 × 512 × 3 image.")
rows = []
for B in [4, 8, 16, 32, 64]:
    d_ = atl[atl.B == B]
    a16, a4 = d_[d_.occlusion == "1/16"], d_[d_.occlusion == "1/4"]
    rows.append([B, f(a16.psnr.mean(), 2), f(a4.psnr.mean(), 2), f(a16.isolation.mean(), 3), f(a16.largest_hole.mean(), 0)])
T["abl_tile"] = dict(caption="Influence of the scrambling tile size B (six images, exact mask, harmonic inpainting).",
                     header=["B", "PSNR 1/16 (dB)", "PSNR 1/4 (dB)", "Isolation (1/16)", "Largest hole (1/16)"],
                     rows=rows, align=["r"] * 5)
p16 = atl[atl.occlusion == "1/16"].groupby("B").psnr.mean()
N["abl_tile_min"], N["abl_tile_max"] = f(p16.min(), 2), f(p16.max(), 2)

rows = [[r.test, r.subtest, f"{r.reference:.6f}", f"{r.ours:.6f}"] for _, r in vne.iterrows()]
T["nist_validation"] = dict(caption="Validation of our NIST SP 800-22 implementation on the first 10^6^ binary digits of "
                                    "e: reference P-values of SP 800-22 Rev. 1a, Appendix B, and ours. For the "
                                    "overlapping-template test our class counts equal those of Sec. 2.8.8 "
                                    f"({', '.join(map(str, val['overlapping_template_nu']))}); Appendix B used the "
                                    "earlier class probabilities.",
                            header=["Test", "Sub-test", "Reference", "Ours"], rows=rows, align=["l", "r", "r", "r"])

json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, ensure_ascii=False)
json.dump(T, open(os.path.join(OUT, "tables.json"), "w"), indent=1, ensure_ascii=False)
print(f"(compact) {len(N)} numbers, {len(T)} tables -> {OUT}")


# =============================================================================================== MDPI / deep-learning additions
N["qp24_cycles_512"] = int(qpp[(qpp.H == 512) & (qpp.W == 512)].cycles.iloc[0])
if os.path.exists(os.path.join(RES, "ml_recovery.csv")):
    e1 = R("ml_distinguisher.csv").set_index("scheme")
    e2 = R("ml_source_id.csv").set_index("scheme")
    e3 = R("ml_recovery.csv").set_index("scheme")
    P1 = lambda x: f"{100 * x:.1f}"  # noqa: E731
    N["ml_e1_train"] = f"{int(e1.train.iloc[0]):,}".replace(",", " ")
    N["ml_e1_test"] = f"{int(e1.test.iloc[0]):,}".replace(",", " ")
    for s, k in [("QUAD-PATH'24", "qp24"), ("LOG-PD", "logpd"), ("AES-256-CTR", "aes"), ("P-CTR", "pctr"),
                 ("QuadPath-AE", "ours")]:
        N[f"ml_e1_{k}"] = P1(e1.loc[s, "accuracy"])
    N["ml_e1_ours_ci"] = f"{100 * e1.loc['QuadPath-AE', 'ci_lo']:.1f}–{100 * e1.loc['QuadPath-AE', 'ci_hi']:.1f}"
    N["ml_e2_train"] = f"{int(e2.train.iloc[0]):,}".replace(",", " ")
    N["ml_e2_test"] = f"{int(e2.test.iloc[0]):,}".replace(",", " ")
    N["ml_e2_chance"] = P1(e2.chance.iloc[0])
    for s, k in [("Plain", "plain"), ("QUAD-PATH'24", "qp24"), ("LOG-PD", "logpd"), ("AES-256-CTR", "aes"),
                 ("QuadPath-AE", "ours")]:
        N[f"ml_e2_{k}"] = P1(e2.loc[s, "accuracy"])
    N["ml_e3_train"] = f"{int(e3.train_pairs.iloc[0]):,}".replace(",", " ")
    N["ml_e3_test"] = f"{int(e3.test_patches.iloc[0]):,}".replace(",", " ")
    N["ml_e3_floor"] = f(e3.floor_psnr.iloc[0], 2)
    for s, k in [("QUAD-PATH'24", "qp24"), ("AES-256-CTR (nonce reuse)", "aes"), ("P-CTR (nonce reuse)", "pctr"),
                 ("LOG-PD", "logpd"), ("QuadPath-AE (nonce reuse)", "ours"), ("QuadPath-AE (fresh nonces)", "fresh")]:
        N[f"ml_e3_{k}_psnr"] = f(e3.loc[s, "psnr"], 2)
        N[f"ml_e3_{k}_bit"] = P1(e3.loc[s, "bit_accuracy"])
        N[f"ml_e3_{k}_ssim"] = f(e3.loc[s, "ssim"], 3)

    def e1c(s):
        return f"{P1(e1.loc[s, 'accuracy'])} ({100 * e1.loc[s, 'ci_lo']:.1f}–{100 * e1.loc[s, 'ci_hi']:.1f})" \
            if s in e1.index else "–"

    def e2c(s):
        return f"{P1(e2.loc[s, 'accuracy'])}" if s in e2.index else "–"

    def e3c(s):
        return (f"{e3.loc[s, 'psnr']:.2f} / {e3.loc[s, 'ssim']:.3f} / {P1(e3.loc[s, 'bit_accuracy'])}"
                if s in e3.index else "–")

    rows = [["Plain image", "–", e2c("Plain"), "–"],
            ["QUAD-PATH'24", e1c("QUAD-PATH'24"), e2c("QUAD-PATH'24"), e3c("QUAD-PATH'24")],
            ["LOG-PD", e1c("LOG-PD"), e2c("LOG-PD"), e3c("LOG-PD")],
            ["AES-256-CTR", e1c("AES-256-CTR"), e2c("AES-256-CTR"), e3c("AES-256-CTR (nonce reuse)")],
            ["P-CTR", e1c("P-CTR"), "–", e3c("P-CTR (nonce reuse)")],
            ["QuadPath-AE", e1c("QuadPath-AE"), e2c("QuadPath-AE"), e3c("QuadPath-AE (nonce reuse)")],
            ["QuadPath-AE, fresh nonces", "–", "–", e3c("QuadPath-AE (fresh nonces)")]]
    T["ml"] = dict(caption="Deep-learning attacks on cipher images (test sets of unseen images). E1: neural "
                           "distinguisher accuracy (%) with 95% Wilson interval; chance 50%. E2: source-identification "
                           f"accuracy (%); chance {N['ml_e2_chance']}%. E3: plaintext recovery under nonce reuse "
                           f"(one key and one nonce for all {N['ml_e3_train']} training pairs), PSNR (dB) / SSIM / "
                           f"correct bits (%); a mean-patch predictor scores {N['ml_e3_floor']} dB. LOG-PD has no "
                           "nonce, so E3 uses its fixed key.",
                   header=["Cipher", "E1 distinguisher (%)", "E2 source ID (%)", "E3 recovery: PSNR / SSIM / bits (%)"],
                   rows=rows, align=["l", "r", "r", "r"])

# independent rerun of the deep-learning attacks by the author on Google Colab (results/colab_rerun/)
CR = os.path.join(RES, "colab_rerun")
if os.path.exists(os.path.join(CR, "ml_recovery.csv")):
    def cmp(name, col):
        a = pd.read_csv(os.path.join(CR, name)).set_index("scheme")[col]
        b = pd.read_csv(os.path.join(RES, name)).set_index("scheme")[col]
        return float((a - b).abs().max())
    N["colab_e1_maxdiff"] = f"{100 * cmp('ml_distinguisher.csv', 'accuracy'):.1f}"
    N["colab_e2_maxdiff"] = f"{100 * cmp('ml_source_id.csv', 'accuracy'):.1f}"
    N["colab_e3_maxdiff"] = f"{cmp('ml_recovery.csv', 'psnr'):.1f}"
    ce = json.load(open(os.path.join(CR, "ml_env.json")))
    N["colab_torch"] = ce["torch"].split("+")[0]
    N["colab_kat_identical"] = "bit for bit" if json.load(open(os.path.join(CR, "kat.json"))) == \
        json.load(open(os.path.join(RES, "kat.json"))) else "NOT identically"

json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, ensure_ascii=False)
json.dump(T, open(os.path.join(OUT, "tables.json"), "w"), indent=1, ensure_ascii=False)
print(f"(mdpi) {len(N)} numbers, {len(T)} tables -> {OUT}")
