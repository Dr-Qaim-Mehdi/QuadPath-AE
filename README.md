# QuadPath-AE

Reference implementation, experiments and results for

> Q. M. Rizvi, M. A. Hassan, T. Fatima, S. Talat and M. Awais, "QuadPath-AE: Authenticated Image Encryption with Keyed Quad-Path Scrambling, Evaluated against
> Statistical and Deep-Learning Attacks," submitted to *Machine Learning and Knowledge Extraction* (MDPI).

Archived on Zenodo: [doi:10.5281/zenodo.23723205](https://doi.org/10.5281/zenodo.23723205)

QuadPath-AE encrypts an `H × W × C` 8-bit image under a 256-bit key and a fresh 128-bit nonce:

1. **Key schedule:** `(Ke ‖ Km) = HKDF-Expand(K, "QPAE-v1" ‖ header ‖ nonce)`. Every secret is drawn from a purpose-labelled ChaCha20 keystream.
2. **Keyed quad-path scrambling:** each 16 × 16 tile is read along one of four paths (zig-zag, snake, spiral, Hilbert) under one of the 8 symmetries of the square, in a keyed tile order. A golden-ratio interleaver then disperses neighbours.
3. **FBF2 diffusion:** forward, backward and forward chaining, `y_i = S[(x_i + y_{i-1} + Q[y_{i-2}]) mod 256] XOR k_i`, with keyed S-boxes `S` and `Q`.
4. **Encrypt-then-MAC:** 32-bit HMAC tags per 32 × 32 cipher tile (damage localisation) and a 128-bit global tag.

Decryption is non-recursive, so one corrupted cipher byte affects at most 7 plain bytes. Strict mode rejects any modified image. Resilient mode returns the image, the exact mask of unreliable pixels, and the harmonically inpainted result.

```python
from qpae import cipher, recovery
import os
key = os.urandom(32)
sealed = cipher.encrypt(image, key)                 # fresh random nonce
blob = sealed.to_bytes()                            # header ‖ nonce ‖ cipher image ‖ tile tags ‖ tag
plain = cipher.decrypt(sealed, key)                 # strict: raises AuthenticationError if modified
plain, mask, failed_tiles, ok = cipher.decrypt_resilient(sealed, key)
repaired = recovery.harmonic_inpaint(plain, mask)
```

## Contents

| Path | What it is |
|---|---|
| `qpae/paths.py` | the four path families, D4 variants, keyed tile scrambling, golden interleaver, QUAD-PATH (2024) permutation |
| `qpae/cipher.py` | key schedule, diffusion kernels and their inverses, error propagation, tags, strict/resilient decryption |
| `qpae/baselines.py` | QUAD-PATH'24, LOG-PD, AES-256-CTR/GCM, ChaCha20-Poly1305, P-CTR ablation |
| `qpae/metrics.py` | entropy, local entropy, χ², adjacent correlation, NPCR/UACI with critical values, PSNR/SSIM |
| `qpae/nist.py` | NIST SP 800-22 Rev. 1a, all 15 tests (validated on the binary expansion of *e*) |
| `qpae/recovery.py` | harmonic inpainting, median filter, dispersion metrics |
| `scripts/exp_*.py` | experiments; each writes CSV/JSON to `results/` |
| `scripts/exp_ml.py` | deep-learning attacks: neural distinguisher (E1), source identification (E2), plaintext recovery under nonce reuse (E3) |
| `scripts/make_figures.py`, `scripts/make_tables.py` | all figures (PDF + 600-dpi PNG) and the paper's tables/numbers |
| `scripts/make_kat.py` | known-answer test vectors (`results/kat.json`) and `--check` |
| `reproduce_colab.ipynb` | one-click reproduction on Google Colab (GPU) |
| `results/` | every number reported in the paper |
| `results/colab_rerun/` | independent rerun of the deep-learning attacks and validation on Google Colab (CUDA, Linux); conclusions identical, known-answer vectors bit-exact |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_kodak.sh data
bash scripts/run_all.sh
```

All test images except Kodak-24 ship with scikit-image. Experiments use a fixed master key and label-derived nonces so that results are reproducible; real use must draw a fresh random nonce per image, as `cipher.encrypt` does by default.
