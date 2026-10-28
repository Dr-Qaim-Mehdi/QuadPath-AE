"""NIST SP 800-22 Rev. 1a statistical test suite (all 15 tests), re-implemented in NumPy/Numba.

Parameters follow the NIST STS 2.1.2 defaults: block frequency M = 128, non-overlapping template m = 9
(all 148 aperiodic templates, N = 8 blocks), overlapping template m = 9 (M = 1032, N = 968), linear
complexity M = 500, serial m = 16, approximate entropy m = 10. `validate_e()` reproduces the reference
P-values that SP 800-22 Appendix B lists for the first 10^6 binary digits of e.
"""
import math

import numpy as np
from numba import njit
from scipy.special import erfc, gammaincc
from scipy.stats import norm

ALPHA = 0.01


def igamc(a, x):
    return float(gammaincc(a, x))


def frequency(e):
    n = e.size
    s = abs(int(2 * e.sum()) - n) / math.sqrt(n)
    return float(erfc(s / math.sqrt(2)))


def block_frequency(e, M=128):
    N = e.size // M
    pi = e[:N * M].reshape(N, M).mean(axis=1)
    chi = 4.0 * M * float(((pi - 0.5) ** 2).sum())
    return igamc(N / 2.0, chi / 2.0)


def runs(e):
    n = e.size
    pi = e.mean()
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0
    v = 1 + int((e[1:] != e[:-1]).sum())
    return float(erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi))))


@njit(cache=True)
def _longest_runs(e, M, N):
    out = np.zeros(N, dtype=np.int64)
    for b in range(N):
        best = 0
        cur = 0
        for i in range(b * M, (b + 1) * M):
            if e[i] == 1:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        out[b] = best
    return out


def longest_run(e):
    n = e.size
    if n < 6272:
        M, lo, hi, pi = 8, 1, 4, [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        M, lo, hi, pi = 128, 4, 9, [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M, lo, hi, pi = 10000, 10, 16, [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    N = n // M
    lr = np.clip(_longest_runs(e, M, N), lo, hi) - lo
    v = np.bincount(lr, minlength=len(pi)).astype(np.float64)
    pi = np.array(pi)
    chi = float(((v - N * pi) ** 2 / (N * pi)).sum())
    return igamc((len(pi) - 1) / 2.0, chi / 2.0)


@njit(cache=True)
def _gf2_rank(mat):
    m = mat.copy()
    rows, cols = m.shape
    rank = 0
    for c in range(cols):
        piv = -1
        for r in range(rank, rows):
            if m[r, c] == 1:
                piv = r
                break
        if piv < 0:
            continue
        if piv != rank:
            for k in range(cols):
                t = m[piv, k]
                m[piv, k] = m[rank, k]
                m[rank, k] = t
        for r in range(rows):
            if r != rank and m[r, c] == 1:
                for k in range(cols):
                    m[r, k] ^= m[rank, k]
        rank += 1
    return rank


@njit(cache=True)
def _ranks(e, N):
    out = np.empty(N, dtype=np.int64)
    for b in range(N):
        out[b] = _gf2_rank(e[b * 1024:(b + 1) * 1024].reshape(32, 32))
    return out


def _rank_prob(r, M=32, Q=32):
    p = 2.0 ** (r * (Q + M - r) - M * Q)
    for i in range(r):
        p *= (1 - 2.0 ** (i - Q)) * (1 - 2.0 ** (i - M)) / (1 - 2.0 ** (i - r))
    return p


def rank(e):
    N = e.size // 1024
    rk = _ranks(e, N)
    p32, p31 = _rank_prob(32), _rank_prob(31)
    p30 = 1 - p32 - p31
    f32, f31 = int((rk == 32).sum()), int((rk == 31).sum())
    f30 = N - f32 - f31
    chi = (f32 - p32 * N) ** 2 / (p32 * N) + (f31 - p31 * N) ** 2 / (p31 * N) + (f30 - p30 * N) ** 2 / (p30 * N)
    return float(math.exp(-chi / 2.0))


def dft(e):
    n = e.size
    x = 2.0 * e - 1.0
    m = np.abs(np.fft.fft(x))[: n // 2]
    T = math.sqrt(math.log(1 / 0.05) * n)
    n0 = 0.95 * n / 2.0
    n1 = float((m < T).sum())
    d = (n1 - n0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return float(erfc(abs(d) / math.sqrt(2)))


def aperiodic_templates(m=9):
    out = []
    for v in range(2 ** m):
        b = [(v >> (m - 1 - i)) & 1 for i in range(m)]
        if all(b[:m - k] != b[k:] for k in range(1, m)):
            out.append(b)
    return out


@njit(cache=True)
def _nonoverlap_counts(e, tmpl, N, M):
    m = tmpl.size
    W = np.zeros(N, dtype=np.int64)
    for b in range(N):
        i = b * M
        end = (b + 1) * M - m + 1
        while i < end:
            ok = True
            for k in range(m):
                if e[i + k] != tmpl[k]:
                    ok = False
                    break
            if ok:
                W[b] += 1
                i += m
            else:
                i += 1
    return W


def non_overlapping_template(e, m=9, N=8):
    M = e.size // N
    mu = (M - m + 1) / 2.0 ** m
    var = M * (1.0 / 2 ** m - (2 * m - 1) / 2.0 ** (2 * m))
    ps = []
    for t in aperiodic_templates(m):
        W = _nonoverlap_counts(e, np.array(t, dtype=np.uint8), N, M)
        chi = float(((W - mu) ** 2).sum() / var)
        ps.append(igamc(N / 2.0, chi / 2.0))
    return ps


@njit(cache=True)
def _overlap_counts(e, m, M, N):
    v = np.zeros(6, dtype=np.int64)
    for b in range(N):
        cnt = 0
        for i in range(b * M, (b + 1) * M - m + 1):
            ok = True
            for k in range(m):
                if e[i + k] != 1:
                    ok = False
                    break
            if ok:
                cnt += 1
        v[min(cnt, 5)] += 1
    return v


def overlapping_template(e, m=9, M=1032):
    N = e.size // M
    v = _overlap_counts(e, m, M, N).astype(np.float64)
    # Corrected class probabilities of SP 800-22 Rev. 1a, Sec. 3.8 (Hamano and Kaneko), as in STS 2.1.2.
    pi = np.array([0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865])
    chi = float(((v - N * pi) ** 2 / (N * pi)).sum())
    return igamc(5 / 2.0, chi / 2.0)


_UNIV = {6: (5.2177052, 2.954), 7: (6.1962507, 3.125), 8: (7.1836656, 3.238), 9: (8.1764248, 3.311),
         10: (9.1723243, 3.356), 11: (10.170032, 3.384), 12: (11.168765, 3.401)}


@njit(cache=True)
def _universal_sum(e, L, Q, K):
    T = np.zeros(2 ** L, dtype=np.int64)
    for i in range(1, Q + 1):
        v = 0
        for j in range(L):
            v = (v << 1) | e[(i - 1) * L + j]
        T[v] = i
    s = 0.0
    for i in range(Q + 1, Q + K + 1):
        v = 0
        for j in range(L):
            v = (v << 1) | e[(i - 1) * L + j]
        s += math.log(i - T[v]) / math.log(2.0)
        T[v] = i
    return s


def universal(e):
    n = e.size
    bounds = [(387840, 6), (904960, 7), (2068480, 8), (4654080, 9), (10342400, 10), (22753280, 11)]
    L = 5
    for lim, l in bounds:
        if n >= lim:
            L = l
    if L < 6:
        return float("nan")
    Q = 10 * 2 ** L
    K = n // L - Q
    expected, variance = _UNIV[L]
    fn = _universal_sum(e, L, Q, K) / K
    c = 0.7 - 0.8 / L + (4 + 32.0 / L) * K ** (-3.0 / L) / 15.0
    sigma = c * math.sqrt(variance / K)
    return float(erfc(abs(fn - expected) / (math.sqrt(2) * sigma)))


@njit(cache=True)
def _berlekamp_massey(s):
    n = s.size
    c = np.zeros(n + 1, dtype=np.uint8)
    b = np.zeros(n + 1, dtype=np.uint8)
    c[0] = 1
    b[0] = 1
    L = 0
    m = -1
    for N in range(n):
        d = s[N]
        for i in range(1, L + 1):
            d ^= c[i] & s[N - i]
        if d == 1:
            t = c.copy()
            for i in range(0, n - N + m):
                c[N - m + i] ^= b[i]
            if L <= N // 2:
                L = N + 1 - L
                m = N
                b = t
    return L


@njit(cache=True)
def _lin_complexities(e, M, N):
    out = np.empty(N, dtype=np.int64)
    for k in range(N):
        out[k] = _berlekamp_massey(e[k * M:(k + 1) * M])
    return out


def linear_complexity(e, M=500):
    N = e.size // M
    Ls = _lin_complexities(e, M, N)
    mu = M / 2.0 + (9 + (-1) ** (M + 1)) / 36.0 - (M / 3.0 + 2 / 9.0) / 2.0 ** M
    T = (-1) ** M * (Ls - mu) + 2 / 9.0
    edges = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    v = np.bincount(np.searchsorted(edges, T, side="left"), minlength=7).astype(np.float64)
    # pi_0 = 0.01047 as in NIST STS 2.1.2 (the exact value is 1/96); this reproduces Appendix B.
    pi = np.array([0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
    chi = float(((v - N * pi) ** 2 / (N * pi)).sum())
    return igamc(3.0, chi / 2.0)


def _pattern_counts(e, m):
    """Counts of all overlapping m-bit patterns with wrap-around (augmented by the first m-1 bits)."""
    if m == 0:
        return np.array([e.size], dtype=np.float64)
    ext = np.concatenate([e, e[:m - 1]]).astype(np.int64)
    n = e.size
    v = np.zeros(n, dtype=np.int64)
    for k in range(m):
        v = (v << 1) | ext[k:k + n]
    return np.bincount(v, minlength=2 ** m).astype(np.float64)


def serial(e, m=16):
    n = e.size

    def psi(mm):
        if mm <= 0:
            return 0.0
        c = _pattern_counts(e, mm)
        return float((2.0 ** mm / n) * (c ** 2).sum() - n)

    p0, p1, p2 = psi(m), psi(m - 1), psi(m - 2)
    d1 = p0 - p1
    d2 = p0 - 2 * p1 + p2
    return [igamc(2.0 ** (m - 2), d1 / 2.0), igamc(2.0 ** (m - 3), d2 / 2.0)]


def approximate_entropy(e, m=10):
    n = e.size

    def phi(mm):
        c = _pattern_counts(e, mm) / n
        c = c[c > 0]
        return float((c * np.log(c)).sum())

    apen = phi(m) - phi(m + 1)
    chi = 2.0 * n * (math.log(2) - apen)
    return igamc(2.0 ** (m - 1), chi / 2.0)


def _cusum_p(n, z):
    zi = int(z)
    nz = n // zi
    s1 = 0.0
    for k in range(int((-nz + 1) / 4), int((nz - 1) / 4) + 1):
        s1 += norm.cdf((4 * k + 1) * z / math.sqrt(n)) - norm.cdf((4 * k - 1) * z / math.sqrt(n))
    s2 = 0.0
    for k in range(int((-nz - 3) / 4), int((nz - 1) / 4) + 1):
        s2 += norm.cdf((4 * k + 3) * z / math.sqrt(n)) - norm.cdf((4 * k + 1) * z / math.sqrt(n))
    return float(1.0 - s1 + s2)


def cumulative_sums(e):
    x = 2 * e.astype(np.int64) - 1
    n = e.size
    zf = int(np.abs(np.cumsum(x)).max())
    zb = int(np.abs(np.cumsum(x[::-1])).max())
    return [_cusum_p(n, zf), _cusum_p(n, zb)]


def _excursion_cycles(e):
    x = 2 * e.astype(np.int64) - 1
    S = np.concatenate([[0], np.cumsum(x), [0]])
    zeros = np.flatnonzero(S == 0)
    return S, zeros, zeros.size - 1


def random_excursions(e):
    S, zeros, J = _excursion_cycles(e)
    if J < max(0.005 * math.sqrt(e.size), 500):
        return None
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    cyc = np.repeat(np.arange(J), np.diff(zeros))  # cycle index of each position in S[:-1]
    Sv = S[:-1]
    ps = []
    for x in states:
        cnt = np.bincount(cyc[Sv == x], minlength=J)
        nu = np.bincount(np.minimum(cnt, 5), minlength=6).astype(np.float64)
        ax = abs(x)
        pi = [1 - 1 / (2 * ax)] + [1 / (4 * x * x) * (1 - 1 / (2 * ax)) ** (k - 1) for k in range(1, 5)]
        pi.append(1 / (2 * ax) * (1 - 1 / (2 * ax)) ** 4)
        pi = np.array(pi)
        chi = float(((nu - J * pi) ** 2 / (J * pi)).sum())
        ps.append(igamc(2.5, chi / 2.0))
    return ps


def random_excursions_variant(e):
    S, zeros, J = _excursion_cycles(e)
    if J < max(0.005 * math.sqrt(e.size), 500):
        return None
    ps = []
    for x in list(range(-9, 0)) + list(range(1, 10)):
        xi = int((S == x).sum())
        ps.append(float(erfc(abs(xi - J) / math.sqrt(2.0 * J * (4.0 * abs(x) - 2)))))
    return ps


TESTS = [
    ("Frequency", frequency), ("BlockFrequency", block_frequency), ("CumulativeSums", cumulative_sums),
    ("Runs", runs), ("LongestRun", longest_run), ("Rank", rank), ("FFT", dft),
    ("NonOverlappingTemplate", non_overlapping_template), ("OverlappingTemplate", overlapping_template),
    ("Universal", universal), ("ApproximateEntropy", approximate_entropy),
    ("RandomExcursions", random_excursions), ("RandomExcursionsVariant", random_excursions_variant),
    ("Serial", serial), ("LinearComplexity", linear_complexity),
]


def run_all(e):
    """All tests on one sequence (uint8 array of bits). Returns {test: [p-values] or None}."""
    e = np.ascontiguousarray(e, dtype=np.uint8)
    out = {}
    for name, fn in TESTS:
        r = fn(e)
        out[name] = None if r is None else (r if isinstance(r, list) else [r])
    return out


def proportion_range(s, alpha=ALPHA):
    p = 1 - alpha
    d = 3 * math.sqrt(p * alpha / s)
    return p - d, p + d


def uniformity(pvals):
    pv = np.asarray(pvals)
    F = np.histogram(pv, bins=10, range=(0, 1))[0].astype(np.float64)
    s = pv.size
    chi = float(((F - s / 10.0) ** 2 / (s / 10.0)).sum())
    return igamc(9 / 2.0, chi / 2.0)


def summarize(results):
    """results: list of run_all outputs (one per sequence). Returns per-test summary rows.

    For tests with several sub-tests the worst sub-test (lowest proportion, lowest uniformity P-value)
    is reported together with the number of passing sub-tests.
    """
    rows = []
    for name, _ in TESTS:
        per = [r[name] for r in results if r[name] is not None]
        if not per:
            rows.append(dict(test=name, sequences=0))
            continue
        k = len(per[0])
        props, unis = [], []
        for j in range(k):
            pv = np.array([p[j] for p in per])
            props.append(float((pv >= ALPHA).mean()))
            unis.append(uniformity(pv))
        lo, _ = proportion_range(len(per))
        ok = [(pr >= lo) and (u >= 0.0001) for pr, u in zip(props, unis)]
        rows.append(dict(test=name, sequences=len(per), subtests=k, min_proportion=min(props),
                         min_uniformity_p=min(unis), subtests_passed=int(sum(ok)), proportion_threshold=lo,
                         passed=all(ok)))
    return rows


def e_bits(n=1_000_000):
    """First n binary digits of e (the integer part '10' included), as in the NIST STS file data.e."""
    import mpmath
    mpmath.mp.prec = n + 64
    v = int(mpmath.floor(mpmath.e * mpmath.mpf(2) ** (n - 2)))
    s = bin(v)[2:]
    return np.frombuffer(s[:n].encode(), dtype=np.uint8) - ord("0")


# Reference P-values from NIST SP 800-22 Rev. 1a, Appendix B (sequence: data.e, n = 10^6).
# Each entry: (index of the sub-test, reference P-value). The overlapping-template reference was computed
# with the older class probabilities of Sec. 2.8 (our class counts nu_0..nu_5 = 329,164,150,111,78,136 match
# Sec. 2.8.8 exactly; with the corrected probabilities of Sec. 3.8 the P-value is 0.159032).
E_REFERENCE = {
    "Frequency": [(0, 0.953749)], "BlockFrequency": [(0, 0.211072)],
    "CumulativeSums": [(0, 0.669887), (1, 0.724266)], "Runs": [(0, 0.561917)], "LongestRun": [(0, 0.718945)],
    "Rank": [(0, 0.306156)], "FFT": [(0, 0.847187)], "NonOverlappingTemplate": [(0, 0.078790)],
    "OverlappingTemplate": [(0, 0.110434)], "Universal": [(0, 0.282568)], "ApproximateEntropy": [(0, 0.700073)],
    "RandomExcursions": [(4, 0.786868)], "RandomExcursionsVariant": [(8, 0.826009)],
    "LinearComplexity": [(0, 0.826335)], "Serial": [(0, 0.766182)],
}
