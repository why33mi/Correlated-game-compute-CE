"""
Pure game-theory verification (no statistical mechanics) -- corrected version.

Snowdrift, player 1's payoff matrix:
        C     D
  C     1     s          0 < s < 1 < t < 2
  D     t     0

Device p(i,j) = exp(-H(i,j)/T)/Z, H = [[-1,-t],[-t,-(t-s)]]
   (only a convenient 1-parameter curve of joint laws:
    T->0 : uniform on {CD,DC};  T->oo : uniform on all four states)

Channel  P(a,b) = [[a, 1-b],[1-a, b]],  p^R = P p P^T.
   alpha := a = P(play C | told C),  gamma := 1-b = P(play C | told D).

With p = [[x,y],[y,w]] and c_dev = x+y (the device's marginal for "told C"):

        c' = p^R(CC)+p^R(CD) = alpha*c_dev + gamma*(1-c_dev)
        x' = p^R(CC)         = alpha^2 x + 2 alpha gamma y + gamma^2 w
        u  = x'(1-s-t) + c'(s+t)                     [decreasing in x']
        CE1 <=> x' <= s c'/(s+t-1)
        CE2 <=> x' <= (c'(t-1+2s)-s)/(s+t-1)
        validity: max(0, 2c'-1) <= x' <= c'
"""

import math
import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=4, suppress=True)


# ------------------------------------------------------------------ basics
def device(s, t, T):
    beta = 0.0 if T == math.inf else 1.0 / T
    H = np.array([[-1.0, -t], [-t, -(t - s)]])
    W = np.exp(-beta * H)
    return W / W.sum()


def pR(p, a, b):
    P = np.array([[a, 1.0 - b], [1.0 - a, b]])
    return P @ p @ P.T


def payoff(q, s, t):
    return q[0, 0] + s * q[0, 1] + t * q[1, 0]


def slacks(q, s, t):
    return (q[0, 0] + s * q[0, 1] - t * q[0, 0],
            t * q[1, 0] - q[1, 0] - s * q[1, 1])


def mixed(s, t):
    d = s + t - 1
    return s / d, s * t / d


def T_crit(s, t):
    if abs(s - (t - 1)) < 1e-12:
        return math.inf
    return (t - 1) / math.log((t - 1) / s) if s < t - 1 else s / math.log(s / (t - 1))


def quad_extrema(A, B, C, lo, hi):
    """min, max of A*t^2+B*t+C on [lo,hi] (A may be ~0)."""
    cand = [lo, hi]
    if abs(A) > 1e-14:
        v = -B / (2 * A)
        if lo < v < hi:
            cand.append(v)
    vals = [A * z * z + B * z + C for z in cand]
    return min(vals), max(vals), cand


def best_channel(s, t, T, nc=400001):
    """max payoff over (a,b) in [0,1]^2 s.t. p^R is a CE.  Exact up to the c' grid."""
    p = device(s, t, T)
    x, y, w = p[0, 0], p[0, 1], p[1, 1]
    cd = x + y                      # device marginal: P(told to play C)
    d = s + t - 1
    best = (-9e9, None)
    for k in range(nc + 1):
        cp = k / nc
        # gamma = u - v*alpha,  u = cp/(1-cd),  v = cd/(1-cd)
        u0 = cp / (1 - cd)
        v0 = cd / (1 - cd)
        # alpha range with gamma in [0,1]
        alo = max(0.0, (cp - (1 - cd)) / cd)
        ahi = min(1.0, cp / cd)
        if alo > ahi + 1e-13:
            continue
        A = x - 2 * v0 * y + v0 * v0 * w
        B = 2 * u0 * y - 2 * u0 * v0 * w
        C = u0 * u0 * w
        xmin, xmax, _ = quad_extrema(A, B, C, alo, ahi)
        l1 = s * cp / d
        l2 = (cp * (t - 1 + 2 * s) - s) / d
        lb = max(0.0, 2 * cp - 1)
        hi = min(l2, l1, cp, xmax)
        lo = max(xmin, lb)
        if lo > hi + 1e-13:
            continue
        xstar = lo
        val = xstar * (1 - s - t) + cp * (s + t)
        if val > best[0]:
            # recover alpha with x'(alpha) = xstar
            al = None
            if abs(A) > 1e-14:
                disc = B * B - 4 * A * (C - xstar)
                if disc >= -1e-12:
                    for r in ((-B + math.sqrt(max(0, disc))) / (2 * A),
                              (-B - math.sqrt(max(0, disc))) / (2 * A)):
                        if alo - 1e-9 <= r <= ahi + 1e-9:
                            al = r
                            break
            if al is None:
                al = (xstar - C) / B if abs(B) > 1e-14 else alo
            al = min(1.0, max(0.0, al))
            ga = min(1.0, max(0.0, u0 - v0 * al))
            best = (val, (al, 1 - ga, cp, xstar, l1 - xstar, l2 - xstar))
    val, (a, b, cp, xstar, s1, s2) = best
    q = pR(p, a, b)
    return val, a, b, q


def binding(q, s, t, tol=3e-4):
    c1, c2 = slacks(q, s, t)
    out = []
    if c1 < tol:
        out.append("CE1")
    if c2 < tol:
        out.append("CE2")
    return "+".join(out) if out else "reachability"


# ---------------------------------------------------------------- main table
print("=" * 106)
print("Best payoff over channels with p^R required to be a CE, vs mixed payoff")
print("=" * 106)
rows = [(0.5, 1.6), (0.5, 1.2), (0.5, 1.4), (0.2, 1.5), (1.2, 1.5)]
for (s, t) in rows:
    pm, um = mixed(s, t)
    Tc = T_crit(s, t)
    print(f"\n### s={s}, t={t}  (s {'<' if s < t-1 else '>' if s > t-1 else '='} t-1={t-1:g})"
          f"   mixed: P(C)={pm:.4f}, payoff={um:.4f}")
    print(f"    device is a CE iff T <= {Tc if Tc == math.inf else round(Tc, 4)}")
    print("      T   | deviceCE? | follow |  best  |   gain   |  (a,b)      |"
          "   c'   | pR(CC) |  binding")
    for T in [0.05, 0.2, 0.5, 1.0, (Tc if Tc != math.inf else 2.0), 5.0, 20.0, 200.0]:
        p = device(s, t, T)
        c1, c2 = slacks(p, s, t)
        ce = "YES" if (c1 >= -1e-12 and c2 >= -1e-12) else "no"
        val, a, b, q = best_channel(s, t, T)
        assert abs(payoff(q, s, t) - val) < 1e-6, (val, payoff(q, s, t))
        q1, q2 = slacks(q, s, t)
        assert q1 > -1e-9 and q2 > -1e-9, (q1, q2)
        cp = q[0, 0] + q[0, 1]
        print(f"   {T:7.2f} | {ce:9s}  | {payoff(p, s, t):6.4f} | {val:6.4f} |"
              f" {val-um:+8.4f} | ({a:4.2f},{b:4.2f}) | {cp:6.4f} |"
              f" {q[0, 0]:6.4f} | {binding(q, s, t)}")

# ------------------------------------------------------------ cross-check grid
print()
print("=" * 106)
print("Cross-check against a brute-force grid over (a,b) (independent code path)")
print("=" * 106)
for (s, t, T) in [(0.5, 1.6, 0.5), (0.5, 1.2, 1.0), (1.2, 1.5, 5.0), (0.2, 1.5, 1.0)]:
    p = device(s, t, T)
    N = 1200
    g = np.linspace(0, 1, N)
    A_, B_ = np.meshgrid(g, g, indexing="ij")
    M = np.zeros(A_.shape + (2, 2))
    M[..., 0, 0], M[..., 0, 1] = A_, 1 - B_
    M[..., 1, 0], M[..., 1, 1] = 1 - A_, B_
    Q = np.einsum("...ik,kl,...jl->...ij", M, p, M)
    uu = Q[..., 0, 0] + s * Q[..., 0, 1] + t * Q[..., 1, 0]
    cc1 = Q[..., 0, 0] + s * Q[..., 0, 1] - t * Q[..., 0, 0]
    cc2 = t * Q[..., 1, 0] - Q[..., 1, 0] - s * Q[..., 1, 1]
    feas = (cc1 >= 0) & (cc2 >= 0)
    ug = np.where(feas, uu, -9e9)
    i, j = np.unravel_index(np.argmax(ug), ug.shape)
    val, a, b, q = best_channel(s, t, T)
    print(f"  s={s}, t={t}, T={T:5.2f}:  analytic={val:.6f} (a,b)=({a:.4f},{b:.4f})   "
          f"grid({N}x{N})={ug[i,j]:.6f} (a,b)=({A_[i,j]:.4f},{B_[i,j]:.4f})")

# ------------------------------------------------------------ closed form table
print()
print("=" * 106)
print("CE boundary payoff formulas (verified by construction)")
print("=" * 106)
for (s, t) in [(0.5, 1.6), (0.5, 1.2), (1.2, 1.5)]:
    d = s + t - 1
    pm, um = mixed(s, t)
    print(f"\n  s={s}, t={t}:  mixed at c={pm:.4f}, payoff={um:.4f}")
    print("     c    | CE1: p(CC)  payoff  (=c*t)  | CE2: p(CC)  payoff (=s+c(1-s))")
    for c in [0.30, 0.40, pm, 0.50, 0.60, 0.70, 0.80, 0.90]:
        x1, x2 = s * c / d, (c * (t - 1 + 2 * s) - s) / d

        def pf(x):
            if not (max(0.0, 2 * c - 1) - 1e-9 <= x <= c + 1e-9):
                return float("nan")
            return payoff(np.array([[x, c - x], [c - x, 1 - 2 * c + x]]), s, t)

        print(f"   {c:5.3f} | {x1:8.4f} {pf(x1):7.4f} {t*c:8.4f}  | "
              f"{x2:8.4f} {pf(x2):7.4f} {s + c*(1-s):8.4f}")
