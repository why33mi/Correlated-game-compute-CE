"""
Does "the device is a correlated equilibrium  =>  payoff better than Nash" hold?

Three separate questions (all answered by linear programs over the CE polytope):

  Q1  Is EVERY correlated equilibrium at least as good as the mixed NE?   -> NO (counterexample)
  Q2  Is the BEST correlated equilibrium strictly better?                 -> YES (closed form for SD)
  Q3  Is the WORST correlated equilibrium at least as good?               -> NO
"""

import numpy as np
from scipy.optimize import linprog

FLIP = np.array([[0.0, 1.0], [1.0, 0.0]])


def payoff_mats(game, s, t=None):
    """return (u1, u2); rows = player 1 action, cols = player 2 action."""
    if game == "SD":                      # symmetric: u2(i,j) = u1(j,i)
        u1 = np.array([[1.0, s], [t, 0.0]])
        return u1, u1.T
    if game == "BOS":                     # asymmetric: u2(i,j) = u1(flip j, flip i)
        u1 = np.array([[1.0, 0.0], [0.0, s]])
        return u1, FLIP @ u1 @ FLIP
    raise ValueError(game)


def ce_A(u1, u2):
    """Rows a with a.p >= 0 for the four obedience inequalities, p=(CC,CD,DC,DD)."""
    return np.array([
        [u1[0, 0] - u1[1, 0], u1[0, 1] - u1[1, 1], 0.0, 0.0],                    # p1 told C
        [0.0, 0.0, u1[1, 0] - u1[0, 0], u1[1, 1] - u1[0, 1]],                    # p1 told D
        [u2[0, 0] - u2[0, 1], 0.0, u2[1, 0] - u2[1, 1], 0.0],                    # p2 told C
        [0.0, u2[0, 1] - u2[0, 0], 0.0, u2[1, 1] - u2[1, 0]],                    # p2 told D
    ])


def pays(p, u1, u2):
    P = np.asarray(p).reshape(2, 2)
    return float((P * u1).sum()), float((P * u2).sum())


def mixed_ne(game, s, t=None):
    """(prob player 1 plays C, payoff of each player) in the mixed NE."""
    if game == "SD":
        d = s + t - 1
        return s / d, s * t / d
    return 1.0 / (1.0 + s), s / (1.0 + s)


def best_ce(game, s, t=None):
    """max over CE of min(u1,u2)  (both players at least v)."""
    u1, u2 = payoff_mats(game, s, t)
    A = ce_A(u1, u2)
    v1, v2 = u1.ravel(), u2.ravel()
    # variables x = (p0..p3, v);  maximise v
    c = np.zeros(5)
    c[4] = -1.0
    A_ub = np.vstack([
        np.hstack([-A, np.zeros((4, 1))]),                    # A p >= 0
        np.hstack([-v1[None, :], [[1.0]]]),                   # v - u1.p <= 0
        np.hstack([-v2[None, :], [[1.0]]]),                   # v - u2.p <= 0
    ])
    b_ub = np.zeros(A_ub.shape[0])
    A_eq = np.array([[1.0, 1.0, 1.0, 1.0, 0.0]])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, 1)] * 4 + [(None, None)], method="highs")
    p = np.clip(res.x[:4], 0, 1)
    p /= p.sum()
    return -res.fun, p, pays(p, u1, u2)


def worst_ce(game, s, t=None):
    """min over CE of the average payoff (u1+u2)/2  -- an LP."""
    u1, u2 = payoff_mats(game, s, t)
    A = ce_A(u1, u2)
    c = 0.5 * (u1.ravel() + u2.ravel())
    res = linprog(c, A_ub=-A, b_ub=np.zeros(4),
                  A_eq=np.array([[1.0, 1.0, 1.0, 1.0]]), b_eq=[1.0],
                  bounds=[(0, 1)] * 4, method="highs")
    p = np.clip(res.x, 0, 1)
    p /= p.sum()
    a, b = pays(p, u1, u2)
    return (a + b) / 2, p, (a, b)


def closed_sd(s, t):
    A_ = s + t
    if A_ >= 2:
        return A_ / 2
    return (A_ * (t - 1) + s) / (s + 2 * t - 2)


# ------------------------------------------------------- Q1: counterexample
print("=" * 100)
print("Q1.  Is EVERY correlated equilibrium >= mixed NE?")
print("=" * 100)
s, t = 0.5, 1.6
u1, u2 = payoff_mats("SD", s, t)
p = np.array([0.0, 5 / 16, 5 / 16, 6 / 16])
print(f"  SD s={s}, t={t};  candidate device p = (CC,CD,DC,DD) = {p}")
print(f"  CE slacks (>=0 required): {np.round(ce_A(u1, u2) @ p, 12)}")
print(f"  payoffs                : {tuple(round(z,6) for z in pays(p, u1, u2))}")
pm, um = mixed_ne("SD", s, t)
print(f"  mixed NE (no device)   : payoff each = {um:.6f}  [P(C)={pm:.4f}]")
print(f"  ---> CE payoff {pays(p,u1,u2)[0]:.6f} < {um:.6f}:  claim is FALSE")

print()
print("  worst achievable CE (min over the CE polytope), vs mixed NE:")
for (s, t) in [(0.5, 1.6), (0.5, 1.2), (0.2, 1.5), (0.9, 1.05)]:
    v, p, (a, b) = worst_ce("SD", s, t)
    pm, um = mixed_ne("SD", s, t)
    print(f"   s={s:4.2f} t={t:4.2f}:  worst CE min-payoff = {v:.4f}   mixed NE = {um:.4f}"
          f"   {'WORSE' if v < um else 'better'}")

# ------------------------------------------------------- Q2: best CE
print()
print("=" * 100)
print("Q2.  Is the BEST correlated equilibrium strictly better than mixed NE?")
print("     SD closed form:  u_max = (s+t)/2  if s+t>=2,  else [(s+t)(t-1)+s]/(s+2t-2)")
print("=" * 100)
bad = 0
worst = (9e9, None)
for s in np.linspace(0.05, 0.95, 19):
    for t in np.linspace(1.05, 1.95, 19):
        v, p, (a, b) = best_ce("SD", s, t)
        pm, um = mixed_ne("SD", s, t)
        cf = closed_sd(s, t)
        if abs(v - cf) > 1e-6:
            print(f"   MISMATCH s={s:.2f} t={t:.2f}: LP={v:.6f} closed={cf:.6f}")
        if v <= um + 1e-12:
            bad += 1
            print(f"   NOT BETTER: s={s:.2f} t={t:.2f} u_max={v:.6f} u_NE={um:.6f}")
        if v - um < worst[0]:
            worst = (v - um, (s, t))
print(f"  19x19 grid: closed form == LP, cases with u_max<=u_NE: {bad}")
print(f"  smallest margin = {worst[0]:.6f} at (s,t)={tuple(round(z,3) for z in worst[1])}")
print()
print("  examples:")
for (s, t) in [(0.2, 1.5), (0.5, 1.2), (0.5, 1.6), (0.9, 1.05), (0.5, 1.1)]:
    v, p, (a, b) = best_ce("SD", s, t)
    pm, um = mixed_ne("SD", s, t)
    print(f"   ({s},{t}):  u_max={v:.6f} (=closed {closed_sd(s,t):.6f})  u_NE={um:.6f}"
          f"  margin={v-um:+.6f}   p*={np.round(p,4)}")

print()
print("=" * 100)
print("Q2b. Battle of the Sexes (asymmetric): u1=[[1,0],[0,s]], u2=[[s,0],[0,1]]")
print("=" * 100)
for s in [0.2, 0.5, 0.9, 1.0, 1.5, 3.0]:
    v, p, (a, b) = best_ce("BOS", s)
    pm, um = mixed_ne("BOS", s)
    print(f"  s={s:4.2f}: u_max(min)= {v:.6f}   mixed NE = {um:.6f}   margin = {v-um:+.6f}"
          f"   p* = {np.round(p,4)}   (u1,u2)=({a:.4f},{b:.4f})")

# ------------------------------------------------------- Q3: which games can improve?
print()
print("=" * 100)
print("Q3.  How big is the CE payoff set?  (max / min average payoff over CE, LP)")
print("     Same calibrated matrix [[1,s],[t,0]] in four parameter regions.")
print("=" * 100)
for name, (s, t) in [("SD", (0.5, 1.6)), ("PD", (-0.5, 1.5)),
                     ("StagHunt", (-0.5, 0.5)), ("Harmony", (0.5, 0.5))]:
    u1 = np.array([[1.0, s], [t, 0.0]])
    u2 = u1.T
    A = ce_A(u1, u2)
    c = 0.5 * (u1.ravel() + u2.ravel())
    hi = linprog(-c, A_ub=-A, b_ub=np.zeros(4), A_eq=np.ones((1, 4)), b_eq=[1.0],
                 bounds=[(0, 1)] * 4, method="highs")
    lo = linprog(c, A_ub=-A, b_ub=np.zeros(4), A_eq=np.ones((1, 4)), b_eq=[1.0],
                 bounds=[(0, 1)] * 4, method="highs")
    ph = np.clip(hi.x, 0, 1); ph /= ph.sum()
    pl = np.clip(lo.x, 0, 1); pl /= pl.sum()
    ne = s * t / (s + t - 1) if s + t > 1 else float("nan")
    print(f"  {name:9s} s={s:4.1f} t={t:4.1f}:  maxCE={-hi.fun:.4f} p={np.round(ph,3)}"
          f"   minCE={lo.fun:.4f} p={np.round(pl,3)}   mixedNE={ne:.4f}")
print()
print("  Harmony: CE collapses to the unique NE (C,C) -> nothing to gain.")
print("  PD      : CE collapses to the unique NE (D,D) -> nothing to gain.")
print("  StagHunt/SD: CE is a fat polytope -> both better and worse points exist.")

# ------------------------------------------------- Q4: private vs public advice
print()
print("=" * 100)
print("Q4.  Third-party mediator with a PUBLIC distribution.  Two readings:")
print("     (V1) recommendations are PRIVATE  (each player hears only his own)  = Aumann CE")
print("     (V2) recommendation profile is PUBLIC  (both hear everything)")
print("          -> obeying is optimal only if the announced profile is a pure NE")
print("          -> achievable payoffs = convex hull of NE payoffs")
print("=" * 100)


def public_device_maxmin(min_payoff, s, t, n=400):
    """max of min(u1,u2) over conv{NE distributions} (SD: CD, DC, mixed NE)."""
    m = s / (s + t - 1)
    delta_CD = np.array([0.0, 1.0, 0.0, 0.0])
    delta_DC = np.array([0.0, 0.0, 1.0, 0.0])
    prod = np.array([m * m, m * (1 - m), (1 - m) * m, (1 - m) * (1 - m)])
    u1 = np.array([[1.0, s], [t, 0.0]])
    u2 = u1.T
    best = -9e9
    for i in range(n + 1):
        for j in range(n + 1 - i):
            a, b = i / n, j / n
            g = 1 - a - b
            p = a * delta_CD + b * delta_DC + g * prod
            x, y = pays(p, u1, u2)
            best = max(best, min(x, y))
    return best


for (s, t) in [(0.1, 1.2), (0.5, 1.2), (0.9, 1.05), (0.5, 1.6)]:
    v1, p1, (a1, b1) = best_ce("SD", s, t)
    v2 = public_device_maxmin(None, s, t)
    pm, um = mixed_ne("SD", s, t)
    base = max(v2, um)
    print(f"  SD s={s:4.2f} t={t:5.2f}: mixedNE={um:.4f}  V2(public)={v2:.4f}"
          f"  V1(private)={v1:.4f}   pure-correlation gain = {v1 - base:+.4f}"
          f"   [p*(private) = {np.round(p1,3)}]")

print()
print("  PD: CE collapses in BOTH readings (D strictly dominates C)")
s, t = -0.5, 1.5
u1 = np.array([[1.0, s], [t, 0.0]])
u2 = u1.T
A = ce_A(u1, u2)
print(f"      slacks at p=(0,0,0,1) : {np.round(A @ np.array([0,0,0,1.]), 12)}")
print(f"      slacks at p=(.5,0,0,.5): {np.round(A @ np.array([.5,0,0,.5]), 12)}  <- negative => not a CE")
print("      => no device, public or private, can put any weight on (C,C).")
