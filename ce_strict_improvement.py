"""
Which payoff-level statement actually survives in a 2x2 game?

I claimed earlier: "NE not unique => some CE is strictly better".  That mixes two
different statements.  This script separates them and tests each one.

(S1) distribution level : NE not unique  =>  the CE *polytope* is strictly bigger
                          than the set of NE distributions.   [proved, and checked]
(S2) payoff level       : NE not unique  =>  some CE payoff strictly dominates
                          something / everything.            [tested here]

For every non-degenerate 2x2 game with 3 NE we test, with an LP:
  D1  is there a CE payoff strictly dominating the mixed-NE payoff m?
  D2  is that achievable already inside conv(NE payoffs)?   (public randomization)
  D3  is there a CE payoff strictly dominating EVERY NE payoff?
  D4  ... achievable inside conv(NE payoffs)?

"strictly dominating x" means > x on both coordinates.  The LP is
   max s  s.t.  (x,y) = sum lam_i pts_i,  lam >= 0, sum lam = 1,  s <= x - t1, s <= y - t2
so s > 0 iff a point of conv(pts) strictly beats the target t.

Run: python ce_strict_improvement.py
"""

import itertools

import numpy as np
from scipy.optimize import linprog

TOL = 1e-9
rng = np.random.default_rng(1)


def ce_constraints(u1, u2):
    return np.array([
        [u1[0, 0] - u1[1, 0], u1[0, 1] - u1[1, 1], 0.0, 0.0],
        [0.0, 0.0, u1[1, 0] - u1[0, 0], u1[1, 1] - u1[0, 1]],
        [u2[0, 0] - u2[0, 1], 0.0, u2[1, 0] - u2[1, 1], 0.0],
        [0.0, u2[0, 1] - u2[0, 0], 0.0, u2[1, 1] - u2[1, 0]],
    ])


def payoff(p, u1, u2):
    P = np.asarray(p, float).reshape(2, 2)
    return np.array([(P * u1).sum(), (P * u2).sum()])


def ce_vertices(u1, u2):
    A = ce_constraints(u1, u2)
    rows = [A[i] for i in range(4)] + [np.eye(4)[i] for i in range(4)]
    out = []
    for S in itertools.combinations(range(8), 3):
        M = np.vstack([rows[k] for k in S] + [np.ones(4)])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        x = np.linalg.solve(M, np.array([0, 0, 0, 1.0]))
        if np.all(x >= -TOL) and np.all(A @ x >= -TOL):
            if not any(np.allclose(x, y, atol=1e-9) for y in out):
                out.append(np.clip(x, 0, 1))
    return out


def pure_ne(u1, u2):
    return [np.eye(2)[i][:, None] * np.eye(2)[j][None, :]
            for i, j in itertools.product(range(2), range(2))
            if (u1[i, j] >= max(u1[r, j] for r in range(2)) - TOL
                and u2[i, j] >= max(u2[i, c] for c in range(2)) - TOL)]


def mixed_ne(u1, u2):
    den_q = (u1[0, 0] - u1[1, 0]) + (u1[1, 1] - u1[0, 1])
    den_p = (u2[0, 0] - u2[0, 1]) + (u2[1, 1] - u2[1, 0])
    if abs(den_q) < TOL or abs(den_p) < TOL:
        return None
    q = (u1[1, 1] - u1[0, 1]) / den_q
    p = (u2[1, 1] - u2[1, 0]) / den_p
    if not (0 < p < 1 and 0 < q < 1):
        return None
    if abs((u1[0, 0] - u1[1, 0]) * q + (u1[0, 1] - u1[1, 1]) * (1 - q)) > 1e-9:
        return None
    if abs((u2[0, 0] - u2[0, 1]) * p + (u2[1, 0] - u2[1, 1]) * (1 - p)) > 1e-9:
        return None
    return np.outer([p, 1 - p], [q, 1 - q])


def best_dominance(pts, target):
    """max s s.t. some (x,y) in conv(pts) has x >= target1+s, y >= target2+s."""
    pts = np.asarray(pts, float)
    k = len(pts)
    c = np.zeros(k + 1)
    c[-1] = -1.0
    A_ub = np.zeros((2, k + 1))
    A_ub[0, :k], A_ub[0, k] = -pts[:, 0], 1.0
    A_ub[1, :k], A_ub[1, k] = -pts[:, 1], 1.0
    b_ub = np.array([-target[0], -target[1]])
    A_eq = np.zeros((1, k + 1))
    A_eq[0, :k] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * k + [(None, None)], method="highs")
    return -res.fun if res.status == 0 else -1e9


def analyse(u1, u2):
    if (u1[0, 0] == u1[1, 0] or u1[0, 1] == u1[1, 1]
            or u2[0, 0] == u2[0, 1] or u2[1, 0] == u2[1, 1]):
        return None
    pne, mne = pure_ne(u1, u2), mixed_ne(u1, u2)
    if mne is None or len(pne) != 2:
        return None                                   # keep only 3-NE games
    verts = ce_vertices(u1, u2)
    ce_pts = [payoff(v, u1, u2) for v in verts]
    ne_pts = [payoff(P.ravel(), u1, u2) for P in pne] + [payoff(mne.ravel(), u1, u2)]
    m = payoff(mne.ravel(), u1, u2)
    top = np.max(np.array(ne_pts), axis=0)            # coordinatewise best NE payoff
    return dict(
        ce_bigger_dist=len(verts) > 1,
        d1_m=best_dominance(ce_pts, m),               # > 0 : CE beats the mixed NE
        d2_m=best_dominance(ne_pts, m),               # > 0 : conv(NE) already does
        d3_all=best_dominance(ce_pts, top),           # > 0 : CE beats every NE
        d4_all=best_dominance(ne_pts, top),
        # just the two pure NE and their equal-weight coin, nothing else:
        d_simple=best_dominance([payoff(P.ravel(), u1, u2) for P in pne] +
                                [np.mean([payoff(P.ravel(), u1, u2) for P in pne], axis=0)], m),
        m=m, top=top, ne_pts=ne_pts, ce_pts=ce_pts, u1=u1, u2=u2, verts=verts,
    )


def run(games, label):
    stats = dict(n=0, s1=0, d1=0, d2=0, d3=0, d4=0, simple=0)
    bad_d1, bad_d2, bad_simple = [], [], []
    for u1, u2 in games:
        r = analyse(u1, u2)
        if r is None:
            continue
        stats["n"] += 1
        stats["s1"] += r["ce_bigger_dist"]
        stats["d1"] += r["d1_m"] > 1e-7
        stats["d2"] += r["d2_m"] > 1e-7
        stats["d3"] += r["d3_all"] > 1e-7
        stats["d4"] += r["d4_all"] > 1e-7
        stats["simple"] += r["d_simple"] > 1e-7
        if r["d1_m"] <= 1e-7:
            bad_d1.append(r)
        if r["d2_m"] <= 1e-7:
            bad_d2.append(r)
        if r["d_simple"] <= 1e-7:
            bad_simple.append(r)
    print(f"{label}: {stats['n']} games with 3 NE")
    print(f"  S1  CE polytope strictly bigger than NE distributions : {stats['s1']}")
    print(f"  D1  some CE payoff strictly beats the mixed NE       : {stats['d1']}")
    print(f"  D2  ... already inside conv(NE payoffs)              : {stats['d2']}")
    print(f"  D*  ... already one of (pure NE 1, pure NE 2, coin)  : {stats['simple']}")
    print(f"  D3  some CE payoff strictly beats EVERY NE payoff    : {stats['d3']}")
    print(f"  D4  ... already inside conv(NE payoffs)              : {stats['d4']}")
    if bad_d1:
        r = bad_d1[0]
        print(f"  counterexample to D1: u1={r['u1'].tolist()} u2={r['u2'].tolist()}")
        print(f"     mixed NE payoff {r['m']},  best CE dominance s = {r['d1_m']:.4f}")
        print(f"     CE payoffs: {sorted({tuple(round(z, 4) for z in x) for x in r['ce_pts']})}")
        print(f"     NE payoffs: {[tuple(round(z, 4) for z in x) for x in r['ne_pts']]}")
    else:
        print("  (no counterexample to D1 in this sample)")
    for name, bad in (("D2", bad_d2), ("D*", bad_simple)):
        if bad:
            r = bad[0]
            print(f"  {name} needs more than conv(NE)/the coin, e.g. "
                  f"u1={r['u1'].tolist()} u2={r['u2'].tolist()}")
            print(f"     mixed NE {np.round(r['m'], 4)}, NE payoffs "
                  f"{[tuple(round(z, 4) for z in x) for x in r['ne_pts']]}, "
                  f"s_CE={r['d1_m']:.4f} vs s_conv={r['d2_m']:.4f}")
    print()
    return stats


mats = [np.array(m, float).reshape(2, 2) for m in itertools.product((0, 1, 2), repeat=4)]
run([(a, b) for a in mats for b in mats], "exhaustive, payoffs in {0,1,2}")
run([(rng.random((2, 2)), rng.random((2, 2))) for _ in range(6000)], "random U[0,1]")

# near-ties: payoffs that are almost degenerate, where the margins are tiny
bases = [np.array(m, float).reshape(2, 2) for m in itertools.product((0.0, 1.0), repeat=4)]
near = []
for _ in range(6000):
    b1, b2 = bases[rng.integers(len(bases))], bases[rng.integers(len(bases))]
    near.append((b1 + 1e-3 * rng.random((2, 2)), b2 + 1e-3 * rng.random((2, 2))))
run(near, "near-ties: entries in {0,1} + U[0,1e-3] noise")

print("BoS, by hand:")
u1b, u2b = np.array([[2.0, 0], [0, 1]]), np.array([[1.0, 0], [0, 2]])
rb = analyse(u1b, u2b)
print(f"  mixed NE payoff {rb['m']}, best CE dominance over it {rb['d1_m']:+.4f}"
      f" (public randomization: {rb['d2_m']:+.4f})")
print(f"  best CE dominance over EVERY NE payoff: {rb['d3_all']:+.4f}")
print("  -> no CE payoff strictly dominates the pure NE (2,1) or (1,2);")
print("     the only 'strictly better' statement that holds is versus the mixed NE.")

print()
print("A game where the midpoint of the two pure NE is NOT above the mixed NE:")
u1x = np.array([[10.0, 100.0], [0.0, 101.0]])
u2x = np.array([[5.0, 4.0], [4.0, 6.0]])
rx = analyse(u1x, u2x)
print(f"  u1={u1x.tolist()} u2={u2x.tolist()}")
print(f"  NE payoffs {[tuple(round(z, 3) for z in x) for x in rx['ne_pts']]}")
print(f"  midpoint of the two pure NE = {np.mean(np.array(rx['ne_pts'])[:2], axis=0)}")
print(f"  mixed NE payoff {np.round(rx['m'], 4)}  -> the best CE dominance over it "
      f"is {rx['d1_m']:+.4f} (achieved by a pure NE, not by the coin)")
