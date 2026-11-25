"""
Does a 2x2 game always give MORE payoff under correlated equilibrium than under Nash?

Answer: no.  This script pins down exactly when and how it fails.

Structure
  Part A  exhaustive scan of all non-degenerate 2x2 games with payoffs in {0,1,2}
  Part B  random continuous payoffs (the integer grid is too coarse to see everything)
  Part C  named examples: BoS, Matching Pennies, Snowdrift, the OR Fig.47.1 game

Run: python ce_2x2_classify.py
"""

import itertools

import numpy as np
from scipy.optimize import linprog

TOL = 1e-9
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ game machinery
def ce_constraints(u1, u2):
    """The 4 obedience inequalities for p = (TL, TR, BL, BR): A p >= 0."""
    return np.array([
        [u1[0, 0] - u1[1, 0], u1[0, 1] - u1[1, 1], 0.0, 0.0],      # p1 told T
        [0.0, 0.0, u1[1, 0] - u1[0, 0], u1[1, 1] - u1[0, 1]],      # p1 told B
        [u2[0, 0] - u2[0, 1], 0.0, u2[1, 0] - u2[1, 1], 0.0],      # p2 told L
        [0.0, u2[0, 1] - u2[0, 0], 0.0, u2[1, 1] - u2[1, 0]],      # p2 told R
    ])


def payoff(p, u1, u2):
    P = np.asarray(p, float).reshape(2, 2)
    return float((P * u1).sum()), float((P * u2).sum())


def ce_vertices(u1, u2):
    """Exact vertex enumeration of {p in the simplex : A p >= 0}  (dim <= 3)."""
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
    ne = []
    for i, j in itertools.product(range(2), range(2)):
        if (u1[i, j] >= max(u1[r, j] for r in range(2)) - TOL
                and u2[i, j] >= max(u2[i, c] for c in range(2)) - TOL):
            ne.append(np.eye(2)[i][:, None] * np.eye(2)[j][None, :])
    return ne


def mixed_ne(u1, u2):
    """The completely mixed NE (unique if it exists), else None."""
    den_q = (u1[0, 0] - u1[1, 0]) + (u1[1, 1] - u1[0, 1])
    den_p = (u2[0, 0] - u2[0, 1]) + (u2[1, 1] - u2[1, 0])
    if abs(den_q) < TOL or abs(den_p) < TOL:
        return None
    q = (u1[1, 1] - u1[0, 1]) / den_q              # P(L)
    p = (u2[1, 1] - u2[1, 0]) / den_p              # P(T)
    if not (0 < p < 1 and 0 < q < 1):
        return None
    if abs((u1[0, 0] - u1[1, 0]) * q + (u1[0, 1] - u1[1, 1]) * (1 - q)) > 1e-9:
        return None                                 # p1 must be indifferent
    if abs((u2[0, 0] - u2[0, 1]) * p + (u2[1, 0] - u2[1, 1]) * (1 - p)) > 1e-9:
        return None                                 # p2 must be indifferent
    return np.outer([p, 1 - p], [q, 1 - q])


def maximin(u1, u2, n=4001):
    """Security levels v_i = max over OWN mixtures of the worst pure payoff."""
    xs = np.linspace(0, 1, n)
    v1 = max(min(x * u1[0, 0] + (1 - x) * u1[1, 0],          # opponent plays L
                 x * u1[0, 1] + (1 - x) * u1[1, 1])          # opponent plays R
             for x in xs)                                    # x = P(T)
    v2 = max(min(q * u2[0, 0] + (1 - q) * u2[0, 1],          # opponent plays T
                 q * u2[1, 0] + (1 - q) * u2[1, 1])          # opponent plays B
             for q in xs)                                    # q = P(L)
    return float(v1), float(v2)


def min_payoff_value(pts):
    """max over conv(pts) of min(x, y)  (tiny LP, fine for collinear points)."""
    pts = np.asarray(pts, float)
    k = len(pts)
    c = np.zeros(k + 1)
    c[-1] = -1.0
    A_ub = np.zeros((2, k + 1))
    A_ub[0, :k], A_ub[0, k] = -pts[:, 0], 1.0      # v - sum lam*x <= 0
    A_ub[1, :k], A_ub[1, k] = -pts[:, 1], 1.0      # v - sum lam*y <= 0
    A_eq = np.zeros((1, k + 1))
    A_eq[0, :k] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(2), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * k + [(None, None)], method="highs")
    return -res.fun


def _hull2d(pts):
    P = sorted({(round(float(x), 12), round(float(y), 12)) for x, y in pts})
    if len(P) <= 2:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for pt in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 1e-12:
            lower.pop()
        lower.append(pt)
    for pt in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 1e-12:
            upper.pop()
        upper.append(pt)
    return lower[:-1] + upper[:-1]


def outside_hull(pt, pts, tol=1e-7):
    H = _hull2d(pts)
    if len(H) == 1:
        return np.hypot(pt[0] - H[0][0], pt[1] - H[0][1]) > tol
    if len(H) == 2:
        a, b = np.array(H[0]), np.array(H[1])
        ap, ab = pt - a, b - a
        proj = float(np.dot(ap, ab) / np.dot(ab, ab))
        return np.hypot(*(ap - np.clip(proj, 0, 1) * ab)) > tol
    n = len(H)
    for k in range(n):
        a, b = np.array(H[k]), np.array(H[(k + 1) % n])
        edge, rel = b - a, pt - a
        if edge[0] * rel[1] - edge[1] * rel[0] < -tol * max(1.0, np.hypot(*edge)):
            return True
    return False


def classify(u1, u2):
    """Full CE-vs-NE picture for one non-degenerate 2x2 game (None if degenerate)."""
    if (u1[0, 0] == u1[1, 0] or u1[0, 1] == u1[1, 1]
            or u2[0, 0] == u2[0, 1] or u2[1, 0] == u2[1, 1]):
        return None
    pne = pure_ne(u1, u2)
    mne = mixed_ne(u1, u2)
    ne_pts = [payoff(P.ravel(), u1, u2) for P in pne]
    if mne is not None:
        ne_pts.append(payoff(mne.ravel(), u1, u2))
    n_ne = len(ne_pts)
    if n_ne not in (1, 3):
        return None
    verts = ce_vertices(u1, u2)
    if not verts:
        return None
    ce_pts = [payoff(v, u1, u2) for v in verts]
    single = len(verts) == 1
    if single:
        ref = [mne.ravel()] if mne is not None else [P.ravel() for P in pne]
        ce_is_ne = any(np.allclose(verts[0], r, atol=1e-7) for r in ref)
    else:
        ce_is_ne = False
    v_star = min_payoff_value(ce_pts)
    v_ne_pts = max(min(x) for x in ne_pts)
    v_ne_conv = min_payoff_value(ne_pts)
    worse_any = worse_both = False
    if mne is not None:
        m1, m2 = payoff(mne.ravel(), u1, u2)
        worse_any = any(x[0] < m1 - 1e-7 or x[1] < m2 - 1e-7 for x in ce_pts)
        worse_both = any(x[0] < m1 - 1e-7 and x[1] < m2 - 1e-7 for x in ce_pts)
    v1, v2 = maximin(u1, u2)
    floor_ok = all(x[0] >= v1 - 1e-7 and x[1] >= v2 - 1e-7 for x in ce_pts)
    return dict(n_ne=n_ne, single=single, ce_is_ne=ce_is_ne,
                ce_bigger_dist=not single,
                gain_ne=v_star > v_ne_pts + 1e-7,
                gain_conv=v_star > v_ne_conv + 1e-7,
                worse_any=worse_any, worse_both=worse_both,
                set_bigger=any(outside_hull(x, ne_pts) for x in ce_pts),
                floor_ok=floor_ok, v_star=v_star, ne_pts=ne_pts, ce_pts=ce_pts,
                verts=verts)


def report(stats, title):
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'#NE':>4} {'games':>7} {'CE = {NE}':>10} {'CE>NE (dist)':>13} "
          f"{'gain vs NE':>11} {'gain vs conv':>13} {'CE<mNE':>7} {'CE<<mNE':>8} "
          f"{'set>conv(NE)':>13} {'floor!':>7}")
    for n in (1, 3):
        s = stats[n]
        if not s["n"]:
            continue
        print(f"{n:>4} {s['n']:>7} {s['ce_is_ne']:>10} {s['ce_bigger_dist']:>13} "
              f"{s['gain_ne']:>11} {s['gain_conv']:>13} {s['worse_any']:>7} "
              f"{s['worse_both']:>8} {s['set_bigger']:>13} {s['floor_bad']:>7}")
    print("  CE = {NE}      : the CE polytope is just the point (the NE)")
    print("  gain vs NE     : max over CE of min(u1,u2) > best single NE")
    print("  gain vs conv   : ... > public randomization over the NE payoffs")
    print("  CE<mNE / <<mNE : some CE payoff below the mixed-NE payoff / below for both players")


# ------------------------------------------------------------------ Part A: exhaustive
print("\n>>> Part A: exhaustive, payoffs in {0,1,2}\n")
values = (0, 1, 2)
mats = [np.array(m, float).reshape(2, 2) for m in itertools.product(values, repeat=4)]
stats = {n: dict(n=0, ce_is_ne=0, ce_bigger_dist=0, gain_ne=0, gain_conv=0,
                 worse_any=0, worse_both=0, set_bigger=0, floor_bad=0) for n in (1, 3)}
for u1 in mats:
    for u2 in mats:
        r = classify(u1, u2)
        if r is None:
            continue
        s = stats[r["n_ne"]]
        s["n"] += 1
        for key in ("ce_is_ne", "ce_bigger_dist", "gain_ne", "gain_conv",
                    "worse_any", "worse_both", "set_bigger"):
            s[key] += bool(r[key])
        s["floor_bad"] += bool(not r["floor_ok"])
report(stats, f"Non-degenerate 2x2 games, payoffs in {{0,1,2}} ({sum(s['n'] for s in stats.values())} games)")

# ------------------------------------------------------------------ Part B: random
print("\n>>> Part B: random continuous payoffs (the integer grid is too coarse)\n")
stats_r = {n: dict(n=0, ce_is_ne=0, ce_bigger_dist=0, gain_ne=0, gain_conv=0,
                   worse_any=0, worse_both=0, set_bigger=0, floor_bad=0) for n in (1, 3)}
for _ in range(6000):
    u1 = rng.random((2, 2))
    u2 = rng.random((2, 2))
    r = classify(u1, u2)
    if r is None:
        continue
    s = stats_r[r["n_ne"]]
    s["n"] += 1
    for key in ("ce_is_ne", "ce_bigger_dist", "gain_ne", "gain_conv",
                "worse_any", "worse_both", "set_bigger"):
        s[key] += bool(r[key])
    s["floor_bad"] += bool(not r["floor_ok"])
report(stats_r, "Random U[0,1] payoffs")

# ------------------------------------------------------------------ Part C: named games
print("\n>>> Part C: named examples\n")
named = {
    "BoS          ": (np.array([[2.0, 0], [0, 1]]), np.array([[1.0, 0], [0, 2]])),
    "MatchingP.   ": (np.array([[1.0, -1], [-1, 1]]), np.array([[-1.0, 1], [1, -1]])),
    "Snowdrift    ": (np.array([[1.0, 0.5], [1.6, 0]]), np.array([[1.0, 1.6], [0.5, 0]])),
    "OR Fig.47.1  ": (np.array([[6.0, 2], [7, 0]]), np.array([[6.0, 7], [2, 0]])),
    "PD           ": (np.array([[1.0, -0.5], [1.5, 0]]), np.array([[1.0, 1.5], [-0.5, 0]])),
}
for name, (u1, u2) in named.items():
    r = classify(u1, u2)
    pm, pn = r["n_ne"], None
    print(f"{name} u1={u1.tolist()} u2={u2.tolist()}")
    print(f"   #NE = {r['n_ne']}   CE polytope vertices = {len(r['verts'])}"
          f"   CE = the NE? {r['ce_is_ne']}")
    print(f"   NE payoffs  : {[tuple(round(z, 4) for z in x) for x in r['ne_pts']]}")
    print(f"   CE payoffs  : {sorted({tuple(round(z, 4) for z in x) for x in r['ce_pts']})}")
    print(f"   best equal split: CE {r['v_star']:.4f} | single NE "
          f"{max(min(x) for x in r['ne_pts']):.4f} | conv(NE) "
          f"{min_payoff_value(r['ne_pts']):.4f}")
    print(f"   some CE below mixed NE? {r['worse_any']}   for both? {r['worse_both']}"
          f"   | CE set > conv(NE)? {r['set_bigger']}   | maxmin floor respected? {r['floor_ok']}")
    if r["worse_both"]:
        for v, c in zip(r["verts"], r["ce_pts"]):
            m1, m2 = payoff(mixed_ne(u1, u2).ravel(), u1, u2) if mixed_ne(u1, u2) is not None else (0, 0)
            if c[0] < m1 - 1e-7 and c[1] < m2 - 1e-7:
                print(f"     e.g. p = {[round(float(z), 4) for z in v]} -> {tuple(round(z, 4) for z in c)}")
    print()

# ------------------------------------------------------------------ why: the algebra
# ------------------------------------------------------------------ Part D: degenerate caveat
print("\n>>> Part D: what breaks without non-degeneracy (ties allowed)\n")
found = []
for u1 in mats:
    for u2 in mats:
        if not (u1[0, 0] == u1[1, 0] or u1[0, 1] == u1[1, 1]
                or u2[0, 0] == u2[0, 1] or u2[1, 0] == u2[1, 1]):
            continue                                    # degenerate only
        pne = pure_ne(u1, u2)
        if len(pne) != 1 or mixed_ne(u1, u2) is not None:
            continue
        verts = ce_vertices(u1, u2)
        if len(verts) < 2:
            continue
        ne_pt = payoff(pne[0].ravel(), u1, u2)
        ce_pts = [payoff(v, u1, u2) for v in verts]
        if any(abs(c[0] - ne_pt[0]) > 1e-9 or abs(c[1] - ne_pt[1]) > 1e-9 for c in ce_pts):
            found.append((u1.copy(), u2.copy(), verts, ce_pts, ne_pt))
print(f"  games with a unique NE, a CE polytope with >= 2 vertices, and CE payoffs")
print(f"  different from the NE payoff: {len(found)}")
for u1, u2, v, c, n in found[:3]:
    print(f"   u1 = {u1.tolist()}  u2 = {u2.tolist()}")
    print(f"     unique NE payoff {tuple(round(z, 4) for z in n)};"
          f"  CE vertices = {len(v)}, CE payoffs =")
    print(f"       {sorted({tuple(round(z, 4) for z in x) for x in c})}")

print()
print(">>> Why a unique NE leaves the CE polytope a single point")
print("    With two actions per player the 4 obedience inequalities are")
print("      p1 told T : [u1(T,L)-u1(B,L)] p_TL + [u1(T,R)-u1(B,R)] p_TR >= 0")
print("      p1 told B : [u1(B,L)-u1(T,L)] p_BL + [u1(B,R)-u1(T,R)] p_BR >= 0")
print("      p2 told L : [u2(T,L)-u2(T,R)] p_TL + [u2(B,L)-u2(B,R)] p_BL >= 0")
print("      p2 told R : [u2(T,R)-u2(T,L)] p_TR + [u2(B,R)-u2(B,L)] p_BR >= 0")
print("    If a player has a strict dominant action, one inequality kills two cells.")
print("    Otherwise both players' preferences flip across L/R and T/B, and the four")
print("    inequalities become  a p_TL >= b p_TR, b p_BR >= a p_BL,")
print("    d p_BL >= c p_TL, c p_TR >= d p_BR  -- multiply them and everything cancels:")
print("    (abcd) p_TL p_TR p_BL p_BR >= (abcd) p_TL p_TR p_BL p_BR, so all are equalities")
print("    and p is the unique mixed NE.")
