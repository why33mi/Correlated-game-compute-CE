"""
Correlated equilibria for 2x2 games, from the eight payoff numbers

        a1 b1 a2 b2 a3 b3 a4 b4

convention (row-major, row = player 1's action, column = player 2's action):

              P2: C          P2: D
    P1: C    (a1, b1)       (a2, b2)
    P1: D    (a3, b3)       (a4, b4)

so           U1 = [[a1, a2], [a3, a4]]      U2 = [[b1, b2], [b3, b4]]

Unknown:  p = (p1, p2, p3, p4) = (p(CC), p(CD), p(DC), p(DD)) on the simplex.

Aumann (private recommendations) obedience inequalities A p >= 0, with

  row 1  player 1 told C : (a1-a3) p1 + (a2-a4) p2                 >= 0
  row 2  player 1 told D : (a3-a1) p3 + (a4-a2) p4                 >= 0
  row 3  player 2 told C : (b1-b2) p1 + (b3-b4) p3                 >= 0
  row 4  player 2 told D : (b2-b1) p2 + (b4-b3) p4                 >= 0

The CE set is  P = { p in simplex : A p >= 0 }  -- a polytope, computed by LP.

Notation for the sign analysis:

    d = (d1, d2) = (a1-a3, a2-a4)      player 1's deviation-gain vector
    e = (e1, e2) = (b1-b2, b3-b4)      player 2's deviation-gain vector

    row 1 :  d.(p1,p2) >= 0        row 2 :  d.(p3,p4) <= 0
    row 3 :  e.(p1,p3) >= 0        row 4 :  e.(p2,p4) <= 0

  d  componentwise > 0  <=>  C strictly dominates D for player 1
  e  componentwise > 0  <=>  C strictly dominates D for player 2
  mixed signs           <=>  no dominant action (the generic case)
"""

import itertools
import numpy as np
from scipy.optimize import linprog

TOL = 1e-9


# ------------------------------------------------------------------ basics
def mats(v):
    """v = (a1,b1,a2,b2,a3,b3,a4,b4) -> (U1, U2)."""
    a = v[0::2]
    b = v[1::2]
    return (np.array([[a[0], a[1]], [a[2], a[3]]], float),
            np.array([[b[0], b[1]], [b[2], b[3]]], float))


def ce_matrix(U1, U2):
    """4x4 matrix A with  (p is a CE)  <=>  A p >= 0."""
    return np.array([
        [U1[0, 0] - U1[1, 0], U1[0, 1] - U1[1, 1], 0.0, 0.0],
        [0.0, 0.0, U1[1, 0] - U1[0, 0], U1[1, 1] - U1[0, 1]],
        [U2[0, 0] - U2[0, 1], 0.0, U2[1, 0] - U2[1, 1], 0.0],
        [0.0, U2[0, 1] - U2[0, 0], 0.0, U2[1, 1] - U2[1, 0]],
    ])


def payoffs(p, U1, U2):
    P = np.asarray(p, float).reshape(2, 2)
    return float((P * U1).sum()), float((P * U2).sum())


def is_ce(v, p, verbose=True):
    U1, U2 = mats(v)
    A = ce_matrix(U1, U2)
    p = np.asarray(p, float)
    ok_norm = abs(p.sum() - 1) < 1e-9 and np.all(p >= -TOL)
    sl = A @ p
    ok = ok_norm and bool(np.all(sl >= -1e-9))
    if verbose:
        labels = ["P1 told C", "P1 told D", "P2 told C", "P2 told D"]
        print(f"  p = {np.round(p, 6)}   sum={p.sum():.9f}")
        for lab, s in zip(labels, sl):
            print(f"    {lab:9s}: slack = {s:+.6e}  "
                  f"{'OK' if s >= -1e-9 else 'VIOLATED'}")
        print(f"  --> {'is a CE' if ok else 'is NOT a CE'}")
    return ok


# ------------------------------------------------------- sign classification
def ce_diagnose(v, verbose=True):
    """Split each obedience inequality into
       vacuous / forcing two variables to zero / a genuine ratio bound.

       A constraint  c.u + c'.w >= 0  with u,w >= 0 is
         - vacuous                if c, c' >= 0
         - forces u = w = 0       if c, c' <= 0
         - a ratio bound          otherwise
    """
    U1, U2 = mats(v)
    d = (U1[0, 0] - U1[1, 0], U1[0, 1] - U1[1, 1])
    e = (U2[0, 0] - U2[0, 1], U2[1, 0] - U2[1, 1])
    if verbose:
        print(f"  d = (a1-a3, a2-a4) = ({d[0]:g}, {d[1]:g})")
        print(f"  e = (b1-b2, b3-b4) = ({e[0]:g}, {e[1]:g})")
    cons = [("P1 told C", (d[0], d[1]),          ("p1", "p2")),
            ("P1 told D", (-d[0], -d[1]),        ("p3", "p4")),
            ("P2 told C", (e[0], e[1]),          ("p1", "p3")),
            ("P2 told D", (-e[0], -e[1]),        ("p2", "p4"))]
    forced, out = set(), []
    for lab, c, (u, w) in cons:
        if c[0] >= -1e-12 and c[1] >= -1e-12:
            kind = "VACUOUS"
            msg = f"{lab:9s}: ({c[0]:+g}){u} + ({c[1]:+g}){w} >= 0  ->  vacuous"
        elif c[0] <= 1e-12 and c[1] <= 1e-12:
            kind = "FORCED"
            msg = (f"{lab:9s}: ({c[0]:+g}){u} + ({c[1]:+g}){w} >= 0  ->  "
                   f"forces {u} = {w} = 0")
            forced |= {u, w}
        else:
            kind = "RATIO"
            if c[0] > 0 > c[1]:
                msg = (f"{lab:9s}: ratio bound   {w}/{u} <= "
                       f"{c[0] / abs(c[1]):g}")
            else:
                msg = (f"{lab:9s}: ratio bound   {w}/{u} >= "
                       f"{abs(c[0]) / c[1]:g}")
        out.append((kind, msg))
        if verbose:
            print("    " + msg)
    if verbose and forced:
        print(f"  => forced to zero: {sorted(forced)}")
    return {"d": d, "e": e, "constraints": out, "forced": forced}


# ------------------------------------------------------------------ vertices
def ce_vertices(v, tol=1e-7):
    """All vertices of P = {p in simplex : A p >= 0}.

    A vertex has 3 linearly independent active constraints, so enumerating
    3-subsets of the 8 inequalities (4 simplex facets + 4 CE facets),
    solving together with  sum p = 1, and keeping the feasible solutions
    finds every vertex (degenerate polytopes included).  A final pass
    removes any point that is a convex combination of the others.
    """
    U1, U2 = mats(v)
    A = ce_matrix(U1, U2)
    rows = [np.eye(4)[i] for i in range(4)] + [A[k] for k in range(4)]
    ones = np.ones(4)
    cand = []
    for combo in itertools.combinations(range(8), 3):
        M = np.vstack([rows[i] for i in combo] + [ones])
        rhs = np.concatenate([np.zeros(3), [1.0]])
        try:
            p = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue
        if np.all(p >= -1e-9) and np.all(A @ p >= -1e-9) and abs(p.sum() - 1) < 1e-9:
            if not any(np.allclose(p, q, atol=tol) for q in cand):
                cand.append(p)
    out = []
    for i, p in enumerate(cand):
        others = np.array([q for j, q in enumerate(cand) if j != i])
        if len(others) == 0 or not _in_hull(p, others):
            out.append(p)
    return np.array(out) if out else np.zeros((0, 4))


def _in_hull(p, pts, tol=1e-7):
    """is p a convex combination of pts?  (LP feasibility)"""
    k = len(pts)
    A_eq = np.vstack([pts.T, np.ones(k)])
    b_eq = np.concatenate([p, [1.0]])
    res = linprog(np.zeros(k), A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * k, method="highs")
    return res.status == 0


# ---------------------------------------------------------------- LP queries
def ce_optimum(v, weights=(0.0, 0.0)):
    """maximise w1*u1 + w2*u2 over the CE set (LP)."""
    U1, U2 = mats(v)
    A = ce_matrix(U1, U2)
    w1, w2 = weights
    c = -(w1 * U1.ravel() + w2 * U2.ravel())
    res = linprog(c, A_ub=-A, b_ub=np.zeros(4),
                  A_eq=np.ones((1, 4)), b_eq=[1.0],
                  bounds=[(0, 1)] * 4, method="highs")
    p = np.clip(res.x, 0, 1)
    p /= p.sum()
    return -res.fun, p


def ce_best_min(v):
    """max over CE of min(u1,u2)."""
    U1, U2 = mats(v)
    A = ce_matrix(U1, U2)
    c = np.zeros(5)
    c[4] = -1.0
    A_ub = np.vstack([
        np.hstack([-A, np.zeros((4, 1))]),
        np.hstack([-U1.ravel()[None, :], [[1.0]]]),
        np.hstack([-U2.ravel()[None, :], [[1.0]]]),
    ])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]),
                  A_eq=np.array([[1.0, 1.0, 1.0, 1.0, 0.0]]), b_eq=[1.0],
                  bounds=[(0, 1)] * 4 + [(None, None)], method="highs")
    p = np.clip(res.x[:4], 0, 1)
    p /= p.sum()
    return -res.fun, p


# -------------------------------------------------------------- Nash (ref.)
def nash(v):
    """pure Nash equilibria and the (unique) completely mixed one, if any."""
    U1, U2 = mats(v)
    out = []
    for i in range(2):
        for j in range(2):
            if (U1[i, j] >= U1[1 - i, j] - 1e-12) and (U2[i, j] >= U2[i, 1 - j] - 1e-12):
                out.append(("pure", (i, j)))
    d1 = U1[0, 0] - U1[1, 0]
    d2 = U1[0, 1] - U1[1, 1]
    e1 = U2[0, 0] - U2[0, 1]
    e2 = U2[1, 0] - U2[1, 1]
    if abs(d1 - d2) > 1e-12 and abs(e1 - e2) > 1e-12:
        q = d2 / (d2 - d1)
        r = e2 / (e2 - e1)
        if 0 <= q <= 1 and 0 <= r <= 1:
            out.append(("mixed", (r, q)))
    return out


# ------------------------------------------------------------------- report
def report(name, v):
    U1, U2 = mats(v)
    print("=" * 96)
    print(f"{name}:  (a1,b1,a2,b2,a3,b3,a4,b4) = {tuple(v)}")
    print(f"  U1 = [[{U1[0,0]:g}, {U1[0,1]:g}], [{U1[1,0]:g}, {U1[1,1]:g}]]   "
          f"U2 = [[{U2[0,0]:g}, {U2[0,1]:g}], [{U2[1,0]:g}, {U2[1,1]:g}]]")
    ce_diagnose(v)
    A = ce_matrix(U1, U2)
    print("  A p >= 0 :")
    for i, lab in enumerate(["P1 told C", "P1 told D", "P2 told C", "P2 told D"]):
        print(f"    {lab:9s}: {A[i,0]:+6.2f} p1 {A[i,1]:+6.2f} p2 {A[i,2]:+6.2f} p3 "
              f"{A[i,3]:+6.2f} p4 >= 0")
    V = ce_vertices(v)
    print(f"  CE set: {len(V)} vertices")
    print("      p(CC)  p(CD)  p(DC)  p(DD) |     u1     u2")
    for p in V:
        u1, u2 = payoffs(p, U1, U2)
        print(f"     {p[0]:6.3f} {p[1]:6.3f} {p[2]:6.3f} {p[3]:6.3f} | {u1:6.4f} {u2:6.4f}")
    tot, ptot = ce_optimum(v, (1, 1))
    vmin, pmin = ce_best_min(v)
    print(f"  max total welfare over CE = {tot:.4f}  at p = {np.round(ptot,4)}")
    print(f"  max-min over CE           = {vmin:.4f}  at p = {np.round(pmin,4)}")
    print(f"  Nash: {nash(v)}")
    if len(V):
        us = np.array([payoffs(p, U1, U2) for p in V])
        print(f"  range over CE:  u1 in [{us[:,0].min():.4f}, {us[:,0].max():.4f}]"
              f"   u2 in [{us[:,1].min():.4f}, {us[:,1].max():.4f}]")


if __name__ == "__main__":
    games = {
        # symmetric-game helper: b = (a1, a3, a2, a4)
        "Prisoner's Dilemma (s=-0.5,t=1.5)": (1, 1, -0.5, 1.5, 1.5, -0.5, 0, 0),
        "Snowdrift (s=0.5,t=1.6)": (1, 1, 0.5, 1.6, 1.6, 0.5, 0, 0),
        "Stag Hunt (s=-0.5,t=0.5)": (1, 1, -0.5, 0.5, 0.5, -0.5, 0, 0),
        "Harmony (s=0.5,t=0.5)": (1, 1, 0.5, 0.5, 0.5, 0.5, 0, 0),
        "Battle of Sexes (s=0.5)": (1, 0.5, 0, 0, 0, 0, 0.5, 1),
        "Matching Pennies": (1, -1, -1, 1, -1, 1, 1, -1),
        "P1 dominant, P2 no dominant": (0, 1, 0, 0, 1, 0, 1, 1),
    }
    for name, v in games.items():
        report(name, v)
        print()
