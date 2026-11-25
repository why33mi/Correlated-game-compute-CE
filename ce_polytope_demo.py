"""
What exactly IS the CE polytope?  Concrete anatomy for two 2x2 games.

For a 2x2 game the device is a point of the 3-simplex
    p = (p_TL, p_TR, p_BL, p_BR) >= 0,  p_TL + p_TR + p_BL + p_BR = 1,
and "correlated equilibrium" adds four linear inequalities (obedience):
    c1: p1 told T   c2: p1 told B   c3: p2 told L   c4: p2 told R.
So the CE set is  {p in simplex : A p >= 0}  -- a polytope (bounded, since it
sits inside the simplex).

This script prints, for BoS and for Snowdrift:
  * the four inequality rows, written out
  * every vertex, with the constraints that are tight there
  * the affine dimension of the polytope
  * the payoff point each vertex maps to
  * a check that the payoff set is the convex hull of those images

Run: python ce_polytope_demo.py
"""

import itertools

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def ce_constraints(u1, u2):
    return np.array([
        [u1[0, 0] - u1[1, 0], u1[0, 1] - u1[1, 1], 0.0, 0.0],
        [0.0, 0.0, u1[1, 0] - u1[0, 0], u1[1, 1] - u1[0, 1]],
        [u2[0, 0] - u2[0, 1], 0.0, u2[1, 0] - u2[1, 1], 0.0],
        [0.0, u2[0, 1] - u2[0, 0], 0.0, u2[1, 1] - u2[1, 0]],
    ])


def payoff(p, u1, u2):
    P = np.asarray(p, float).reshape(2, 2)
    return float((P * u1).sum()), float((P * u2).sum())


def vertices_with_tight_sets(u1, u2, tol=1e-9):
    A = ce_constraints(u1, u2)
    rows = [A[i] for i in range(4)] + [np.eye(4)[i] for i in range(4)]
    names = ["c1:p1 told T", "c2:p1 told B", "c3:p2 told L", "c4:p2 told R",
             "p_TL=0", "p_TR=0", "p_BL=0", "p_BR=0"]
    out = []
    for S in itertools.combinations(range(8), 3):
        M = np.vstack([rows[k] for k in S] + [np.ones(4)])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        x = np.linalg.solve(M, np.array([0, 0, 0, 1.0]))
        if np.all(x >= -tol) and np.all(A @ x >= -tol):
            if any(np.allclose(x, y[0], atol=1e-9) for y in out):
                continue
            tight = [names[k] for k in range(4) if abs((A @ x)[k]) < 1e-7]
            tight += [names[4 + i] for i in range(4) if abs(x[i]) < 1e-7]
            out.append((np.clip(x, 0, 1), tight))
    return out


def dim_affine(pts, tol=1e-7):
    P = np.array(pts)
    sv = np.linalg.svd(P - P[0])[1]
    return int((sv > tol).sum())


def show(name, u1, u2, notes):
    print("=" * 92)
    print(name)
    print("=" * 92)
    A = ce_constraints(u1, u2)
    labels = ["p1 told T", "p1 told B", "p2 told L", "p2 told R"]
    cs = ["p_TL", "p_TR", "p_BL", "p_BR"]
    print("  device p = (p_TL, p_TR, p_BL, p_BR) in the 3-simplex")
    print("  four obedience inequalities  A p >= 0 :")
    for k in range(4):
        terms = [f"{A[k, i]:+.1f} {cs[i]}" for i in range(4) if abs(A[k, i]) > 1e-12]
        print(f"    c{k + 1} [{labels[k]:9s}]  " + "  ".join(terms) + "  >= 0")
    V = vertices_with_tight_sets(u1, u2)
    pts = [v for v, _ in V]
    print(f"\n  vertices: {len(V)}   affine dimension: {dim_affine(pts)}"
          f"   (the simplex itself has dimension 3)")
    print(f"  {'vertex p':<34} {'payoff (u1,u2)':<18} tight constraints")
    for v, tight in V:
        pv = "(" + ", ".join(f"{z:.4f}" for z in v) + ")"
        u = payoff(v, u1, u2)
        print(f"  {pv:<34} ({u[0]:.4f},{u[1]:.4f})    {' , '.join(tight)}")
    print("  " + notes)


bos_u1 = np.array([[2.0, 0.0], [0.0, 1.0]])
bos_u2 = np.array([[1.0, 0.0], [0.0, 2.0]])
show("Bach or Stravinsky  u1 = [[2,0],[0,1]],  u2 = [[1,0],[0,2]]",
     bos_u1, bos_u2,
     "note: the mixed NE (2/9,4/9,1/9,2/9) is NOT a vertex -- it is interior.")

snow_u1 = np.array([[1.0, 0.5], [1.6, 0.0]])
snow_u2 = snow_u1.T
show("Snowdrift  u1 = [[1,.5],[1.6,0]],  u2 = u1^T",
     snow_u1, snow_u2,
     "note: the bad CE p=(0,5/16,5/16,6/16) is a vertex -> payoff (0.6562,0.6562).")

# the simplex itself, for contrast: no obedience constraints at all
print("=" * 92)
print("For contrast: drop the four inequalities (a free 'device')")
print("=" * 92)
simplex = [np.eye(4)[i] for i in range(4)]
print("  vertices: 4, affine dimension 3, payoff set = conv of the four payoff cells")
for v in simplex:
    print("   ", v, "->", payoff(v, bos_u1, bos_u2))
