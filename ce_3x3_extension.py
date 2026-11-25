"""
Does the 2x2 theorem survive in 3x3 games?

The 2x2 proof used, essentially, "each player has exactly ONE alternative action",
so the obedience conditions form a chain of 4 inequalities (a,b,c,d) that closes.
In 3x3 each player has TWO alternative actions, so there are 12 inequalities and
no chain.  Question: is "unique NE  =>  CE polytope is that single point" still true?

Method
  1. random 3x3 payoffs
  2. NE by support enumeration (all 49 support pairs, solve indifference systems)
  3. among games with a UNIQUE NE: is the CE polytope a single point?
     (test: maximise and minimise every coordinate p_ij over the CE polytope; if
      every max == min == the NE's value, the polytope IS that point)
  4. if it is bigger: does some CE payoff strictly beat the unique NE payoff?

Run: python ce_3x3_extension.py
"""

import itertools

import numpy as np
from scipy.optimize import linprog

TOL = 1e-9
rng = np.random.default_rng(7)


def ce_matrix_constraints(u1, u2):
    """Obedience inequalities A p >= 0 for p = vec of the m x n joint device."""
    m, n = u1.shape
    rows = []
    for i in range(m):                       # player 1 told action i
        for i2 in range(m):
            if i2 == i:
                continue
            row = np.zeros((m, n))
            row[i, :] = u1[i, :] - u1[i2, :]
            rows.append(row.ravel())
    for j in range(n):                       # player 2 told action j
        for j2 in range(n):
            if j2 == j:
                continue
            row = np.zeros((m, n))
            row[:, j] = u2[:, j] - u2[:, j2]
            rows.append(row.ravel())
    return np.array(rows)


def payoffs_of(p, u1, u2):
    P = np.asarray(p, float).reshape(u1.shape)
    return float((P * u1).sum()), float((P * u2).sum())


def extreme_values_over_ce(u1, u2):
    """max and min of every coordinate p_ij over the CE polytope."""
    m, n = u1.shape
    A = ce_matrix_constraints(u1, u2)
    d = m * n
    lo, hi = np.zeros(d), np.zeros(d)
    for k in range(d):
        c = np.zeros(d)
        c[k] = 1.0
        r1 = linprog(c, A_ub=-A, b_ub=np.zeros(len(A)), A_eq=[np.ones(d)],
                     b_eq=[1.0], bounds=[(0, 1)] * d, method="highs")
        r2 = linprog(-c, A_ub=-A, b_ub=np.zeros(len(A)), A_eq=[np.ones(d)],
                     b_eq=[1.0], bounds=[(0, 1)] * d, method="highs")
        lo[k], hi[k] = r1.fun, -r2.fun
    return lo, hi


def nelist(u1, u2):
    """All NE of a 3x3 game by support enumeration.  Returns list of (P1mix, P2mix)."""
    m, n = u1.shape
    out = []
    for s1 in range(1, m + 1):
        for s2 in range(1, n + 1):
            for S1 in itertools.combinations(range(m), s1):
                for S2 in itertools.combinations(range(n), s2):
                    if s1 != s2:
                        continue                 # generically square supports only
                    S1, S2 = list(S1), list(S2)
                    # P2's mix q over S2 makes P1 indifferent over S1
                    M = np.zeros((s1, s2))
                    for a, i in enumerate(S1):
                        for b, j in enumerate(S2):
                            M[a, b] = u1[S1[0] if a == 0 else i, j] - u1[S1[0], j]
                    Aeq = np.vstack([M, np.ones(s2)])
                    beq = np.zeros(s1 + 1)
                    beq[-1] = 1.0
                    try:
                        q = np.linalg.lstsq(Aeq, beq, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        continue
                    if np.any(q <= 1e-9) or abs(q.sum() - 1) > 1e-7:
                        continue
                    # P1's mix p over S1 makes P2 indifferent over S2
                    M2 = np.zeros((s2, s1))
                    for b, j in enumerate(S2):
                        for a, i in enumerate(S1):
                            M2[b, a] = u2[i, S2[0] if b == 0 else j] - u2[i, S2[0]]
                    Aeq2 = np.vstack([M2, np.ones(s1)])
                    beq2 = np.zeros(s2 + 1)
                    beq2[-1] = 1.0
                    try:
                        p = np.linalg.lstsq(Aeq2, beq2, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        continue
                    if np.any(p <= 1e-9) or abs(p.sum() - 1) > 1e-7:
                        continue
                    P1 = np.zeros(m)
                    P1[S1] = p
                    P2 = np.zeros(n)
                    P2[S2] = q
                    # verify it really is an equilibrium: only the SUPPORT actions
                    # have to be best responses
                    vals1 = u1 @ P2
                    if np.any(vals1[S1] < vals1.max() - 1e-7):
                        continue
                    vals2 = P1 @ u2
                    if np.any(vals2[S2] < vals2.max() - 1e-7):
                        continue
                    if not any(np.allclose(P1, X, atol=1e-7) and np.allclose(P2, Y, atol=1e-7)
                               for X, Y in out):
                        out.append((P1, P2))
    return out


unique_checked = 0
collapsed = 0
bigger = []
strict_improve = 0
for trial in range(3000):
    u1 = rng.random((3, 3))
    u2 = rng.random((3, 3))
    nes = nelist(u1, u2)
    if len(nes) != 1:
        continue
    unique_checked += 1
    P1, P2 = nes[0]
    prod = np.outer(P1, P2)
    lo, hi = extreme_values_over_ce(u1, u2)
    if np.all(hi - lo < 1e-7) and np.allclose((lo + hi) / 2, prod.ravel(), atol=1e-7):
        collapsed += 1
    else:
        bigger.append((u1, u2, P1, P2, prod, lo, hi))
    # does some CE payoff strictly beat the unique NE payoff for both players?
    A = ce_matrix_constraints(u1, u2)
    d = 9
    m = payoffs_of(prod.ravel(), u1, u2)
    c = np.zeros(d + 1)
    c[-1] = -1.0
    A_ub = np.vstack([np.hstack([-A, np.zeros((len(A), 1))]),
                      np.hstack([-u1.ravel()[None, :], [[1.0]]]),
                      np.hstack([-u2.ravel()[None, :], [[1.0]]])])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]),
                  A_eq=[list(np.ones(d)) + [0.0]], b_eq=[1.0],
                  bounds=[(0, 1)] * d + [(None, None)], method="highs")
    if res.status == 0 and min(res.x[-1] - m[0], res.x[-1] - m[1]) > 1e-7:
        strict_improve += 1

print("=" * 78)
print("random 3x3 games with a UNIQUE Nash equilibrium")
print("=" * 78)
print(f"  games with unique NE      : {unique_checked}")
print(f"  CE polytope == that point : {collapsed}")
print(f"  CE polytope strictly bigger: {len(bigger)}")
print(f"  some CE payoff beats the unique NE for both players: {strict_improve}")

if bigger:
    u1, u2, P1, P2, prod, lo, hi = bigger[0]
    np.set_printoptions(precision=4, suppress=True)
    print("\n  counterexample to 'unique NE => CE = {NE}' in 3x3:")
    print("   u1 =", u1.tolist())
    print("   u2 =", u2.tolist())
    print(f"   unique NE: P1 = {P1}, P2 = {P2}  -> payoff {payoffs_of(prod.ravel(), u1, u2)}")
    k = int(np.argmax(hi - lo))
    print(f"   CE polytope is bigger: coordinate {k} ranges over [{lo[k]:.6f}, {hi[k]:.6f}]"
          f" but the NE has {prod.ravel()[k]:.6f}")
    c = np.zeros(9)
    c[k] = 1.0
    A = ce_matrix_constraints(u1, u2)
    r = linprog(-c, A_ub=-A, b_ub=np.zeros(len(A)), A_eq=[np.ones(9)], b_eq=[1.0],
                bounds=[(0, 1)] * 9, method="highs")
    p = np.clip(r.x, 0, 1)
    p = p / p.sum()
    print("   a CE point different from the NE:")
    print("     p =", np.round(p.reshape(3, 3), 4).tolist())
    print("     payoff =", tuple(round(z, 4) for z in payoffs_of(p, u1, u2)))
    print("     NE payoff =", tuple(round(z, 4) for z in payoffs_of(prod.ravel(), u1, u2)))
