"""
Undergraduate-level demo for the correlated-equilibrium handout.

Two things only:
  1. BoS with a PUBLIC coin  -> CE payoff (1.5, 1.5) vs mixed NE (2/3, 2/3).
  2. The "3 states, payoff (5,5)" game of Osborne-Rubinstein Fig.47.1
     -> verify the obedience inequalities and the payoffs, and check that
        (5,5) lies OUTSIDE the convex hull of the Nash payoffs.

Run:  python ce_intro_demo.py
"""

import numpy as np
from scipy.optimize import linprog

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------- utilities
def ce_constraints(u1, u2):
    """Rows A with (A p) >= 0 for the 4 obedience inequalities, p = (11,12,21,22)."""
    return np.array([
        [u1[0, 0] - u1[1, 0], u1[0, 1] - u1[1, 1], 0.0, 0.0],   # p1 told row 0
        [0.0, 0.0, u1[1, 0] - u1[0, 0], u1[1, 1] - u1[0, 1]],   # p1 told row 1
        [u2[0, 0] - u2[0, 1], 0.0, u2[1, 0] - u2[1, 1], 0.0],   # p2 told col 0
        [0.0, u2[0, 1] - u2[0, 0], 0.0, u2[1, 1] - u2[1, 0]],   # p2 told col 1
    ])


def payoff(p, u1, u2):
    P = np.asarray(p, float).reshape(2, 2)
    return float((P * u1).sum()), float((P * u2).sum())


def is_ce(p, u1, u2, tol=1e-12):
    return bool(np.all(ce_constraints(u1, u2) @ np.asarray(p, float) >= -tol))


def best_ce_min(u1, u2):
    """max over CE of min(u1, u2)  (both players get at least v)."""
    A, v1, v2 = ce_constraints(u1, u2), u1.ravel(), u2.ravel()
    c = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
    A_ub = np.vstack([np.hstack([-A, np.zeros((4, 1))]),
                      np.hstack([-v1[None, :], [[1.0]]]),
                      np.hstack([-v2[None, :], [[1.0]]])])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]),
                  A_eq=[[1, 1, 1, 1, 0]], b_eq=[1.0],
                  bounds=[(0, 1)] * 4 + [(None, None)], method="highs")
    p = np.clip(res.x[:4], 0, 1)
    return -res.fun, p / p.sum()


def direction_max(u1, u2, w1, w2):
    """max over CE of w1*u1 + w2*u2  (an extreme point of the CE payoff set)."""
    c = -(w1 * u1.ravel() + w2 * u2.ravel())
    res = linprog(c, A_ub=-ce_constraints(u1, u2), b_ub=np.zeros(4),
                  A_eq=[[1, 1, 1, 1]], b_eq=[1.0], bounds=[(0, 1)] * 4, method="highs")
    p = np.clip(res.x, 0, 1)
    return payoff(p / p.sum(), u1, u2)


def ce_payoff_polygon(u1, u2, n=720):
    """Outer boundary of the CE payoff set, by sweeping directions."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.array([direction_max(u1, u2, np.cos(t), np.sin(t)) for t in th])
    return pts


def mixed_ne(u1, u2):
    """Fully mixed NE of a 2x2 game: (P(row 1), P(col 1), payoff)."""
    q = (u1[0, 0] - u1[1, 0]) / ((u1[0, 0] - u1[1, 0]) + (u1[1, 1] - u1[0, 1]))
    p = (u2[0, 0] - u2[0, 1]) / ((u2[0, 0] - u2[0, 1]) + (u2[1, 1] - u2[1, 0]))
    P = np.outer([1 - p, p], [1 - q, q])
    return p, q, payoff(P, u1, u2)


def in_convex_hull(pt, pts, tol=1e-9):
    """Is pt in the convex hull of pts?"""
    A = np.array(pts).T                       # 2 x k
    res = linprog(np.zeros(len(pts)), A_eq=np.vstack([A, np.ones(len(pts))]),
                  b_eq=[pt[0], pt[1], 1.0], bounds=[(0, None)] * len(pts), method="highs")
    return res.status == 0


# ---------------------------------------------------------------- Example 1: BoS + public coin
print("=" * 78)
print("Example 1 (BoS): a public coin beats the mixed equilibrium")
print("=" * 78)
u1 = np.array([[2.0, 0.0], [0.0, 1.0]])          # rows/cols: 0 = Bach, 1 = Stravinsky
u2 = np.array([[1.0, 0.0], [0.0, 2.0]])

p_coin = np.array([0.5, 0.0, 0.0, 0.5])         # public coin: (B,B) or (S,S) each 1/2
print(f"  public-coin distribution   p = {p_coin.reshape(2, 2).tolist()}")
print(f"  obedience slacks (>=0)     : {np.round(ce_constraints(u1, u2) @ p_coin, 10)}")
print(f"  is a CE?                   : {is_ce(p_coin, u1, u2)}")
print(f"  payoff                     : {tuple(round(z, 4) for z in payoff(p_coin, u1, u2))}")

p, q, (a, b) = mixed_ne(u1, u2)
print(f"  mixed NE: P1 plays Bach {1-p:.4f}, P2 plays Bach {1-q:.4f}"
      f"  -> payoff {tuple(round(z, 4) for z in (a, b))}")
print(f"  gain from the coin         : {payoff(p_coin, u1, u2)[0] - a:+.4f} per player")

# ---------------------------------------------------------------- Example 2: the (5,5) game
print()
print("=" * 78)
print("Example 2 (OR Fig.47.1): a CE payoff outside the hull of the Nash payoffs")
print("=" * 78)
u1 = np.array([[6.0, 2.0], [7.0, 0.0]])          # rows/cols: 0 = T / L, 1 = B / R
u2 = np.array([[6.0, 7.0], [2.0, 0.0]])

ne_pure = []
for i in range(2):
    for j in range(2):
        # a pure profile is a Nash equilibrium iff neither player can profit by deviating
        if (u1[i, j] >= max(u1[r, j] for r in range(2)) - 1e-12
                and u2[i, j] >= max(u2[i, c] for c in range(2)) - 1e-12):
            ne_pure.append((i, j, payoff(np.eye(2)[i][:, None] * np.eye(2)[j][None, :], u1, u2)))
print("  pure Nash equilibria (row, col, payoff):", [(f"{'TB'[i]}{'LR'[j]}", tuple(round(z, 3) for z in v))
                                                    for i, j, v in ne_pure])
p_, q_, (a_, b_) = mixed_ne(u1, u2)
print(f"  mixed NE: P(T)={1 - p_:.4f}, P(L)={1 - q_:.4f}  -> payoff {tuple(round(z, 4) for z in (a_, b_))}")

# the 3-state, 3-outcome device:  x -> (B,L),  y -> (T,L),  z -> (T,R), each 1/3
p3 = np.array([[1 / 3, 1 / 3],      # T: (T,L), (T,R)
               [1 / 3, 0.0]])       # B: (B,L)
print()
print(f"  device p (rows T,B / cols L,R) = {p3.tolist()}")
print(f"  obedience slacks (>=0)     : {np.round(ce_constraints(u1, u2) @ p3.ravel(), 10)}")
print(f"  is a CE?                   : {is_ce(p3.ravel(), u1, u2)}")
v = payoff(p3.ravel(), u1, u2)
print(f"  payoff                     : {tuple(round(z, 4) for z in v)}")

vmax, pbest = best_ce_min(u1, u2)
print(f"\n  best CE for min-payoff (LP): v = {vmax:.4f},  p = {pbest.reshape(2, 2).tolist()}")
print(f"  payoff (u1,u2)             : {tuple(round(z, 4) for z in payoff(pbest, u1, u2))}")

hull_pts = [(a_, b_)] + [v_ for _, _, v_ in ne_pure]
print(f"\n  Nash payoffs                : {[tuple(round(z, 4) for z in t) for t in hull_pts]}")
print(f"  is (5,5) in conv(Nash)?     : {in_convex_hull((5.0, 5.0), hull_pts)}")
print(f"  is (5,5) in the CE payoff set? (obedience above) -> yes")

# where the CE payoff set sits: extreme rays
print("\n  CE payoff set boundary (sweep of directions, u1 right / u2 up):")
poly = ce_payoff_polygon(u1, u2, n=72)
print("   ", [tuple(round(z, 2) for z in t) for t in poly[::12]])

# ---------------------------------------------------------------- figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # left: BoS
    ax = axes[0]
    u1b = np.array([[2.0, 0.0], [0.0, 1.0]])
    u2b = np.array([[1.0, 0.0], [0.0, 2.0]])
    ax.plot(*ce_payoff_polygon(u1b, u2b).T, lw=2, label="CE payoffs")
    ax.scatter([2, 1], [1, 2], c="k", zorder=5, label="pure NE")
    ax.scatter([2 / 3], [2 / 3], c="tab:red", zorder=5, label="mixed NE")
    ax.scatter([1.5], [1.5], c="tab:green", marker="*", s=200, zorder=5, label="public coin")
    ax.set_title("BoS: the coin reaches (1.5, 1.5)")
    ax.set_xlabel("player 1 payoff")
    ax.set_ylabel("player 2 payoff")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # right: the (5,5) game
    ax = axes[1]
    poly = ce_payoff_polygon(u1, u2)
    ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, color="tab:blue")
    ax.plot(poly[:, 0], poly[:, 1], lw=2, color="tab:blue", label="CE payoffs")
    ax.scatter([7, 2, 14 / 3], [2, 7, 14 / 3], c="k", zorder=5, label="Nash payoffs")
    ax.scatter([5], [5], c="tab:green", marker="*", s=200, zorder=5, label="3-state device")
    ax.set_title("Fig.47.1 game: (5,5) lies outside conv(Nash)")
    ax.set_xlabel("player 1 payoff")
    ax.set_ylabel("player 2 payoff")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("ce_payoff_sets.png", dpi=150)
    print("\n  figure written to ce_payoff_sets.png")
except ImportError:
    print("\n  (matplotlib not installed -- skipping the figure)")
