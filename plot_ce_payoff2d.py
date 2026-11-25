"""
2-D picture in payoff space for the game

            P2: S      P2: B
  P1: S     (6, 6)    (2, 7)
  P1: B     (7, 2)    (0, 0)

axes: u1 = payoff of player 1 (A),  u2 = payoff of player 2 (B)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
import ce_tool as T

v = (6, 6, 2, 7, 7, 2, 0, 0)
U1, U2 = T.mats(v)
cells = np.array([[6, 6], [2, 7], [7, 2], [0, 0]], float)
V = T.ce_vertices(v)
Uv = np.array([T.payoffs(p, U1, U2) for p in V])
feas = cells[ConvexHull(cells).vertices]
cepf = Uv[ConvexHull(Uv).vertices]

fig, ax = plt.subplots(figsize=(10.2, 8.6))

# ---------------------------------------------------------------- feasible set
ax.add_patch(Polygon(feas, closed=True, facecolor="0.91", edgecolor="0.55", lw=1.1, zorder=1))
ax.plot(np.append(feas[:, 0], feas[0, 0]), np.append(feas[:, 1], feas[0, 1]),
        color="0.55", lw=1.1, zorder=2)
ax.plot([2, 6, 7], [7, 6, 2], color="0.25", lw=2.6, zorder=3)

# ---------------------------------------------------------------- CE payoff set
ax.add_patch(Polygon(cepf, closed=True, facecolor="#4C8BF5", alpha=0.32,
                     edgecolor="#17458f", lw=2.2, zorder=4))

# ---------------------------------------------------------------- diagonal
ax.plot([0, 8], [0, 8], ls=":", color="0.65", lw=1.2, zorder=2)
ax.text(6.55, 7.15, "u1 = u2", color="0.55", fontsize=9, rotation=42)

# ---------------------------------------------------------------- corners
for (x, y), nm, off in [((6, 6), "(S,S)  6,6", (10, 2)),
                        ((2, 7), "(S,B)  2,7", (12, 2)),
                        ((7, 2), "(B,S)  7,2", (10, 2)),
                        ((0, 0), "(B,B)  0,0", (12, -4))]:
    ax.scatter(x, y, marker="o", facecolor="white", edgecolor="0.5", s=75, zorder=6)
    ax.annotate(nm, (x, y), xytext=off, textcoords="offset points", fontsize=9.5,
                color="0.35")

# ---------------------------------------------------------------- special points
def dot(x, y, m, c, s):
    ax.scatter(x, y, marker=m, color=c, s=s, zorder=8, edgecolor="white", lw=1.0)

def note(txt, xy, off, color, fs=10, arrow=False):
    kw = dict(fontsize=fs, color=color, zorder=9,
              bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.85))
    if arrow:
        kw["arrowprops"] = dict(arrowstyle="-", color=color, lw=0.9)
    ax.annotate(txt, xy, xytext=off, textcoords="offset points", **kw)

dot(2, 7, "*", "crimson", 340); dot(7, 2, "*", "crimson", 340)
note("pure NE", (2, 7), (16, 4), "crimson")
note("pure NE", (7, 2), (-70, 10), "crimson")

dot(4.5, 4.5, "^", "0.45", 110)
note("4.500", (4.5, 4.5), (-46, 8), "0.30", fs=9.5, arrow=True)
dot(14 / 3, 14 / 3, "s", "darkorange", 105)
note("4.667", (14 / 3, 14 / 3), (10, -16), "darkorange", fs=9.5, arrow=True)
dot(5.25, 5.25, "P", "seagreen", 190)
note("BEST CE  5.250", (5.25, 5.25), (-118, 12), "seagreen", fs=10, arrow=True)
dot(3.6, 3.6, "o", "#17458f", 75)
note("3.600", (3.6, 3.6), (10, -20), "#17458f", fs=9.5, arrow=True)

ax.annotate("(S,S) is Pareto efficient\nbut NOT a correlated equilibrium",
            (6, 6), xytext=(-118, 48), textcoords="offset points", fontsize=9.5,
            color="0.25",
            arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.0,
                            connectionstyle="arc3,rad=-0.3"))

ax.text(1.25, 2.25, "feasible payoffs\n(any joint distribution)",
        fontsize=9.5, color="0.45", ha="center")

# ---------------------------------------------------------------- device table
rows = [("public randomisation", "(0, 1/2, 1/2, 0)", "4.500"),
        ("mixed NE (no device)", "(4/9, 2/9, 2/9, 1/9)", "4.667"),
        ("best CE", "(1/2, 1/4, 1/4, 0)", "5.250"),
        ("lowest symmetric CE", "(0, 2/5, 2/5, 1/5)", "3.600")]
tbl = "each player's payoff\n" + "\n".join(f"{a:24s}{c:>7s}   {b}" for a, b, c in rows)
ax.text(4.15, 1.02, tbl, fontsize=8.6, family="monospace", color="0.25",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.8", alpha=0.95))

handles = [Line2D([], [], color="0.25", lw=2.6, label="Pareto frontier"),
           Patch(facecolor="#4C8BF5", alpha=0.32, edgecolor="#17458f",
                 label="CE payoff set"),
           Line2D([], [], marker="*", color="crimson", ls="", ms=15, label="pure NE"),
           Line2D([], [], marker="s", color="darkorange", ls="", ms=9, label="mixed NE"),
           Line2D([], [], marker="^", color="0.45", ls="", ms=10, label="public randomisation"),
           Line2D([], [], marker="P", color="seagreen", ls="", ms=11, label="best CE")]
ax.legend(handles=handles, loc="upper left", fontsize=9.3, framealpha=0.95,
          borderpad=0.7, labelspacing=0.55)

ax.set_xlabel("u1  =  payoff of player 1 (A)", fontsize=12.5)
ax.set_ylabel("u2  =  payoff of player 2 (B)", fontsize=12.5)
ax.set_title("Correlated equilibria of  [[6,6],[2,7],[7,2],[0,0]]  in payoff space",
             fontsize=13.5, pad=12)
ax.set_xlim(-0.25, 7.7); ax.set_ylim(-0.25, 7.7)
ax.set_xticks(range(0, 8)); ax.set_yticks(range(0, 8))
ax.grid(alpha=0.22); ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("ce_payoff2d.png", dpi=185, bbox_inches="tight")
print("saved ce_payoff2d.png")
