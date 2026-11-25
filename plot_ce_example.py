"""
CE set of the 2x2 game

            P2: S      P2: B
  P1: S     (6, 6)    (2, 7)
  P1: B     (7, 2)    (0, 0)

p = (p1,p2,p3,p4) = (p(SS), p(SB), p(BS), p(BB)),  sum = 1, p >= 0

obedience (Aumann, private recommendations):
  P1 told S :  -p1 + 2p2 >= 0   ->  p2 >= p1/2
  P1 told B :   p3 - 2p4 >= 0   ->  p3 >= 2p4
  P2 told S :  -p1 + 2p3 >= 0   ->  p3 >= p1/2
  P2 told B :    p2 - 2p4 >= 0  ->  p2 >= 2p4
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import ce_tool as T

v = (6, 6, 2, 7, 7, 2, 0, 0)
U1, U2 = T.mats(v)
V = np.array(sorted(T.ce_vertices(v).tolist(), key=lambda x: (round(x[0], 6), round(x[2], 6))))

# ---------------------------------------------------------------- text output
print("CE polytope vertices")
print(f"{'p(SS)':>7} {'p(SB)':>7} {'p(BS)':>7} {'p(BB)':>7} | {'u1':>6} {'u2':>6}")
for p in V:
    u1, u2 = T.payoffs(p, U1, U2)
    print(f"{p[0]:7.4f} {p[1]:7.4f} {p[2]:7.4f} {p[3]:7.4f} | {u1:6.3f} {u2:6.3f}")

fig = plt.figure(figsize=(20.5, 6.6))
g3 = fig.add_subplot(1, 3, 1, projection="3d")
ax1 = fig.add_subplot(1, 3, 2)
ax2 = fig.add_subplot(1, 3, 3)

# --------------------------------------------------------------- (a) simplex
corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], float)
names = ["SS", "SB", "BS", "BB"]
for i in range(4):
    for j in range(i + 1, 4):
        g3.plot(*zip(corners[i], corners[j]), color="0.78", lw=0.9, zorder=1)
for c, nm in zip(corners, names):
    g3.scatter(*c, color="0.5", s=20, depthshade=False, zorder=5)
    g3.text(c[0], c[1], c[2], " " + nm, fontsize=11, color="0.35")

hull = ConvexHull(V[:, :3])
tris = [[V[s, :3] for s in sim] for sim in hull.simplices]
g3.add_collection3d(Poly3DCollection(tris, facecolor="#4C8BF5", alpha=0.30,
                                     edgecolor="#17458f", lw=1.1, zorder=3))
for p in V:
    g3.scatter(p[0], p[1], p[2], color="#17458f", s=80, depthshade=False, zorder=6)
for p in [[0, 1, 0], [0, 0, 1], [4 / 9, 2 / 9, 2 / 9]]:
    g3.scatter(p[0], p[1], p[2], marker="*", color="crimson", s=270,
               depthshade=False, zorder=7)
b = V[3]
g3.text(b[0], b[1], b[2], "  best CE\n  (1/2,1/4,1/4,0)", fontsize=9, color="#17458f",
        zorder=8)
g3.set_xlabel("p(SS)", labelpad=3); g3.set_ylabel("p(SB)", labelpad=3)
g3.set_zlabel("p(BS)", labelpad=3)
g3.set_title("(a) the CE polytope inside the 3-simplex\n"
             "star = Nash equilibrium,   dot = vertex", fontsize=11)
g3.view_init(elev=20, azim=-120)
g3.set_box_aspect((1, 1, 1))
g3.set_xlim(0, 1); g3.set_ylim(0, 1); g3.set_zlim(0, 1)
for axis in (g3.xaxis, g3.yaxis, g3.zaxis):
    axis.set_tick_params(labelsize=7)

# --------------------------------------------------------------- (b) payoff region
Uv = np.array([T.payoffs(p, U1, U2) for p in V])
poly = Uv[ConvexHull(Uv).vertices]
ax1.fill(poly[:, 0], poly[:, 1], color="#4C8BF5", alpha=0.15, zorder=1)
ax1.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
         color="#17458f", lw=1.5, zorder=2)
ax1.plot([0, 8.2], [0, 8.2], ls=":", color="0.65", lw=1, zorder=0)
ax1.plot([2, 7], [7, 2], ls="--", color="0.55", lw=1.1, zorder=1)
for a, b in [(2, 7), (7, 2), (3.6, 3.6), (4.667, 4.667), (5.25, 5.25)]:
    ax1.scatter(a, b, color="#17458f", s=55, zorder=4)
ax1.annotate("(2, 7)\npure NE", (2, 7), xytext=(20, -6), textcoords="offset points",
             fontsize=9, color="#17458f")
ax1.annotate("(7, 2)\npure NE", (7, 2), xytext=(-52, 10), textcoords="offset points",
             fontsize=9, color="#17458f")
ax1.annotate("(3.6, 3.6)\nlowest CE vertex", (3.6, 3.6), xytext=(-40, -34),
             textcoords="offset points", fontsize=9, color="#17458f")
ax1.annotate("(4.667, 4.667)\nmixed NE", (4.667, 4.667), xytext=(10, -34),
             textcoords="offset points", fontsize=9, color="darkorange")
ax1.annotate("(5.25, 5.25)   BEST CE\np = (1/2, 1/4, 1/4, 0)", (5.25, 5.25),
             xytext=(-96, 20), textcoords="offset points", fontsize=9,
             color="seagreen")
ax1.scatter([2, 7], [7, 2], marker="*", s=300, color="crimson", zorder=5,
            label="pure NE")
ax1.scatter(14 / 3, 14 / 3, marker="s", s=80, color="darkorange", zorder=5,
            label="mixed NE: 4.667")
ax1.scatter(4.5, 4.5, marker="^", s=100, color="0.45", zorder=5,
            label="public randomisation: 4.500")
ax1.scatter(5.25, 5.25, marker="P", s=140, color="seagreen", zorder=6,
            label="best CE: 5.250")
ax1.set_xlabel("u1"); ax1.set_ylabel("u2")
ax1.set_title("(b) the set of CE payoffs", fontsize=11)
ax1.set_xlim(1.4, 8.0); ax1.set_ylim(1.4, 8.0)
ax1.grid(alpha=0.3); ax1.set_aspect("equal")
ax1.legend(fontsize=8.5, loc="lower left", framealpha=0.92)

# --------------------------------------------------------------- (c) symmetric slice
quad = np.array([[0, 0.4], [0, 0.5], [0.5, 0.25], [4 / 9, 2 / 9]])
ax2.fill(quad[:, 0], quad[:, 1], color="#4C8BF5", alpha=0.20, zorder=1)
ax2.plot(np.append(quad[:, 0], quad[0, 0]), np.append(quad[:, 1], quad[0, 1]),
         color="#17458f", lw=1.6, zorder=3)
xs = np.linspace(-0.03, 0.6, 300)
ax2.plot(xs, xs / 2, ls="--", color="0.6", lw=1, label="p2 = p1/2   (P1 told S)")
ax2.plot(xs, 0.4 - 0.4 * xs, ls="--", color="0.45", lw=1, label="p2 = (2-2p1)/5   (P1 told B)")
ax2.plot(xs, (1 - xs) / 2, ls=":", color="0.6", lw=1, label="p4 = 0")
lbl = [("(0, 2/5)  u = 3.6", (0, 0.4), (16, -14)),
       ("(0, 1/2)  u = 4.5", (0, 0.5), (16, 4)),
       ("(1/2, 1/4)  u = 5.25", (0.5, 0.25), (-108, -34)),
       ("(4/9, 2/9)  u = 4.667", (4 / 9, 2 / 9), (-136, 12))]
for txt, (a, b), off in lbl:
    ax2.scatter(a, b, color="#17458f", s=50, zorder=6)
    ax2.annotate(txt, (a, b), xytext=off, textcoords="offset points",
                 fontsize=9, color="#17458f",
                 arrowprops=dict(arrowstyle="-", color="#17458f", lw=0.6))
X, Q = np.meshgrid(np.linspace(-0.03, 0.6, 400), np.linspace(0.15, 0.56, 400))
cs = ax2.contour(X, Q, 6 * X + 9 * Q, levels=[3.6, 4.0, 4.5, 5.0, 5.25],
                 colors="darkorange", linewidths=0.9, alpha=0.9)
ax2.clabel(cs, fmt="%.2f", fontsize=7.5)
ax2.set_xlabel("p1 = p(SS)          (q = p2 = p3,  p4 = 1 - p1 - 2q)")
ax2.set_ylabel("q = p(SB) = p(BS)")
ax2.set_title("(c) symmetric slice p2 = p3\n"
              "blue = CE region,  orange = level curves of u = 6p1 + 9q", fontsize=11)
ax2.set_xlim(-0.03, 0.6); ax2.set_ylim(0.15, 0.56)
ax2.grid(alpha=0.3); ax2.legend(fontsize=8, loc="upper right", framealpha=0.92)

fig.suptitle("Correlated equilibria of the game   [[6,6],[2,7],[7,2],[0,0]]",
             fontsize=13.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig("ce_6_2_7_0.png", dpi=170, bbox_inches="tight")
print("saved ce_6_2_7_0.png")
