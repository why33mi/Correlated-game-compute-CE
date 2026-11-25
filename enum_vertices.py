import numpy as np, itertools
np.set_printoptions(precision=4, suppress=True)
IA = np.eye(4)
CE = np.array([[-1,2,0,0],[0,0,1,-2],[-1,0,2,0],[0,1,0,-2]])
rows = np.vstack([IA, CE])
namer = ["p1=0","p2=0","p3=0","p4=0","CE1","CE2","CE3","CE4"]
ones = np.ones(4)
def feasible(p):
    return np.all(p>=-1e-9) and abs(p.sum()-1)<1e-9 and np.all(CE@p>=-1e-9)
combos=list(itertools.combinations(range(8),3))
sing=infeas=0; found=[]
for combo in combos:
    M=np.vstack([rows[i] for i in combo]+[ones]); rhs=np.zeros(4); rhs[3]=1
    if abs(np.linalg.det(M))<1e-12: sing+=1; continue
    p=np.linalg.solve(M,rhs)
    if feasible(p): found.append((combo,p))
    else: infeas+=1
print(f"total 3-subsets : {len(combos)}")
print(f"singular systems: {sing}")
print(f"infeasible      : {infeas}")
seen=[]
for combo,p in found:
    if not any(np.allclose(p,q,atol=1e-9) for q in seen):
        seen.append(p)
        tag="+".join(namer[i] for i in combo)
        print(f"  {tag:28s} -> {np.round(p,4)}")
print("distinct vertices:", len(seen))
print()
for p in seen:
    tight=[namer[i] for i in range(8) if abs(rows[i]@p)<1e-12]
    print(f"  {np.round(p,4)}  tight: {tight}")
