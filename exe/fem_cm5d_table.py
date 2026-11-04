import json,os,sys
R=sys.argv[1]
V=[("r","재현 2x6"),("c3","2x3"),("c12","2x12"),("c24","2x24"),("p0","2x6 εP 0"),("n4","4x6"),("u","시작점 U 2x6"),("u3","시작점 U 2x3")]
T=["0001","0006","0014","0015","0017","0018"]
print("| 궤적 | "+" | ".join(n for _,n in V)+" |"); print("|---"*(len(V)+1)+"|")
for t in T:
    c=[]
    for v,_ in V:
        p=f"{R}/{v}_{t}/summary.json"
        if not os.path.exists(p): c.append("미완"); continue
        s=json.load(open(p)); dv=s.get("diverged")
        c.append(("**x%d** (%.2f)"%(dv,100*s["err_mean_done"])) if dv else "%.2f"%(100*s["err_mean"]))
    print("| %s | "%t+" | ".join(c)+" |")
