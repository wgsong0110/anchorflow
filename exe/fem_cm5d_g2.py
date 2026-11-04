import json,sys
def show(path, lo=0, hi=999):
    rows=[json.loads(l) for l in open(path+"/fem.jsonl")]
    for r in rows:
        if not (lo<=r["t"]<=hi): continue
        d=r.get("diag"); 
        it=d["it"] if d else []
        s=" ".join("%.0e>%.0e"%(x["g2"], it[i+1]["g2"] if i+1<len(it) and i%2==0 else float('nan')) for i,x in enumerate(it) if i%2==0)
        e=r["err"]
        print(r["t"], "%.2f"%(100*e) if e==e else "nan", r["nbad"], r["it"]["capP"], r.get("y0k","-"), "| g2 걸음별 (뉴턴1>뉴턴2):", s)
show(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
