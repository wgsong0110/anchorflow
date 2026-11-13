import numpy as np,sys,os,glob
run=sys.argv[1]
prev=None
for p in sorted(glob.glob(os.path.join(run,'esc','f*.npz'))):
    z=np.load(p); f=int(p[-7:-4]); x=z['p1_x']; x0=z['p0_x']
    lo=np.quantile(x,0.001,axis=0); hi=np.quantile(x,0.999,axis=0)
    out=((x<lo-0.08)|(x>hi+0.08)).any(1)
    v=np.linalg.norm(z['p1_v'],axis=1)
    new=out if prev is None else out&~prev
    print(f'f{f}: 4칸 밖 {out.sum()} 새 {new.sum()} |v|>3 {np.sum(v>3)} 최대 |v| {v.max():.2f} 상자 {int(z["IP"][9])}', end='')
    if new.any():
        i=np.where(new)[0][:3]
        print(' 새 예:', [(int(j), np.round(x0[j],3).tolist(), np.round(z['p0_v'][j],2).tolist(), np.round(z['p1_v'][j],2).tolist()) for j in i])
    else: print()
    prev=out
