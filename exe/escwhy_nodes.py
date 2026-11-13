import numpy as np,sys,glob,os
run=sys.argv[1]; fr=[int(q) for q in sys.argv[2].split(':')]
for f in range(fr[0],fr[1]+1):
    p=os.path.join(run,'esc',f'f{f:03d}.npz')
    if not os.path.exists(p): continue
    z=np.load(p)
    if 'nid' not in z.files: print(f,'no rec'); continue
    K=int(z['K']); mn=z['mn']; med=np.median(mn)
    for k in range(K):
        dy=np.linalg.norm(z['Y'][k]-z['Ut'][k],axis=1)
        dU=np.linalg.norm(z['U1'][k]-z['U0'][k],axis=1)
        ext=z['W8'][k]>=0; pm=z['PM'][k]; kb=z['KB'][k]; v1=np.linalg.norm(z['V1'][k],axis=1)
        r=mn/med
        big=dy>0.02
        print(f'f{f} k{k} K{K}: 노드 {len(mn)} |y-Ũ|>1칸 {big.sum()} (외부 {np.sum(big&ext)}, pm2 {np.sum(big&(pm==2))}, kb<1 {np.sum(big&(kb<1))}) |v1|>3 {np.sum(v1>3)}',
              '| 질량비 분포(큰 dy):', np.round(np.quantile(r[big],[0,.25,.5,.75,1]),5) if big.any() else '-',
              '| 전체 노드 외부 몫 %.2f'%ext.mean(), 'dy 최대 %.3f'%dy.max())
        # dy vs mass ratio bins
        bins=[0,1e-4,1e-3,1e-2,1e-1,1,1e9]
        row=[]
        for a,b in zip(bins[:-1],bins[1:]):
            s=(r>=a)&(r<b)
            if s.any(): row.append(f'[{a:g},{b:g}) n{s.sum()} dy99 {np.quantile(dy[s],0.99):.4f} max {dy[s].max():.3f} ext {ext[s].mean():.2f}')
        print('    ', ' ; '.join(row))
