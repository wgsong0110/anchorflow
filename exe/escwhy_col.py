import numpy as np,sys,os
run,f=sys.argv[1],int(sys.argv[2]); px=np.array([float(q) for q in sys.argv[3].split(',')])
z=np.load(os.path.join(run,'esc',f'f{f:03d}.npz'))
IP,A,Ai=z['IP'],z['A'],z['Ai']; nb1,nb2=IP[7],IP[8]
nid=z['nid'].astype(np.int64); g=np.stack([nid//(nb1*nb2),(nid//nb2)%nb1,nid%nb2],1)+IP[0:3]
pos=g@A.T; med=np.median(z['mn']); k=int(sys.argv[4]) if len(sys.argv)>4 else 0
gp=np.round(px@Ai.T).astype(int)
print('grid of point',gp,'A diag',np.diag(A))
for dx in (-1,0,1):
  for dy_ in (-1,0,1):
    sel=np.where((g[:,0]==gp[0]+dx)&(g[:,1]==gp[1]+dy_))[0]
    sel=sel[np.argsort(g[sel,2])]
    print(f'-- 기둥 dx {dx} dy {dy_}')
    for n in sel:
        if pos[n][2]>0.7: continue
        print(f"  z {pos[n][2]-0.48:+.3f} m x{z['mn'][n]/med:.1e} 외 {int(z['W8'][k][n]>=0)} pm {z['PM'][k][n]} sk {z['SK'][k][n]:.2f} kb {z['KB'][k][n]:.2f} ul {z['UL'][k][n]} bG {z['BG'][k][n]:+.1e} bK {z['BK'][k][n]:.1e} v0z {z['V0'][k][n][2]:+.3f} v1 {np.round(z['V1'][k][n],3)} y-Ũ {np.round(z['Y'][k][n]-z['Ut'][k][n],4)} |g0| {np.linalg.norm(z['G0'][k][n]):.1e} fz {z['Fn'][k][n][2]:+.2e} an {z['AN'][k][n]:.1e}")
