import numpy as np,sys,os
run,f=sys.argv[1],int(sys.argv[2]); k=int(sys.argv[3]) if len(sys.argv)>3 else 0
z=np.load(os.path.join(run,'esc',f'f{f:03d}.npz'))
IP=z['IP']; nb1,nb2=int(IP[7]),int(IP[8]); nid=z['nid'].astype(np.int64)
nrel=np.stack([nid//(nb1*nb2),(nid//nb2)%nb1,nid%nb2],1)
cb=z['cb'].astype(np.int64); nc=len(cb); QS=z['QS'][k]; Vq=z['Vq']; mq=z['mq']
st=QS[:,3].astype(int); dX=QS[:,4]
# per node accumulate over cells c with cb in [n-2,n]
key=lambda a:(a[:,0]*4096+a[:,1])*4096+a[:,2]
ck=key(cb); order=np.argsort(ck); cks=ck[order]
Vap=np.zeros(len(nid)); Vall=np.zeros(len(nid)); Vel=np.zeros(len(nid)); minJ=np.full(len(nid),9.)
Vc=Vq.reshape(-1,8); stc=st.reshape(-1,8); dXc=dX.reshape(-1,8)
for di in range(3):
  for dj in range(3):
    for dk in range(3):
      c=nrel-np.array([di,dj,dk]); kk=key(c); j=np.searchsorted(cks,kk); j=np.clip(j,0,nc-1); ok=cks[j]==kk; ci=order[j]
      v=np.where(ok[:,None],Vc[ci],0); s=stc[ci]
      Vall+=v.sum(1); Vap+=(v*(s==2)).sum(1); Vel+=(v*(s==0)).sum(1)
      minJ=np.minimum(minJ,np.where(ok,dXc[ci].min(1),9.))
fap=np.where(Vall>0,Vap/np.maximum(Vall,1e-30),np.nan)
dy=np.linalg.norm(z['Y'][k]-z['Ut'][k],axis=1); mn=z['mn']; med=np.median(mn); ext=z['W8'][k]>=0
g0=np.linalg.norm(z['G0'][k],axis=1); w=mn*9.8
print(f'f{f} k{k}: 노드 {len(nid)}; 꼭짓점 부피 몫 분포', np.round(np.nanquantile(fap,[.1,.5,.9]),3))
for lo,hi in ((0,.25),(.25,.5),(.5,.75),(.75,.95),(.95,1.01)):
    s=(fap>=lo)&(fap<hi)&(~ext)
    s2=(fap>=lo)&(fap<hi)&ext
    if s.any(): print(f'  꼭짓점 몫 [{lo},{hi}) 내부·고립 n{s.sum()} |y-Ũ| 99% {np.quantile(dy[s],.99):.4f} 최대 {dy[s].max():.3f} | 외부 n{s2.sum()} 99% {np.quantile(dy[s2],.99) if s2.any() else 0:.4f} 최대 {dy[s2].max() if s2.any() else 0:.3f} | 잔차/무게 중앙 {np.median(g0[s]/np.maximum(w[s],1e-12)):.1f} 99% {np.quantile(g0[s]/np.maximum(w[s],1e-12),.99):.1f}')
big=dy>0.03
print('  |y-Ũ|>0.03 노드:', big.sum(), '꼭짓점 몫 중앙', np.nanmedian(fap[big]) if big.any() else '-', '외부 몫', ext[big].mean() if big.any() else '-', 'min det X 중앙', np.median(minJ[big]) if big.any() else '-')
print('  전체 점 상태 몫 (부피):', [round(float(Vq[st==s_].sum()/Vq.sum()),3) for s_ in (0,1,2)], 'det X<0.1 점', int((dX<0.1).sum()), 'det X>1.5 점', int((dX>1.5).sum()))
