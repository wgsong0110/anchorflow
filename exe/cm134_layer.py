"""절 CM134 층별 측정 (CPU, 읽기만; torch 없이 exe/cm130_ptload.py 로 pred.pt 읽음). 사용: python exe/cm134_layer.py <wolf.pt> <판 pred.pt 가 든 디렉토리 묶음 루트> <판 ...>
1) 방출 뒤 시간 tau (정답 입자별 처음 dz < -0.5 mm 프레임 기준) 별 평균 v_z -- 낙하 가속 (정답 대 판)  2) 높이 띠별 수평 속력 (바닥 고착·퍼짐).
(경로는 /dev/shm/cm134a/{wolf.pt,<판>/pred.pt} 로 둠 -- 인자 판 이름)"""
# ---- 1) 방출 뒤 시간별 v_z ----
import sys; sys.path.insert(0,'exe')
import cm130_ptload as L, numpy as np
d=L.load('/dev/shm/cm134a/wolf.pt'); GT=np.array(d['x'][:51],dtype=np.float32)
runs={'gt':GT}
for n in sys.argv[1:]:
    Pp=np.array(L.load(f'/dev/shm/cm134a/{n}/pred.pt')['pred'],dtype=np.float32); runs[n]=np.concatenate([GT[:1],Pp],0)
z0=GT[0][:,2]
# GT onset: first frame where dz < -0.5mm
dz=GT[:,:,2]-z0[None]
on=np.where((dz<-5e-4).any(0),(dz<-5e-4).argmax(0),999)
sel=(on<40)&(z0>0.75)   # released, started high so no floor contact early
print('particles',sel.sum(),'onset hist',np.bincount(on[sel])[:40])
h=0.04
for tau in range(0,6,1):
    row=[]
    for k,X in runs.items():
        vs=[]
        for o in np.unique(on[sel]):
            f=o+tau
            if f>=len(X) or f<1: continue
            m=sel&(on==o)
            vs.append(((X[f][m,2]-X[f-1][m,2])/h).mean()*m.sum())
        n=sum(((on==o)&sel).sum() for o in np.unique(on[sel]) if o+tau<len(X))
        row.append(f"{k}:{sum(vs)/max(n,1):6.2f}")
    print(f"tau {tau:2d} (g*tau*h={9.8*tau*h:5.2f})"," ".join(row))

# ---- 2) 높이 띠별 수평 속력 (별도 실행: python exe/cm134_layer.py 와 같은 인자, 아래 블록을 따로 떼어 쓰면 됨) ----
# import sys; sys.path.insert(0,'exe')
# import cm130_ptload as L, numpy as np
# d=L.load('/dev/shm/cm134a/wolf.pt'); GT=np.array(d['x'][:51],dtype=np.float32)
# runs={'gt':GT}
# for n in sys.argv[1:]:
#     Pp=np.array(L.load(f'/dev/shm/cm134a/{n}/pred.pt')['pred'],dtype=np.float32); runs[n]=np.concatenate([GT[:1],Pp],0)
# h=0.04
# print('particles by z at frame f-1 band; mean horizontal speed |vxy| (m/s), mean vz; fraction of mass')
# for f in [20,26,32,40]:
#     for k,X in runs.items():
#         if f>=len(X): continue
#         v=(X[f]-X[f-1])/h; z=X[f-1][:,2]
#         out=[]
#         for a,b in [(0.48,0.50),(0.50,0.54),(0.54,0.62),(0.62,2)]:
#             m=(z>=a)&(z<b)
#             out.append(f"[{a:.2f},{b:.2f}) {m.mean():.2f} |vxy| {np.linalg.norm(v[m,:2],axis=1).mean():.2f} vz {v[m,2].mean():+.2f}")
#         print(f"f{f} {k:6s}"," | ".join(out))
