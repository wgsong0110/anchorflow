"""절 AF 1 차: 확정 설정 (절 AD) 을 다른 씬에 -- N 스윕 (비측정, 카드 0~6). 줄 형식: <EX_ROOT> <태그> <fem_ex_sim 인자...>
씬: lego 홀드아웃 정답 궤적 lbpt/test/0000~0019 (lbdata/test h5 를 lb_to_pt 로 옮긴 것, 정답 생성 config lbdata/config.json 같음 -> L000 과 같은 물리 인자).
AF_DATA: .pt 가 있는 곳 (계산 노드 /tmp 사본, 없으면 /home/dkta/work/lbpt/test). AF_R 반복 (기본 3), AF_N N 목록."""
import os

D = os.environ.get("AF_DATA", "/home/dkta/work/lbpt/test")
REC = "--mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5"   # 절 AD 확정 플래그 그대로
LEGO = "--floor vel --node_h 0.028284 --frames 80 --no_dump"                                            # lego L000 과 같은 씬 인자
NS = [int(q) for q in os.environ.get("AF_N", "64 96 128 160 192 224 256").split()]
TR = [f"{i:04d}" for i in range(20)]
NR = int(os.environ.get("AF_R", "3"))
for r in range(1, NR + 1):
    for N in NS:
        for t in TR:
            print(f"fem_af_lego af_{t}_n{N}_r{r} --data {D}/{t}.pt --tag T{t} {LEGO} {REC} --N {N}")
