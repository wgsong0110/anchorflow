#!/bin/bash
# 절 BD 보충 둘째 (bigdt2m2; 커밋 $1, 잡 130567): bigdt2m 이 wolf·bread 정답 MPM 을 FileNotFoundError 로 2 분 만에 끝낸 것과 wolf K 8 우리 판 중단을 다시.
#  (1) 정답 MPM: exe/gt_sens.py 가 PG gs_simulation 출력 하위 폴더 simulation_ply/ 를 찾고, --pg_cfg 원래 서브스텝은 PG 와 똑같이 int(frame_dt/substep_dt) 걸음 x substep_dt
#      (wolf 1999 걸음), 모래 입자 부피는 PG get_particle_volume(unifrom=sand) 대로 균일, config 재질 매개 (friction_angle 등) 그대로.
#      원래 서브스텝 mpmK0 이 정답과 평균 오차 < 0.005 % (표에 0.00) 이고 50/300 프레임 완주면 그 씬만 K 1·2·4·8 을 띄움.
#  (2) wolf K 8: 첫 시도는 프레임 40 에서 노드 상자가 용량 (첫 프레임 상자 x 2) 을 넘어 다시 할당한 직후 CUDA 불법 접근 (wolf 재질 판에서 다시 할당이 처음 일어난 판).
#      같은 명령 + --cap_mul 4 (결정적 판이라 결과 불변 -- 프레임 1..39 오차가 첫 시도와 같은지 자동 대조) 로 표 칸을 채움.
#      진단: dgK1c1 (K 1 + --cap_mul 1 -> 이른 프레임에 다시 할당; 죽으면 wolf 재질 판 다시 할당 경로 결함 확정), sanK1c1 (같은 것을 compute-sanitizer memcheck 아래, 20 프레임).
# 출력 /dev/shm/dkta/lbours/fem_an/bigdt2m2 (R2 bigdt2m2/) -> 끝나면 bigdt2/ 로 덮어씀 (있던 같은 이름은 <이름>.fail1), 표 bigdt2_table.md, 끝 표식 bigdt2/bigdt2m.END (옛 것은 bigdt2m_fail1.END).
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1
PY=/home/dkta/.conda/envs/af/bin/python
NM=bigdt2m2
R=/dev/shm/dkta/lbours/fem_an/$NM; RR=r2:storage/result/anchorflow/lbours/fem_an/$NM
R2=/dev/shm/dkta/lbours/fem_an/bigdt2; RR2=r2:storage/result/anchorflow/lbours/fem_an/bigdt2
mkdir -p $R /dev/shm/dkta/an_lists
log() { echo "$*" | tee -a $R/$NM.log; rclone copy $R/$NM.log $RR -q; }
log "[$NM 시작] $C $(date '+%F %T')"
[ -f $R2/list.txt ] || { mkdir -p $R2; rclone copy $RR2 $R2 -q; log "bigdt2 를 R2 에서 내려받음"; }
if [ -f $R2/bigdt2m.END ]; then mv $R2/bigdt2m.END $R2/bigdt2m_fail1.END; fi
rclone moveto $RR2/bigdt2m.END $RR2/bigdt2m_fail1.END -q 2>/dev/null
UNTIL=$(date -d '2026-11-05 12:09 KST' +%s); HARD=$(date -d '2026-11-05 12:20 KST' +%s)
pool() { AN_UNTIL=$UNTIL AN_CARDS="${AN_CARDS:-1 2 3 4 5 6}" AN_PAR=${AN_PAR:-3} AN_GAP=4 bash exe/fem_an_bigdt_pool.sh $NM $1 >> $R/pool_$2.log 2>&1; rclone copy $R/pool_$2.log $RR -q; }
HW="--ga hht --rho 0.8 --newton 2 --cg 4 --tolN 1e-2 --tolP 1e-1 --det"                 # 절 BD bigdt2 wolf 줄 그대로 (K 만 다름)
# 가: memcheck 진단 (느림) 은 따로 돌림
LA=/dev/shm/dkta/an_lists/${NM}_a.txt
echo "sanK1c1 wolf $HW --K 1 --cap_mul 1.0 --frames 20" > $LA
pool $LA a &
PA=$!
# 나: 정답 원래 서브스텝 확인 + wolf K 8 (용량 4) + 다시 할당 시험
LB=/dev/shm/dkta/an_lists/${NM}_b.txt
{ echo "mpmK0 wolf --sub_chk --rpic_cfg"; echo "mpmK0 bread --sub_chk --rpic_cfg"
  echo "hhK8 wolf $HW --K 8 --cap_mul 4.0"; echo "dgK1c1 wolf $HW --K 1 --cap_mul 1.0"; } > $LB
pool $LB b
# 원래 서브스텝 재현 판정 -> 통과한 씬만 K 1·2·4·8
LC=/dev/shm/dkta/an_lists/${NM}_c.txt; : > $LC
for t in wolf bread; do
  v=$($PY - $R/mpmK0_$t/err.json <<'EOF'
import json, math, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception as e:
    print("없음", type(e).__name__); sys.exit()
er = s["err"]
m = 100 * sum(er) / max(len(er), 1)
ok = s.get("div") is None and len(er) > 0 and all(math.isfinite(e) for e in er) and m < 0.005
print(("통과" if ok else "실패"), f"{m:.4f}%", f"{len(er)} 프레임", f"서브스텝 {s['nsub']} x {s['dt']:.4e}", f"최대 {100 * max(er):.4f}%")
EOF
)
  log "정답 MPM 원래 서브스텝 $t: $v"
  case "$v" in 통과*) for K in 1 2 4 8; do echo "mpmK$K $t --nsub $K --sub_chk --rpic_cfg" >> $LC; done;; esac
done
[ -s $LC ] && pool $LC c
# wolf K 8 용량 4 판이 첫 시도와 프레임 1..39 같은지
$PY - $R2/hhK8_wolf/fem.jsonl $R/hhK8_wolf/fem.jsonl <<'EOF' | while read -r q; do log "$q"; done
import json, sys
def rd(p):
    try:
        return [json.loads(q) for q in open(p) if q.strip()]
    except Exception:
        return []
a, b = rd(sys.argv[1]), rd(sys.argv[2])
n = min(len(a), len(b))
same = sum(1 for i in range(n) if a[i]["err"] == b[i]["err"])
print(f"wolf K 8 용량 4 판 대조: 첫 시도 {len(a)} 프레임, 새 판 {len(b)} 프레임, 겹친 {n} 프레임 중 오차 비트 같음 {same}" + (" -- 결과 불변 확인" if n and same == n else " -- 다름 (확인 필요)"))
EOF
for v in dgK1c1_wolf hhK8_wolf; do log "$v: $(grep -a '용량 늘려\|롤아웃\|rror' $R/$v/log.txt 2>/dev/null | tail -3 | cut -c1-200 | tr '\n' ' ')"; done
# 진단 가 는 마감까지 기다림
while kill -0 $PA 2>/dev/null && [ "$(date +%s)" -lt "$HARD" ]; do sleep 60; done
log "sanK1c1_wolf: $(grep -a 'Invalid\|at 0x\|by thread\|ERROR SUMMARY\|용량 늘려\|rror' $R/sanK1c1_wolf/log.txt 2>/dev/null | head -12 | cut -c1-200 | tr '\n' ' ')"
# bigdt2/ 로 덮어쓰기 (같은 이름은 .fail1)
for d in $R/mpmK*_wolf $R/mpmK*_bread $R/hhK8_wolf; do
  [ -f $d/run_info.txt ] || continue; n=$(basename $d)
  if [ -d $R2/$n ] && [ ! -d $R2/$n.fail1 ]; then mv $R2/$n $R2/$n.fail1; rclone move $RR2/$n $RR2/$n.fail1 -q; fi
  rm -rf $R2/$n; cp -r $d $R2/$n; rclone copy $R2/$n $RR2/$n -q
done
$PY exe/fem_an_bigdt2_table.py $R2 > $R2/bigdt2_table.md 2>&1
{ echo; echo "## 보충 bigdt2m2 ($C) 기록"; echo; sed 's/^/- /' $R/$NM.log; } >> $R2/bigdt2_table.md
rclone copy $R2/bigdt2_table.md $RR2 -q
echo "[bigdt2m 끝] $C $(date '+%F %T') (bigdt2m2 판; 첫 판 실패 기록 bigdt2m_fail1.END)$(kill -0 $PA 2>/dev/null && echo ' 진단 sanitizer 아직 돎 (마감)')" | tee -a $R/$NM.log > $R2/bigdt2m.END
rclone copy $R/$NM.log $RR -q; rclone copy $R2/bigdt2m.END $RR2 -q
