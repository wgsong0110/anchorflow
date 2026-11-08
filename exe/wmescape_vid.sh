#!/bin/bash
# 수박 입자 날아감 비교 영상 (사용자 11-08 요청): E · F (= E + W̄ + R2 + A2, 절 CM49) · F + 벌점 셋 γ̄ 1e-1 (@g3 0.1, 절 CM51) 을 판마다 덤프 켜고 2 판씩 (비결정).
# 판 명령 = exe/stab_ghost3.sh 의 수박 방식 g 그대로 (같은 감싸개 사슬·진단 D1·D2·D5·G1, 플래그 해석은 그 파일에서 sed 로 읽음) + --dump (옛 실행기: pred.pt 에 위치 + F).
# 판별: 이탈 최대 수 (e1.jsonl nfar 의 프레임 최대 = 절 CM51 표 '이탈 최대 수' 와 같은 정의) 가 큰 판 (같으면 거리 큰 판) -> 3DGS 일반 렌더 (exe/wm_render.py, 절 CM46 영상과 같은 렌더러·카메라) + 점 그림용 위치 (exe/wmescape_pts.py).
# 잡 안, 커밋 디렉토리에서. 사용: bash exe/wmescape_vid.sh <출력> <R2>
O=$1; RO=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
W=/home/dkta/work; CD=$(pwd -P)
eval "$(sed -n '/^WMD=/p;/^WMB=/p;/^BW4=/p;/^V2N=/p;/^DFL=/p;/^nw()/p;/^dfl()/p' exe/stab_ghost3.sh)"
[ "$AF_DRY" = 1 ] || mkdir -p $WMD
[ "$AF_DRY" = 1 ] || [ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
log() { echo "[$(date '+%F %T')] $*" >> $O/vid.log; rclone copyto $O/vid.log $RO/vid.log -q; }
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
cmd_of() {   # cmd_of 출력디렉토리 플래그 -> stab_ghost3.sh run() 의 수박 방식 g 와 같은 명령 (+ --dump)
  local Od=$1 x="$(dfl M3 "$2")" b wo w2 w3 w5 wg
  wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+,-]*' | grep -v '^@e[235]' | grep -v '^@g3' | sed 's/^@/--/' | tr '\n' ' ')"; wg="$(echo " $x " | grep -o '@g3[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w5="$(echo " $x " | grep -o '@e5[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9.e+,-]*//g')"
  b="${WMB/--cell 1.5/--cell 3}"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  echo "exe/ghost3_wrap.py $wg $w5 $w3 $w2 $wo $b --erec --dump $x --out $Od"; }
[ "$AF_DRY" = 1 ] && { cmd_of /X/E " E "; cmd_of /X/F " E @e3w 1 @e5r 2 @e5a 2 "; cmd_of /X/G " E @e3w 1 @e5r 2 @e5a 2 @g3 1e-1 "; exit 0; }
RUNS="E_a|E|E
E_b|E|E
F_a|F|E @e3w 1 @e5r 2 @e5a 2
F_b|F|E @e3w 1 @e5r 2 @e5a 2
G_a|G|E @e3w 1 @e5r 2 @e5a 2 @g3 1e-1
G_b|G|E @e3w 1 @e5r 2 @e5a 2 @g3 1e-1"
run() { local n=$1 x="$2" g=$3 Od=$O/$1 c
  [ "$(cat $Od/rc.txt 2>/dev/null)" = 0 ] && [ -s $Od/pred.pt ] && return                    # 끝난 판 (rc 0 + 덤프) 은 다시 안 돌림
  rm -rf $Od; mkdir -p $Od
  c="$(cmd_of $Od " $x ")"; echo "AF_E2_DG=1 AF_E5_DG=1 AF_E1_DG=1 AF_G1_DG=1 $PY -u $c" > $Od/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $Od/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  env AF_E2_DG=1 AF_E5_DG=1 AF_E1_DG=1 AF_G1_DG=1 CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c < /dev/null > $Od/log.txt 2>&1
  local rc=$?; echo $rc > $Od/rc.txt
  log "$n rc=$rc 덤프 $(ls -la $Od/pred.pt 2>&1 | awk '{print $5}') $(grep -ah '롤아웃\]' $Od/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $Od/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $Od $RO/$n -q --transfers 8 --exclude pred.pt; }
while IFS='|' read -r n v x; do
  ( me=$BASHPID; g=$(claim_wait $me); run $n "$x" $g; rm -f $CL/$g ) & sleep 5
done <<< "$RUNS"
wait
log "판 끝 (남은 자리 $(df -h --output=avail $O | tail -1))"
# 판별 이탈 최대 수·거리 -> 고른 판
$PY - $O <<'PYX' > $O/pick.json 2> $O/pick.err
import json, os, sys
O = sys.argv[1]; R = {}
for v in "EFG":
    c = []
    for r in "ab":
        n = f"{v}_{r}"; p = os.path.join(O, n, "e1.jsonl")
        if not os.path.exists(p): continue
        e = [json.loads(l) for l in open(p) if l.strip()]; e = [q for q in e if "nfar" in q]
        if not e: continue
        s = json.load(open(os.path.join(O, n, "summary.json"))) if os.path.exists(os.path.join(O, n, "summary.json")) else {}
        c.append(dict(run=n, nfar=max(q["nfar"] for q in e), dmax=max((q.get("dmax") or 0) for q in e), frames=len(e), err=s.get("err_mean_done"), diverged=s.get("diverged")))
    R[v] = dict(runs=c, pick=max(c, key=lambda q: (q["nfar"], q["dmax"]))["run"] if c else None)
print(json.dumps(R, ensure_ascii=False, indent=1))
PYX
rclone copy $O $RO -q --include "pick.*"
log "고른 판 $(tr -d '\n ' < $O/pick.json | cut -c1-600)"
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json
GTM=/tmp/dkta/ag_data/gt_wm; [ -f $GTM/meta.json ] || GTM=/dev/shm/dkta/ag_data/gt_wm
log "렌더 모델 $WMM 설정 $CFG 정답 궤적 $GTM $(ls $GTM 2>&1 | head -3 | tr '\n' ' ')"
rend() { local n=$1 g=$2 Od=$O/$1 V=$O/$1/vj
  mkdir -p $V
  $PY - $Od/pred.pt $Od/pred_v.pt <<'PYX' > $V/pad.log 2>&1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
n = 0
while n < P.shape[0] and bool(torch.isfinite(P[n]).all()) and bool(torch.isfinite(F[n]).all()): n += 1
L = 100
P, F = P[:n], F[:n]
if n < L:
    P = torch.cat([P, P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F, F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("유한 프레임", n, "->", P.shape[0])
PYX
  $PY -u exe/lb_roll_to_h5.py --dump $Od/pred_v.pt --traj $GTM --out $Od/h5_fem > $V/h5.log 2>&1; rm -f $Od/pred_v.pt
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --sim $Od/h5_fem --out $V/plain.mp4 < /dev/null > $V/r_plain.log 2>&1)
  log "[렌더] $n plain $(ls -la $V/plain.mp4 2>&1 | awk '{print $5}') $(tail -1 $V/r_plain.log | cut -c1-160) $(tail -1 $V/h5.log | cut -c1-120)"
  rm -rf $Od/h5_fem                                                                        # 렌더용 변환본 (pred.pt 에서 다시 만들 수 있음) -- 자리 때문에 지움
  rclone copy $V $RO/$n/vj -q --transfers 8; }
for v in E F G; do
  n=$($PY -c "import json; print(json.load(open('$O/pick.json'))['$v']['pick'] or '')")
  [ -n "$n" ] || { log "$v 고른 판 없음"; continue; }
  ( me=$BASHPID; g=$(claim_wait $me); rend $n $g; rm -f $CL/$g )                          # 차례로 (변환본 자리)
done
# 점 그림용 위치 (CPU; 정답 + 판 6 개 모두)
CELL=$(grep -aom1 "셀 [0-9.]* m" $O/E_a/log.txt | awk '{print $2}'); [ -n "$CELL" ] || CELL=0.02
RA=""; for n in E_a E_b F_a F_b G_a G_b; do [ -f $O/$n/pred.pt ] && RA="$RA --run $n=$O/$n/pred.pt"; done
$PY -u exe/wmescape_pts.py --cell $CELL --gt $WMD/wm.pt --outdir $O/pts $RA > $O/pts.log 2>&1
log "[위치] 셀 $CELL $(ls -la $O/pts | awk 'NR>1{printf "%s %s · ", $9, $5}') $(tail -7 $O/pts.log | cut -c1-160 | tr '\n' ' ')"
rclone copy $O/pts $RO/pts -q --transfers 8; rclone copy $O $RO -q --include "pts.log"
wait
log "VID_END"
