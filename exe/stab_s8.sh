#!/bin/bash
# 절 S.8 묶음 (vast.ai A6000 4 장 인스턴스, exe/vast_s6.sh S6_PART=s8 가 세팅 뒤 부름; 커밋 $1 디렉토리에서):
#   (1) 우리 (B) L000 K 4 발산 진단 exe/stab_k4diag.sh (마지막 카드를 anerr_claim 으로 잡음) -> R2 lbours/stab_k4diag
#   (2) stab 빈칸 채우기 = exe/stab.sh 이어 돌리기 (방법 mpm·vrgs·bnv·ours·phystwin·simplicits x L000·ficus·wolf·bread; 끝난 판 건너뜀):
#       정답 MPM L000·ficus 덤프·시간 판 다시 시도 (stab.py RETRY), wolf·bread 시각 품질 (pg_h5_to_pt 정렬 수정 8e478e0 뒤 정답 렌더 디렉토리 다시)
#   (3) Spring-Gaus 연기 (L000·ficus, 3 프레임) -> 통과하면 본 판 (같은 R2 lbours/stab). 실패하면 /root/s6/sg_commit 에 고친 커밋이 들어올 때까지
#       (최대 S8_SGWAIT 초) 기다렸다 그 커밋으로 다시 (최대 5 회).
#   (4) 모두 끝나면 (stab.py 프로세스 없음, /root/s6/hold 없음) 모든 방법·씬으로 한 번 더 (전부 이어 돌리기 -> 표 다시 만듦), 표식 s8_*.
set -u
C=$1
MK=r2:storage/result/anchorflow/lbours/vast_s6/marks; R=r2:storage/result/anchorflow
LOG=/root/s6/s8.log
say() { echo "[s8 $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $MK/s8.log -q 2>/dev/null; }
cdir() {   # 커밋 디렉토리 (github clone -> 커밋별; 인스턴스 편집 없음)
  local c=$1 d=/dev/shm/dkta/impwt/$1
  git -C /root/af fetch -q origin implicit-bspline 2>/dev/null; git -C /root/af pull -q origin implicit-bspline 2>/dev/null
  [ -f $d/.done ] || { rm -rf $d; git clone -q /root/af $d && git -C $d checkout -q $c && echo $c > $d/.done; }
  echo $d; }
D=$(cdir $C); cd $D || exit 1
P3D=$(rclone lsf $R/vastpkg --include "pytorch3d-*.whl" 2>/dev/null | head -1)
[ -n "$P3D" ] && { rclone copy $R/vastpkg/$P3D /root/s6/ -q; export ST_P3D_WHL=/root/s6/$P3D; }
export ST_NO_BUILD=1
mkdir -p /dev/shm/dkta/anerr_claim /dev/shm/dkta/pylib
say "[시작] 커밋 $C pytorch3d ${P3D:-없음}"
bash exe/stab_bl_setup.sh /dev/shm/dkta/pylib/bl_env.sh > /root/s6/bl_setup_s8.log 2>&1; say "[bl_setup] $(tail -1 /root/s6/bl_setup_s8.log)"
rclone copy /root/s6/bl_setup_s8.log $R/lbours/stab_k4diag/ -q
# ---- (1) 진단 (마지막 카드; 4 장이면 3)
KC=$(( $(nvidia-smi -L | wc -l) - 1 ))
( echo $BASHPID > /dev/shm/dkta/anerr_claim/$KC
  K4O=/dev/shm/dkta/lbours/stab_k4diag
  ( while true; do sleep 300; rclone copy $K4O $R/lbours/stab_k4diag -q; done ) & KS=$!
  bash exe/stab_k4diag.sh $KC $K4O; kill $KS; rclone copy $K4O $R/lbours/stab_k4diag -q
  rm -f /dev/shm/dkta/anerr_claim/$KC; echo "$(date)" | rclone rcat $MK/s8_k4diag.done; say "[k4diag 끝] $(tail -1 $K4O/k4diag.log)" ) &
K4=$!
sleep 20
# ---- (2) 빈칸 채우기
MS="mpm,vrgs,bnv,ours,phystwin,simplicits"; SC="L000,ficus,wolf,bread"
( ST_ARGS="--par 3 --methods $MS --scenes $SC" bash exe/stab.sh $C; echo "$(date)" | rclone rcat $MK/s8_main.done; say "[빈칸 채우기 끝] $(cat /dev/shm/dkta/lbours/stab/stab.END 2>/dev/null)" ) &
MA=$!
sleep 60
# ---- (3) Spring-Gaus 연기 -> 본 판
SGOK=""; SGC=$C
for tr in 1 2 3 4 5 6; do
  SGD=$(cdir $SGC)
  say "[Spring-Gaus 연기 $tr] 커밋 $SGC"
  ( cd $SGD && ST_OUT=/dev/shm/dkta/lbours/stab_smoke_sg ST_UNTIL=$(( $(date +%s) + 4 * 3600 )) ST_HARD=$(( $(date +%s) + 5 * 3600 )) \
      ST_ARGS="--smoke 3 --par 2 --methods springgaus --scenes L000,ficus" bash exe/stab.sh $SGC )
  SGOK=$(python - /dev/shm/dkta/lbours/stab_smoke_sg/stab.json <<'PYX'
import json, sys
R = json.load(open(sys.argv[1]))
ok = [t for t in ("L000", "ficus") if (R.get(f"springgaus_{t}") or {}).get("nmin") is not None and ((R.get(f"springgaus_{t}") or {}).get("time") or {}).get("ms") is not None
      and ((R.get(f"springgaus_{t}") or {}).get("vis") or {}).get("psnr") is not None]
print("springgaus" if len(ok) == 2 else "")
PYX
)
  echo "${SGOK:-실패} 시도 $tr 커밋 $SGC" | rclone rcat $MK/s8_sg_smoke.done
  say "[Spring-Gaus 연기 판정 $tr] ${SGOK:-실패} ($(grep -h '준비\]\|끝\]' /dev/shm/dkta/lbours/stab_smoke_sg/stab.log | tail -2 | tr '\n' ' '))"
  [ -n "$SGOK" ] && break
  # 연기 실패: 지난 연기 판 지우고 (준비 판 res.json 없음 -> 다시) 고친 커밋을 기다림
  rclone purge r2:storage/result/anchorflow/lbours/stab_smoke_sg/p -q 2>/dev/null; rm -rf /dev/shm/dkta/lbours/stab_smoke_sg/p
  W0=$(date +%s); NEW=""
  while [ $(( $(date +%s) - W0 )) -lt ${S8_SGWAIT:-7200} ]; do
    NEW=$(cat /root/s6/sg_commit 2>/dev/null); [ -n "$NEW" ] && [ "$NEW" != "$SGC" ] && break; NEW=""; sleep 30; done
  [ -z "$NEW" ] && { say "[Spring-Gaus 보류] 고친 커밋 없음"; break; }
  SGC=$NEW; echo $SGC > /root/s6/final_commit
done
if [ -n "$SGOK" ]; then
  say "[Spring-Gaus 본 판 띄움] 커밋 $SGC"
  ( cd $(cdir $SGC) && ST_ARGS="--par 2 --methods springgaus --scenes L000,ficus" bash exe/stab.sh $SGC )
  say "[Spring-Gaus 본 판 끝]"
fi
wait $K4 $MA
# ---- (4) 마무리 (손으로 띄운 판이 있으면 /root/s6/hold 를 만들어 두고 끝나면 지움)
for i in $(seq 1440); do pgrep -f "exe/stab.py" >/dev/null || [ -f /root/s6/hold ] || break; sleep 60; done
FC=$(cat /root/s6/final_commit 2>/dev/null || echo $C)
say "[마무리] 커밋 $FC 방법 $MS${SGOK:+,springgaus}"
( cd $(cdir $FC) && ST_ARGS="--par 3 --methods $MS${SGOK:+,springgaus} --scenes $SC" bash exe/stab.sh $FC )
echo "$(date) 커밋 $FC springgaus ${SGOK:-없음}" | rclone rcat $MK/s8.done
say "[s8 끝] $(cat /dev/shm/dkta/lbours/stab/stab.END 2>/dev/null)"
