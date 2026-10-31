#!/bin/bash
# 절 AK 최종 판 덤프·렌더: 작업 목록 (AK_KIND=F fem_ak_jobs.py v:...) 을 차례로 GPU0 에서 (fem_ab_run.sh EX_TIMED=1 -- gpu0.lock flock, GPU0 이 빌 때까지 기다림,
# 시작·끝 7 장 상태 기록). F 판은 이어서 3DGS 알파블렌딩 렌더 (lb_roll_to_h5 -> lb_render, 카드 1~6 중 가장 덜 쓰는 카드), R2 크기 확인 뒤 /tmp 큰 덤프만 지움.
# 사용: bash exe/fem_ak_render.sh v:pqf3:z:1:: L000,0003 (바닥 없는 ficus 는 v:pqf3::1::)
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
W=/home/dkta/work; ST=${1:?단계}; SCS=${2:-}
echo "[단독 $ST] 커밋 $(git log -1 --oneline) $(date '+%F %T')"
while read -r ROOT TG ARGS; do
  O=/tmp/dkta/lbours/$ROOT/$TG; RR=r2:storage/result/anchorflow/lbours/$ROOT/$TG
  EX_TIMED=1 EX_ROOT=$ROOT bash exe/fem_ab_run.sh "$TG" 0 $ARGS < /dev/null > /dev/null 2>&1
  echo "$(date +%T) $TG $(grep -a '롤아웃\]\|\[발산\]' $O/v_fem.log | tail -1 | cut -c1-170) | 시작 카드 $(tr '\n' ' ' < $O/gpu_at_start.txt)"
  case $TG in *_F*) ;; *) continue;; esac
  [ -f $O/roll_fem.pt ] || { echo "[덤프 없음] $TG"; continue; }
  case $TG in
    ficus_*) GTD=$W/lbours/ficus_v0/gt; CFG=$GTD/config.json; SH=ficus;;
    L000_*) GTD=$W/lbdata/test_low/L000; CFG=$W/lbdata/config.json; SH=lego;;
    *) t=${TG%%_*}; GTD=$W/lbdata/test/$t; CFG=$W/lbdata/config.json; SH=lego;;
  esac
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1>=1 && $1<=6' | sort -t, -k2 -n | head -1 | cut -d, -f1)
  [ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
  mkdir -p $O/vj
  [ -f $O/vj/fem_$SH.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $CFG --shape $SH --out $O/vj/fem_$SH.mp4 \
     --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
  [ -f $O/vj/fem_$SH.mp4 ] && touch $O/vj/RENDER_DONE || tail -3 $O/vj/r_fem.log
  rclone copy $O $RR -q
  if rclone check $O $RR --one-way --size-only -q 2>/dev/null; then rm -rf $O/roll_fem.pt $O/h5_fem; echo "[렌더 카드 $G · R2 확인·/tmp 정리] $TG"; else echo "[R2 확인 실패 -- /tmp 유지] $TG"; fi
done < <(AK_KIND=F python exe/fem_ak_jobs.py "$ST" $SCS)
echo "[단독 끝 $ST] $(date +%T)"
