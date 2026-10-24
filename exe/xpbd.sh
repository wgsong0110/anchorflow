#!/bin/bash
# XPBD 한 스텝 (사용자 10-24): lego L000 (80 프레임) · ficus_v0 (125 프레임) x {N1K1, N1K10, N4K1, N16K1, N64K1}.
#   데이터·격자·초기 상태·경계조건은 기존 출력만 최적화 실행과 같게 (lego = lbours/pfip_noin/L000/cmd.txt 의 --data one/L000·--node_h 0.028284·--v0_traj,
#   바닥 = cfg surface_collider; ficus = /home/dkta/work/ficus_v0.sh 의 data F_pg_v0·--node_h 0.056569·--cfg_bc·--pg_region_density·--pg_vdamp·--v0_traj).
#   순서: 단위 시험 (CPU) -> 실행마다 빈 GPU 하나 (메모리 < 500 MiB, 프로세스 없음, 내가 잡은 카드 제외; 없으면 기다림)에서
#         xpbd_roll -> h5 -> 3DGS 렌더 -> 탄성 색 (공통 고정 범위) -> 정리 (xpbd_post: 표·그림·tex 조각)
#   출력 lbours/xpbd (처음부터 R2 통째 동기화, 제외 없음, 5 분 루프). 멱등: 다시 띄우면 끝난 단계는 건너뛴다.
#   표식: XPBD_LAUNCHED, XPBD_FAIL, XPBD_CLUSTER_DONE
# 사용 (부트가 워크트리를 만들고 이 파일을 부른다): AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/xpbd_boot.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
C=${AF_XPBD_COMMIT:?커밋}; WT=$W/wt/xpbd_$C
OB=$W/lbours/xpbd; R=r2:storage/result/anchorflow/lbours/xpbd; mkdir -p $OB
fail(){ echo "[오류] $1 $(date)"; echo "$(date '+%m-%d %H:%M') $1" >> $OB/XPBD_FAIL; rclone copy $OB $R -q; exit 1; }
cd $WT || fail "워크트리 없음 $WT"
if [ -f $OB/driver.hb ] && [ $(( $(date +%s) - $(cat $OB/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $OB/driver.pid
( while true; do date +%s > $OB/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $OB $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
rm -f $OB/gpu_busy_*
echo "[시작] $(date) 커밋 $(git rev-parse --short HEAD) 노드 $(hostname) 잡 ${PBS_JOBID:-?} | $(df -h /home | tail -1)"
# 0) 단위 시험 (CPU)
if [ ! -f $OB/unit_$C.ok ]; then
  CUDA_VISIBLE_DEVICES= python -u exe/test_xpbd.py > $OB/unit_$C.log 2>&1; cat $OB/unit_$C.log
  grep -q "XPBD_TEST PASS" $OB/unit_$C.log || fail "단위 시험 실패 $OB/unit_$C.log"
  touch $OB/unit_$C.ok; rclone copy $OB $R -q --max-depth 1
fi
# 씬 설정
LEGO_PT=$W/one/L000/L000.pt; LEGO_GTD=$W/lbdata/test_low/L000; LEGO_OV=$W/lbours/spstick/roll_ov_L000.pt; LEGO_CFG=$W/lbdata/config.json
FIC_PT=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt; FIC_GTD=$W/lbours/ficus_v0/gt; FIC_OV=$W/lbours/ficus_v0/roll_ov.pt; FIC_TOPO=$W/topo/pg_ficus_elastic_v0
LEGO_RANGE=$W/lbours/lego_psi/range.json; FIC_RANGE="-2.9999966621398926 4.933352713584899"
for f in $LEGO_PT $LEGO_GTD/meta.json $LEGO_OV $LEGO_CFG $FIC_PT $FIC_GTD/config.json $FIC_OV $LEGO_RANGE $W/lbours/ficus_v0/psi/psi_gt.npy $W/lbours/ficus_v0/psi/psi_ov.npy; do
  [ -e $f ] || fail "입력 없음 $f"; done
python - <<PY | tee $OB/cmd_ref.txt
import json
print("[기준 lego] " + open("$W/lbours/pfip_noin/L000/cmd.txt").read().strip()[:4000])
print("[기준 ficus] " + "".join(l for l in open("$W/ficus_v0.sh") if l.startswith(("REST=", "PGF=", "COMMON=", "FLOOR="))).strip())
r = json.load(open("$W/lbours/ficus_v0/psi/range.json")); print(f"[ficus 탄성 색 범위] {r['lo']} {r['hi']} (이 실행 $FIC_RANGE)")
r = json.load(open("$LEGO_RANGE")); print(f"[lego 탄성 색 범위] {r['lo']} {r['hi']} ($LEGO_RANGE)")
PY
scene_vars(){ if [ $1 = lego_L000 ]; then PT=$LEGO_PT; GTD=$LEGO_GTD; TAG=L000; LEN=80; NH=0.028284; FLAGS=""; RCFG=$LEGO_CFG; SHAPE=lego
  else PT=$FIC_PT; GTD=$FIC_GTD; TAG=F_pg_v0; LEN=125; NH=0.056569; FLAGS="--pg_region_density --cfg_bc --pg_vdamp"; RCFG=$FIC_GTD/config.json; SHAPE=ficus; fi; }
free_gpu(){ for c in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<500{print $1}'); do
    [ -e $OB/gpu_busy_$c ] && continue
    u=$(nvidia-smi -i $c --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && continue
    echo $c; return; done; }
wait_gpu(){ local g=""; while [ -z "$g" ]; do g=$(free_gpu); [ -z "$g" ] && { echo "[대기] 빈 GPU 없음 $(date +%T)" >> $OB/wait.log; sleep 120; }; done
  touch $OB/gpu_busy_$g
  echo "[GPU 배정] $g $(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $OB/gpu.log; echo $g; }
run_one(){ # 씬 키 GPU
  local sc=$1 k=$2 g=$3; scene_vars $sc
  local N=$(sed 's/^n\([0-9]*\)k.*/\1/' <<<$k) K=$(sed 's/.*k\([0-9]*\)$/\1/' <<<$k) D=$OB/$sc/$k
  mkdir -p $D
  if ! grep -q XPBD_ROLL_DONE $D/run.log 2>/dev/null; then
    [ -f $D/run.log ] && mv $D/run.log $D/run.log.partial_$(date +%m%d_%H%M%S)
    local CMD="python -u exe/xpbd_roll.py --pt $PT --frames $LEN --node_h $NH --sub $N --iters $K $FLAGS --out $D --tag $TAG"
    echo "[실행 명령 $(date '+%F_%T') 커밋 $C GPU $g] CUDA_VISIBLE_DEVICES=$g $CMD" > $D/cmd.txt
    CUDA_VISIBLE_DEVICES=$g $CMD > $D/run.log 2>&1 < /dev/null
  fi
  grep -a "\[롤아웃\]\|\[발산\]\|Traceback\|Error" $D/run.log | tail -3
  grep -q XPBD_ROLL_DONE $D/run.log || { echo "[오류] $sc $k 롤아웃"; return 1; }
  [ -f $D/roll_xpbd.pt ] || { echo "[오류] $sc $k 덤프 없음 (첫 프레임에 발산?)"; return 1; }
  [ -f $D/h5/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $D/roll_xpbd.pt --traj $GTD --out $D/h5 > $D/h5.log 2>&1
  [ -f $D/${sc}_$k.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $D/h5 --config $RCFG --shape $SHAPE --out $D/${sc}_$k.mp4 > $D/render.log 2>&1 < /dev/null
  [ -f $D/${sc}_$k.mp4 ] || { echo "[오류] $sc $k 렌더 $(tail -2 $D/render.log)"; return 1; }
  mkdir -p $D/psi
  if [ $sc = lego_L000 ]; then
    grep -q PSI_COLOR_DONE $D/psi/psi_color.log 2>/dev/null || CUDA_VISIBLE_DEVICES=$g python -u exe/psi_color_lb.py --gt_dir $GTD --pt $PT --dump $k=$D/roll_xpbd.pt --out $D/psi \
      --eps 1e-3 --q 1 99 --cmap turbo --range $LEGO_RANGE --stats_name defstats_L000.txt --stats_frames 1 40 80 > $D/psi/psi_color.log 2>&1 < /dev/null
    grep -q PSI_COLOR_DONE $D/psi/psi_color.log || { echo "[오류] $sc $k psi $(tail -2 $D/psi/psi_color.log)"; return 1; }
    [ -f $D/psi/psi_${k}_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $D/h5 --config $RCFG --shape lego --colors $D/psi/col_$k.npy \
      --out $D/psi/psi_${k}_lego.mp4 > $D/psi/r_$k.log 2>&1 < /dev/null
  else
    grep -q PSI_COLOR_DONE $D/psi/psi_color.log 2>/dev/null || CUDA_VISIBLE_DEVICES=$g python -u exe/psi_color.py --gt_h5 $FIC_TOPO --dump $D/roll_xpbd.pt --config $GTD/config.json --out $D/psi \
      --eps 1e-3 --q 1 99 --range $FIC_RANGE --cmap turbo --stats_frames 1 40 120 > $D/psi/psi_color.log 2>&1 < /dev/null
    grep -q PSI_COLOR_DONE $D/psi/psi_color.log || { echo "[오류] $sc $k psi $(tail -2 $D/psi/psi_color.log)"; return 1; }
    [ -f $D/psi/psi_${k}_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $D/h5 --config $RCFG --shape ficus --colors $D/psi/col_ov.npy \
      --out $D/psi/psi_${k}_ficus.mp4 > $D/psi/r_$k.log 2>&1 < /dev/null
  fi
  ls $D/psi/*.mp4 > /dev/null 2>&1 || { echo "[오류] $sc $k psi 렌더"; return 1; }
  touch $D/DONE; echo "[끝] $sc $k $(date +%T)"; }
gt_psi_lego(){ # 정답·출력만 최적화 탄성 색 (lego; 같은 고정 범위) -- ov 덤프로 psi_color_lb 한 번 + 정답·ov 렌더
  local g=$1 D=$OB/lego_L000/ov_psi; scene_vars lego_L000; mkdir -p $D
  grep -q PSI_COLOR_DONE $D/psi_color.log 2>/dev/null || CUDA_VISIBLE_DEVICES=$g python -u exe/psi_color_lb.py --gt_dir $GTD --pt $PT --dump ov=$LEGO_OV --out $D \
      --eps 1e-3 --q 1 99 --cmap turbo --range $LEGO_RANGE --stats_name defstats_L000.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null
  grep -q PSI_COLOR_DONE $D/psi_color.log || { echo "[오류] ov psi $(tail -2 $D/psi_color.log)"; return 1; }
  [ -f $D/psi_gt_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $GTD --config $RCFG --shape lego --colors $D/col_gt.npy --out $D/psi_gt_lego.mp4 > $D/r_gt.log 2>&1 < /dev/null
  [ -f $D/psi_ov_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $W/lbours/spstick/h5_ov_L000 --config $RCFG --shape lego --colors $D/col_ov.npy --out $D/psi_ov_lego.mp4 > $D/r_ov.log 2>&1 < /dev/null
  [ -f $D/psi_gt_lego.mp4 ] && [ -f $D/psi_ov_lego.mp4 ] || { echo "[오류] 정답/ov psi 렌더"; return 1; }
  touch $D/DONE; }
# 1) 실행 (빈 GPU 마다 하나씩, 동시에)
KEYS="n1k1 n4k1 n1k10 n16k1 n64k1"
echo "[실행 시작] $(date '+%F %T') 커밋 $C 씬 lego_L000 ficus_v0 x $KEYS" | tee -a $OB/XPBD_LAUNCHED; rclone copy $OB $R -q --max-depth 1
PIDS=""
for sc in lego_L000 ficus_v0; do for k in $KEYS; do
  [ -f $OB/$sc/$k/DONE ] && continue
  g=$(wait_gpu); echo "[띄움] $sc $k GPU$g $(date +%T)"
  ( run_one $sc $k $g > $OB/task_${sc}_$k.log 2>&1; rm -f $OB/gpu_busy_$g ) & PIDS="$PIDS $!"
  sleep 20
done; done
if [ ! -f $OB/lego_L000/ov_psi/DONE ]; then g=$(wait_gpu); ( gt_psi_lego $g > $OB/task_lego_ov_psi.log 2>&1; rm -f $OB/gpu_busy_$g ) & PIDS="$PIDS $!"; fi
wait $PIDS
cat $OB/task_*.log | grep -a "\[오류\]\|\[끝\]\|\[롤아웃\]\|\[발산\]" | cut -c1-240
rclone copy $OB $R -q
# 2) 정리 (CPU): 씬별 json -> 합친 그림·표·tex
P=$OB/post; mkdir -p $P
runs_of(){ local sc=$1 out=""; for k in $KEYS; do D=$OB/$sc/$k; [ -f $D/roll_xpbd.pt ] || continue
    if [ $sc = lego_L000 ]; then ps=$D/psi/psi_$k.npy; else ps=$D/psi/psi_ov.npy; fi
    out="$out --run $k=$D/roll_xpbd.pt:$D/summary_xpbd.json:$D/stats_xpbd.jsonl:$ps"; done; echo $out; }
[ -f $P/lego_L000.json ] || python -u exe/xpbd_post.py --scene lego_L000 --gt_pt $LEGO_PT --ov_dump $LEGO_OV --ov_psi $OB/lego_L000/ov_psi/psi_ov.npy \
   --gt_psi $OB/lego_L000/ov_psi/psi_gt.npy $(runs_of lego_L000) --frames 80 --out_json $P/lego_L000.json > $P/lego_L000.log 2>&1
[ -f $P/ficus_v0.json ] || python -u exe/xpbd_post.py --scene ficus_v0 --gt_pt $FIC_PT --region_density --ov_dump $FIC_OV --ov_psi $W/lbours/ficus_v0/psi/psi_ov.npy \
   --gt_psi $W/lbours/ficus_v0/psi/psi_gt.npy $(runs_of ficus_v0) --frames 125 --out_json $P/ficus_v0.json > $P/ficus_v0.log 2>&1
cat $P/lego_L000.log $P/ficus_v0.log | grep -av "it/s" | tail -20
[ -f $P/lego_L000.json ] && [ -f $P/ficus_v0.json ] || fail "정리 실패 $(tail -3 $P/lego_L000.log $P/ficus_v0.log)"
grep -a "\] ov:" $P/lego_L000.log $P/ficus_v0.log | tee $OB/ov_check.txt
python -u exe/xpbd_post.py --merge $P/lego_L000.json $P/ficus_v0.json --fig $P/xpbd_cmp.png --table $P/xpbd_cmp.md --tex $P/xpbd_cmp.tex --commit $C > $P/merge.log 2>&1
grep -q XPBD_POST_DONE $P/merge.log || fail "합친 정리 실패 $(tail -3 $P/merge.log)"
cat $P/xpbd_cmp.md
kill $SYNC 2>/dev/null; rclone copy $OB $R -q
touch $OB/XPBD_CLUSTER_DONE; rclone copy $OB $R -q
rclone check $OB $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo "XPBD_CLUSTER_DONE $(date)"
