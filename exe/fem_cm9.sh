#!/bin/bash
# 절 CM9 묶음 (커밋 $1; 사용자 승인 11-05 정의 변경 --gq_mass: 노드 질량·운동량 (HHT 가속도 몫) 을 입자 -> 자기 팔분 가우스 점 -> 노드, 힘과 같은 점·가중·기저).
#  W. 수박 (지금 판 = 절 AS 전체 판, 씬 파일 exe/fem_cm9_scn.sh wm, 옛 재질 실행기, 비결정): 끔 w0a·w0b, 켬 w1a (덤프)·w1b -- 빈 카드 1-6, 진단 --gq_chk (결과 불변)
#     동시에 단독 시간 (카드 0, gpu0.lock): zw0a·zw1a·zw0b·zw1b
#  R. 수박 켬 판 둘 다 100 프레임 완주면: 회귀 (새 실행기, 채택 판 (B) = 절 CM8 --v=b, 결정적) L000·0016·drop1 (2x4)·wolf (2x5)·ficus (2x24)·bread (2x2) 끔 r0·켬 r1
#     + 수박 영상 (w1a 덤프 -> h5 -> GF 3DGS 알파블렌딩 wm_render -> 정답 | 우리 합성 -> R2 result/anchorflow/v20261009/fem_cm9_watermelon_cmp.mp4)
#     완주 못 하면 회귀·영상 안 함 (원인 수치는 표의 발산 진단 열).
# 결과 R2 lbours/fem_cm/cm9 (판)·cm9z (시간), 표 marks/cm9_table.md (5 분마다 다시), 끝 표식 marks/cm9.END. CM9_UNTIL (epoch, 잡 만료) 이 있으면 그 25 분 전 뒤 새 단계 안 띄우고 5 분 전 마감 표·표식.
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_cm9 XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
B=/dev/shm/dkta/lbours/fem_cm; M=$B/marks; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; L=/dev/shm/dkta/cm_lists/cm9; mkdir -p $M $L $CUPY_CACHE_DIR
LOG=$M/cm9.log; say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
UNTIL=${CM9_UNTIL:-0}; [ "$UNTIL" -gt 0 ] || UNTIL=$(( $(date +%s) + 72 * 3600 ))
NOSTART=$(( UNTIL - 1500 )); TBL=$(( UNTIL - 300 ))
say "[cm9 시작] $C $(date '+%F %T') $(hostname) 마감 $(date -d @$UNTIL '+%F %T')"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $LOG; df -h /tmp /dev/shm | tail -2 >> $LOG; free -g | head -2 >> $LOG
export CM_SCN=exe/fem_cm9_scn.sh
source exe/fem_cm9_scn.sh
cm_data wm; say "[자료] wm $(ls -la $WMD/wm.pt 2>&1 | awk '{print $5}') $(date '+%T')"
table() { $PY exe/fem_cm9_table.py $B/cm9 $B/cm9z > $M/cm9_table.md.tmp 2>&1; { echo "# cm9 ($C) -- 절 CM9 --gq_mass 켜고 끔 ($(date '+%F %T') 기준)"; echo; cat $M/cm9_table.md.tmp; } > $M/cm9_table.md; rclone copy $M $MR -q --include "cm9*"; }
( while [ $(date +%s) -lt $TBL ]; do sleep 300; table; done
  [ -f $M/cm9.END ] || { table; echo "[cm9 끝] $(date '+%F %T') (마감 표; 돌던 판은 표에 '도는 중')" > $M/cm9.END; rclone copy $M/cm9.END $MR -q; say "[마감 표·끝 표식] $(date '+%T')"; } ) &
PT=$!
# W 수박
: > $L/w.txt
echo "w0a wm --gq_chk" >> $L/w.txt; echo "w1a wm --gq_mass --gq_chk --dump" >> $L/w.txt
echo "w0b wm --gq_chk" >> $L/w.txt; echo "w1b wm --gq_mass --gq_chk" >> $L/w.txt
cat $L/w.txt >> $LOG
( CM_CARDS="1 2 3 4 5 6" CM_PAR=4 CM_GAP=20 CM_TMO=3600 bash exe/fem_cm_pool.sh cm9 $L/w.txt > $M/cm9w_pool.log 2>&1; say "[수박 판 끝] $(date '+%T')" ) &
PW=$!
: > $L/z.txt; for r in a b; do echo "zw0$r wm" >> $L/z.txt; echo "zw1$r wm --gq_mass" >> $L/z.txt; done
( CM_TMO=3600 bash exe/fem_cm_q.sh cm9z $L/z.txt > $M/cm9z_q.log 2>&1; say "[수박 단독 시간 끝] $(date '+%T')" ) &
PZ=$!
wait $PW
table
OK=1; for v in w1a w1b; do $PY - $B/cm9/${v}_wm/summary.json <<'PYX' || OK=0
import json, sys
s = json.load(open(sys.argv[1])); sys.exit(0 if (s.get("diverged") is None and s.get("frames_done") == s.get("frames") == 100) else 1)
PYX
done
say "[수박 켬 판 완주] $OK $(date '+%T')"
if [ $OK = 1 ] && [ $(date +%s) -lt $NOSTART ]; then
  # 영상 카드 하나 잡기 (풀과 같은 잡기 규약) -> 회귀 풀은 나머지 카드
  CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL; VC=$(( $(nvidia-smi -L | wc -l) - 1 )); [ $VC -gt 6 ] && VC=6; echo $$ > $CL/$VC   # 클러스터 8 장이면 6 그대로, vast 4 장이면 3
  if [ -n "${S6_WM_MARK:-}" ]; then for i in $(seq 240); do [ -f $S6_WM_MARK ] && break; sleep 30; done; say "[수박 자료] $(ls $S6_WM_MARK 2>&1)"; fi   # vast (절 S.6): 클러스터에서 R2 로 옮긴 수박 3DGS·정답이 내려올 때까지 (최대 2 h)
  ( O=$B/cm9/w1a_wm; W=/home/dkta/work; WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json
    GTD=/tmp/dkta/ag_data/gt_wm; [ -f $GTD/meta.json ] || GTD=/dev/shm/dkta/ag_data/gt_wm
    if [ ! -f $GTD/meta.json ]; then mkdir -p /dev/shm/dkta/ag_data
      CUDA_VISIBLE_DEVICES=$VC $PY -u exe/pg_h5_to_pt.py --h5_dir $W/wm2/gf/simulation_ply --config $CFG --out_pt /dev/shm/dkta/ag_data/wm_rend_tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1
      rm -f /dev/shm/dkta/ag_data/wm_rend_tmp.pt; fi
    mkdir -p $O/vj
    $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
    (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$VC $PY -u $D/exe/wm_render.py --model_path $WMM --config $CFG --sim $O/h5_fem --out $O/vj/fem_watermelon.mp4 < /dev/null > $O/vj/r_fem.log 2>&1)
    RR=r2:storage/result/anchorflow/lbours/fem_cm/cm9/w1a_wm
    rclone copy $O/vj $RR/vj -q; rclone copy $O/h5_fem $RR/h5_fem -q; rclone copy $O/pred.pt $RR -q
    if [ -f $O/vj/fem_watermelon.mp4 ]; then
      FF=$(command -v ffmpeg || $PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)
      T=/dev/shm/dkta/cm9cmp; mkdir -p $T; rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_wm/gt/gt_watermelon.mp4 $T/gt.mp4 -q
      rclone copyto r2:storage/result/anchorflow/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $T/font.otf -q
      E=$($PY -c "import json; s=json.load(open('$O/summary.json')); print('%.2f' % (100*s['err_mean']))")
      H=$($PY -c "import imageio.v3 as iio; print(iio.improps('$T/gt.mp4', plugin='pyav').shape[1])" 2>/dev/null || echo 720)
      $FF -y -loglevel error -i $T/gt.mp4 -i $O/vj/fem_watermelon.mp4 -filter_complex \
        "[0:v]scale=-2:$H,drawtext=fontfile=$T/font.otf:text='정답':x=20:y=20:fontsize=36:fontcolor=black[a];[1:v]scale=-2:$H,drawtext=fontfile=$T/font.otf:text='우리 (절 CM9 가우스 점 질량, 오차 $E %%)':x=20:y=20:fontsize=36:fontcolor=black[b];[a][b]hstack=inputs=2:shortest=1" \
        -c:v libx264 -pix_fmt yuv420p -crf 20 $T/out.mp4 > $T/ff.log 2>&1
      [ -s $T/out.mp4 ] && rclone copyto $T/out.mp4 r2:storage/result/anchorflow/v20261009/fem_cm9_watermelon_cmp.mp4 -q && say "[영상] result/anchorflow/v20261009/fem_cm9_watermelon_cmp.mp4" || say "[영상 합성 실패] $(tail -3 $T/ff.log | tr '\n' ' ') -- 렌더는 R2 lbours/fem_cm/cm9/w1a_wm/vj/fem_watermelon.mp4"
    else say "[렌더 실패] $(tail -5 $O/vj/r_fem.log | tr '\n' ' ') $(tail -2 $O/h5_fem.log | tr '\n' ' ')"; fi
    rm -f $CL/$VC ) &
  PV=$!
  G="--ga hht --rho 0.8 --K 2 --newton 2"; T_="--tolN 1e-2 --tolP 1e-1"
  : > $L/r.txt
  for v in r0 r1; do X=""; [ $v = r1 ] && X="--gq_mass --gq_chk"
    for t in L000 0016 drop1; do echo "$v $t $G --cg 4 $T_ --det --v=b $X" >> $L/r.txt; done
    echo "$v wolf $G --cg 5 $T_ --det --hgt --v=b $X" >> $L/r.txt
    echo "$v ficus $G --cg 24 $T_ --det $X" >> $L/r.txt
    echo "$v bread $G --cg 2 $T_ --det $X" >> $L/r.txt; done
  cat $L/r.txt >> $LOG
  CM_CARDS="1 2 3 4 5" CM_PAR=5 CM_GAP=5 bash exe/fem_cm_pool.sh cm9 $L/r.txt > $M/cm9r_pool.log 2>&1
  say "[회귀 끝] $(date '+%T')"
  wait $PV
else
  say "[회귀·영상 안 함] 수박 켬 판 완주 $OK (또는 마감 가까움)"
fi
wait $PZ
table
[ $(date +%s) -lt $TBL ] && { echo "[cm9 끝] $(date '+%F %T') (모두 완주)" > $M/cm9.END; rclone copy $M/cm9.END $MR -q; say "[cm9 끝] $(date '+%F %T')"; }
kill $PT 2>/dev/null
rclone copy $B/cm9 r2:storage/result/anchorflow/lbours/fem_cm/cm9 -q --exclude pred.pt
