#!/bin/bash
# 절 CM8 (A) 판 lego 영상 (조정자 11-05 요청): (A) 바닥 = 가우스 점 고착 (exe/fem_cm8_scn.sh --v=a), 나머지 절 CM8 판정 판 a6 와 같은 명령 (K 2 x 뉴턴 2 x PCG 6, εN 1e-2 · εP 1e-1, 결정적)
#   + 덤프 (--dump --dump_F --dump_part: a6 판정에서 L000·0016 은 용량 늘려 다시 중 GPU 메모리 부족으로 멈췄다 -> 그때까지 프레임을 덤프, 풀이 불변)
#   -> h5 (exe/lb_roll_to_h5.py) -> 3DGS 알파블렌딩 (exe/lb_render.py, 절 CM4 V 와 같은 인자) -> 정답 | 우리 합성 (ffmpeg; 우리 칸은 멈춘 프레임에서 정지).
# CM8_RENDER_ONLY=1: 덤프 단계 건너뛰고 R2 의 h5 로 렌더·합성만 (11-05 인스턴스에 matplotlib 이 없어 렌더가 실패했던 것 다시). 커밋 $1, 잡/인스턴스 안. 출력 R2 lbours/fem_cm/cm8av/v_<씬> (덤프·h5·영상 포함), 영상 result/anchorflow/v20261009/fem_cm8a_lego_{L000,0016}_cmp.mp4, 표식 lbours/fem_cm/marks/cm8av.END.
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python; W=/home/dkta/work
B=/dev/shm/dkta/lbours/fem_cm; R=$B/cm8av; RR=r2:storage/result/anchorflow/lbours/fem_cm/cm8av; M=$B/marks; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks
L=/dev/shm/dkta/cm_lists/cm8av; CL=/dev/shm/dkta/anerr_claim; mkdir -p $R $M $L $CL
LOG=$M/cm8av.log; say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
say "[cm8av 시작] $C $(date '+%F %T') $(hostname)"
export CM_SCN=exe/fem_cm8_scn.sh
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
: > $L/j.txt; for t in L000 0016; do echo "v $t $G --cg 6 $T --det --v=a --dump --dump_F --dump_part" >> $L/j.txt; done
cat $L/j.txt >> $LOG
[ -n "$CM8_RENDER_ONLY" ] || CM_CARDS="1 2 3 4 5 6" CM_PAR=2 CM_GAP=5 CM_TMO=3600 bash exe/fem_cm_pool.sh cm8av $L/j.txt > $M/cm8av_pool.log 2>&1
if [ -n "$CM8_RENDER_ONLY" ]; then for t in L000 0016; do rclone copy $RR/v_$t $R/v_$t -q --exclude "vj/**" --exclude pred.pt; done; fi   # 렌더만 다시: h5·summary 를 R2 에서 (pred.pt 는 h5 가 있으면 불필요)
say "[덤프 끝] $(date '+%T') $(cat $R/done.txt 2>/dev/null | tr '\n' ' ')"
free_card() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || return
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r i u; do
    [ "$i" = 0 ] && continue
    echo "$apps" | grep -q "$u" && continue
    if [ -f $CL/$i ] && kill -0 "$(cat $CL/$i)" 2>/dev/null; then continue; fi
    echo $i; break
  done
}
gtd() { case $1 in L000) echo "$W/lbdata/test_low/L000";; *) echo "$W/lbdata/test/$1";; esac; }
gtv() { case $1 in L000) echo lbours/fem2_lego/gt_lego.mp4;; 0016) echo lbours/fem_af_lego/gt_0016/gt_lego.mp4;; esac; }
FF=$(command -v ffmpeg || $PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
TF=/dev/shm/dkta/cm8avcmp; mkdir -p $TF; rclone copyto r2:storage/result/anchorflow/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $TF/font.otf -q
rend() { local Gc=$1 t=$2; local O=$R/v_$t
  [ -f $O/pred.pt ] || [ -f $O/h5/meta.json ] || { say "[덤프 없음] v_$t: $(tail -2 $O/log.txt | tr '\n' ' ' | cut -c1-300)"; return; }
  local GTD; GTD=$(gtd $t)
  [ -f $O/h5/meta.json ] || $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5 > $O/h5.log 2>&1
  mkdir -p $O/vj
  CUDA_VISIBLE_DEVICES=$Gc $PY -u exe/lb_render.py --sim $O/h5 --config $W/lbdata/config.json --shape lego --out $O/vj/fem_lego.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r.log 2>&1
  rclone copy $O $RR/v_$t -q
  [ -f $O/vj/fem_lego.mp4 ] || { say "[렌더 실패] v_$t $(tail -3 $O/vj/r.log | tr '\n' ' ')"; return; }
  rclone copyto r2:storage/result/anchorflow/$(gtv $t) $TF/gt_$t.mp4 -q
  local INF; INF=$($PY - $O/summary.json <<'PYX'
import json, sys
s = json.load(open(sys.argv[1])); d = s.get("diverged"); e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
oom = " GPU 메모리 부족" if s.get("oom") else ""
print(f"오차 {100 * e:.2f} %%" + (f" · 프레임 {d} 에서 멈춤{oom} (그 뒤 정지 화면)" if d else " · 완주"))
PYX
)
  local H; H=$($PY -c "import imageio.v3 as iio; print(iio.improps('$TF/gt_$t.mp4', plugin='pyav').shape[1])" 2>/dev/null || echo 800)
  $FF -y -loglevel error -i $TF/gt_$t.mp4 -i $O/vj/fem_lego.mp4 -filter_complex \
    "[0:v]scale=-2:$H,drawtext=fontfile=$TF/font.otf:text='정답 (MPM) lego $t':x=20:y=20:fontsize=30:fontcolor=black[a];[1:v]scale=-2:$H,tpad=stop_mode=clone:stop=-1,drawtext=fontfile=$TF/font.otf:text='우리 (A) 가우스 점 고착 바닥 · K 2 x 뉴턴 2 x PCG 6':x=20:y=20:fontsize=30:fontcolor=black,drawtext=fontfile=$TF/font.otf:text='$INF':x=20:y=60:fontsize=26:fontcolor=black[b];[a][b]hstack=inputs=2:shortest=1" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 $TF/out_$t.mp4 > $TF/ff_$t.log 2>&1
  if [ -s $TF/out_$t.mp4 ]; then rclone copyto $TF/out_$t.mp4 r2:storage/result/anchorflow/v20261009/fem_cm8a_lego_${t}_cmp.mp4 -q && rclone copy $TF/out_$t.mp4 $RR/v_$t/vj -q
    say "[영상] result/anchorflow/v20261009/fem_cm8a_lego_${t}_cmp.mp4 ($INF)"
  else say "[합성 실패] $t $(tail -3 $TF/ff_$t.log | tr '\n' ' ')"; fi; }
for t in L000 0016; do
  while true; do Gc=$(free_card); [ -n "$Gc" ] && break; sleep 20; done
  rend $Gc $t & echo $! > $CL/$Gc; sleep 10
done
wait
rclone copy $R $RR -q
echo "[cm8av 끝] $(date '+%F %T')" > $M/cm8av.END; rclone copy $M/cm8av.END $MR -q; say "[cm8av 끝] $(date '+%T')"
