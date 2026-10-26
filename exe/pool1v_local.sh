#!/bin/bash
# pool1 변형 (사용자 10-26) 로컬 감시·재기동·후처리: bash exe/pool1v_local.sh <s32|s32g3>   로그 /tmp/pool1<변형>_gunet_local.log (tmux pool1<변형>)
#   s32   = pool1_gunet + --pool_rdrop 0 --pool_size 32                      -> lbours/pool1s32_gunet,   완료 POOL1S32_DONE
#   s32g3 = s32 + --pool_gate 1e-3                                           -> lbours/pool1s32g3_gunet, 완료 POOL1S32G3_DONE
#   10 분마다 하트비트·[풀 상태]·[pgrad] 로그 확인, 죽었으면 같은 부트 (멱등), 새 탈락 영상 합성 (pool1_drop/<실행>/), 렌더 실패분 재렌더 잡.
#   끝나면: <실행>_{L000,0004}.mp4 (정답 | pool1_gunet | 이 변형), lego_psi_<실행>_*.mp4 (공통 고정 범위), 학습 곡선, 풀 관찰 그림, 표 -> outputs.md·tex·notify.
VAR=${1:?s32 · s32g3 · s32r · s32g3r}
#   s32r·s32g3r (10-26) = s32·s32g3 에서 --pool_init_traj 빼고 예전 풀 낙하 분포 시작 -> 탈락 영상은 정답 칸 없이 풀 상태만 (시작이 L000 이 아니다)
case $VAR in s32) LABV="무작위 폐기 끔·풀 32"; MARK=POOL1S32_DONE ;; s32g3) LABV="무작위 폐기 끔·풀 32·전진 문턱 1e-3"; MARK=POOL1S32G3_DONE ;;
  s32r) LABV="무작위 폐기 끔·풀 32·낙하 분포 시작"; MARK=POOL1S32R_DONE ;; s32g3r) LABV="무작위 폐기 끔·풀 32·전진 문턱 1e-3·낙하 분포 시작"; MARK=POOL1S32G3R_DONE ;;
  *) echo "[오류] $VAR"; exit 1 ;; esac
RUN=pool1${VAR}_gunet
exec 9>/tmp/${RUN}_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
C=372d53f
RL=r2:storage/result/anchorflow/lbours; RV=$RL/$RUN; RNI=$RL/noinertial; R5=$RL/gunet5_near
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; T=/tmp/${RUN}_dl; mkdir -p $T $D
TEX=/home/wgsong/workspace/writing/anchorflow/learned_results.tex
REPO=/home/wgsong/workspace/execute/anchorflow/master
LOGIN=dkta@147.46.121.39
F=$(fc-list :lang=ko file | head -1 | cut -d: -f1)
get(){ rclone copyto $1 $2 -q 2>/dev/null; }
lab(){ printf '%s' "$1" > $T/$2.txt; echo "drawtext=fontfile=$F:textfile=$T/$2.txt:expansion=none:x=10:y=8:fontsize=20:fontcolor=black"; }
cell(){ ffmpeg -loglevel error -y -i $1 -vf "pad=iw:ih+40:0:40:white,$(lab "$3" $2)" -c:v libx264 -pix_fmt yuv420p $T/c_$2.mp4; }
stack3(){ OUT=$1; TTL=$2; ffmpeg -loglevel error -y -i $T/c_a.mp4 -i $T/c_b.mp4 -i $T/c_c.mp4 -filter_complex "hstack=inputs=3:shortest=1" -c:v libx264 -pix_fmt yuv420p $T/stack.mp4 \
  && ffmpeg -loglevel error -y -i $T/stack.mp4 -vf "pad=iw:ih+44:0:44:white,$(lab "$TTL" title),drawtext=fontfile=$F:text='프레임 %{frame_num}':x=w-tw-14:y=h-th-12:fontsize=44:fontcolor=red" -c:v libx264 -pix_fmt yuv420p $OUT \
  && ls -la $OUT; }
texadd(){ python3 - "$TEX" "$1" <<'PY'
import sys
p, txt = sys.argv[1:]
s = open(p).read()
s = s.replace(r"\end{document}", txt + "\n" + r"\end{document}")
open(p, "w").write(s)
PY
}
q(){ echo "$1" | sed 's/%/\\%/g'; }
fail(){ echo "[실패] $1 $(date)"; notify "$RUN 실패" "$1" --priority high --tags x; }
# psi 3칸 + 색막대 + 분포 패널 (gunet_local.sh psi3 와 같은 식): psi3 <디렉토리> <라벨2> <라벨3> <출력> <제목 접두>
psi3(){ P=$1; OUT=$4; X=$5
  printf '정답 (MPM)' > $T/l_gt.txt; printf '%s' "$2" > $T/l_2.txt; printf '%s' "$3" > $T/l_3.txt
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $P/psi_gt_lego.mp4)
  NS=""; for n in gt p1 q1; do NS="$NS $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $P/psi_${n}_lego.mp4)"; done
  NG=$(echo $NS | cut -d' ' -f1); [ $(echo $NS | tr ' ' '\n' | sort -u | wc -l) = 1 ] || { fail "psi 프레임 수 다름 ($NS) $P"; return 1; }
  TW=$((W*3)); [ -e $P/frames ] && mv $P/frames $P/frames_prev_$(date +%H%M%S)
  ( cd $REPO && python3 exe/psi_dist_vid.py --hist $P/psi_hist_p1.npz --hist2 $P/psi_hist_q1.npz --range $P/range.json --out_dir $P/frames --cbar $P/cbar.png \
    --w $TW --h 560 --lab_gt "정답 (MPM)" --lab_ov "${2%% ·*}" --lab2 "${3%% ·*}" ) || { fail "분포 패널 $P"; return 1; }
  DT="fontfile=$F:fontsize=26:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6:expansion=none"
  ffmpeg -y -v error -r 15 -i $P/psi_gt_lego.mp4 -r 15 -i $P/psi_p1_lego.mp4 -r 15 -i $P/psi_q1_lego.mp4 -loop 1 -r 15 -i $P/cbar.png -framerate 15 -i $P/frames/%04d.png \
   -filter_complex "[0:v]setpts=N/15/TB,drawtext=textfile=$T/l_gt.txt:x=20:y=20:$DT[a];[1:v]setpts=N/15/TB,drawtext=textfile=$T/l_2.txt:x=20:y=20:$DT[b];\
[2:v]setpts=N/15/TB,drawtext=textfile=$T/l_3.txt:x=20:y=20:$DT[c];[a][b][c]hstack=inputs=3[top];[3:v]scale=$TW:-2,setsar=1[cb];[4:v]scale=$TW:-2,setsar=1[d];[top][cb][d]vstack=inputs=3:shortest=1,\
drawtext=fontfile=$F:text='$X 프레임 %{frame_num}':x=w-tw-20:y=h-th-20:fontsize=44:fontcolor=red:box=1:boxcolor=white@0.8:boxborderw=6,format=yuv420p" \
   -frames:v $NG -c:v libx264 -crf 18 -r 15 $OUT || { fail "psi ffmpeg $OUT"; return 1; }
  ffprobe -v error -count_frames -show_entries stream=nb_read_frames,width,height -of csv=p=0 $OUT; }


DROPD=$D/pool1_drop
dropvids(){ for m in gunet; do
    RTV=$RV/pool/pool_trace/vid; X=$T/drop; OD=$DROPD/$RUN; mkdir -p $X $OD
    [ -f $T/gt_L000.mp4 ] || get $RV/gt_L000.mp4 $T/gt_L000.mp4
    [ -f $T/gt_L000.mp4 ] || return 0
    for f in $(rclone lsf $RTV --include "*.mp4" 2>/dev/null); do n=${f%.mp4}; OUT=$OD/$n.mp4
      [ -e $OUT ] && continue
      get $RTV/$n.mp4 $X/$n.mp4; get $RTV/$n.json $X/$n.json
      [ -f $X/$n.mp4 ] && [ -f $X/$n.json ] || continue
      read A NF LAB <<<"$(python3 -c "
import json; m=json.load(open('$X/$n.json')); fr=m['frames']; tf=m['trace_from']
a = fr[0] if (tf or m.get('has_start')) else 0     # 영상 첫 프레임의 프레임 번호
nf = len(fr) + (0 if (tf or m.get('has_start')) else 1)
k = '최대 나이 갱신' if m.get('kind') == 'best' else ('탈락 ' + m['reason'] + ' (' + m.get('select', '') + ' 선택)')
print(a, nf, f\"$RUN · uid {m['uid']} · {k} · 반복 {m['it']} · 나이 {m['age']} 프레임 · 정체 {m['stuck']} · 태어난 반복 {m['it_born']}\" + (f' · 이력은 프레임 {fr[0]} 부터 (이어 돌리기 전 상태)' if tf else ''))")"
      [ -n "$NF" ] || continue
      printf '정답 L000 (MPM)' > $T/dl_g.txt; printf '풀 상태 (모델로 전진)' > $T/dl_p.txt; printf '%s' "$LAB" > $T/dl_t.txt
      DT="fontfile=$F:fontsize=22:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=5:expansion=none"
      if [ "${VAR: -1}" = r ]; then   # 낙하 분포 시작: 정답 칸 없음
      ffmpeg -loglevel error -y -i $X/$n.mp4 -filter_complex \
"[0:v]trim=end_frame=$NF,setpts=PTS-STARTPTS,drawtext=textfile=$T/dl_p.txt:x=14:y=14:$DT,pad=iw:ih+44:0:44:white,\
drawtext=textfile=$T/dl_t.txt:x=10:y=10:fontfile=$F:fontsize=22:fontcolor=black:expansion=none,\
drawtext=fontfile=$F:text='프레임 %{eif\:n+$A\:d}':x=w-tw-16:y=h-th-14:fontsize=48:fontcolor=red,format=yuv420p" \
        -c:v libx264 -crf 20 $OUT && echo "[탈락 영상] $OUT ($NF 프레임)" || { echo "[탈락 영상 실패] $n"; [ -e $OUT ] && mv $OUT $OUT.fail_$(date +%H%M%S); continue; }
      else
      ffmpeg -loglevel error -y -i $T/gt_L000.mp4 -i $X/$n.mp4 -filter_complex \
"[0:v]trim=start_frame=$A,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop=$NF,trim=end_frame=$NF,setpts=PTS-STARTPTS,drawtext=textfile=$T/dl_g.txt:x=14:y=14:$DT[g];\
[1:v]trim=end_frame=$NF,setpts=PTS-STARTPTS,drawtext=textfile=$T/dl_p.txt:x=14:y=14:$DT[q];[g][q]hstack=inputs=2,pad=iw:ih+44:0:44:white,\
drawtext=textfile=$T/dl_t.txt:x=10:y=10:fontfile=$F:fontsize=22:fontcolor=black:expansion=none,\
drawtext=fontfile=$F:text='프레임 %{eif\:n+$A\:d}':x=w-tw-16:y=h-th-14:fontsize=48:fontcolor=red,format=yuv420p" \
        -c:v libx264 -crf 20 $OUT && echo "[탈락 영상] $OUT ($NF 프레임)" || { echo "[탈락 영상 실패] $n"; [ -e $OUT ] && mv $OUT $OUT.fail_$(date +%H%M%S); continue; }
      fi
      rclone copy $OUT r2:storage/result/anchorflow/v20261009/pool1_drop/$RUN -q
      if [ ! -f $OD/.outputs_noted ]; then touch $OD/.outputs_noted
        cat >> $RR/outputs.md <<EOM

## $(date +%F) pool1_drop/$RUN: 풀 상태 탈락·최고 기록 이력 영상 ($LABV)
- Path: $OD/<이름>.mp4 (정답 L000 같은 프레임 | 풀 상태, 3DGS, 프레임 번호 빨간색, 사유·반복·나이 라벨); R2 v20261009/pool1_drop/$RUN/
- 원본: $RV/pool/pool_trace/ (drop_*·best_*.pt 위치·F float16 이력, drops.jsonl, saved.jsonl, vid/); 규칙은 pool1_drop 과 같음 (500 반복마다 나이 상위 2 + 무작위 1, 최대 나이 갱신)
EOM
      fi
    done
  done; }

post(){ echo "[후처리] $(date)"
  XD=$T/run; mkdir -p $XD
  rclone copy $RV $XD --max-depth 1 --include "*.txt" --include "*.json" --include "*.jsonl" --include "*.mp4" --include "POOL1_*" -q || { fail "내려받기"; return 1; }
  rclone copy $RV/psi $XD/psi --include "*/psi_*_lego.mp4" --include "*/*.npz" --include "*/*.json" --include "*/*.txt" --include "*/*.log" -q || { fail "psi 내려받기"; return 1; }
  python3 -c "import json;d=json.load(open('$XD/curves.json'));json.dump({'$RUN':list(d.values())[0]},open('$XD/curves_named.json','w'),ensure_ascii=False)" || { fail "곡선 이름"; return 1; }
  # 표·비교용: pool1 두 실행, sf1 두 실행, 다섯 궤적 두 실행 (pool1 후처리와 같은 배치), 그리고 다른 변형(있으면)
  for m in gnn gunet; do Y=$T/$m; mkdir -p $Y
    rclone copy $RL/pool1_$m $Y --max-depth 1 --include "errors.txt" --include "curves.json" --include "inv_*.jsonl" --include "pool1_${m}_*.mp4" -q
    rclone copy $RL/pool1_$m/psi $Y/psi --include "*/psi_*_lego.mp4" --include "*/*.npz" --include "*/*.json" --include "*/defstats_*.txt" -q
    Y=$T/sf1_$m; mkdir -p $Y
    rclone copy $RL/sf1_$m $Y --max-depth 1 --include "errors.txt" --include "curves.json" --include "inv_*.jsonl" -q
    for t in L000 0004; do get $RL/sf1_$m/psi/$t/defstats_$t.txt $Y/defstats_$t.txt; done
  done
  N=$T/ni; G5=$T/g5; mkdir -p $N $G5
  for t in L000 0004; do
    get $RNI/psi/$t/defstats_$t.txt $N/defstats_$t.txt; get $R5/psi/$t/defstats_$t.txt $G5/defstats_$t.txt
    get $RNI/inv_noinertial_multi_$t.jsonl $N/inv_$t.jsonl; get $R5/inv_gunet5_multi_$t.jsonl $G5/inv_$t.jsonl
  done
  get $RNI/errors_multi.txt $N/errors.txt; get $R5/errors_multi.txt $G5/errors.txt
  ER(){ grep -a "^$2 " $1 | grep -o '[0-9.]*%'; }
  EV_L=$(ER $XD/errors.txt pool1_gunet_L000); EV_4=$(ER $XD/errors.txt pool1_gunet_0004)      # 이 변형 (pool1.sh 의 이름표는 pool1_gunet_*)
  EQ_L=$(ER $T/gunet/errors.txt pool1_gunet_L000); EQ_4=$(ER $T/gunet/errors.txt pool1_gunet_0004)
  EP_L=$(ER $T/gnn/errors.txt pool1_gnn_L000); EP_4=$(ER $T/gnn/errors.txt pool1_gnn_0004)
  EG_L=$(ER $T/sf1_gnn/errors.txt sf1_gnn_L000); EG_4=$(ER $T/sf1_gnn/errors.txt sf1_gnn_0004)
  EU_L=$(ER $T/sf1_gunet/errors.txt sf1_gunet_L000); EU_4=$(ER $T/sf1_gunet/errors.txt sf1_gunet_0004)
  EN_L=$(ER $N/errors.txt noinertial_multi_L000); EN_4=$(ER $N/errors.txt noinertial_multi_0004)
  E5_L=$(ER $G5/errors.txt gunet5_multi_L000); E5_4=$(ER $G5/errors.txt gunet5_multi_0004)
  [ -n "$EV_L" ] && [ -n "$EV_4" ] || { fail "$RUN 오차 값 없음"; return 1; }
  echo "[오차] $RUN $EV_L / $EV_4 | pool1_gunet $EQ_L / $EQ_4"
  for x in L000 0004; do OUT=$D/${RUN}_$x.mp4; [ -e $OUT ] && { echo "[건너뜀] $OUT"; continue; }
    [ $x = L000 ] && { A=$EQ_L; B_=$EV_L; TT="L000 (학습 궤적)"; } || { A=$EQ_4; B_=$EV_4; TT="0004 (학습 안 쓴 궤적)"; }
    cell $XD/gt_$x.mp4 a "정답"; cell $T/gunet/pool1_gunet_$x.mp4 b "Graph U-Net 한 궤적 풀 (풀 128·무작위 폐기 0.01·문턱 1e-4) · $A"; cell $XD/pool1_gunet_$x.mp4 c "$LABV · $B_"
    stack3 $OUT "$TT 롤아웃 80프레임 · Graph U-Net 한 궤적 풀 · 관성 예측 없음 · % = 위치 오차" || { fail "3칸 영상 $x"; return 1; }
  done
  for x in L000 0004; do P=$T/psi_$x; mkdir -p $P; OUT=$D/lego_psi_${RUN}_$x.mp4; [ -e $OUT ] && continue
    cp $XD/psi/$x/psi_gt_lego.mp4 $XD/psi/$x/range.json $P/
    cp $T/gunet/psi/$x/psi_q1_lego.mp4 $P/psi_p1_lego.mp4; cp $T/gunet/psi/$x/psi_hist_q1.npz $P/psi_hist_p1.npz      # 칸 2 = pool1_gunet
    cp $XD/psi/$x/psi_q1_lego.mp4 $P/psi_q1_lego.mp4; cp $XD/psi/$x/psi_hist_q1.npz $P/psi_hist_q1.npz                  # 칸 3 = 이 변형
    python3 -c "
import json
r=[json.load(open(p)) for p in ('$XD/psi/$x/range.json','$T/gunet/psi/$x/range.json','$XD/psi/$x/range_noinertial.json')]
assert all(abs(a['lo']-r[0]['lo'])<1e-9 and abs(a['hi']-r[0]['hi'])<1e-9 for a in r); print('[범위 동일]', r[0]['lo'], r[0]['hi'])" || { fail "psi 범위 다름 $x"; return 1; }
    [ $x = L000 ] && { A=$EQ_L; B_=$EV_L; } || { A=$EQ_4; B_=$EV_4; }
    psi3 $P "pool1 Graph U-Net · 위치 오차 $A" "$LABV · 위치 오차 $B_" $OUT $x || return 1
  done
  OTH=""; case $VAR in s32g3) OTH=s32 ;; s32g3r) OTH=s32r ;; esac
  CJ="$T/gunet/curves.json $XD/curves_named.json"
  if [ -n "$OTH" ] && rclone lsf $RL/pool1${OTH}_gunet/curves.json >/dev/null 2>&1; then
    get $RL/pool1${OTH}_gunet/curves.json $T/oth_curves.json && python3 -c "import json;d=json.load(open('$T/oth_curves.json'));json.dump({'pool1${OTH}_gunet':list(d.values())[0]},open('$T/oth_named.json','w'),ensure_ascii=False)" && CJ="$CJ $T/oth_named.json"
  fi
  CV=$D/${RUN}_curves.png; [ -e $CV ] || python3 $REPO/exe/pool1_plot.py $T/gnn/curves.json $CJ $T/sf1_gnn/curves.json $T/sf1_gunet/curves.json $CV || { fail "곡선"; return 1; }
  PV=$D/${RUN}_pool.png; [ -e $PV ] || python3 $REPO/exe/pool1v_poolplot.py $PV $CJ || { fail "풀 관찰 그림"; return 1; }
  # 표: pool1 표 여섯 행 + (s32g3 이면 s32 행, 있으면) + 이 변형
  XR="[[\"Graph U-Net · 한 궤적 풀, $LABV ($RUN)\", \"q1\", \"$XD/psi/{t}/defstats_{t}.txt\", \"$XD/inv_pool1_gunet_{t}.jsonl\", \"$EV_L\", \"$EV_4\"]]"
  if [ -n "$OTH" ]; then Z=$T/oth; mkdir -p $Z
    rclone copy $RL/pool1${OTH}_gunet $Z --max-depth 1 --include "errors.txt" --include "inv_*.jsonl" -q
    for t in L000 0004; do get $RL/pool1${OTH}_gunet/psi/$t/defstats_$t.txt $Z/defstats_$t.txt; done
    EO_L=$(ER $Z/errors.txt pool1_gunet_L000); EO_4=$(ER $Z/errors.txt pool1_gunet_0004)
    [ -n "$EO_L" ] && XR="[[\"Graph U-Net · 한 궤적 풀, 비교 변형 (pool1${OTH}_gunet)\", \"q1\", \"$Z/defstats_{t}.txt\", \"$Z/inv_pool1_gunet_{t}.jsonl\", \"$EO_L\", \"$EO_4\"], ${XR:1:-1}]"
  fi
  TB=$D/${RUN}_table.txt
  [ -e $TB ] || POOL1_EXTRA_ROWS="$XR" python3 $REPO/exe/pool1_table.py $T $TB "$EN_L" "$EN_4" "$E5_L" "$E5_4" "$EG_L" "$EG_4" "$EU_L" "$EU_4" "$EP_L" "$EP_4" "$EQ_L" "$EQ_4" || { fail "표"; return 1; }
  cat $TB
  for f in $D/${RUN}_L000.mp4 $D/${RUN}_0004.mp4 $D/lego_psi_${RUN}_L000.mp4 $D/lego_psi_${RUN}_0004.mp4 $CV $PV $TB; do
    rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $RV/final -q; done
  rclone copy $REPO/exe/pool1v_local.sh $RV/final -q; rclone copy $REPO/exe/pool1v_poolplot.py $RV/final -q
  VAL=$(python3 -c "
import json; s=sorted(json.load(open('$XD/curves_named.json'))['$RUN']['L000롤아웃_위치오차%']); b=min(s,key=lambda p:p[1]); print(f'마지막 {s[-1][1]:.2f}% 최저 {b[1]:.2f}% (반복 {b[0]})')")
  POOLS=$(grep -a "\[풀 상태 1[45][0-9]*\]\|\[pgrad 1[45][0-9]*\]" $XD/pool_tail.txt 2>/dev/null | tail -2 | tr '\n' ' ')
  cat >> $RR/outputs.md <<EOM

## $(date +%F) $RUN: lego 한 궤적 풀 학습 Graph U-Net, $LABV
- Path: $D/${RUN}_L000.mp4, $D/${RUN}_0004.mp4 (정답 | pool1_gunet | $RUN, 3DGS), $D/lego_psi_${RUN}_{L000,0004}.mp4 (탄성 에너지 색, 공통 고정 범위), 학습 곡선 $CV, 풀 관찰 $PV, 표 $TB, 탈락 영상 $D/pool1_drop/$RUN/
- R2 원본: $RV/ (pool/MAIN_last.pt, pool/pool_trace/, hist/, hist_eval.jsonl, tb/, pool.log, roll_pool1_gunet_*.pt, psi/, variant.txt, cmdline.txt, final/)
- Commit: $C (POOL1_VARIANT=$VAR exe/pool1.sh gunet; 부트 /home/dkta/work/${RUN}_run_boot.sh)
- Config: pool1_gunet 명령에서 $(cat $XD/variant.txt 2>/dev/null | sed 's/^\[[^]]*\] //' | tr '\n' ' ')
- 위치 오차: L000 $EV_L, 0004 $EV_4 (pool1_gunet $EQ_L / $EQ_4); L000 롤아웃 곡선 $VAL
EOM
  texadd "
\\subsection*{한 궤적 풀 학습 Graph U-Net · $(q "$LABV") (2026-10-26, $C)}
pool1\\_gunet 과 같은 설정에서 $(q "$LABV") 로 바꿨다. 위치 오차 L000 $(q $EV_L), 0004 $(q $EV_4) (pool1\\_gunet $(q $EQ_L), $(q $EQ_4)). L000 롤아웃 학습 곡선: $(q "$VAL").
{\\scriptsize\\begin{verbatim}
$(grep -v '^#' $TB)
\\end{verbatim}}
"
  touch $D/$MARK; echo $MARK
  notify "$RUN 끝" "L000 $EV_L / 0004 $EV_4 (pool1_gunet $EQ_L / $EQ_4)" --tags white_check_mark; }

relaunch(){ local LG=/home/dkta/work/${RUN}_run_$(date +%m%d_%H%M).log
  echo "[재기동] $(date '+%F %T') -> $LG"
  ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && read J N < /home/dkta/work/pool1_job.txt; AF_JOB=\$J AF_NODE=\$N AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/${RUN}_run_boot.sh $LG" 2>&1 | tail -3
  date +%s > /tmp/${RUN}_last_launch; }
echo "[시작] $(date) 커밋 $C $RUN"
[ -f /tmp/${RUN}_last_launch ] || date +%s > /tmp/${RUN}_last_launch
while true; do
  [ -f $D/$MARK ] && { echo "$MARK (이미) $(date)"; break; }
  ST=$(ssh -o ConnectTimeout=20 $LOGIN "W=/home/dkta/work; now=\$(date +%s); U=\$W/lbours/$RUN
    read J N < \$W/pool1_job.txt; timeout 30 ssh -o ConnectTimeout=15 \$N 'pgrep -u dkta -f \"Singularity runtime parent\" >/dev/null' || echo NOJOB
    [ -f \$U/POOL1_GUNET_CLUSTER_DONE ] && echo DONE; [ -f \$U/POOL1_GUNET_FAIL ] && echo \"FAIL \$(head -c 200 \$U/POOL1_GUNET_FAIL)\"
    echo \"HB \$(( now - \$(cat \$U/driver.hb 2>/dev/null || echo 0) ))\"
    echo \"IT \$(tail -c 3000 \$U/pool.log 2>/dev/null | tr '\r' '\n' | grep -a -o '학습: *[0-9]*%[^,]*' | tail -1)\"
    grep -a '\[풀 상태\|\[pgrad' \$U/pool.log 2>/dev/null | tail -2" 2>&1)
  echo "[점검] $(date '+%F %T') $(echo "$ST" | tr '\n' ' ' | cut -c1-700)"
  grep -a '\[풀 상태\|\[pgrad' <<<"$ST" >> $T/pool_obs.log
  grep -q "^FAIL" <<<"$ST" && [ ! -f /tmp/${RUN}_fail_seen ] && { touch /tmp/${RUN}_fail_seen; fail "$RUN 드라이버: $(grep '^FAIL' <<<"$ST")"; }
  if ! grep -q 'NOJOB\|Connection\|timed out' <<<"$ST" && ! grep -q "^DONE\|^FAIL" <<<"$ST"; then
    AGE=$(grep -o "^HB [0-9]*" <<<"$ST" | awk '{print $2}'); SINCE=$(( $(date +%s) - $(cat /tmp/${RUN}_last_launch 2>/dev/null || echo 0) ))
    [ -n "$AGE" ] && [ "$AGE" -gt 900 ] && [ "$SINCE" -gt 1500 ] && relaunch
  fi
  dropvids 2>&1 | tail -20
  if [ $(( $(date +%s) - $(cat /tmp/${RUN}_rf_last 2>/dev/null || echo 0) )) -gt 1800 ]; then
    NF=$(timeout 60 ssh -o ConnectTimeout=20 $LOGIN "n=0; V=/home/dkta/work/lbours/$RUN/pool/pool_trace/vid; for f in \$(ls \$V/*.fail 2>/dev/null); do [ -f \${f%.fail}.mp4 ] || n=\$((n+1)); done; echo \$n" 2>/dev/null)
    if [ -n "$NF" ] && [ "$NF" -gt 0 ]; then date +%s > /tmp/${RUN}_rf_last
      echo "[재렌더 잡] 실패 이력 $NF 개 $(date)"
      timeout 110 ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && read J N < /home/dkta/work/pool1_job.txt; AF_JOB=\$J AF_NODE=\$N AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/pool1_render_fail_boot.sh /home/dkta/work/pool1_render_fail_\$(date +%m%d_%H%M).log" 2>&1 | tail -2
    fi
  fi
  if grep -q "^DONE" <<<"$ST" && [ -n "$(rclone lsf $RV/POOL1_GUNET_CLUSTER_DONE 2>/dev/null)" ]; then
    mkdir -p $T/run; timeout 60 ssh -o ConnectTimeout=20 $LOGIN "grep -a '\[풀 상태\|\[pgrad' /home/dkta/work/lbours/$RUN/pool.log" > $T/run/pool_tail.txt 2>/dev/null
    NP=$(( $(cat /tmp/${RUN}_post_tries 2>/dev/null || echo 0) + 1 )); echo $NP > /tmp/${RUN}_post_tries
    post && { echo "$MARK $(date)"; break; }
    [ $NP -ge 3 ] && { echo "${MARK%_DONE}_POST_FAILED 후처리 3 번 실패 $(date)"; notify "$RUN 후처리 3 번 실패" "로그 /tmp/${RUN}_local.log" --priority high --tags x; break; }
  fi
  sleep 600
done
