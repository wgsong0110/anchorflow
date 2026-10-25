#!/bin/bash
# pool1 (lego 한 궤적 풀 학습, 일반 GNN · Graph U-Net; 사용자 10-25) 로컬 감시·재기동·후처리. 로그 /tmp/pool1_local.log (tmux pool1)
#   10 분마다 클러스터 하트비트 확인 (ssh ControlMaster 재사용) -> 죽었고 안 끝났으면 /home/dkta/work/pool1_relaunch.txt 의 같은 부트 (멱등, 이어 돈다)
#   PBS 서버가 응답하지 않을 수 있어 잡·노드를 AF_JOB/AF_NODE 로 준다 (/home/dkta/work/pool1_job.txt: "<잡> <노드>")
#   둘 다 POOL1_*_CLUSTER_DONE 이면:
#     pool1_{L000,0004}.mp4 (정답 | 일반 GNN 한 궤적 풀 | Graph U-Net 한 궤적 풀, 3DGS, 프레임 번호 크게 빨간색)
#     lego_psi_pool1_{L000,0004}.mp4 (탄성 에너지 색, 공통 고정 범위 lbours/lego_psi/range.json, 분포 패널)
#     pool1_curves.png (여섯 실행 학습 곡선), pool1_table.txt (sf1 두 실행·다섯 궤적 두 실행 포함)
#   -> outputs.md, learned_results.tex 절, notify, POOL1_DONE. 기존 결과 파일은 덮어쓰지 않는다. 아무것도 지우지 않는다.
exec 9>/tmp/pool1_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
C=7154293
RL=r2:storage/result/anchorflow/lbours; RG=$RL/pool1_gnn; RU=$RL/pool1_gunet; RNI=$RL/noinertial; R5=$RL/gunet5_near
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; T=/tmp/pool1_dl; mkdir -p $T $D
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
fail(){ echo "[실패] $1 $(date)"; notify "pool1 실패" "$1" --priority high --tags x; }
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

post(){ echo "[후처리] $(date)"
  for m in gnn gunet; do X=$T/$m; mkdir -p $X
    rclone copy $RL/pool1_$m $X --max-depth 1 --include "*.txt" --include "*.json" --include "*.jsonl" --include "*.mp4" --include "POOL1_*" -q || { fail "내려받기 $m"; return 1; }
    rclone copy $RL/pool1_$m/psi $X/psi --include "*/psi_*_lego.mp4" --include "*/*.npz" --include "*/*.json" --include "*/*.txt" --include "*/*.log" -q || { fail "psi 내려받기 $m"; return 1; }
  done
  for m in gnn gunet; do X=$T/sf1_$m; mkdir -p $X
    rclone copy $RL/sf1_$m $X --max-depth 1 --include "errors.txt" --include "curves.json" --include "inv_*.jsonl" -q
    for t in L000 0004; do get $RL/sf1_$m/psi/$t/defstats_$t.txt $X/defstats_$t.txt; done
  done
  N=$T/ni; G5=$T/g5; mkdir -p $N $G5
  for t in L000 0004; do
    get $RNI/psi/$t/defstats_$t.txt $N/defstats_$t.txt; get $R5/psi/$t/defstats_$t.txt $G5/defstats_$t.txt
    get $RNI/inv_noinertial_multi_$t.jsonl $N/inv_$t.jsonl; get $R5/inv_gunet5_multi_$t.jsonl $G5/inv_$t.jsonl
  done
  get $RNI/errors_multi.txt $N/errors.txt; get $R5/errors_multi.txt $G5/errors.txt
  ER(){ grep -a "^$2 " $1 | grep -o '[0-9.]*%'; }
  EP_L=$(ER $T/gnn/errors.txt pool1_gnn_L000); EP_4=$(ER $T/gnn/errors.txt pool1_gnn_0004)
  EQ_L=$(ER $T/gunet/errors.txt pool1_gunet_L000); EQ_4=$(ER $T/gunet/errors.txt pool1_gunet_0004)
  EG_L=$(ER $T/sf1_gnn/errors.txt sf1_gnn_L000); EG_4=$(ER $T/sf1_gnn/errors.txt sf1_gnn_0004)
  EU_L=$(ER $T/sf1_gunet/errors.txt sf1_gunet_L000); EU_4=$(ER $T/sf1_gunet/errors.txt sf1_gunet_0004)
  EN_L=$(ER $N/errors.txt noinertial_multi_L000); EN_4=$(ER $N/errors.txt noinertial_multi_0004)
  E5_L=$(ER $G5/errors.txt gunet5_multi_L000); E5_4=$(ER $G5/errors.txt gunet5_multi_0004)
  [ -n "$EP_L" ] && [ -n "$EP_4" ] && [ -n "$EQ_L" ] && [ -n "$EQ_4" ] || { fail "pool1 오차 값 없음"; return 1; }
  echo "[오차] 한 궤적 풀 GNN $EP_L / $EP_4, U-Net $EQ_L / $EQ_4 | sf1 GNN $EG_L / $EG_4, U-Net $EU_L / $EU_4 | 다섯 궤적 GNN $EN_L / $EN_4, U-Net $E5_L / $E5_4"
  for x in L000 0004; do OUT=$D/pool1_$x.mp4; [ -e $OUT ] && { echo "[건너뜀] $OUT"; continue; }
    [ $x = L000 ] && { A=$EP_L; B_=$EQ_L; TT="L000 (학습 궤적)"; } || { A=$EP_4; B_=$EQ_4; TT="0004 (학습 안 쓴 궤적)"; }
    cell $T/gnn/gt_$x.mp4 a "정답"; cell $T/gnn/pool1_gnn_$x.mp4 b "일반 GNN · L000 한 궤적 풀 학습 · $A"; cell $T/gunet/pool1_gunet_$x.mp4 c "Graph U-Net · L000 한 궤적 풀 학습 · $B_"
    stack3 $OUT "$TT 롤아웃 80프레임 · 한 궤적 풀 학습 · 관성 예측 없음 · 바닥 softplus + 미끄러짐 방지 · % = 위치 오차" || { fail "3칸 영상 $x"; return 1; }
  done
  for x in L000 0004; do P=$T/psi_$x; mkdir -p $P; OUT=$D/lego_psi_pool1_$x.mp4; [ -e $OUT ] && continue
    for f in psi_gt_lego.mp4 range.json; do cp $T/gnn/psi/$x/$f $P/; done
    cp $T/gnn/psi/$x/psi_p1_lego.mp4 $T/gnn/psi/$x/psi_hist_p1.npz $P/; cp $T/gunet/psi/$x/psi_q1_lego.mp4 $T/gunet/psi/$x/psi_hist_q1.npz $P/
    python3 -c "
import json
r=[json.load(open(p)) for p in ('$T/gnn/psi/$x/range.json','$T/gunet/psi/$x/range.json','$T/gnn/psi/$x/range_noinertial.json')]
assert all(abs(a['lo']-r[0]['lo'])<1e-9 and abs(a['hi']-r[0]['hi'])<1e-9 for a in r); print('[범위 동일]', r[0]['lo'], r[0]['hi'])" || { fail "psi 범위 다름 $x"; return 1; }
    [ $x = L000 ] && { A=$EP_L; B_=$EQ_L; } || { A=$EP_4; B_=$EQ_4; }
    psi3 $P "일반 GNN 한 궤적 풀 · 위치 오차 $A" "Graph U-Net 한 궤적 풀 · 위치 오차 $B_" $OUT $x || return 1
  done
  CV=$D/pool1_curves.png; [ -e $CV ] || python3 $REPO/exe/pool1_plot.py $T/gnn/curves.json $T/gunet/curves.json $T/sf1_gnn/curves.json $T/sf1_gunet/curves.json $CV || { fail "곡선"; return 1; }
  TB=$D/pool1_table.txt
  [ -e $TB ] || python3 $REPO/exe/pool1_table.py $T $TB "$EN_L" "$EN_4" "$E5_L" "$E5_4" "$EG_L" "$EG_4" "$EU_L" "$EU_4" "$EP_L" "$EP_4" "$EQ_L" "$EQ_4" || { fail "표"; return 1; }
  cat $TB
  for f in $D/pool1_L000.mp4 $D/pool1_0004.mp4 $D/lego_psi_pool1_L000.mp4 $D/lego_psi_pool1_0004.mp4 $CV $TB; do
    rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $RG/final -q; rclone copy $f $RU/final -q; done
  rclone copy $REPO/exe/pool1_local.sh $RG/final -q; rclone copy $REPO/exe/pool1_plot.py $RG/final -q; rclone copy $REPO/exe/pool1_table.py $RG/final -q
  VAL=$(python3 -c "
import json; M={}; [M.update(json.load(open(p))) for p in ('$T/gnn/curves.json','$T/gunet/curves.json')]
for r in ('pool1_gnn','pool1_gunet'):
    s=sorted(M[r]['L000롤아웃_위치오차%']); b=min(s,key=lambda p:p[1]); print(f'{r} 마지막 {s[-1][1]:.2f}% 최저 {b[1]:.2f}% (반복 {b[0]})')")
  RT=$(for m in gnn gunet; do head -1 $T/$m/resume_test.txt | cut -c1-300; done | tr '\n' ' ')
  cat >> $RR/outputs.md <<EOM

## $(date +%F) pool1: lego 한 궤적 풀 학습 (일반 GNN · Graph U-Net), 바닥 softplus 0.01 + 미끄러짐 방지, 관성 예측 없음
- Path: $D/pool1_L000.mp4, $D/pool1_0004.mp4 (정답 | 일반 GNN 한 궤적 풀 | Graph U-Net 한 궤적 풀, 3DGS), $D/lego_psi_pool1_L000.mp4, $D/lego_psi_pool1_0004.mp4 (탄성 에너지 색, 공통 고정 범위 lbours/lego_psi/range.json), 학습 곡선 $CV, 표 $TB
- R2 원본: $RG/, $RU/ (pool/MAIN_last.pt·MAIN_best.pt (풀 포함 체크포인트), hist/step_*.pt (500 마다 망만), hist_eval.jsonl, tb/, pool.log, roll_pool1_*_*.pt F32 덤프, inv_/tun_*.jsonl, h5_*, psi/{L000,0004}/, resume_test.txt, base_equiv.txt, cmdline.txt, curves.json, final/)
- Commit: $C (exe/pool1.sh gnn|gunet; 부트 /home/dkta/work/pool1_{gnn,gunet}_run_boot.sh, 다시 띄우기 /home/dkta/work/pool1_relaunch.txt)
- Config: 예전 lego 풀 학습 spstick pool(13471f4) 실행 인자에서 --net_inertial 뺌 + --pool_init_traj lbpt/test_low/L000.pt (풀 새 상태 = L000 0 프레임 하나; 예전은 --pool_drop 낙하 분포에서 매번 새로, 학습 궤적 없음); U-Net 은 + --gnn_layer unet --gunet_levels 3 --bc_feat near --bc_near_h 1.0 (자동 대조: 망·바닥 인자는 sf1·다섯 궤적과 같고 차이는 풀 학습 전용 인자뿐); 15000 반복, 배치 8, 풀 128, 게이트 pgrad 1e-4
- 위치 오차: 일반 GNN L000 $EP_L, 0004 $EP_4 | Graph U-Net L000 $EQ_L, 0004 $EQ_4 (한 궤적 상태 적합 sf1: $EG_L / $EG_4, $EU_L / $EU_4; 다섯 궤적: $EN_L / $EN_4, $E5_L / $E5_4)
- 학습 중 L000 롤아웃 (500 반복마다 저장한 망): $(echo "$VAL" | tr '\n' ';')
- 이어 돌리기 시험: $RT
EOM
  texadd "
\\subsection*{L000 한 궤적 풀 학습 · 일반 GNN 과 Graph U-Net (2026-10-25, $C)}
예전 lego 풀 학습(spstick pool)의 인자에서 관성 예측(\\texttt{--net\\_inertial})을 빼고, 풀의 새 상태를 낙하 분포 대신 L000 의 0 프레임 하나로 고정했다(\\texttt{--pool\\_init\\_traj}). 망·바닥 인자는 한 궤적 상태 적합(sf1)·다섯 궤적 상태 적합과 같다 (softplus 0.01 + 미끄러짐 방지, 15000 반복).
\\begin{center}
\\begin{tabular}{lcccccc}
\\hline
위치 오차 & GNN 다섯 궤적 & U-Net 다섯 궤적 & GNN sf1 & U-Net sf1 & GNN 풀 & U-Net 풀 \\\\
\\hline
L000 (학습 궤적) & $(q $EN_L) & $(q $E5_L) & $(q $EG_L) & $(q $EU_L) & $(q $EP_L) & $(q $EQ_L) \\\\
0004 (학습 안 쓴 궤적) & $(q $EN_4) & $(q $E5_4) & $(q $EG_4) & $(q $EU_4) & $(q $EP_4) & $(q $EQ_4) \\\\
\\hline
\\end{tabular}
\\end{center}
풀 학습 중 L000 롤아웃 (500 반복마다 저장한 망, 정답 시작 80 프레임): $(q "$(echo "$VAL" | sed 's/^pool1_gnn/일반 GNN/; s/^pool1_gunet/Graph U-Net/' | tr '\n' ';')").
탄성 에너지·뒤집힘 (프레임 40·80 \$\\psi\$ 평균 [Pa], 뒤집힌 사면체 최대, det 최소, 프레임 80 det\$<0\$ 입자):
{\\scriptsize\\begin{verbatim}
$(grep -v '^#' $TB)
\\end{verbatim}}
"
  touch $D/POOL1_DONE; echo POOL1_DONE
  notify "pool1 한 궤적 풀 학습 끝" "일반 GNN L000 $EP_L / 0004 $EP_4, Graph U-Net L000 $EQ_L / 0004 $EQ_4 (sf1 GNN $EG_L / $EG_4)" --tags white_check_mark; }

relaunch(){ local m=$1 LG=/home/dkta/work/pool1_${1}_run_$(date +%m%d_%H%M).log
  echo "[재기동 $m] $(date '+%F %T') -> $LG"
  ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && read J N < /home/dkta/work/pool1_job.txt; AF_JOB=\$J AF_NODE=\$N AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/pool1_${m}_run_boot.sh $LG" 2>&1 | tail -3
  date +%s > /tmp/pool1_last_launch_$m; }
echo "[시작] $(date) 커밋 $C"
for m in gnn gunet; do [ -f /tmp/pool1_last_launch_$m ] || date +%s > /tmp/pool1_last_launch_$m; done
while true; do
  [ -f $D/POOL1_DONE ] && { echo "POOL1_DONE (이미) $(date)"; break; }
  ST=$(ssh -o ConnectTimeout=20 $LOGIN 'W=/home/dkta/work; now=$(date +%s)
    read J N < $W/pool1_job.txt; timeout 30 ssh -o ConnectTimeout=15 $N "pgrep -u dkta -f \"Singularity runtime parent\" >/dev/null" || echo NOJOB
    for m in gnn gunet; do U=$W/lbours/pool1_$m; M=$(echo $m | tr a-z A-Z)
      [ -f $U/POOL1_${M}_CLUSTER_DONE ] && echo "${m}DONE"; [ -f $U/POOL1_${M}_FAIL ] && echo "${m}FAIL $(head -c 200 $U/POOL1_${M}_FAIL)"
      echo "${m}HB $(( now - $(cat $U/driver.hb 2>/dev/null || echo 0) ))"
      echo "${m}IT $(tail -c 3000 $U/pool.log 2>/dev/null | tr "\r" "\n" | grep -a -o "학습: *[0-9]*%[^,]*" | tail -1)"; done' 2>&1)
  echo "[점검] $(date '+%F %T') $(echo "$ST" | tr '\n' ' ' | cut -c1-500)"
  for m in gnn gunet; do
    grep -q "^${m}FAIL" <<<"$ST" && [ ! -f /tmp/pool1_fail_seen_$m ] && { touch /tmp/pool1_fail_seen_$m; fail "$m 드라이버: $(grep "^${m}FAIL" <<<"$ST")"; }
  done
  if ! grep -q 'NOJOB\|PBSERR\|Connection\|timed out' <<<"$ST"; then
    for m in gnn gunet; do
      grep -q "^${m}DONE\|^${m}FAIL" <<<"$ST" && continue
      AGE=$(grep -o "^${m}HB [0-9]*" <<<"$ST" | awk '{print $2}'); SINCE=$(( $(date +%s) - $(cat /tmp/pool1_last_launch_$m 2>/dev/null || echo 0) ))
      [ -n "$AGE" ] && [ "$AGE" -gt 900 ] && [ "$SINCE" -gt 1500 ] && relaunch $m
    done
  fi
  if [ -n "$(rclone lsf $RG/POOL1_GNN_CLUSTER_DONE 2>/dev/null)" ] && [ -n "$(rclone lsf $RU/POOL1_GUNET_CLUSTER_DONE 2>/dev/null)" ]; then
    NP=$(( $(cat /tmp/pool1_post_tries 2>/dev/null || echo 0) + 1 )); echo $NP > /tmp/pool1_post_tries
    post && { echo "POOL1_DONE $(date)"; break; }
    [ $NP -ge 3 ] && { echo "POOL1_POST_FAILED 후처리 3 번 실패 $(date)"; notify "pool1 후처리 3 번 실패" "로그 /tmp/pool1_local.log" --priority high --tags x; break; }
  fi
  sleep 600
done
