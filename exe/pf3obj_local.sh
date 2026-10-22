#!/bin/bash
# 목적함수 넷 비교 (exe/pf3obj.sh full) 로컬 감시·재기동·후처리. 로그 /tmp/pf3obj_local.log (tmux pf3obj)
#   10 분마다 (ssh 한 번, ControlMaster 재사용): 클러스터 드라이버 하트비트 -> 죽었고 안 끝났고 오류 없으면 돌고 있는 잡에
#   exe/job_run.sh(AF_DETACH=1)로 다시 띄움 (잡이 바뀌어도 드라이버는 프레임 파일에서 이어 돈다), R2 표식 PF3OBJ_LAUNCHED / FAIL / CLUSTER_DONE
#   끝나면: 3DGS 6칸 영상 2 개, psi 색 6칸 영상 2 개 (공통 고정 범위), 변형 통계, 프레임별 곡선, outputs.md, learned_results.tex -> PF3OBJ_DONE
# 중복 방지 flock. 기존 결과 파일은 덮어쓰지 않는다. 이 루프는 끝없는 동기화를 기다리지 않는다 (틱마다 조회만).
exec 9>/tmp/pf3obj_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
C=${PF3_COMMIT:?커밋}
R=r2:storage/result/anchorflow/lbours/pf3obj; RSP=r2:storage/result/anchorflow/lbours/spstick
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; OVN=$RR/ovnet; T=/tmp/pf3obj_dl; mkdir -p $T $D $OVN
TEX=/home/wgsong/workspace/writing/anchorflow/learned_results.tex
REPO=/home/wgsong/workspace/execute/anchorflow/master
LOGIN=dkta@147.46.121.39
F=$(fc-list :lang=ko file | head -1 | cut -d: -f1)
KS="ip nt l2ov l2gt"
get(){ rclone copyto $1 $2 -q 2>/dev/null; }
lab(){ printf '%s' "$1" > $T/$2.txt; echo "drawtext=fontfile=$F:textfile=$T/$2.txt:expansion=none:x=10:y=8:fontsize=20:fontcolor=black"; }
cell(){ ffmpeg -loglevel error -y -i $1 -vf "pad=iw:ih+40:0:40:white,$(lab "$3" $2)" -c:v libx264 -pix_fmt yuv420p $T/c_$2.mp4; }
texadd(){ python3 - "$TEX" "$1" <<'PY'
import sys
p, txt = sys.argv[1:]
s = open(p).read()
s = s.replace(r"\end{document}", txt + "\n" + r"\end{document}")
open(p, "w").write(s)
PY
}
q(){ echo "$1" | sed 's/%/\\%/g'; }
fail(){ echo "[실패] $1 $(date)"; echo PF3OBJ_FAIL; notify "목적함수 넷 비교 실패" "$1" --priority high --tags x; exit 1; }
relaunch(){
  local LG=/home/dkta/work/pf3obj_driver_$(date +%m%d_%H%M).log
  echo "[재기동] $(date '+%F %T') -> $LG"
  ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/pf3obj_boot.sh $LG" 2>&1 | tail -3
  echo $LG > /tmp/pf3obj_cur_driver_log; date +%s > /tmp/pf3obj_last_launch; }
pe(){ python3 -c "import json;print(f\"{json.load(open('$T/summary.json'))['$1_$2']['pos_err_mean']:.2f}%\")"; }
post(){
  echo "[후처리] $(date)"
  rclone copy $R $T --max-depth 1 --include "*.txt" --include "*.json" --include "PF3OBJ_*" -q || fail "내려받기"
  for k in $KS; do for x in L000 0004; do
    rclone copy $R/$k/$x $T/$k/$x --max-depth 1 --include "records.json" --include "net.mp4" --include "cmd.txt" -q; done; done
  for x in L000 0004; do rclone copy $R/psi/$x $T/psi/$x --include "psi_*_lego.mp4" --include "*.npz" --include "*.json" --include "*.txt" --include "*.log" -q
    get $RSP/ov_$x.mp4 $T/ov_$x.mp4; get $RSP/gt_$x.mp4 $T/gt_$x.mp4; done
  cat $T/summary.txt
  declare -A E; for k in $KS; do for x in L000 0004; do E[${k}_$x]=$(pe $k $x); [ -n "${E[${k}_$x]}" ] || fail "요약 값 없음 $k $x"; done; done
  NM_ip="증분 포텐셜"; NM_nt="뉴턴 목표 L2"; NM_l2ov="출력만 최적화 L2"; NM_l2gt="정답 L2"
  # (a) 3DGS 6칸: 정답 | 출력만 최적화 | 증분 포텐셜 | 뉴턴 목표 L2 | 출력만 최적화 L2 | 정답 L2
  for x in L000 0004; do
    OUT=$D/pf3obj_$x.mp4; [ -e $OUT ] && { echo "[건너뜀] $OUT 이미 있음 -- 덮어쓰지 않음"; continue; }
    [ $x = L000 ] && { EO=11.47%; TT="L000"; } || { EO=6.92%; TT="0004 (다섯 궤적 학습에 안 쓴 궤적)"; }
    cell $T/gt_$x.mp4 a "정답"; cell $T/ov_$x.mp4 b "출력만 최적화 $EO"
    n=c; for k in $KS; do v=NM_$k; cell $T/$k/$x/net.mp4 $n "${!v} ${E[${k}_$x]}"; n=$(echo $n | tr 'cdef' 'defg'); done
    ffmpeg -loglevel error -y -i $T/c_a.mp4 -i $T/c_b.mp4 -i $T/c_c.mp4 -i $T/c_d.mp4 -i $T/c_e.mp4 -i $T/c_f.mp4 -filter_complex "hstack=inputs=6:shortest=1" -c:v libx264 -pix_fmt yuv420p $T/stack.mp4 \
     && ffmpeg -loglevel error -y -i $T/stack.mp4 -vf "pad=iw:ih+44:0:44:white,$(lab "$TT 80프레임 · 망 파라미터 프레임별 최적화 (Adam 1e-3, 2000 회/프레임, 새 망) · 목적함수별 · % = 위치 오차 평균" title),drawtext=fontfile=$F:text='프레임 %{frame_num}':x=w-tw-12:y=h-th-10:fontsize=20:fontcolor=black" -c:v libx264 -pix_fmt yuv420p $OUT \
     && ls -la $OUT || fail "6칸 영상 $x"
  done
  # (b) psi 색 6칸 (공통 고정 범위 lbours/lego_psi/range.json, 색 막대 + 분포 패널)
  DT="fontfile=$F:fontsize=26:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6:expansion=none"
  for x in L000 0004; do P=$T/psi/$x; OUT=$D/lego_psi_pf3obj_$x.mp4
    [ -e $OUT ] && { echo "[건너뜀] $OUT 이미 있음 -- 덮어쓰지 않음"; continue; }
    python3 -c "import json;a=json.load(open('$P/range_spstick.json'));b=json.load(open('$P/range.json'));assert abs(a['lo']-b['lo'])<1e-9 and abs(a['hi']-b['hi'])<1e-9;print('[범위 동일]',a['lo'],a['hi'])" || fail "psi 범위 다름 $x"
    [ $x = L000 ] && EO=11.47% || EO=6.92%
    printf '정답 (MPM)' > $T/l_gt.txt; printf '출력만 최적화 · %s' "$EO" > $T/l_sp.txt
    for k in $KS; do v=NM_$k; printf '%s · %s' "${!v}" "${E[${k}_$x]}" > $T/l_$k.txt; done
    W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $P/psi_gt_lego.mp4)
    NS=""; for n in gt sp $KS; do NS="$NS $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $P/psi_${n}_lego.mp4)"; done
    echo "[psi 렌더 $x] 폭 $W, 프레임$NS"; NG=$(echo $NS | cut -d' ' -f1)
    [ $(echo $NS | tr ' ' '\n' | sort -u | wc -l) = 1 ] || fail "$x psi 프레임 수 다름"
    TW=$((W*6)); rm -rf $P/frames
    ( cd $REPO && python3 exe/psi_dist_vid.py --hist $P/psi_hist_sp.npz --hist2 $P/psi_hist_ip.npz $P/psi_hist_nt.npz $P/psi_hist_l2ov.npz $P/psi_hist_l2gt.npz --range $P/range.json --out_dir $P/frames --cbar $P/cbar.png \
      --w $TW --h 560 --lab_gt "정답 (MPM)" --lab_ov "출력만 최적화" --lab2 "증분 포텐셜" "뉴턴 목표 L2" "출력만 최적화 L2" "정답 L2" ) || fail "분포 패널 $x"
    NP=$(ls $P/frames/*.png | wc -l); [ "$NP" = "$NG" ] || fail "패널 $NP 장 vs 렌더 $NG"
    ffmpeg -y -v error -r 15 -i $P/psi_gt_lego.mp4 -r 15 -i $P/psi_sp_lego.mp4 -r 15 -i $P/psi_ip_lego.mp4 -r 15 -i $P/psi_nt_lego.mp4 -r 15 -i $P/psi_l2ov_lego.mp4 -r 15 -i $P/psi_l2gt_lego.mp4 \
      -loop 1 -r 15 -i $P/cbar.png -framerate 15 -i $P/frames/%04d.png \
     -filter_complex "[0:v]setpts=N/15/TB,drawtext=textfile=$T/l_gt.txt:x=20:y=20:$DT[a];[1:v]setpts=N/15/TB,drawtext=textfile=$T/l_sp.txt:x=20:y=20:$DT[b];\
[2:v]setpts=N/15/TB,drawtext=textfile=$T/l_ip.txt:x=20:y=20:$DT[c];[3:v]setpts=N/15/TB,drawtext=textfile=$T/l_nt.txt:x=20:y=20:$DT[e];\
[4:v]setpts=N/15/TB,drawtext=textfile=$T/l_l2ov.txt:x=20:y=20:$DT[f];[5:v]setpts=N/15/TB,drawtext=textfile=$T/l_l2gt.txt:x=20:y=20:$DT[h];[a][b][c][e][f][h]hstack=inputs=6[top];\
[6:v]scale=$TW:-2,setsar=1[cb];[7:v]scale=$TW:-2,setsar=1[d];[top][cb][d]vstack=inputs=3:shortest=1,\
drawtext=fontfile=$F:text='$x 프레임 %{frame_num}':x=w-tw-20:y=h-th-20:fontsize=28:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6,format=yuv420p" \
     -frames:v $NG -c:v libx264 -crf 18 -r 15 $OUT || fail "psi ffmpeg $x"
    ffprobe -v error -count_frames -show_entries stream=nb_read_frames,width,height -of csv=p=0 $OUT
  done
  # (c) 변형 통계: 칸 이름·덤프 경로 확인
  for x in L000 0004; do S_=$T/psi/$x/defstats_$x.txt
    [ -f $S_ ] || fail "defstats 없음 $x"
    grep -q "^# sp = .*/lbours/spstick/roll_ov_$x.pt" $S_ || fail "defstats $x sp 경로"
    for k in $KS; do grep -q "^# $k = .*/lbours/pf3obj/$k/$x/roll_net.pt" $S_ || fail "defstats $x $k 경로"; done
    for k in gt sp $KS; do [ "$(awk -v k=$k '$1==k' $S_ | wc -l)" = 3 ] || fail "defstats $x 칸 $k 행 수"; done
  done
  DS=$D/lego_psi_pf3obj_defstats.txt
  [ -e $DS ] || { echo "# 칸: gt = 정답(MPM), sp = 출력만 최적화, ip = 증분 포텐셜, nt = 뉴턴 목표 L2, l2ov = 출력만 최적화 L2, l2gt = 정답 L2 (망 파라미터 프레임별 최적화)"
                  cat $T/psi/L000/defstats_L000.txt $T/psi/0004/defstats_0004.txt; } > $DS; cat $DS
  # (d) 프레임별 곡선
  CV=$OVN/pf3obj.png; [ -e $CV ] || python3 $REPO/exe/pf3obj_plot.py $T $CV || fail "곡선"
  for f in $D/pf3obj_L000.mp4 $D/pf3obj_0004.mp4 $D/lego_psi_pf3obj_L000.mp4 $D/lego_psi_pf3obj_0004.mp4 $DS; do rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $R/final -q; done
  rclone copy $CV r2:storage/result/anchorflow/ovnet -q; rclone copy $CV $R/final -q
  rclone copy $REPO/exe/pf3obj_local.sh $R/final -q
  SUM=$(grep -a "^\[" $T/summary.txt | tr '\n' ' ')
  rat(){ python3 -c "import json;print(f\"{json.load(open('$T/summary.json'))['$1_$2']['ratio_E_Estar_median']:.2f}\")"; }
  cat >> $RR/outputs.md <<EOM

## $(date +%F) 목적함수 넷 비교 (망 파라미터 프레임별 최적화, pf3obj)
- Path: $D/pf3obj_L000.mp4, $D/pf3obj_0004.mp4 (정답 | 출력만 최적화 | 증분 포텐셜 | 뉴턴 목표 L2 | 출력만 최적화 L2 | 정답 L2, 3DGS), $D/lego_psi_pf3obj_L000.mp4, $D/lego_psi_pf3obj_0004.mp4 (탄성 에너지 색, 공통 고정 범위 lbours/lego_psi/range.json), 변형 통계 $DS, 프레임별 곡선 $CV
- R2 원본: $R/ ({ip,nt,l2ov,l2gt}/{L000,0004}/frames/frame_XX.pt = 프레임마다 망 파라미터 + 모델 상태 전/후 + E* 해 + 목적함수 곡선(뉴턴은 CG 기록), records.json, roll_net.pt F32 덤프, h5_net, net.mp4; psi/; cmdline_diff.txt; test_result.txt; summary.json)
- Commit: $C (exe/train_deform.py _ovnet_fit = AF_OVNET_FIT, AF_OV_L2_TGT, --ov_obj newton; 드라이버 exe/pf3obj.sh, 감시 exe/pf3obj_local.sh)
- Config: spstick BASE 글자 그대로 + 예전 B3j adam3 (--out_var --ov_init predict --ov_vars net --ov_opt adam --out_var_lr 1e-3 --ov_roll 2000), 새 망(출력층 0), --ov_obj ip / newton (CG 50, 감쇠 1e-3·평균 노드질량/h², 목표 갱신 간격은 cmd.txt) / l2 + AF_OV_L2_TGT=spstick roll_ov_T.pt / l2; E* = 같은 모델 상태 출력만 최적화 L-BFGS 500 (spstick OV 플래그)
- 결과: $SUM
EOM
  texadd "
\\subsection*{목적함수 넷 비교: 망 파라미터 프레임별 최적화 (2026-10-22, $C)}
매 프레임 모델 자신의 롤아웃 상태에서 망 파라미터를 그 프레임 목적함수로 내리고(Adam \$10^{-3}\$, 프레임당 2000 회, 새 망에서 시작, 망은 프레임 사이에 이어 씀) 그 망 출력으로 한 스텝 옮긴다. 설정은 다섯 궤적 학습과 같은 BASE(softplus 바닥 + 미끄러짐 방지). \$E/E^*\$ 는 같은 모델 상태에서 출력만 최적화로 푼 최솟값 대비 망 출력의 증분 포텐셜 (프레임 중앙값).
\\begin{center}
\\begin{tabular}{lcccccc}
\\hline
위치 오차 평균 & 출력만 최적화 & 다섯 궤적 학습 & 증분 포텐셜 & 뉴턴 목표 L2 & 출력만 최적화 L2 & 정답 L2 \\\\
\\hline
L000 & 11.47\\% & 12.21\\% & $(q ${E[ip_L000]}) & $(q ${E[nt_L000]}) & $(q ${E[l2ov_L000]}) & $(q ${E[l2gt_L000]}) \\\\
0004 & 6.92\\% & 25.57\\% & $(q ${E[ip_0004]}) & $(q ${E[nt_0004]}) & $(q ${E[l2ov_0004]}) & $(q ${E[l2gt_0004]}) \\\\
\$E/E^*\$ L000 & & & $(rat ip L000) & $(rat nt L000) & $(rat l2ov L000) & $(rat l2gt L000) \\\\
\$E/E^*\$ 0004 & & & $(rat ip 0004) & $(rat nt 0004) & $(rat l2ov 0004) & $(rat l2gt 0004) \\\\
\\hline
\\end{tabular}
\\end{center}
"
  touch $D/PF3OBJ_DONE; echo PF3OBJ_DONE
  notify "목적함수 넷 비교 끝" "L000 증분 ${E[ip_L000]} 뉴턴 ${E[nt_L000]} 출력L2 ${E[l2ov_L000]} 정답L2 ${E[l2gt_L000]} / 0004 ${E[ip_0004]} ${E[nt_0004]} ${E[l2ov_0004]} ${E[l2gt_0004]}" --tags white_check_mark
}
SEEN_L=$( [ -f /tmp/pf3obj_seen_launched ] && echo 1 || echo 0 ); ERRN=0
echo "[시작] $(date) 커밋 $C"
while true; do
  date +%s > /tmp/pf3obj_local.hb
  ST=$(ssh -o ConnectTimeout=20 $LOGIN "bash /home/dkta/work/pf3obj_check.sh $(cat /tmp/pf3obj_cur_driver_log 2>/dev/null)" 2>&1)
  echo "[점검] $(date '+%F %T') $(echo "$ST" | tr '\n' ' ' | cut -c1-700)"
  L=$(rclone lsf $R --max-depth 1 2>/dev/null)
  if [ $SEEN_L = 0 ] && grep -qx 'PF3OBJ_LAUNCHED' <<<"$L"; then
    get $R/PF3OBJ_LAUNCHED $T/PF3OBJ_LAUNCHED; cut -c1-300 $T/PF3OBJ_LAUNCHED; echo PF3OBJ_LAUNCHED; touch /tmp/pf3obj_seen_launched; SEEN_L=1
    notify "목적함수 넷 비교 시작" "$(wc -l < $T/PF3OBJ_LAUNCHED) 실행 띄움" --tags rocket; fi
  if grep -qx 'PF3OBJ_CLUSTER_DONE' <<<"$L"; then post; break; fi
  if grep -qx 'PF3OBJ_FAIL' <<<"$L" || grep -q '^FAILMARK' <<<"$ST"; then
    get $R/PF3OBJ_FAIL $T/PF3OBJ_FAIL; fail "클러스터 드라이버: $(cat $T/PF3OBJ_FAIL 2>/dev/null) ($(cat /tmp/pf3obj_cur_driver_log))"; fi
  if ! grep -q '^DONE' <<<"$ST" && ! grep -q 'NOJOB\|PBSERR' <<<"$ST" && grep -q '^HB ' <<<"$ST"; then
    AGE=$(grep -o '^HB [0-9]*' <<<"$ST" | awk '{print $2}'); NE=$(grep -o '^ERR [0-9]*' <<<"$ST" | awk '{print $2}')
    SINCE=$(( $(date +%s) - $(cat /tmp/pf3obj_last_launch 2>/dev/null || echo 0) ))
    if [ -n "$AGE" ] && [ "$AGE" -gt 900 ] && [ "$SINCE" -gt 1500 ]; then
      if [ "${NE:-0}" -gt 0 ]; then
        [ $ERRN = 0 ] && { echo "[멈춤] 드라이버가 오류로 끝남 -- 다시 띄우지 않음"; notify "목적함수 넷 비교 오류" "드라이버 오류: $(cat /tmp/pf3obj_cur_driver_log)" --priority high --tags x; }
        ERRN=1
      else relaunch; fi
    fi
  fi
  grep -q NOJOB <<<"$ST" && echo "    (돌고 있는 잡 없음 -- 잡 할당 대기)"
  grep -q PBSERR <<<"$ST" && echo "    (PBS 조회 실패 -- 이번 틱 건너뜀)"
  sleep 600
done
