#!/bin/bash
# 프레임별 맞춤 (exe/perframe.sh) 로컬 감시·재기동·후처리. 로그 /tmp/perframe_local.log (tmux perframe)
#   10 분마다: (1) 클러스터 드라이버 하트비트 확인 -> 죽었고 안 끝났고 오류 표식 없으면 돌고 있는 잡에 exe/job_run.sh(AF_DETACH=1)로 다시 띄움
#                  (잡이 바뀌어도 job_run.sh 가 돌고 있는 잡을 찾고, 드라이버는 프레임 파일에서 이어 돈다)
#              (2) R2 표식 PERFRAME_LAUNCHED / PERFRAME_FAIL / PERFRAME_CLUSTER_DONE
#   궤적마다 (0004 먼저, L000 은 그 뒤) 클러스터 표식 PERFRAME_<T>_DONE 이 보이면: 3DGS 4칸 영상, psi 색 4칸 영상 (공통 고정 범위), 변형 통계,
#   프레임별 곡선, outputs.md, learned_results.tex -> /tmp/perframe_local.log 에 PERFRAME_DONE (0004 끝), 클러스터 전부 끝나면 종료.
#   이번 실행은 --net_inertial 뺌 (망 출력 = 노드 변위); 비교 다섯 궤적 모델 = lbours/noinertial (같은 설정의 다섯 궤적 학습).
# 중복 방지: flock. ssh 는 ~/.ssh/config ControlMaster 재사용, 10 분 간격. 기존 결과 파일은 덮어쓰지 않는다.
exec 9>/tmp/perframe_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
C=${PERFRAME_COMMIT:?커밋}
R=r2:storage/result/anchorflow/lbours/perframe_fit/ni; RSP=r2:storage/result/anchorflow/lbours/spstick; RNI=r2:storage/result/anchorflow/lbours/noinertial
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; OVN=$RR/ovnet; T=/tmp/perframe_dl; mkdir -p $T $D $OVN
TEX=/home/wgsong/workspace/writing/anchorflow/learned_results.tex
REPO=/home/wgsong/workspace/execute/anchorflow/master
LOGIN=dkta@147.46.121.39
F=$(fc-list :lang=ko file | head -1 | cut -d: -f1)
get(){ rclone copyto $1 $2 -q 2>/dev/null; }
lab(){ printf '%s' "$1" > $T/$2.txt; echo "drawtext=fontfile=$F:textfile=$T/$2.txt:expansion=none:x=10:y=8:fontsize=20:fontcolor=black"; }
cell(){ ffmpeg -loglevel error -y -i $1 -vf "pad=iw:ih+40:0:40:white,$(lab "$3" $2)" -c:v libx264 -pix_fmt yuv420p $T/c_$2.mp4; }
stack4(){ OUT=$1; TTL=$2; ffmpeg -loglevel error -y -i $T/c_a.mp4 -i $T/c_b.mp4 -i $T/c_c.mp4 -i $T/c_d.mp4 -filter_complex "hstack=inputs=4:shortest=1" -c:v libx264 -pix_fmt yuv420p $T/stack.mp4 \
  && ffmpeg -loglevel error -y -i $T/stack.mp4 -vf "pad=iw:ih+44:0:44:white,$(lab "$TTL" title),drawtext=fontfile=$F:text='프레임 %{frame_num}':x=w-tw-12:y=h-th-10:fontsize=20:fontcolor=black" -c:v libx264 -pix_fmt yuv420p $OUT \
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
fail(){ echo "[실패] $1 $(date)"; echo PERFRAME_FAIL; notify "프레임별 맞춤 실패" "$1" --priority high --tags x; exit 1; }
relaunch(){
  local LG=/home/dkta/work/perframe_driver_$(date +%m%d_%H%M).log
  echo "[재기동] $(date '+%F %T') -> $LG"
  ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/perframe_boot.sh $LG" 2>&1 | tail -3
  echo $LG > /tmp/perframe_cur_driver_log; date +%s > /tmp/perframe_last_launch; }
post(){ # post <T>
  x=$1; echo "[후처리 $x] $(date)"
  rclone copy $R $T --max-depth 1 --include "*.txt" --include "*.json" --include "PERFRAME_*" -q || fail "내려받기"
  for N in 1000 3000; do
    rclone copy $R/n$N/$x $T/n$N/$x --max-depth 1 --include "records.json" --include "pf.mp4" --include "cmd.txt" -q
    rclone copy $R/n$N/$x/psi $T/n$N/$x/psi --include "psi_*_lego.mp4" --include "*.npz" --include "*.json" --include "*.txt" --include "*.log" -q
  done
  cat $T/summary_$x.txt; tail -3 $T/decision.txt
  NC=$(grep -q "^DECISION n3000" $T/decision.txt && echo 3000 || echo 1000)
  get $RSP/ov_$x.mp4 $T/ov_$x.mp4; get $RNI/noinertial_multi_$x.mp4 $T/mm_$x.mp4; get $RSP/gt_$x.mp4 $T/gt_$x.mp4
  EE=$(python3 -c "import json;print(f\"{json.load(open('$T/summary_$x.json'))['${x}_n$NC']['pos_err_mean']:.2f}%\")") || fail "요약 값 없음 $x"
  [ $x = L000 ] && { EO=11.47%; EM=13.26%; EI=12.21%; TT="L000 (다섯 궤적 학습에 쓴 궤적)"; } || { EO=6.92%; EM=24.82%; EI=25.57%; TT="0004 (다섯 궤적 학습에 안 쓴 궤적)"; }
  # (a) 3DGS 4칸: 정답 | 출력만 최적화 | 다섯 궤적 모델 (망 출력만) | 프레임별 맞춤 모델
  OUT=$D/perframe_fit_$x.mp4
  if [ -e $OUT ]; then echo "[건너뜀] $OUT 이미 있음 -- 덮어쓰지 않음"; else
    cell $T/gt_$x.mp4 a "정답"; cell $T/ov_$x.mp4 b "출력만 최적화 $EO"; cell $T/mm_$x.mp4 c "다섯 궤적 학습 모델 (망 출력만) $EM"
    cell $T/n$NC/$x/pf.mp4 d "프레임별 맞춤 모델 ($NC 반복/프레임) $EE"
    stack4 $OUT "$TT 80프레임 · 관성 예측 없음 · softplus 바닥 + 미끄러짐 방지 · % = 위치 오차 평균" || fail "4칸 영상 $x"; fi
  # (b) psi 색 4칸 (공통 고정 범위 lbours/lego_psi/range.json, 색 막대 + 분포 패널)
  DT="fontfile=$F:fontsize=26:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6:expansion=none"
  P=$T/n$NC/$x/psi; OUT=$D/lego_psi_perframe_fit_$x.mp4
  if [ -e $OUT ]; then echo "[건너뜀] $OUT 이미 있음 -- 덮어쓰지 않음"; else
    for rj in range_spstick range_noinertial; do
      python3 -c "import json;a=json.load(open('$P/$rj.json'));b=json.load(open('$P/range.json'));assert abs(a['lo']-b['lo'])<1e-9 and abs(a['hi']-b['hi'])<1e-9;print('[범위 동일 $rj]',a['lo'],a['hi'])" || fail "psi 범위 다름 $x $rj"; done
    printf '정답 (MPM)' > $T/l_gt.txt; printf '출력만 최적화 · %s' "$EO" > $T/l_sp.txt; printf '다섯 궤적 학습 모델 (망 출력만) · %s' "$EM" > $T/l_mm.txt
    printf '프레임별 맞춤 모델 (%s 반복) · %s' "$NC" "$EE" > $T/l_pf.txt
    W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $P/psi_gt_lego.mp4)
    NS=""; for n in gt sp ni pf; do NS="$NS $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $P/psi_${n}_lego.mp4)"; done
    echo "[psi 렌더 $x] 폭 $W, 프레임$NS"; NG=$(echo $NS | cut -d' ' -f1)
    [ $(echo $NS | tr ' ' '\n' | sort -u | wc -l) = 1 ] || fail "$x psi 프레임 수 다름"
    TW=$((W*4)); rm -rf $P/frames
    ( cd $REPO && python3 exe/psi_dist_vid.py --hist $P/psi_hist_sp.npz --hist2 $P/psi_hist_ni.npz $P/psi_hist_pf.npz --range $P/range.json --out_dir $P/frames --cbar $P/cbar.png \
      --w $TW --h 560 --lab_gt "정답 (MPM)" --lab_ov "출력만 최적화" --lab2 "다섯 궤적 학습 모델 (망 출력만)" "프레임별 맞춤 모델" ) || fail "분포 패널 $x"
    NP=$(ls $P/frames/*.png | wc -l); [ "$NP" = "$NG" ] || fail "패널 $NP 장 vs 렌더 $NG"
    ffmpeg -y -v error -r 15 -i $P/psi_gt_lego.mp4 -r 15 -i $P/psi_sp_lego.mp4 -r 15 -i $P/psi_ni_lego.mp4 -r 15 -i $P/psi_pf_lego.mp4 -loop 1 -r 15 -i $P/cbar.png -framerate 15 -i $P/frames/%04d.png \
     -filter_complex "[0:v]setpts=N/15/TB,drawtext=textfile=$T/l_gt.txt:x=20:y=20:$DT[a];[1:v]setpts=N/15/TB,drawtext=textfile=$T/l_sp.txt:x=20:y=20:$DT[b];\
[2:v]setpts=N/15/TB,drawtext=textfile=$T/l_mm.txt:x=20:y=20:$DT[c];[3:v]setpts=N/15/TB,drawtext=textfile=$T/l_pf.txt:x=20:y=20:$DT[e];[a][b][c][e]hstack=inputs=4[top];\
[4:v]scale=$TW:-2,setsar=1[cb];[5:v]scale=$TW:-2,setsar=1[d];[top][cb][d]vstack=inputs=3:shortest=1,\
drawtext=fontfile=$F:text='$x 프레임 %{frame_num}':x=w-tw-20:y=h-th-20:fontsize=28:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6,format=yuv420p" \
     -frames:v $NG -c:v libx264 -crf 18 -r 15 $OUT || fail "psi ffmpeg $x"
    ffprobe -v error -count_frames -show_entries stream=nb_read_frames,width,height -of csv=p=0 $OUT; fi
  # (c) 변형 통계: 칸 이름·덤프 경로 확인 (sp = 출력만 최적화, ni = 다섯 궤적 모델 망 출력만, pf = 프레임별 맞춤)
  S_=$P/defstats_$x.txt; [ -f $S_ ] || fail "defstats 없음 $x"
  grep -q "^# sp = .*/lbours/spstick/roll_ov_$x.pt" $S_ || fail "defstats $x sp 경로"
  grep -q "^# ni = .*/lbours/noinertial/roll_noinertial_multi_$x.pt" $S_ || fail "defstats $x ni 경로"
  grep -q "^# pf = .*/lbours/perframe_fit/ni/n$NC/$x/roll_pf.pt" $S_ || fail "defstats $x pf 경로가 이번 실행이 아니다"
  for k in gt sp ni pf; do [ "$(awk -v k=$k '$1==k' $S_ | wc -l)" = 3 ] || fail "defstats $x 칸 $k 행 수"; done
  DS=$D/lego_psi_perframe_fit_defstats_$x.txt
  [ -e $DS ] || { echo "# 칸: gt = 정답(MPM), sp = 출력만 최적화, ni = 다섯 궤적 학습 모델 (망 출력만), pf = 프레임별 맞춤 모델 ($NC 반복/프레임, 망 출력만)"; cat $S_; } > $DS; cat $DS
  # (d) 프레임별 곡선 (0004 때 ovnet/perframe_fit.png, L000 까지 끝나면 ovnet/perframe_fit_withL000.png -- 덮어쓰지 않음)
  [ $x = 0004 ] && CV=$OVN/perframe_fit.png || CV=$OVN/perframe_fit_withL000.png
  [ -e $CV ] || python3 $REPO/exe/perframe_plot.py $T $CV || fail "곡선"
  for f in $D/perframe_fit_$x.mp4 $D/lego_psi_perframe_fit_$x.mp4 $DS; do rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $R/final -q; done
  rclone copy $CV r2:storage/result/anchorflow/ovnet -q; rclone copy $CV $R/final -q; rclone copy $REPO/exe/perframe_local.sh $R/final -q
  SUM=$(grep -a "^\[$x 반복" $T/summary_$x.txt | tr '\n' ' ')
  DEC=$(grep -a "판정 근거\|^DECISION" $T/decision.txt | tr '\n' ' ')
  cat >> $RR/outputs.md <<EOM

## $(date +%F) 프레임별 맞춤 $x (관성 예측 없음: 한 프레임씩 망 갱신 -> 모델 자신의 상태에서 한 스텝)
- Path: $D/perframe_fit_$x.mp4 (정답 | 출력만 최적화 | 다섯 궤적 모델 망 출력만 | 프레임별 맞춤, 3DGS), $D/lego_psi_perframe_fit_$x.mp4 (탄성 에너지 색, 공통 고정 범위 lbours/lego_psi/range.json), 변형 통계 $DS, 프레임별 곡선 $CV
- R2 원본: $R/ (n$NC/$x/frames/frame_XX.pt = 프레임마다 망·Adam 상태 전체 + 옮기기 전/후 모델 상태 x,v,F,p; records.json; roll_pf.pt F32 덤프; h5_pf; psi/), estar_$x.json (E*), summary_$x.json, decision.txt, repro_test.txt, resume_test.txt, cmdline_diff.txt
- Commit: $C (exe/train_deform.py _perframe_fit = AF_PERFRAME_FIT, 드라이버 exe/perframe.sh, 로컬 감시 exe/perframe_local.sh)
- Config: spstick BASE 에서 --net_inertial 만 뺌 (= noinertial 다섯 궤적 학습 BASE, 글자 그대로 대조), 손실 phys_w·E + det 벌점 (배치 1), Adam lr 3e-4, 기울기 1.0 자르기, 프레임당 $NC 반복; 프레임 0 새 망(입력 표준화 = noinertial 다섯 궤적 체크포인트의 in_mu/in_sd), 이후 망·Adam 이어감; 갱신 상태 = lbours/spstick/roll_ov_$x.pt; E* = 그 출력만 최적화 실행 설정으로 상태마다 한 스텝 L-BFGS 500 (AF_OV_MIN_STATES)
- 반복 수 판정: $DEC
- 결과: $SUM
EOM
  texadd "
\\subsection*{프레임별 맞춤 $x (관성 예측 없음, 2026-10-22, $C)}
프레임 \$t\$ 마다 출력만 최적화 상태 하나로 망을 갱신(학습과 같은 증분 포텐셜 손실, Adam \$3\\times10^{-4}\$, 프레임당 $NC 반복, 직전 프레임 망에서 이어서)한 뒤 그 망으로 모델 자신의 롤아웃 상태에서 한 스텝을 간다. 망 출력이 곧 노드 변위다 (\\texttt{--net\\_inertial} 없음).
\\begin{center}
\\begin{tabular}{lcccc}
\\hline
위치 오차 평균 & 출력만 최적화 & 다섯 궤적 (관성 예측 + 망) & 다섯 궤적 (망 출력만) & 프레임별 맞춤 (망 출력만) \\\\
\\hline
$x & $(q $EO) & $(q $EI) & $(q $EM) & $(q $EE) \\\\
\\hline
\\end{tabular}
\\end{center}
$(q "$SUM")
"
  touch $D/PERFRAME_${x}_DONE
  [ $x = 0004 ] && echo PERFRAME_DONE
  notify "프레임별 맞춤 $x 끝" "위치 오차 $EE (출력만 최적화 $EO, 다섯 궤적 망 출력만 $EM)" --tags white_check_mark
}
SEEN_L=$( [ -f /tmp/perframe_seen_launched ] && echo 1 || echo 0 ); ERRN=0
echo "[시작] $(date) 커밋 $C"
while true; do
  date +%s > /tmp/perframe_local.hb
  ST=$(ssh -o ConnectTimeout=20 $LOGIN "bash /home/dkta/work/perframe_check.sh $(cat /tmp/perframe_cur_driver_log 2>/dev/null)" 2>&1)
  echo "[점검] $(date '+%F %T') $(echo "$ST" | tr '\n' ' ' | cut -c1-400)"
  L=$(rclone lsf $R --max-depth 1 2>/dev/null)
  if [ $SEEN_L = 0 ] && grep -qx 'PERFRAME_LAUNCHED' <<<"$L"; then
    for f in PERFRAME_LAUNCHED repro_test.txt resume_test.txt cmdline_diff.txt; do get $R/$f $T/$f; done
    cat $T/cmdline_diff.txt $T/repro_test.txt $T/resume_test.txt $T/PERFRAME_LAUNCHED; echo PERFRAME_LAUNCHED; touch /tmp/perframe_seen_launched; SEEN_L=1
    notify "프레임별 맞춤 시작" "$(head -1 $T/PERFRAME_LAUNCHED | cut -c1-150)" --tags rocket; fi
  for x in 0004 L000; do
    grep -qx "PERFRAME_${x}_DONE" <<<"$L" && [ ! -f $D/PERFRAME_${x}_DONE ] && post $x; done
  if grep -qx 'PERFRAME_CLUSTER_DONE' <<<"$L" && [ -f $D/PERFRAME_0004_DONE ] && [ -f $D/PERFRAME_L000_DONE ]; then echo "[전부 끝] $(date)"; break; fi
  if grep -qx 'PERFRAME_FAIL' <<<"$L" || grep -q '^FAILMARK' <<<"$ST"; then
    get $R/PERFRAME_FAIL $T/PERFRAME_FAIL; fail "클러스터 드라이버: $(cat $T/PERFRAME_FAIL 2>/dev/null) ($(cat /tmp/perframe_cur_driver_log))"; fi
  if ! grep -q '^DONE' <<<"$ST" && ! grep -q 'NOJOB\|PBSERR' <<<"$ST" && grep -q '^HB ' <<<"$ST"; then
    AGE=$(grep -o '^HB [0-9]*' <<<"$ST" | awk '{print $2}'); NE=$(grep -o '^ERR [0-9]*' <<<"$ST" | awk '{print $2}')
    SINCE=$(( $(date +%s) - $(cat /tmp/perframe_last_launch 2>/dev/null || echo 0) ))
    if [ -n "$AGE" ] && [ "$AGE" -gt 900 ] && [ "$SINCE" -gt 1500 ]; then
      if [ "${NE:-0}" -gt 0 ]; then
        [ $ERRN = 0 ] && { echo "[멈춤] 드라이버가 오류로 끝남 -- 다시 띄우지 않음"; notify "프레임별 맞춤 오류" "드라이버 오류: $(cat /tmp/perframe_cur_driver_log)" --priority high --tags x; }
        ERRN=1
      else relaunch; fi
    fi
  fi
  grep -q NOJOB <<<"$ST" && echo "    (돌고 있는 잡 없음 -- 잡 할당 대기)"
  grep -q PBSERR <<<"$ST" && echo "    (PBS 조회 실패 -- 이번 틱 건너뜀)"
  sleep 600
done
