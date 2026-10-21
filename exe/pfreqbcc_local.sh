#!/bin/bash
# pfreq_bcc (--pfreq 6 + --bc_feat contact) 로컬 감시·재기동·후처리. 로그 /tmp/pfreqbcc_local.log (tmux pfreqbcc)
#   후처리는 pfreq 후처리(PFREQ_DONE)가 끝난 뒤에 한다 (4칸 영상에 pfreq 칸이 필요)
#   10 분마다: (1) 클러스터 드라이버 하트비트 확인 -> 죽었고 안 끝났고 오류 표식 없으면 지금 돌고 있는 잡에 exe/job_run.sh(AF_DETACH=1)로 다시 띄움
#                  (잡이 바뀌어도 job_run.sh 가 돌고 있는 잡을 찾는다, 드라이버는 체크포인트에서 이어 돈다)
#              (2) R2 표식 PFREQBCC_LAUNCHED / PFREQBCC_FAIL / PFREQBCC_CLUSTER_DONE
#   끝나면: 3DGS 4칸 영상 2 개, psi 색 영상 2 개, 변형 통계, 학습 곡선, outputs.md, learned_results.tex -> PFREQBCC_DONE
# 중복 방지: flock /tmp/pfreqbcc_local.lock (두 번째 실행은 바로 끝). ssh 는 ~/.ssh/config ControlMaster 재사용, 10 분 간격.
exec 9>/tmp/pfreqbcc_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
C=f5f427c; KF=6
R=r2:storage/result/anchorflow/lbours/pfreq_bcc; RPF=r2:storage/result/anchorflow/lbours/pfreq; RSP=r2:storage/result/anchorflow/lbours/spstick
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; T=/tmp/pfreqbcc_dl; mkdir -p $T $D
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
fail(){ echo "[실패] $1 $(date)"; echo PFREQBCC_FAIL; notify "pfreq_bcc 실패" "$1" --priority high --tags x; exit 1; }
relaunch(){
  local LG=/home/dkta/work/pfreqbcc_driver_$(date +%m%d_%H%M).log
  echo "[재기동] $(date '+%F %T') -> $LG"
  ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/pfreqbcc_boot.sh $LG" 2>&1 | tail -3
  echo $LG > /tmp/pfreqbcc_cur_driver_log; date +%s > /tmp/pfreqbcc_last_launch; }
post(){
  echo "[후처리] $(date)"
  rclone copy $R $T --max-depth 1 --include "*.txt" --include "*.json" --include "*.mp4" --include "PFREQBCC_*" -q || fail "내려받기"
  rclone copy $R/psi $T/psi --include "*/psi_*_lego.mp4" --include "*/*.npz" --include "*/*.json" --include "*/*.txt" --include "*/*.log" -q || fail "psi 내려받기"
  rclone copy $RPF/psi $T/psi --include "*/psi_pf_lego.mp4" --include "*/psi_hist_pf.npz" --include "*/defstats_*.txt" --include "*/range.json" -q 2>/dev/null
  for x in L000 0004; do get $RSP/m_multi_$x.mp4 $T/sp_multi_$x.mp4; get $RPF/pfreq_multi_$x.mp4 $T/pf_multi_$x.mp4; done
  get $RPF/errors_multi.txt $T/pf_errors.txt
  for x in L000 0004; do rclone copyto $RPF/psi/$x/defstats_$x.txt $T/psi/$x/pf_defstats_$x.txt -q; done
  cat $T/errors_multi.txt $T/pf_errors.txt $T/err_spstick_mm.txt
  EL=$(grep -a '^pfreq_bcc_multi_L000 ' $T/errors_multi.txt | grep -o '[0-9.]*%'); E4=$(grep -a '^pfreq_bcc_multi_0004 ' $T/errors_multi.txt | grep -o '[0-9.]*%')
  PL=$(grep -a '^pfreq_multi_L000 ' $T/pf_errors.txt | grep -o '[0-9.]*%'); P4=$(grep -a '^pfreq_multi_0004 ' $T/pf_errors.txt | grep -o '[0-9.]*%')
  [ -n "$EL" ] && [ -n "$E4" ] && [ -n "$PL" ] && [ -n "$P4" ] || fail "오차 값 없음"
  # (a) 3DGS 4칸: 정답 | 기존 다섯 궤적 | pfreq | pfreq+접촉 바닥
  for x in L000 0004; do
    OUT=$D/pfreq_bcc_multi_$x.mp4; [ -e $OUT ] && { echo "[건너뜀] $OUT 이미 있음 -- 덮어쓰지 않음"; continue; }
    [ $x = L000 ] && { EE=$EL; EP=$PL; EO=12.21%; TT="L000"; } || { EE=$E4; EP=$P4; EO=25.57%; TT="0004 (학습 안 쓴 궤적)"; }
    cell $T/gt_$x.mp4 a "정답"; cell $T/sp_multi_$x.mp4 b "기존 다섯 궤적 (입자 모멘트) $EO"; cell $T/pf_multi_$x.mp4 c "pfreq (주파수 인코딩 + 바닥 4) $EP"
    cell $T/pfreq_bcc_multi_$x.mp4 d "pfreq + 접촉 셀 바닥 5 $EE"
    stack4 $OUT "$TT 다섯 궤적 학습 모델 롤아웃 80프레임 · softplus 바닥 + 미끄러짐 방지 · % = 위치 오차" || fail "4칸 영상 $x"
  done
  # (b) psi 색 (공통 고정 범위, 같은 칸 4 개 + 색 막대 + 분포 패널)
  DT="fontfile=$F:fontsize=26:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6:expansion=none"
  for x in L000 0004; do P=$T/psi/$x; OUT=$D/lego_psi_pfreq_bcc_$x.mp4
    [ -e $OUT ] && { echo "[건너뜀] $OUT 이미 있음 -- 덮어쓰지 않음"; continue; }
    python3 -c "import json;a=json.load(open('$P/range_spstick.json'));b=json.load(open('$P/range.json'));assert abs(a['lo']-b['lo'])<1e-9 and abs(a['hi']-b['hi'])<1e-9;print('[범위 동일]',a['lo'],a['hi'])" || fail "psi 범위 다름 $x"
    [ -f $P/psi_pf_lego.mp4 ] && [ -f $P/psi_hist_pf.npz ] || fail "pfreq psi 파일 없음 $x"
    EM=$(grep -a "재롤아웃 $x" $T/err_spstick_mm.txt | grep -o '평균 [0-9.]*%' | head -1 | grep -o '[0-9.]*%'); [ $x = L000 ] && { EE=$EL; EP=$PL; } || { EE=$E4; EP=$P4; }
    printf '정답 (MPM)' > $T/l_gt.txt; printf '기존 다섯 궤적 · 위치 오차 %s' "$EM" > $T/l_mm.txt; printf 'pfreq · 위치 오차 %s' "$EP" > $T/l_pf.txt; printf 'pfreq + 접촉 바닥 · 위치 오차 %s' "$EE" > $T/l_pb.txt
    W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $P/psi_gt_lego.mp4)
    NS=""; for n in gt mm pf pb; do NS="$NS $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $P/psi_${n}_lego.mp4)"; done
    echo "[psi 렌더 $x] 폭 $W, 프레임$NS"; NG=$(echo $NS | cut -d' ' -f1)
    [ $(echo $NS | tr ' ' '\n' | sort -u | wc -l) = 1 ] || fail "$x psi 프레임 수 다름"
    TW=$((W*4)); rm -rf $P/frames
    ( cd $REPO && python3 exe/psi_dist_vid.py --hist $P/psi_hist_mm.npz --hist2 $P/psi_hist_pf.npz $P/psi_hist_pb.npz --range $P/range.json --out_dir $P/frames --cbar $P/cbar.png \
      --w $TW --h 560 --lab_gt "정답 (MPM)" --lab_ov "기존 다섯 궤적 (입자 모멘트)" --lab2 "pfreq" "pfreq + 접촉 바닥" ) || fail "분포 패널 $x"
    NP=$(ls $P/frames/*.png | wc -l); [ "$NP" = "$NG" ] || fail "패널 $NP 장 vs 렌더 $NG"
    ffmpeg -y -v error -r 15 -i $P/psi_gt_lego.mp4 -r 15 -i $P/psi_mm_lego.mp4 -r 15 -i $P/psi_pf_lego.mp4 -r 15 -i $P/psi_pb_lego.mp4 -loop 1 -r 15 -i $P/cbar.png -framerate 15 -i $P/frames/%04d.png \
     -filter_complex "[0:v]setpts=N/15/TB,drawtext=textfile=$T/l_gt.txt:x=20:y=20:$DT[a];[1:v]setpts=N/15/TB,drawtext=textfile=$T/l_mm.txt:x=20:y=20:$DT[b];\
[2:v]setpts=N/15/TB,drawtext=textfile=$T/l_pf.txt:x=20:y=20:$DT[c];[3:v]setpts=N/15/TB,drawtext=textfile=$T/l_pb.txt:x=20:y=20:$DT[e];[a][b][c][e]hstack=inputs=4[top];\
[4:v]scale=$TW:-2,setsar=1[cb];[5:v]scale=$TW:-2,setsar=1[d];[top][cb][d]vstack=inputs=3:shortest=1,\
drawtext=fontfile=$F:text='$x 프레임 %{frame_num}':x=w-tw-20:y=h-th-20:fontsize=28:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6,format=yuv420p" \
     -frames:v $NG -c:v libx264 -crf 18 -r 15 $OUT || fail "psi ffmpeg $x"
    ffprobe -v error -count_frames -show_entries stream=nb_read_frames,width,height -of csv=p=0 $OUT
  done
  DS=$D/lego_psi_pfreq_bcc_defstats.txt
  [ -e $DS ] || { cat $T/psi/L000/defstats_L000.txt $T/psi/0004/defstats_0004.txt; echo "# pfreq (pf) 행:"; awk '$1=="pf"' $T/psi/L000/pf_defstats_L000.txt $T/psi/0004/pf_defstats_0004.txt; } > $DS; cat $DS
  CV=$D/pfreq_bcc_multi_curves.png; [ -e $CV ] || python3 $REPO/exe/pfreq_plot.py $T/curves.json $CV $KF || fail "곡선"
  for f in $D/pfreq_bcc_multi_L000.mp4 $D/pfreq_bcc_multi_0004.mp4 $D/lego_psi_pfreq_bcc_L000.mp4 $D/lego_psi_pfreq_bcc_0004.mp4 $DS $CV; do rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $R/final -q; done
  rclone copy $REPO/exe/pfreqbcc_local.sh $R/final -q; rclone copy $REPO/exe/pfreq_plot.py $R/final -q
  VAL=$(python3 -c "
import json; M=json.load(open('$T/curves.json'))
for r in ('spstick','pfreq','pfreq_bcc'):
    if r in M: print(r, ' '.join(f\"{t} {M[r].get('다궤적적합_롤아웃_위치오차%/'+t,[[0,float('nan')]])[-1][1]:.2f}%\" for t in ('L000','0000','0001','0002','0003')))")
  DSL=$(awk '$1=="pb" || $1=="pf" || $1=="mm"' $DS | tr -s ' ' | tr '\n' ';')
  cat >> $RR/outputs.md <<EOM

## $(date +%F) pfreq_bcc 다섯 궤적 학습 -- --pfreq $KF + 접촉 셀 바닥 특징 (--bc_feat contact)
- Path: $D/pfreq_bcc_multi_L000.mp4, $D/pfreq_bcc_multi_0004.mp4 (정답 | 기존 spstick 다섯 궤적 | pfreq | pfreq+접촉 바닥, 3DGS), $D/lego_psi_pfreq_bcc_L000.mp4, $D/lego_psi_pfreq_bcc_0004.mp4 (탄성 에너지 색, 같은 칸), 변형 통계 $DS, 학습 곡선 $CV
- R2 원본: $R/ (multi/ 체크포인트·tb·run.log, roll_pfreq_bcc_multi_*.pt F32 덤프, h5_*, psi/{L000,0004}/, unit_test.txt, resume_test.txt, cmdline_diff.txt, curves.json, final/)
- Commit: $C (드라이버 exe/pfreq_bcc.sh, 단위 시험 exe/test_pfreq_bcc.py); 클러스터 /home/dkta/work/pfreqbcc_boot.sh, 로컬 감시 exe/pfreqbcc_local.sh
- Config: pfreq 학습(824be7c) 명령줄 글자 그대로 + --bc_feat contact 하나 (명령줄 자동 대조); 입력 = 주파수 인코딩 $((15*(1+2*KF))) + 접촉 바닥 5 (접촉 셀 꼭짓점에만 [플래그, 거리, 법선], 나머지 0) = $((15*(1+2*KF)+5))
- 위치 오차: L000 $EL (pfreq $PL, 기존 12.21%), 0004 $E4 (pfreq $P4, 기존 25.57%)
- 마지막 검증(학습 중 정답 시작 롤아웃): $(echo "$VAL" | tr '\n' ';')
- 변형 통계 (mm = 기존, pf = pfreq, pb = 이번; 프레임 |σ-1|max 중앙 99% det<0 ψ평균 ψ중앙): $DSL
- psi 색: log10(psi+1e-3) 고정 범위 lbours/lego_psi/range.json, turbo, 칸별 정규화 없음; 정답·기존 칸은 lego_psi_spstick, pfreq 칸은 lbours/pfreq/psi 렌더 재사용
- 이어 돌리기 시험: $(head -2 $T/resume_test.txt | tr '\n' ' ')
EOM
  texadd "
\\subsection*{주파수 인코딩 입력 + 접촉 셀 바닥 특징 (2026-10-21, $C)}
--pfreq $KF 학습과 같은 설정에서 바닥 4 채널(모든 노드의 높이·법선) 대신 접촉 셀 바닥 특징 5 채널(바닥과 교차하거나 이번 스텝 관성 예측으로 닿는 셀의 꼭짓점 노드에만 [플래그, 거리, 법선], 나머지 0)을 썼다. 입력 $((15*(1+2*KF)+5)) 채널.
\\begin{center}
\\begin{tabular}{lccc}
\\hline
위치 오차 & 입자 모멘트 (기존) & 주파수 인코딩 & 주파수 인코딩 + 접촉 바닥 \\\\
\\hline
L000 & 12.21\\% & $(q $PL) & $(q $EL) \\\\
0004 (학습 안 쓴 궤적) & 25.57\\% & $(q $P4) & $(q $E4) \\\\
\\hline
\\end{tabular}
\\end{center}
학습 중 검증(정답 시작 80프레임, 마지막): $(q "$(echo "$VAL" | sed 's/^spstick/기존/; s/^pfreq_bcc/주파수+접촉/; s/^pfreq/주파수/' | tr '\n' ';')")
"
  touch $D/PFREQBCC_DONE; echo PFREQBCC_DONE
  notify "pfreq_bcc 끝" "L000 $EL (pfreq $PL, 기존 12.21%), 0004 $E4 (pfreq $P4, 기존 25.57%)" --tags white_check_mark
}
SEEN_L=$( [ -f /tmp/pfreqbcc_seen_launched ] && echo 1 || echo 0 ); ERRN=0
echo "[시작] $(date) 커밋 $C"
while true; do
  date +%s > /tmp/pfreqbcc_local.hb
  ST=$(ssh -o ConnectTimeout=20 $LOGIN "bash /home/dkta/work/pfreqbcc_check.sh $(cat /tmp/pfreqbcc_cur_driver_log 2>/dev/null)" 2>&1)
  echo "[점검] $(date '+%F %T') $(echo "$ST" | tr '\n' ' ' | cut -c1-300)"
  L=$(rclone lsf $R --max-depth 1 2>/dev/null)
  if [ $SEEN_L = 0 ] && grep -qx 'PFREQBCC_LAUNCHED' <<<"$L"; then
    for f in PFREQBCC_LAUNCHED unit_test.txt resume_test.txt cmdline_diff.txt; do get $R/$f $T/$f; done
    cat $T/unit_test.txt $T/cmdline_diff.txt $T/resume_test.txt $T/PFREQBCC_LAUNCHED; echo PFREQBCC_LAUNCHED; touch /tmp/pfreqbcc_seen_launched; SEEN_L=1
    notify "pfreq_bcc 학습 시작" "$(head -1 $T/PFREQBCC_LAUNCHED | cut -c1-150)" --tags rocket; fi
  if grep -qx 'PFREQBCC_CLUSTER_DONE' <<<"$L"; then
    if [ -f $D/PFREQ_DONE ]; then post; break; else echo "    (클러스터 끝, pfreq 후처리 PFREQ_DONE 기다림)"; fi; fi
  if grep -qx 'PFREQBCC_FAIL' <<<"$L" || grep -q '^FAILMARK' <<<"$ST"; then
    get $R/PFREQBCC_FAIL $T/PFREQBCC_FAIL; fail "클러스터 드라이버: $(cat $T/PFREQBCC_FAIL 2>/dev/null) ($(cat /tmp/pfreqbcc_cur_driver_log))"; fi
  if ! grep -q '^DONE' <<<"$ST" && ! grep -q 'NOJOB\|PBSERR' <<<"$ST" && grep -q '^HB ' <<<"$ST"; then
    AGE=$(grep -o '^HB [0-9]*' <<<"$ST" | awk '{print $2}'); NE=$(grep -o '^ERR [0-9]*' <<<"$ST" | awk '{print $2}')
    SINCE=$(( $(date +%s) - $(cat /tmp/pfreqbcc_last_launch 2>/dev/null || echo 0) ))
    if [ -n "$AGE" ] && [ "$AGE" -gt 900 ] && [ "$SINCE" -gt 1500 ]; then
      if [ "${NE:-0}" -gt 0 ]; then
        [ $ERRN = 0 ] && { echo "[멈춤] 드라이버가 오류로 끝남 -- 다시 띄우지 않음"; notify "pfreq_bcc 오류" "드라이버 오류: $(cat /tmp/pfreqbcc_cur_driver_log)" --priority high --tags x; }
        ERRN=1
      else relaunch; fi
    fi
  fi
  grep -q NOJOB <<<"$ST" && echo "    (돌고 있는 잡 없음 -- 잡 할당 대기)"
  grep -q PBSERR <<<"$ST" && echo "    (PBS 조회 실패 -- 이번 틱 건너뜀)"
  sleep 600
done
