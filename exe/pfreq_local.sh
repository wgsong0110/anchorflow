#!/bin/bash
# pfreq (노드 입력 = 입자 물리량 주파수 인코딩 --pfreq 6) 로컬 감시·재기동·후처리. 로그 /tmp/pfreq_local.log (tmux pfreq)
#   10 분마다: (1) 클러스터 드라이버 하트비트 확인 -> 죽었고 안 끝났고 오류 표식 없으면 지금 돌고 있는 잡에 exe/job_run.sh(AF_DETACH=1)로 다시 띄움
#                  (잡이 바뀌어도 job_run.sh 가 돌고 있는 잡을 찾는다, 드라이버는 체크포인트에서 이어 돈다)
#              (2) R2 표식 PFREQ_LAUNCHED / PFREQ_FAIL / PFREQ_CLUSTER_DONE
#   끝나면: 3DGS 3칸 영상 2 개, psi 색 영상 2 개, 변형 통계, 학습 곡선, outputs.md, learned_results.tex -> PFREQ_DONE
# 중복 방지: flock /tmp/pfreq_local.lock (두 번째 실행은 바로 끝). ssh 는 ~/.ssh/config ControlMaster 재사용, 10 분 간격.
exec 9>/tmp/pfreq_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
C=824be7c; KF=6
R=r2:storage/result/anchorflow/lbours/pfreq; RSP=r2:storage/result/anchorflow/lbours/spstick
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; T=/tmp/pfreq_dl; mkdir -p $T $D
TEX=/home/wgsong/workspace/writing/anchorflow/learned_results.tex
REPO=/home/wgsong/workspace/execute/anchorflow/master
LOGIN=dkta@147.46.121.39
F=$(fc-list :lang=ko file | head -1 | cut -d: -f1)
get(){ rclone copyto $1 $2 -q 2>/dev/null; }
lab(){ printf '%s' "$1" > $T/$2.txt; echo "drawtext=fontfile=$F:textfile=$T/$2.txt:expansion=none:x=10:y=8:fontsize=20:fontcolor=black"; }
cell(){ ffmpeg -loglevel error -y -i $1 -vf "pad=iw:ih+40:0:40:white,$(lab "$3" $2)" -c:v libx264 -pix_fmt yuv420p $T/c_$2.mp4; }
stack3(){ OUT=$1; TTL=$2; ffmpeg -loglevel error -y -i $T/c_a.mp4 -i $T/c_b.mp4 -i $T/c_c.mp4 -filter_complex "hstack=inputs=3:shortest=1" -c:v libx264 -pix_fmt yuv420p $T/stack.mp4 \
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
fail(){ echo "[실패] $1 $(date)"; echo PFREQ_FAIL; notify "pfreq 실패" "$1" --priority high --tags x; exit 1; }
relaunch(){
  local LG=/home/dkta/work/pfreq_driver_$(date +%m%d_%H%M).log
  echo "[재기동] $(date '+%F %T') -> $LG"
  ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/pfreq_boot.sh $LG" 2>&1 | tail -3
  echo $LG > /tmp/pfreq_cur_driver_log; date +%s > /tmp/pfreq_last_launch; }
post(){
  echo "[후처리] $(date)"
  rclone copy $R $T --max-depth 1 --include "*.txt" --include "*.json" --include "*.mp4" --include "PFREQ_*" -q || fail "내려받기"
  rclone copy $R/psi $T/psi --include "*/psi_*_lego.mp4" --include "*/*.npz" --include "*/*.json" --include "*/*.txt" --include "*/*.log" -q || fail "psi 내려받기"
  for x in L000 0004; do get $RSP/m_multi_$x.mp4 $T/sp_multi_$x.mp4; done
  cat $T/errors_multi.txt $T/err_spstick_mm.txt
  EL=$(grep -a '^pfreq_multi_L000 ' $T/errors_multi.txt | grep -o '[0-9.]*%'); E4=$(grep -a '^pfreq_multi_0004 ' $T/errors_multi.txt | grep -o '[0-9.]*%')
  [ -n "$EL" ] && [ -n "$E4" ] || fail "오차 값 없음"
  # (a) 3DGS 3칸: 정답 | 기존 다섯 궤적 | 이번
  for x in L000 0004; do
    OUT=$D/pfreq_multi_$x.mp4; [ -e $OUT ] && { echo "[건너뜀] $OUT 이미 있음 -- 덮어쓰지 않음"; continue; }
    [ $x = L000 ] && { EE=$EL; EO=12.21%; TT="L000"; } || { EE=$E4; EO=25.57%; TT="0004 (학습 안 쓴 궤적)"; }
    cell $T/gt_$x.mp4 a "정답"; cell $T/sp_multi_$x.mp4 b "기존 다섯 궤적 (입자 모멘트 입력) $EO"; cell $T/pfreq_multi_$x.mp4 c "이번: 입자 주파수 인코딩 입력 $EE"
    stack3 $OUT "$TT 다섯 궤적 학습 모델 롤아웃 80프레임 · softplus 바닥 + 미끄러짐 방지 · % = 위치 오차" || fail "3칸 영상 $x"
  done
  # (b) psi 색 (공통 고정 범위, 원본 형식: 색 막대 + 분포 패널)
  DT="fontfile=$F:fontsize=26:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6:expansion=none"
  for x in L000 0004; do P=$T/psi/$x; OUT=$D/lego_psi_pfreq_$x.mp4
    [ -e $OUT ] && { echo "[건너뜀] $OUT 이미 있음 -- 덮어쓰지 않음"; continue; }
    python3 -c "import json;a=json.load(open('$P/range_spstick.json'));b=json.load(open('$P/range.json'));assert abs(a['lo']-b['lo'])<1e-9 and abs(a['hi']-b['hi'])<1e-9;print('[범위 동일]',a['lo'],a['hi'])" || fail "psi 범위 다름 $x"
    EM=$(grep -a "재롤아웃 $x" $T/err_spstick_mm.txt | grep -o '평균 [0-9.]*%' | head -1 | grep -o '[0-9.]*%'); [ $x = L000 ] && EE=$EL || EE=$E4
    printf '정답 (MPM)' > $T/l_gt.txt; printf '기존 다섯 궤적 (입자 모멘트) · 위치 오차 %s' "$EM" > $T/l_mm.txt; printf '이번 입자 주파수 인코딩 · 위치 오차 %s' "$EE" > $T/l_pf.txt
    W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $P/psi_gt_lego.mp4)
    NS=""; for n in gt mm pf; do NS="$NS $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $P/psi_${n}_lego.mp4)"; done
    echo "[psi 렌더 $x] 폭 $W, 프레임$NS"; NG=$(echo $NS | cut -d' ' -f1)
    [ $(echo $NS | tr ' ' '\n' | sort -u | wc -l) = 1 ] || fail "$x psi 프레임 수 다름"
    TW=$((W*3)); rm -rf $P/frames
    ( cd $REPO && python3 exe/psi_dist_vid.py --hist $P/psi_hist_mm.npz --hist2 $P/psi_hist_pf.npz --range $P/range.json --out_dir $P/frames --cbar $P/cbar.png \
      --w $TW --h 560 --lab_gt "정답 (MPM)" --lab_ov "기존 다섯 궤적 (입자 모멘트)" --lab2 "이번 입자 주파수 인코딩" ) || fail "분포 패널 $x"
    NP=$(ls $P/frames/*.png | wc -l); [ "$NP" = "$NG" ] || fail "패널 $NP 장 vs 렌더 $NG"
    ffmpeg -y -v error -r 15 -i $P/psi_gt_lego.mp4 -r 15 -i $P/psi_mm_lego.mp4 -r 15 -i $P/psi_pf_lego.mp4 -loop 1 -r 15 -i $P/cbar.png -framerate 15 -i $P/frames/%04d.png \
     -filter_complex "[0:v]setpts=N/15/TB,drawtext=textfile=$T/l_gt.txt:x=20:y=20:$DT[a];[1:v]setpts=N/15/TB,drawtext=textfile=$T/l_mm.txt:x=20:y=20:$DT[b];\
[2:v]setpts=N/15/TB,drawtext=textfile=$T/l_pf.txt:x=20:y=20:$DT[c];[a][b][c]hstack=inputs=3[top];\
[3:v]scale=$TW:-2,setsar=1[cb];[4:v]scale=$TW:-2,setsar=1[d];[top][cb][d]vstack=inputs=3:shortest=1,\
drawtext=fontfile=$F:text='$x 프레임 %{frame_num}':x=w-tw-20:y=h-th-20:fontsize=28:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6,format=yuv420p" \
     -frames:v $NG -c:v libx264 -crf 18 -r 15 $OUT || fail "psi ffmpeg $x"
    ffprobe -v error -count_frames -show_entries stream=nb_read_frames,width,height -of csv=p=0 $OUT
  done
  DS=$D/lego_psi_pfreq_defstats.txt
  [ -e $DS ] || cat $T/psi/L000/defstats_L000.txt $T/psi/0004/defstats_0004.txt > $DS; cat $DS
  # (c) 학습 곡선
  CV=$D/pfreq_multi_curves.png; [ -e $CV ] || python3 $REPO/exe/pfreq_plot.py $T/curves.json $CV $KF || fail "곡선"
  for f in $D/pfreq_multi_L000.mp4 $D/pfreq_multi_0004.mp4 $D/lego_psi_pfreq_L000.mp4 $D/lego_psi_pfreq_0004.mp4 $DS $CV; do rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $R/final -q; done
  rclone copy $REPO/exe/pfreq_local.sh $R/final -q; rclone copy $REPO/exe/pfreq_plot.py $R/final -q
  VAL=$(python3 -c "
import json; M=json.load(open('$T/curves.json'))
for r in ('spstick','pfreq'):
    print(r, ' '.join(f\"{t} {M[r].get('다궤적적합_롤아웃_위치오차%/'+t,[[0,float('nan')]])[-1][1]:.2f}%\" for t in ('L000','0000','0001','0002','0003')))")
  DSL=$(awk '$1=="pf" || $1=="mm"' $DS | tr -s ' ' | tr '\n' ';')
  BEQ=$(cat $T/base_equiv.txt 2>/dev/null | tr '\n' ' ')
  cat >> $RR/outputs.md <<EOM

## $(date +%F) pfreq 다섯 궤적 학습 -- 노드 입력 = 입자 물리량 주파수 인코딩 (--pfreq $KF)
- Path: $D/pfreq_multi_L000.mp4, $D/pfreq_multi_0004.mp4 (정답 | 기존 spstick 다섯 궤적 | 이번, 3DGS), $D/lego_psi_pfreq_L000.mp4, $D/lego_psi_pfreq_0004.mp4 (탄성 에너지 색, 같은 칸), 변형 통계 $DS, 학습 곡선 $CV
- R2 원본: $R/ (multi/ 체크포인트·tb·run.log, roll_pfreq_multi_*.pt F32 덤프, h5_*, psi/{L000,0004}/, unit_test.txt, base_equiv.txt, resume_test.txt, cmdline_diff.txt, curves.json, final/)
- Commit: 647db8c (--pfreq 코드·단위 시험) + $C (드라이버 exe/pfreq.sh); 클러스터 /home/dkta/work/pfreq_boot.sh, 로컬 감시 exe/pfreq_local.sh
- Config: spstick multi 글자 그대로 + --pfreq $KF 하나 (명령줄 자동 대조); 15000 반복 배치 8, 학습 상태 = lbours/spstick/roll_ov_{L000,0000..0003}.pt
- 노드 입력: 입자 모멘트 29 채널 뺌, 입자-노드 15 스칼라 [(x_p-x_j)/h 3, v_p/vel_scale 3, F_p-I 9] 각각 [s, sin/cos(2^k π s), k<$KF] 를 λ·m 가중 평균 = $((15*(1+2*KF))) 채널 + 바닥 4 (높이·법선, 빼면 바닥을 모름) = $((15*(1+2*KF)+4))
- 위치 오차: L000 $EL (기존 12.21%), 0004 $E4 (기존 25.57%)
- 마지막 검증(학습 중 정답 시작 롤아웃): $(echo "$VAL" | tr '\n' ';')
- 변형 통계 (mm = 기존, pf = 이번; 프레임 |σ-1|max 중앙 99% det<0 ψ평균 ψ중앙): $DSL
- psi 색: log10(psi+1e-3) 고정 범위 lbours/lego_psi/range.json, turbo, 칸별 정규화 없음; 정답·기존 칸은 lego_psi_spstick 렌더 재사용
- 기본 경로 대조: $BEQ
- 이어 돌리기 시험: $(head -2 $T/resume_test.txt | tr '\n' ' ')
EOM
  texadd "
\\subsection*{노드 입력 = 입자 물리량 주파수 인코딩 (2026-10-21, $C)}
노드 입력의 입자 모멘트 29 채널을 빼고, 입자 \$p\$ 가 자기 사면체 꼭짓점 \$j\$ 에 갖는 15 스칼라 \$s\\in\\{(x_p-x_j)/h,\\ v_p/v_s,\\ F_p-I\\}\$ 를 \$[s,\\ \\sin(2^k\\pi s),\\ \\cos(2^k\\pi s)]_{k<$KF}\$ 로 인코딩해 \$\\lambda_{pj}m_p\$ 가중 평균한 $((15*(1+2*KF))) 채널로 바꾸었다. 바닥 4 채널(높이·법선)은 그대로 둔다. 나머지는 spstick 다섯 궤적 학습과 같다.
\\begin{center}
\\begin{tabular}{lcc}
\\hline
위치 오차 & 입자 모멘트 (기존) & 주파수 인코딩 (--pfreq $KF) \\\\
\\hline
L000 & 12.21\\% & $(q $EL) \\\\
0004 (학습 안 쓴 궤적) & 25.57\\% & $(q $E4) \\\\
\\hline
\\end{tabular}
\\end{center}
학습 중 검증(정답 시작 80프레임, 마지막): $(q "$(echo "$VAL" | sed 's/^spstick/기존/; s/^pfreq/이번/' | tr '\n' ';')")
"
  touch $D/PFREQ_DONE; echo PFREQ_DONE
  notify "pfreq 끝" "L000 $EL (기존 12.21%), 0004 $E4 (기존 25.57%)" --tags white_check_mark
}
SEEN_L=$( [ -f /tmp/pfreq_seen_launched ] && echo 1 || echo 0 ); ERRN=0
echo "[시작] $(date) 커밋 $C"
while true; do
  date +%s > /tmp/pfreq_local.hb
  ST=$(ssh -o ConnectTimeout=20 $LOGIN "bash /home/dkta/work/pfreq_check.sh $(cat /tmp/pfreq_cur_driver_log 2>/dev/null)" 2>&1)
  echo "[점검] $(date '+%F %T') $(echo "$ST" | tr '\n' ' ' | cut -c1-300)"
  L=$(rclone lsf $R --max-depth 1 2>/dev/null)
  if [ $SEEN_L = 0 ] && grep -qx 'PFREQ_LAUNCHED' <<<"$L"; then
    for f in PFREQ_LAUNCHED unit_test.txt base_equiv.txt resume_test.txt cmdline_diff.txt; do get $R/$f $T/$f; done
    cat $T/unit_test.txt $T/cmdline_diff.txt $T/base_equiv.txt $T/resume_test.txt $T/PFREQ_LAUNCHED; echo PFREQ_LAUNCHED; touch /tmp/pfreq_seen_launched; SEEN_L=1
    notify "pfreq 학습 시작" "$(head -1 $T/PFREQ_LAUNCHED | cut -c1-150)" --tags rocket; fi
  if grep -qx 'PFREQ_CLUSTER_DONE' <<<"$L"; then post; break; fi
  if grep -qx 'PFREQ_FAIL' <<<"$L" || grep -q '^FAILMARK' <<<"$ST"; then
    get $R/PFREQ_FAIL $T/PFREQ_FAIL; fail "클러스터 드라이버: $(cat $T/PFREQ_FAIL 2>/dev/null) ($(cat /tmp/pfreq_cur_driver_log))"; fi
  if ! grep -q '^DONE' <<<"$ST" && ! grep -q 'NOJOB\|PBSERR' <<<"$ST" && grep -q '^HB ' <<<"$ST"; then
    AGE=$(grep -o '^HB [0-9]*' <<<"$ST" | awk '{print $2}'); NE=$(grep -o '^ERR [0-9]*' <<<"$ST" | awk '{print $2}')
    SINCE=$(( $(date +%s) - $(cat /tmp/pfreq_last_launch 2>/dev/null || echo 0) ))
    if [ -n "$AGE" ] && [ "$AGE" -gt 900 ] && [ "$SINCE" -gt 1500 ]; then
      if [ "${NE:-0}" -gt 0 ]; then
        [ $ERRN = 0 ] && { echo "[멈춤] 드라이버가 오류로 끝남 -- 다시 띄우지 않음"; notify "pfreq 오류" "드라이버 오류: $(cat /tmp/pfreq_cur_driver_log)" --priority high --tags x; }
        ERRN=1
      else relaunch; fi
    fi
  fi
  grep -q NOJOB <<<"$ST" && echo "    (돌고 있는 잡 없음 -- 잡 할당 대기)"
  grep -q PBSERR <<<"$ST" && echo "    (PBS 조회 실패 -- 이번 틱 건너뜀)"
  sleep 600
done
