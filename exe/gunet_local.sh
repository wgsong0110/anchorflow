#!/bin/bash
# Graph U-Net 두 실행 (사용자 10-24) 로컬 감시·재기동·후처리. 로그 /tmp/gunet_local.log (tmux gunet)
#   (0) 짧은 시험+본 실행 부트(/home/dkta/work/gunet_go_boot.sh)를 아직 안 띄웠으면, 돌고 있는 잡(129788 이 아닌 것)에 띄운다
#   (1) pfip_gunet/L000 (증분 포텐셜 프레임별, 몸통 Graph U-Net)   (2) gunet5 (다섯 궤적 학습, 몸통 Graph U-Net)
#   10 분마다 gunet_check.sh 로 하트비트 확인 -> 죽었고 안 끝났으면 job_run.sh(AF_DETACH=1)로 같은 부트 다시 (멱등, 이어 돈다)
#   끝나면: (1) pf_ip_gunet_L000.mp4 (정답 | 출력만 최적화 | Graph U-Net), lego_psi_pf_ip_gunet_L000.mp4 -> GUNET1_DONE
#           (2) gunet5_multi_{L000,0004}.mp4 (정답 | 기준 noinertial | Graph U-Net), lego_psi_gunet5_*.mp4, 변형 통계, 학습 곡선 -> GUNET5_DONE
#   outputs.md·learned_results.tex 에 덧붙이고 notify. 기존 결과 파일은 덮어쓰지 않는다. ssh 는 ControlMaster 재사용, 10 분 간격.
exec 9>/tmp/gunet_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
C=2f8fa21; OLDJOB=129788
RL=r2:storage/result/anchorflow/lbours; R1=$RL/pfip_gunet/L000; R5=$RL/gunet5; RNI=$RL/noinertial; RSP=$RL/spstick; RS=$RL/gunet_smoke_$C
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; T=/tmp/gunet_dl; mkdir -p $T $D
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
FAILED=0
fail(){ echo "[실패] $1 $(date)"; notify "gunet 실패" "$1" --priority high --tags x; FAILED=1; }
# psi 3칸 + 색막대 + 분포 패널: psi3 <psi 디렉토리> <칸2 이름> <칸3 이름> <라벨2> <라벨3> <출력> <제목 접두>
psi3(){ P=$1; n2=$2; n3=$3; OUT=$6; X=$7
  printf '정답 (MPM)' > $T/l_gt.txt; printf '%s' "$4" > $T/l_2.txt; printf '%s' "$5" > $T/l_3.txt
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $P/psi_gt_lego.mp4)
  NS=""; for n in gt $n2 $n3; do NS="$NS $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $P/psi_${n}_lego.mp4)"; done
  NG=$(echo $NS | cut -d' ' -f1); [ $(echo $NS | tr ' ' '\n' | sort -u | wc -l) = 1 ] || { fail "psi 프레임 수 다름 ($NS) $P"; return 1; }
  TW=$((W*3)); rm -rf $P/frames
  ( cd $REPO && python3 exe/psi_dist_vid.py --hist $P/psi_hist_$n2.npz --hist2 $P/psi_hist_$n3.npz --range $P/range.json --out_dir $P/frames --cbar $P/cbar.png \
    --w $TW --h 560 --lab_gt "정답 (MPM)" --lab_ov "${4%% ·*}" --lab2 "${5%% ·*}" ) || { fail "분포 패널 $P"; return 1; }
  DT="fontfile=$F:fontsize=26:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6:expansion=none"
  ffmpeg -y -v error -r 15 -i $P/psi_gt_lego.mp4 -r 15 -i $P/psi_${n2}_lego.mp4 -r 15 -i $P/psi_${n3}_lego.mp4 -loop 1 -r 15 -i $P/cbar.png -framerate 15 -i $P/frames/%04d.png \
   -filter_complex "[0:v]setpts=N/15/TB,drawtext=textfile=$T/l_gt.txt:x=20:y=20:$DT[a];[1:v]setpts=N/15/TB,drawtext=textfile=$T/l_2.txt:x=20:y=20:$DT[b];\
[2:v]setpts=N/15/TB,drawtext=textfile=$T/l_3.txt:x=20:y=20:$DT[c];[a][b][c]hstack=inputs=3[top];[3:v]scale=$TW:-2,setsar=1[cb];[4:v]scale=$TW:-2,setsar=1[d];[top][cb][d]vstack=inputs=3:shortest=1,\
drawtext=fontfile=$F:text='$X 프레임 %{frame_num}':x=w-tw-20:y=h-th-20:fontsize=44:fontcolor=red:box=1:boxcolor=white@0.8:boxborderw=6,format=yuv420p" \
   -frames:v $NG -c:v libx264 -crf 18 -r 15 $OUT || { fail "psi ffmpeg $OUT"; return 1; }
  ffprobe -v error -count_frames -show_entries stream=nb_read_frames,width,height -of csv=p=0 $OUT; }

# ---------------- (1) 증분 포텐셜 프레임별 ----------------
post1(){ echo "[후처리 (1)] $(date)"; X=$T/g1; mkdir -p $X
  rclone copy $R1 $X --max-depth 1 --include "records.json" --include "net.mp4" --include "cmd*.txt" -q
  rclone copy $R1/psi $X/psi --include "psi_*_lego.mp4" --include "*.npz" --include "*.json" --include "*.txt" --include "*.log" -q
  get $RL/pfip_noin/L000/records.json $X/noin_records.json
  [ -f $T/ov_L000.mp4 ] || get $RSP/ov_L000.mp4 $T/ov_L000.mp4; [ -f $T/gt_L000.mp4 ] || get $RSP/gt_L000.mp4 $T/gt_L000.mp4
  SUM=$(python3 - $X/records.json $X/noin_records.json <<'PY'
import json, sys, numpy as np
r = json.load(open(sys.argv[1])); assert len(r) == 80, len(r)
try: b = json.load(open(sys.argv[2]))
except Exception: b = []
pe = lambda rr: 100 * np.mean([q["pos_err_model"] for q in rr])
ra = lambda rr: np.median([q["ratio_E_Estar"] for q in rr if q.get("ratio_E_Estar") is not None])
n = len(b); rs = [q for q in r if q["t"] <= max([p["t"] for p in b], default=-1)]
print(f"{pe(r):.2f}%")
print(f"80 프레임 위치 오차 평균 {pe(r):.2f}%, E/E* 중앙 {ra(r):.2f}, det 최소 {min(q['F_stats']['det_min'] for q in r):.3f}")
if n: print(f"기준판(관성 예측 없음, 중단 {n} 프레임)과 같은 {n} 프레임: Graph U-Net 위치 오차 평균 {pe(rs):.2f}% E/E* 중앙 {ra(rs):.2f} | 기준판 {pe(b):.2f}% E/E* 중앙 {ra(b):.2f}")
PY
) || { fail "(1) 기록 요약"; return 1; }
  EN=$(echo "$SUM" | sed -n 1p); S2=$(echo "$SUM" | sed -n 2p); S3=$(echo "$SUM" | sed -n 3p); echo "$SUM"
  OUT=$D/pf_ip_gunet_L000.mp4
  if [ -e $OUT ]; then echo "[건너뜀] $OUT"; else
    cell $T/gt_L000.mp4 a "정답"; cell $T/ov_L000.mp4 b "출력만 최적화 · 위치 오차 평균 11.47%"; cell $X/net.mp4 c "Graph U-Net 몸통 · 위치 오차 평균 $EN"
    stack3 $OUT "L000 80프레임 · 증분 포텐셜 프레임별 망 최적화 · 관성 예측 없음 · 몸통 Graph U-Net (h·2h·4h)" || { fail "(1) 3칸 영상"; return 1; }; fi
  P=$X/psi; OUT=$D/lego_psi_pf_ip_gunet_L000.mp4
  if [ ! -e $OUT ]; then
    python3 -c "import json;a=json.load(open('$P/range_spstick.json'));b=json.load(open('$P/range.json'));assert abs(a['lo']-b['lo'])<1e-9 and abs(a['hi']-b['hi'])<1e-9;print('[범위 동일]',a['lo'],a['hi'])" || { fail "(1) psi 범위 다름"; return 1; }
    psi3 $P sp gunet "출력만 최적화 · 11.47%" "Graph U-Net 몸통 · $EN" $OUT L000 || return 1; fi
  for f in $D/pf_ip_gunet_L000.mp4 $D/lego_psi_pf_ip_gunet_L000.mp4; do rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $R1/final -q; done
  PS=$(awk '$1=="gunet" && ($2==40||$2==80){printf "프레임 %s ψ 평균 %s det<0 %s; ", $2, $6, $5}' $P/defstats_L000.txt)
  cat >> $RR/outputs.md <<EOM

## $(date +%F) pf_ip_gunet: 증분 포텐셜 프레임별 망 최적화 · 몸통 Graph U-Net
- Path: $D/pf_ip_gunet_L000.mp4 (정답 | 출력만 최적화 | Graph U-Net, 3DGS), $D/lego_psi_pf_ip_gunet_L000.mp4 (탄성 에너지 색, 공통 고정 범위 lbours/lego_psi/range.json)
- R2 원본: $R1/ (frames/frame_XX.pt, records.json, roll_net.pt, net.mp4, psi/), 짧은 시험 $RS/
- Commit: $C (exe/pfnew.sh gunet; 부트 /home/dkta/work/gunet1_boot.sh)
- Config: 기준판 pfip_noin 명령 + --gnn_layer unet --gunet_levels 3 (몸통 h·2h·4h, 층 2+2+2+1+1, 폭 128, 파라미터 1,176,451 vs 기존 1,110,659); Adam 2000/프레임, --ov_obj ip
- 결과: $S2; $S3; $PS
EOM
  texadd "
\\subsection*{증분 포텐셜 프레임별 망 최적화 · 몸통 Graph U-Net (2026-10-24, $C)}
기준판(관성 예측 없음, \\texttt{--ov\\_obj ip}, Adam 2000/프레임)에서 몸통만 Graph U-Net 으로 바꿨다: 수준 0(원 격자 \$h\$)·1(\$2h\$)·2(\$4h\$), 풀링 = 정수 격자좌표를 2 로 내림 나눈 칸으로 묶기(자식 평균), 언풀링 = 부모 복사 후 같은 수준 인코더 특징과 이어붙여 선형(잔차), 층 배분 인코더 2+2, 바닥 2, 디코더 1+1 (메시지 전달 층 8, 폭 128, 파라미터 1{,}176{,}451; 기존 8 층 1{,}110{,}659). $(q "$S2"). $(q "$S3"). $(q "$PS")
"
  touch $D/GUNET1_DONE; echo GUNET1_DONE
  notify "Graph U-Net (1) 끝" "프레임별 증분 포텐셜 L000 위치 오차 $EN (출력만 최적화 11.47%)" --tags white_check_mark; }

# ---------------- (2) 다섯 궤적 학습 ----------------
post5(){ echo "[후처리 (2)] $(date)"; X=$T/g5; mkdir -p $X
  rclone copy $R5 $X --max-depth 1 --include "*.txt" --include "*.json" --include "*.mp4" --include "GUNET5_*" -q || { fail "(2) 내려받기"; return 1; }
  rm -rf $X/psi; rclone copy $R5/psi $X/psi --include "*/psi_*_lego.mp4" --include "*/*.npz" --include "*/*.json" --include "*/*.txt" --include "*/*.log" -q || { fail "(2) psi 내려받기"; return 1; }
  for x in L000 0004; do get $RNI/noinertial_multi_$x.mp4 $X/ni_multi_$x.mp4; done
  cat $X/errors_multi.txt
  EL=$(grep -a '^gunet5_multi_L000 ' $X/errors_multi.txt | grep -o '[0-9.]*%'); E4=$(grep -a '^gunet5_multi_0004 ' $X/errors_multi.txt | grep -o '[0-9.]*%')
  [ -n "$EL" ] && [ -n "$E4" ] || { fail "(2) 오차 값 없음"; return 1; }
  for x in L000 0004; do
    OUT=$D/gunet5_multi_$x.mp4; [ -e $OUT ] && { echo "[건너뜀] $OUT"; continue; }
    [ $x = L000 ] && { EE=$EL; EO=13.26%; TT="L000"; } || { EE=$E4; EO=24.82%; TT="0004 (학습 안 쓴 궤적)"; }
    cell $X/gt_$x.mp4 a "정답"; cell $X/ni_multi_$x.mp4 b "기준 (메시지 패싱 8 층) $EO"; cell $X/gunet5_multi_$x.mp4 c "Graph U-Net 몸통 $EE"
    stack3 $OUT "$TT 다섯 궤적 학습 모델 롤아웃 80프레임 · 관성 예측 없음 · % = 위치 오차" || { fail "(2) 3칸 영상 $x"; return 1; }
  done
  for x in L000 0004; do P=$X/psi/$x; OUT=$D/lego_psi_gunet5_$x.mp4; [ -e $OUT ] && continue
    python3 -c "import json;a=json.load(open('$P/range_noinertial.json'));b=json.load(open('$P/range.json'));assert abs(a['lo']-b['lo'])<1e-9 and abs(a['hi']-b['hi'])<1e-9;print('[범위 동일]',a['lo'],a['hi'])" || { fail "(2) psi 범위 다름 $x"; return 1; }
    [ $x = L000 ] && { EE=$EL; EO=13.26%; } || { EE=$E4; EO=24.82%; }
    psi3 $P ni gu "기준 (메시지 패싱) · 위치 오차 $EO" "Graph U-Net 몸통 · 위치 오차 $EE" $OUT $x || return 1
  done
  for x in L000 0004; do S_=$X/psi/$x/defstats_$x.txt
    [ -f $S_ ] || { fail "(2) defstats 없음 $x"; return 1; }
    grep -q "^# ni = .*/lbours/noinertial/roll_noinertial_multi_$x.pt" $S_ && grep -q "^# gu = .*/lbours/gunet5/roll_gunet5_multi_$x.pt" $S_ || { fail "(2) defstats $x 덤프 경로 다름"; return 1; }
  done
  DS=$D/lego_psi_gunet5_defstats.txt
  [ -e $DS ] || { echo "# 칸: gt = 정답(MPM), ni = 기준 noinertial (메시지 패싱 8 층 128), gu = Graph U-Net 몸통 (이번)"; cat $X/psi/L000/defstats_L000.txt $X/psi/0004/defstats_0004.txt; } > $DS; cat $DS
  CV=$D/gunet5_multi_curves.png; [ -e $CV ] || python3 $REPO/exe/gunet5_plot.py $X/curves.json $CV || { fail "(2) 곡선"; return 1; }
  for f in $D/gunet5_multi_L000.mp4 $D/gunet5_multi_0004.mp4 $D/lego_psi_gunet5_L000.mp4 $D/lego_psi_gunet5_0004.mp4 $DS $CV; do rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $R5/final -q; done
  rclone copy $REPO/exe/gunet_local.sh $R5/final -q; rclone copy $REPO/exe/gunet5_plot.py $R5/final -q
  VAL=$(python3 -c "
import json; M=json.load(open('$X/curves.json'))
for r in ('noinertial','gunet5'):
    print(r, ' '.join(f\"{t} {M[r].get('다궤적적합_롤아웃_위치오차%/'+t,[[0,float('nan')]])[-1][1]:.2f}%\" for t in ('L000','0000','0001','0002','0003')))")
  DSL=$(awk '$1=="ni" || $1=="gu"' $DS | tr -s ' ' | tr '\n' ';')
  cat >> $RR/outputs.md <<EOM

## $(date +%F) gunet5 다섯 궤적 학습 -- 몸통 Graph U-Net (기준 noinertial)
- Path: $D/gunet5_multi_L000.mp4, $D/gunet5_multi_0004.mp4 (정답 | 기준 noinertial | Graph U-Net, 3DGS), $D/lego_psi_gunet5_L000.mp4, $D/lego_psi_gunet5_0004.mp4 (탄성 에너지 색), 변형 통계 $DS, 학습 곡선 $CV
- R2 원본: $R5/ (multi/ 체크포인트·tb·run.log, roll_gunet5_multi_*.pt F32 덤프, h5_*, psi/{L000,0004}/, resume_test.txt, cmdline_diff.txt, curves.json, final/), 짧은 시험 $RS/
- Commit: $C (exe/gunet5.sh; 부트 /home/dkta/work/gunet5_boot.sh, 다시 띄우기 /home/dkta/work/gunet_relaunch.txt)
- Config: noinertial 실행 학습 명령줄 + --gnn_layer unet --gunet_levels 3 (명령줄 자동 대조); 15000 반복 배치 8, 손실 증분 포텐셜; 몸통 h·2h·4h, 층 2+2+2+1+1, 폭 128, 파라미터 1,176,451 (기준 1,110,659)
- 위치 오차: L000 $EL (기준 13.26%), 0004 $E4 (기준 24.82%)
- 마지막 검증(학습 중 정답 시작 80프레임 롤아웃): $(echo "$VAL" | tr '\n' ';')
- 변형 통계 (ni = 기준, gu = 이번; 프레임 |σ-1|max 중앙 99% det<0 ψ평균 ψ중앙): $DSL
- psi 색: log10(psi+1e-3) 고정 범위 lbours/lego_psi/range.json, turbo, 칸별 정규화 없음; 정답·기준 칸은 noinertial psi 렌더 재사용
- 이어 돌리기 시험: $(head -2 $X/resume_test.txt | tr '\n' ' ')
EOM
  texadd "
\\subsection*{다섯 궤적 학습 · 몸통 Graph U-Net (2026-10-24, $C)}
관성 예측 없이 학습한 다섯 궤적 실행(noinertial)과 모든 설정을 같게 두고 몸통만 Graph U-Net(수준 \$h\$·\$2h\$·\$4h\$, 격자 묶기 풀링, 스킵 이어붙이기, 층 2+2+2+1+1, 폭 128)으로 바꿨다.
\\begin{center}
\\begin{tabular}{lcc}
\\hline
위치 오차 & 기준 (메시지 패싱 8 층) & Graph U-Net \\\\
\\hline
L000 & 13.26\\% & $(q $EL) \\\\
0004 (학습 안 쓴 궤적) & 24.82\\% & $(q $E4) \\\\
\\hline
\\end{tabular}
\\end{center}
학습 중 검증(정답 시작 80프레임, 마지막): $(q "$(echo "$VAL" | sed 's/^noinertial/기준/; s/^gunet5/Graph U-Net/' | tr '\n' ';')"). 변형 통계(ni = 기준, gu = Graph U-Net; 프레임, \$|\\sigma-1|\$ 최대·중앙·99\\%, det\$<0\$, \$\\psi\$ 평균·중앙): $(q "$DSL")
"
  touch $D/GUNET5_DONE; echo GUNET5_DONE
  notify "Graph U-Net (2) 끝" "다섯 궤적 학습 L000 $EL (기준 13.26%), 0004 $E4 (기준 24.82%)" --tags white_check_mark; }

relaunch(){ local k=$1 LG=/home/dkta/work/gunet${1}_driver_$(date +%m%d_%H%M).log
  echo "[재기동 ($k)] $(date '+%F %T') -> $LG"
  ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/gunet${k}_boot.sh $LG" 2>&1 | tail -3
  date +%s > /tmp/gunet_last_launch_$k; }
echo "[시작] $(date) 커밋 $C"
while true; do
  date +%s > /tmp/gunet_local.hb; FAILED=0
  ST=$(ssh -o ConnectTimeout=20 $LOGIN "bash /home/dkta/work/gunet_check.sh; /opt/pbs/bin/qstat -answ1 -u dkta 2>/dev/null | awk '\$10==\"R\"{print \"JOB\", \$1}'" 2>&1)
  echo "[점검] $(date '+%F %T') $(echo "$ST" | tr '\n' ' ' | cut -c1-400)"
  JOB=$(grep -o '^JOB [0-9]*' <<<"$ST" | awk '{print $2}')
  VER=$(rclone cat $RS/verdict.txt 2>/dev/null)
  if [ ! -f /tmp/gunet_go_launched ] && [ -n "$JOB" ] && [ "$JOB" != $OLDJOB ]; then
    LG=/home/dkta/work/gunet_go_$(date +%m%d_%H%M).log; echo "[짧은 시험+본 실행 부트] 잡 $JOB -> $LG"
    ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/gunet_go_boot.sh $LG" 2>&1 | tail -3
    touch /tmp/gunet_go_launched; date +%s > /tmp/gunet_last_launch_1; date +%s > /tmp/gunet_last_launch_5
  fi
  if grep -q GUNET_SMOKE_FAIL <<<"$VER" && [ ! -f /tmp/gunet_smoke_fail_seen ]; then
    touch /tmp/gunet_smoke_fail_seen; fail "짧은 시험 기준 미달: $(echo "$VER" | head -1)"; fi
  if grep -q GUNET_FULL_LAUNCHED <<<"$VER" && [ ! -f /tmp/gunet_full_seen ]; then
    touch /tmp/gunet_full_seen; echo "[본 실행 띄움] $VER"; notify "Graph U-Net 본 실행 시작" "$(echo "$VER" | head -1 | cut -c1-150)" --tags rocket; fi
  [ -f $D/GUNET1_DONE ] || { [ -n "$(rclone lsf $R1/.done 2>/dev/null)" ] && post1; }
  [ -f $D/GUNET5_DONE ] || { [ -n "$(rclone lsf $R5/GUNET5_CLUSTER_DONE 2>/dev/null)" ] && post5; }
  grep -q '^G5FAIL' <<<"$ST" && [ ! -f /tmp/gunet_g5fail_seen ] && { touch /tmp/gunet_g5fail_seen; fail "(2) 드라이버: $(grep '^G5FAIL' <<<"$ST")"; }
  if [ -f /tmp/gunet_full_seen ] && ! grep -q 'NOJOB\|PBSERR' <<<"$ST"; then
    for k in 1 5; do
      [ $k = 1 ] && { grep -q '^G1DONE' <<<"$ST" && continue; }
      [ $k = 5 ] && { grep -q '^G5DONE\|^G5FAIL' <<<"$ST" && continue; }
      AGE=$(grep -o "^G${k}HB [0-9]*" <<<"$ST" | awk '{print $2}'); SINCE=$(( $(date +%s) - $(cat /tmp/gunet_last_launch_$k 2>/dev/null || echo 0) ))
      [ -n "$AGE" ] && [ "$AGE" -gt 900 ] && [ "$SINCE" -gt 1500 ] && relaunch $k
    done
  fi
  [ -f $D/GUNET1_DONE ] && [ -f $D/GUNET5_DONE ] && { echo "GUNET_ALL_DONE $(date)"; notify "Graph U-Net 두 실행 끝" "v20261009/pf_ip_gunet_L000.mp4, gunet5_multi_*.mp4" --tags white_check_mark; break; }
  sleep 600
done
