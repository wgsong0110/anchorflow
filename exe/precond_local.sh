#!/bin/bash
# exe/pfnew.sh 출력 전처리 세 판 prediag · prems · prelbfgs (L000 한 궤적) 로컬 감시·후처리 (사용자 10-23). 로그 /tmp/precond_local.log (tmux precond)
#   10 분마다 R2 의 lbours/{pfip_prediag,pfip_prems,pfip_prelbfgs,pfip_noin}/L000/.done 만 본다 (ssh 안 씀). 한 판과 기준판 noin 이 끝나면:
#   pf_ip_<판>_L000.mp4 (정답 | 관성 예측 없는 기준판 | 이번 것, 3DGS, 프레임 번호 빨간색),
#   lego_psi_pf_ip_<판>_L000.mp4 (탄성 에너지 색 정답 | 기준판 | 이번 것 + 색막대 + 분포 패널, 공통 고정 범위),
#   outputs.md·learned_results.tex 에 덧붙이고 알림. 셋 다 끝나면 기준판 대비 비교표 (위치 오차 평균, E/E* 중앙, 프레임 40·80 ψ 평균·det<0)
#   와 러닝 커브 페이지 precond/precond_curve.html (네 판 곡선 겹침 + 스텝 슬라이더 + 정답|기준판|세 판 프레임).
#   기존 결과 파일은 덮어쓰지 않는다.
exec 9>/tmp/precond_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
RL=r2:storage/result/anchorflow/lbours; RSP=$RL/spstick
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; T=/tmp/precond_dl; mkdir -p $T $D
TEX=/home/wgsong/workspace/writing/anchorflow/learned_results.tex
REPO=/home/wgsong/workspace/execute/anchorflow/master
F=$(fc-list :lang=ko file | head -1 | cut -d: -f1)
X=L000
KS="prediag prems prelbfgs"
declare -A DIR=([prediag]=pfip_prediag [prems]=pfip_prems [prelbfgs]=pfip_prelbfgs [noin]=pfip_noin)
declare -A NM=([prediag]="출력 전처리 · 대각" [prems]="출력 전처리 · 다중 해상도" [prelbfgs]="출력 전처리 · L-BFGS 역헤시안")
declare -A DS=([prediag]="증분 포텐셜 프레임별 망 최적화(Adam 2000) · 망 출력 z 를 u_i = z_i·sqrt(중앙 d / d_i), d_i = M_i/h² + 탄성 헤시안 대각 (프레임마다 헤시안-벡터 곱 24 회로 정확히) · 관성 예측 없음"
               [prems]="증분 포텐셜 프레임별 망 최적화(Adam 2000) · 망 출력 z 에 계층 평활 u = (z + Σ 2^l I_l R_l z)/(1+Σ 2^l), 거친 격자 2h·4h·8h 삼선형 · 관성 예측 없음"
               [prelbfgs]="증분 포텐셜 프레임별 망 최적화(Adam 2000) · 프레임마다 노드 출력 L-BFGS 50 회로 곡률 쌍을 모아 출력 기울기에 2-loop 역헤시안 적용 (순전파 그대로) · 관성 예측 없음")
NB="관성 예측 없는 기준판"
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
fail(){ echo "[실패] $1 $(date)"; notify "출력 전처리 후처리 실패" "$1" --priority high --tags x; FAILED=1; }
# 요약 한 줄: "위치오차%|E/E* 중앙|det 최소"
summ(){ python3 - "$1" <<'PY'
import json, sys, numpy as np
r = json.load(open(sys.argv[1]))
assert len(r) == 80, len(r)
pe = 100 * np.mean([q["pos_err_model"] for q in r])
ra = [q["ratio_E_Estar"] for q in r if q.get("ratio_E_Estar") is not None]
print(f"{pe:.2f}%|{np.median(ra):.3f}|{min(q['F_stats']['det_min'] for q in r):.3f}")
PY
}
# defstats 에서 프레임 f 의 (ψ 평균, det<0)
dstat(){ awk -v k=$2 -v f=$3 '$1==k && $2==f{print $6" "$5}' $1; }
fetch(){ local dd=$1
  rclone copy $RL/$dd/$X $T/$dd/$X --max-depth 1 --include "records.json" --include "net.mp4" --include "cmd.txt" -q
  rclone copy $RL/$dd/$X/psi $T/$dd/$X/psi --include "psi_*_lego.mp4" --include "*.npz" --include "*.json" --include "*.txt" --include "*.log" -q; }
post(){ k=$1; vn=ip_$k; dd=${DIR[$k]}; B=$T/pfip_noin/$X
  echo "[후처리 $k] $(date)"
  fetch $dd; fetch pfip_noin
  [ -f $T/gt_$X.mp4 ] || rclone copyto $RSP/gt_$X.mp4 $T/gt_$X.mp4 -q
  o=$(summ $T/$dd/$X/records.json) || { fail "$k 기록"; return 1; }; EN=${o%%|*}
  ob=$(summ $B/records.json) || { fail "noin 기록"; return 1; }; EB=${ob%%|*}
  C=$(grep -ao '커밋 [0-9a-f]*' $T/$dd/$X/cmd.txt | tail -1 | cut -d' ' -f2)
  OUT=$D/pf_${vn}_$X.mp4
  if [ -e $OUT ]; then echo "[건너뜀] $OUT 이미 있음"; else
    cell $T/gt_$X.mp4 a "정답"; cell $B/net.mp4 b "$NB · 위치 오차 평균 $EB"; cell $T/$dd/$X/net.mp4 c "${NM[$k]} · 위치 오차 평균 $EN"
    ffmpeg -loglevel error -y -i $T/c_a.mp4 -i $T/c_b.mp4 -i $T/c_c.mp4 -filter_complex "hstack=inputs=3:shortest=1" -c:v libx264 -pix_fmt yuv420p $T/stack.mp4 \
     && ffmpeg -loglevel error -y -i $T/stack.mp4 -vf "pad=iw:ih+44:0:44:white,$(lab "$X 80프레임 · ${DS[$k]}" title),drawtext=fontfile=$F:text='프레임 %{frame_num}':x=w-tw-14:y=h-th-12:fontsize=44:fontcolor=red" -c:v libx264 -pix_fmt yuv420p $OUT \
     && ls -la $OUT || { fail "$k 3칸 영상"; return 1; }; fi
  P=$T/$dd/$X/psi; PB=$B/psi; OUT=$D/lego_psi_pf_${vn}_$X.mp4
  if [ -e $OUT ]; then echo "[건너뜀] $OUT 이미 있음"; else
    python3 -c "import json;a=json.load(open('$PB/range.json'));b=json.load(open('$P/range.json'));assert abs(a['lo']-b['lo'])<1e-9 and abs(a['hi']-b['hi'])<1e-9;print('[범위 동일]',a['lo'],a['hi'])" || { fail "$k psi 범위 다름"; return 1; }
    printf '정답 (MPM)' > $T/l_gt.txt; printf '%s · %s' "$NB" "$EB" > $T/l_b.txt; printf '%s · %s' "${NM[$k]}" "$EN" > $T/l_k.txt
    W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $P/psi_gt_lego.mp4)
    NS=""; for f in $P/psi_gt_lego.mp4 $PB/psi_noin_lego.mp4 $P/psi_${k}_lego.mp4; do NS="$NS $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $f)"; done
    NG=$(echo $NS | cut -d' ' -f1); [ $(echo $NS | tr ' ' '\n' | sort -u | wc -l) = 1 ] || { fail "$k psi 프레임 수 다름 ($NS)"; return 1; }
    TW=$((W*3)); rm -rf $P/frames_vs_noin
    ( cd $REPO && python3 exe/psi_dist_vid.py --hist $PB/psi_hist_noin.npz --hist2 $P/psi_hist_$k.npz --range $P/range.json --out_dir $P/frames_vs_noin --cbar $P/cbar.png \
      --w $TW --h 560 --lab_gt "정답 (MPM)" --lab_ov "$NB" --lab2 "${NM[$k]}" ) || { fail "$k 분포 패널"; return 1; }
    DT="fontfile=$F:fontsize=26:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6:expansion=none"
    ffmpeg -y -v error -r 15 -i $P/psi_gt_lego.mp4 -r 15 -i $PB/psi_noin_lego.mp4 -r 15 -i $P/psi_${k}_lego.mp4 -loop 1 -r 15 -i $P/cbar.png -framerate 15 -i $P/frames_vs_noin/%04d.png \
     -filter_complex "[0:v]setpts=N/15/TB,drawtext=textfile=$T/l_gt.txt:x=20:y=20:$DT[a];[1:v]setpts=N/15/TB,drawtext=textfile=$T/l_b.txt:x=20:y=20:$DT[b];\
[2:v]setpts=N/15/TB,drawtext=textfile=$T/l_k.txt:x=20:y=20:$DT[c];[a][b][c]hstack=inputs=3[top];[3:v]scale=$TW:-2,setsar=1[cb];[4:v]scale=$TW:-2,setsar=1[d];[top][cb][d]vstack=inputs=3:shortest=1,\
drawtext=fontfile=$F:text='$X 프레임 %{frame_num}':x=w-tw-20:y=h-th-20:fontsize=44:fontcolor=red:box=1:boxcolor=white@0.8:boxborderw=6,format=yuv420p" \
     -frames:v $NG -c:v libx264 -crf 18 -r 15 $OUT || { fail "$k psi ffmpeg"; return 1; }
  fi
  for f in $D/pf_${vn}_$X.mp4 $D/lego_psi_pf_${vn}_$X.mp4; do rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $RL/$dd/final -q; done
  PS=$(for f in 40 80; do s=($(dstat $P/defstats_$X.txt $k $f)); echo -n "프레임 $f ψ 평균 ${s[0]} det<0 ${s[1]}; "; done)
  PSB=$(for f in 40 80; do s=($(dstat $PB/defstats_$X.txt noin $f)); echo -n "프레임 $f ψ 평균 ${s[0]} det<0 ${s[1]}; "; done)
  R=${o#*|}; RB=${ob#*|}
  cat >> $RR/outputs.md <<EOM

## $(date +%F) pf_${vn}: ${NM[$k]} (망 파라미터 프레임별 최적화, L000)
- Path: $D/pf_${vn}_$X.mp4 (정답 | $NB | 이번 것, 3DGS), $D/lego_psi_pf_${vn}_$X.mp4 (탄성 에너지 색, 공통 고정 범위 lbours/lego_psi/range.json)
- R2 원본: $RL/$dd/$X/ (frames/frame_XX.pt, records.json, roll_net.pt, net.mp4, psi/)
- Commit: $C (exe/pfnew.sh $k)
- Config: ${DS[$k]}; 명령 = pfip_noin 명령 + 해당 인자만
- 결과: 위치 오차 평균 $EN (기준판 $EB), E/E* 중앙 ${R%%|*} (기준판 ${RB%%|*}), det 최소 ${R#*|} (기준판 ${RB#*|}); $PS (기준판 $PSB)
EOM
  texadd "
\\subsection*{$(q "${NM[$k]}") (2026-10-23, $C)}
$(q "${DS[$k]}"). L000 위치 오차 평균 $(q $EN) (관성 예측 없는 기준판 $(q $EB)), E/E\$^*\$ 중앙 ${R%%|*} (기준판 ${RB%%|*}). $(q "$PS") (기준판 $(q "$PSB"))
"
  touch $D/PRECOND_${k}_DONE
  notify "pfnew $k 끝" "L000 $EN (기준판 $EB), E/E* 중앙 ${R%%|*}" --tags white_check_mark
}
table(){ # 둘 다 끝나면 비교표 (outputs.md·tex·로그)
  local rows="" texrows=""
  for k in noin prediag prems prelbfgs; do
    dd=${DIR[$k]}; o=$(summ $T/$dd/$X/records.json); s40=($(dstat $T/$dd/$X/psi/defstats_$X.txt $k 40)); s80=($(dstat $T/$dd/$X/psi/defstats_$X.txt $k 80))
    nm=$([ $k = noin ] && echo "$NB" || echo "${NM[$k]}")
    rows="$rows| $nm | ${o%%|*} | $(echo $o | cut -d'|' -f2) | ${s40[0]} | ${s40[1]} | ${s80[0]} | ${s80[1]} |
"
    texrows="$texrows$(q "$nm") & $(q ${o%%|*}) & $(echo $o | cut -d'|' -f2) & ${s40[0]} & ${s40[1]} & ${s80[0]} & ${s80[1]} \\\\
"
  done
  printf '%s' "$rows"
  cat >> $RR/outputs.md <<EOM

## $(date +%F) 출력 전처리 세 판 vs 관성 예측 없는 기준판 (L000, 80 프레임)
| 판 | 위치 오차 평균 | E/E* 중앙 | 프레임 40 ψ 평균 | 프레임 40 det<0 | 프레임 80 ψ 평균 | 프레임 80 det<0 |
|---|---|---|---|---|---|---|
$rows
EOM
  texadd "
\\subsection*{출력 전처리 (대각 · 다중 해상도 · L-BFGS 역헤시안) vs 관성 예측 없는 기준판 (L000, 80 프레임)}
\\begin{tabular}{lrrrrrr}
판 & 위치 오차 평균 & \$E/E^*\$ 중앙 & ψ 평균 (40) & det\$<0\$ (40) & ψ 평균 (80) & det\$<0\$ (80) \\\\ \\hline
$texrows\\end{tabular}
"
  touch $D/PRECOND_TABLE_DONE
  notify "출력 전처리 비교표" "$(printf '%s' "$rows" | cut -d'|' -f2,3 | tr '\n' ' ')" --tags white_check_mark
}
curve(){ # 러닝 커브 페이지: 네 판 곡선 겹침 + 스텝 슬라이더 + 프레임 이미지 (정답 | 기준판 | 세 판)
  local PC=$RR/precond; mkdir -p $PC/fr_L000
  for k in noin prediag prems prelbfgs; do cp $T/${DIR[$k]}/$X/records.json $PC/rec_$k.json; done
  cell $T/gt_$X.mp4 g "정답"; cell $T/pfip_noin/$X/net.mp4 n "$NB"
  for k in prediag prems prelbfgs; do cell $T/${DIR[$k]}/$X/net.mp4 $k "${NM[$k]}"; done
  ffmpeg -loglevel error -y -i $T/c_g.mp4 -i $T/c_n.mp4 -i $T/c_prediag.mp4 -i $T/c_prems.mp4 -i $T/c_prelbfgs.mp4 \
    -filter_complex "hstack=inputs=5:shortest=1,scale=1800:-2" -q:v 3 $PC/fr_L000/%03d.jpg || { fail "곡선 페이지 프레임"; return 1; }
  ( cd $REPO && python3 exe/precond_curve.py --rec "$NB=$PC/rec_noin.json" --rec "${NM[prediag]}=$PC/rec_prediag.json" \
      --rec "${NM[prems]}=$PC/rec_prems.json" --rec "${NM[prelbfgs]}=$PC/rec_prelbfgs.json" --frames fr_L000 --out $PC/precond_curve.html ) || { fail "곡선 페이지"; return 1; }
  rclone copy $PC r2:storage/result/anchorflow/precond -q
  cat >> $RR/outputs.md <<EOM

## $(date +%F) precond_curve.html (출력 전처리 세 판 + 기준판 프레임별 러닝 커브 인터랙티브, L000)
- Path: $PC/precond_curve.html (+ fr_L000 프레임 이미지: 정답 | 기준판 | 대각 | 다중 해상도 | L-BFGS, rec_*.json), 같은 형식 ipcurve/ip_curve.html
EOM
  touch $D/PRECOND_CURVE_DONE
}
done_r2(){ [ -n "$(rclone lsf $RL/$1/$X/.done 2>/dev/null)" ]; }
while true; do
  left=0; FAILED=0
  for k in $KS; do
    [ -f $D/PRECOND_${k}_DONE ] && continue
    if done_r2 ${DIR[$k]} && done_r2 pfip_noin; then post $k; [ -f $D/PRECOND_${k}_DONE ] || left=$((left+1)); else left=$((left+1)); fi
  done
  echo "[틱] $(date '+%F %T') 남은 판 $left"
  if [ $left = 0 ]; then
    [ -f $D/PRECOND_TABLE_DONE ] || table
    [ -f $D/PRECOND_CURVE_DONE ] || curve
    echo PRECOND_ALL_DONE; break
  fi
  [ $FAILED = 1 ] && { echo "[멈춤] 후처리 실패"; break; }
  sleep 600
done
