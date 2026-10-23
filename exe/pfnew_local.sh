#!/bin/bash
# exe/pfnew.sh 다섯 판(nodestar nodealt floorpe floorcontact noin) 로컬 감시·후처리 (사용자 10-23). 로그 /tmp/pfnew_local.log (tmux pfnew)
#   10 분마다 R2 의 <판>/<T>/.done 만 본다 (ssh 안 씀). 한 판의 L000·0004 가 둘 다 끝나면:
#   pf_<이름>_<T>.mp4 (정답 | 출력만 최적화 | 이번 것, 3DGS), lego_psi_pf_<이름>_<T>.mp4 (탄성 에너지 색 + 색막대 + 분포 패널, 공통 고정 범위),
#   outputs.md·learned_results.tex 에 덧붙이고 알림. 기존 결과 파일은 덮어쓰지 않는다.
exec 9>/tmp/pfnew_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
RL=r2:storage/result/anchorflow/lbours; RSP=$RL/spstick
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; T=/tmp/pfnew_dl; mkdir -p $T $D
TEX=/home/wgsong/workspace/writing/anchorflow/learned_results.tex
REPO=/home/wgsong/workspace/execute/anchorflow/master
F=$(fc-list :lang=ko file | head -1 | cut -d: -f1)
KS=${PFNEW_KS:-"nodestar nodealt floorpe floorcontact noin"}
declare -A DIR=([nodestar]=nodestar [nodealt]=nodealt [floorpe]=pfip_floorpe [floorcontact]=pfip_floorcontact [noin]=pfip_noin)
declare -A VN=([nodestar]=nodestar [nodealt]=nodealt [floorpe]=ip_floorpe [floorcontact]=ip_floorcontact [noin]=ip_noin)
declare -A NM=([nodestar]="모델 상태 L-BFGS 목표 노드 L2" [nodealt]="L-BFGS·Adam 번갈아"
               [floorpe]="증분 포텐셜 · 바닥 위치 인코딩" [floorcontact]="증분 포텐셜 · 접촉 셀만 바닥" [noin]="증분 포텐셜 · 관성 예측 없음")
declare -A DS=([nodestar]="매 프레임 모델 상태에서 L-BFGS 500 으로 노드 출력 u* 를 풀고 망을 노드 L2 로 따라가게 한 뒤 옮김"
               [nodealt]="매 프레임 L-BFGS 10 회 -> 망 Adam 50 회 노드 L2 따라가기를 40 바퀴 한 뒤 옮김"
               [floorpe]="증분 포텐셜 프레임별 망 최적화 · 바닥 거리 원래 값 + sin/cos 6 주파수 · 관성 예측 없음"
               [floorcontact]="증분 포텐셜 프레임별 망 최적화 · 바닥 정보는 바닥에 닿은 셀 꼭짓점만 · 관성 예측 없음"
               [noin]="증분 포텐셜 프레임별 망 최적화 · 바닥 입력 원래 그대로 · 관성 예측 없음")
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
fail(){ echo "[실패] $1 $(date)"; notify "pfnew 후처리 실패" "$1" --priority high --tags x; FAILED=1; }
summ(){ python3 - "$1" "$2" <<'PY'
import json, sys, numpy as np
r = json.load(open(sys.argv[1])); k = sys.argv[2]
assert len(r) == 80, len(r)
pe = 100 * np.mean([q["pos_err_model"] for q in r])
ra = [q["ratio_E_Estar"] for q in r if q.get("ratio_E_Estar") is not None]
nu = [q["node_vs_ustar"]["rms"] for q in r if "node_vs_ustar" in q]
s = f"{pe:.2f}%|E/E* 중앙 {np.median(ra):.2f}" + (f", 노드-u* 상대 rms 평균 {100*np.mean(nu):.3f}% EXT (중앙 {100*np.median(nu):.3f}%)" if nu else "")
s += f", det 최소 {min(q['F_stats']['det_min'] for q in r):.3f}"
if k == "nodealt":
    a0 = np.median([q["alt"][-1]["E_lbfgs_after"] / max(q["alt"][-1]["E_lbfgs_before"], 1e-30) for q in r])
    s += f", 끝 바퀴 L-BFGS E 후/전 중앙 {a0:.3f}"
print(s)
PY
}
post(){ k=$1; vn=${VN[$k]}; dd=${DIR[$k]}; declare -A EN; declare -A RT
  echo "[후처리 $k] $(date)"
  for x in L000 0004; do
    rclone copy $RL/$dd/$x $T/$dd/$x --max-depth 1 --include "records.json" --include "net.mp4" --include "cmd.txt" -q
    rclone copy $RL/$dd/$x/psi $T/$dd/$x/psi --include "psi_*_lego.mp4" --include "*.npz" --include "*.json" --include "*.txt" --include "*.log" -q
    [ -f $T/ov_$x.mp4 ] || rclone copyto $RSP/ov_$x.mp4 $T/ov_$x.mp4 -q; [ -f $T/gt_$x.mp4 ] || rclone copyto $RSP/gt_$x.mp4 $T/gt_$x.mp4 -q
    o=$(summ $T/$dd/$x/records.json $k) || { fail "$k $x 기록"; return 1; }
    EN[$x]=${o%%|*}; RT[$x]=${o#*|}; done
  C=$(grep -ao '커밋 [0-9a-f]*' $T/$dd/L000/cmd.txt | tail -1 | cut -d' ' -f2)
  for x in L000 0004; do
    [ $x = L000 ] && { EO=11.47%; TT="L000"; } || { EO=6.92%; TT="0004"; }
    OUT=$D/pf_${vn}_$x.mp4
    if [ -e $OUT ]; then echo "[건너뜀] $OUT 이미 있음"; else
      cell $T/gt_$x.mp4 a "정답"; cell $T/ov_$x.mp4 b "출력만 최적화 · 위치 오차 평균 $EO"; cell $T/$dd/$x/net.mp4 c "${NM[$k]} · 위치 오차 평균 ${EN[$x]}"
      ffmpeg -loglevel error -y -i $T/c_a.mp4 -i $T/c_b.mp4 -i $T/c_c.mp4 -filter_complex "hstack=inputs=3:shortest=1" -c:v libx264 -pix_fmt yuv420p $T/stack.mp4 \
       && ffmpeg -loglevel error -y -i $T/stack.mp4 -vf "pad=iw:ih+44:0:44:white,$(lab "$TT 80프레임 · ${DS[$k]}" title),drawtext=fontfile=$F:text='프레임 %{frame_num}':x=w-tw-14:y=h-th-12:fontsize=44:fontcolor=red" -c:v libx264 -pix_fmt yuv420p $OUT \
       && ls -la $OUT || { fail "$k 3칸 영상 $x"; return 1; }; fi
    P=$T/$dd/$x/psi; OUT=$D/lego_psi_pf_${vn}_$x.mp4
    if [ -e $OUT ]; then echo "[건너뜀] $OUT 이미 있음"; continue; fi
    python3 -c "import json;a=json.load(open('$P/range_spstick.json'));b=json.load(open('$P/range.json'));assert abs(a['lo']-b['lo'])<1e-9 and abs(a['hi']-b['hi'])<1e-9;print('[범위 동일]',a['lo'],a['hi'])" || { fail "$k psi 범위 다름 $x"; return 1; }
    printf '정답 (MPM)' > $T/l_gt.txt; printf '출력만 최적화 · %s' "$EO" > $T/l_sp.txt; printf '%s · %s' "${NM[$k]}" "${EN[$x]}" > $T/l_k.txt
    W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $P/psi_gt_lego.mp4)
    NS=""; for n in gt sp $k; do NS="$NS $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $P/psi_${n}_lego.mp4)"; done
    NG=$(echo $NS | cut -d' ' -f1); [ $(echo $NS | tr ' ' '\n' | sort -u | wc -l) = 1 ] || { fail "$k $x psi 프레임 수 다름 ($NS)"; return 1; }
    TW=$((W*3)); rm -rf $P/frames
    ( cd $REPO && python3 exe/psi_dist_vid.py --hist $P/psi_hist_sp.npz --hist2 $P/psi_hist_$k.npz --range $P/range.json --out_dir $P/frames --cbar $P/cbar.png \
      --w $TW --h 560 --lab_gt "정답 (MPM)" --lab_ov "출력만 최적화" --lab2 "${NM[$k]}" ) || { fail "$k 분포 패널 $x"; return 1; }
    DT="fontfile=$F:fontsize=26:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6:expansion=none"
    ffmpeg -y -v error -r 15 -i $P/psi_gt_lego.mp4 -r 15 -i $P/psi_sp_lego.mp4 -r 15 -i $P/psi_${k}_lego.mp4 -loop 1 -r 15 -i $P/cbar.png -framerate 15 -i $P/frames/%04d.png \
     -filter_complex "[0:v]setpts=N/15/TB,drawtext=textfile=$T/l_gt.txt:x=20:y=20:$DT[a];[1:v]setpts=N/15/TB,drawtext=textfile=$T/l_sp.txt:x=20:y=20:$DT[b];\
[2:v]setpts=N/15/TB,drawtext=textfile=$T/l_k.txt:x=20:y=20:$DT[c];[a][b][c]hstack=inputs=3[top];[3:v]scale=$TW:-2,setsar=1[cb];[4:v]scale=$TW:-2,setsar=1[d];[top][cb][d]vstack=inputs=3:shortest=1,\
drawtext=fontfile=$F:text='$x 프레임 %{frame_num}':x=w-tw-20:y=h-th-20:fontsize=44:fontcolor=red:box=1:boxcolor=white@0.8:boxborderw=6,format=yuv420p" \
     -frames:v $NG -c:v libx264 -crf 18 -r 15 $OUT || { fail "$k psi ffmpeg $x"; return 1; }
  done
  for f in $D/pf_${vn}_L000.mp4 $D/pf_${vn}_0004.mp4 $D/lego_psi_pf_${vn}_L000.mp4 $D/lego_psi_pf_${vn}_0004.mp4; do rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $RL/$dd/final -q; done
  PS=$(for x in L000 0004; do awk -v k=$k -v x=$x '$1==k && ($2==40||$2==80){printf "%s 프레임 %s ψ 평균 %s det<0 %s; ", x, $2, $6, $5}' $T/$dd/$x/psi/defstats_$x.txt; done)
  cat >> $RR/outputs.md <<EOM

## $(date +%F) pf_${vn}: ${NM[$k]} (망 파라미터 프레임별 최적화)
- Path: $D/pf_${vn}_L000.mp4, $D/pf_${vn}_0004.mp4 (정답 | 출력만 최적화 | 이번 것, 3DGS), $D/lego_psi_pf_${vn}_L000.mp4, $D/lego_psi_pf_${vn}_0004.mp4 (탄성 에너지 색, 공통 고정 범위 lbours/lego_psi/range.json)
- R2 원본: $RL/$dd/{L000,0004}/ (frames/frame_XX.pt, records.json, roll_net.pt, net.mp4, psi/)
- Commit: $C (exe/pfnew.sh $k)
- Config: ${DS[$k]}; 명령 = pf3obj ip 명령에서 --net_inertial 뺌 (기존 pf3obj 실행은 --net_inertial 있었음)
- 결과: 위치 오차 평균 L000 ${EN[L000]}, 0004 ${EN[0004]} (출력만 최적화 11.47% / 6.92%); L000 ${RT[L000]}; 0004 ${RT[0004]}; $PS
EOM
  texadd "
\\subsection*{$(q "${NM[$k]}") (2026-10-23, $C)}
$(q "${DS[$k]}"). 기존 pf3obj 실행과 달리 관성 예측 더하기(--net\\_inertial)가 없다. 위치 오차 평균 L000 $(q ${EN[L000]}), 0004 $(q ${EN[0004]}) (출력만 최적화 11.47\\% / 6.92\\%). L000: $(q "${RT[L000]}"). 0004: $(q "${RT[0004]}"). $(q "$PS")
"
  touch $D/PFNEW_${k}_DONE
  notify "pfnew $k 끝" "L000 ${EN[L000]}, 0004 ${EN[0004]}" --tags white_check_mark
}
while true; do
  left=0; FAILED=0
  for k in $KS; do
    [ -f $D/PFNEW_${k}_DONE ] && continue
    ok=1; for x in L000 0004; do rclone lsf $RL/${DIR[$k]}/$x/.done >/dev/null 2>&1 && [ -n "$(rclone lsf $RL/${DIR[$k]}/$x/.done 2>/dev/null)" ] || ok=0; done
    if [ $ok = 1 ]; then post $k; [ -f $D/PFNEW_${k}_DONE ] || left=$((left+1)); else left=$((left+1)); fi
  done
  echo "[틱] $(date '+%F %T') 남은 판 $left"
  [ $left = 0 ] && { echo PFNEW_ALL_DONE; notify "pfnew 다섯 판 후처리 끝" "v20261009/pf_*" --tags white_check_mark; break; }
  [ $FAILED = 1 ] && { echo "[멈춤] 후처리 실패"; break; }
  sleep 600
done
