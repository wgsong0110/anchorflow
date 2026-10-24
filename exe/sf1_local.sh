#!/bin/bash
# sf1 (lego L000 한 궤적 학습, 일반 GNN · Graph U-Net; 사용자 10-24) 로컬 감시·재기동·후처리. 로그 /tmp/sf1_local.log (tmux sf1)
#   10 분마다 클러스터 하트비트 확인 (ssh ControlMaster 재사용) -> 죽었고 안 끝났으면 /home/dkta/work/sf1_relaunch.txt 의 같은 부트 (멱등, 이어 돈다)
#   둘 다 SF1_*_CLUSTER_DONE 이면:
#     sf1_{L000,0004}.mp4 (정답 | 일반 GNN 한 궤적 | Graph U-Net 한 궤적, 3DGS, 프레임 번호 크게 빨간색)
#     lego_psi_sf1_{L000,0004}.mp4 (탄성 에너지 색, 공통 고정 범위 lbours/lego_psi/range.json, 분포 패널)
#     sf1_curves.png (네 실행 학습 곡선), sf1_table.txt (다섯 궤적 두 실행 noinertial·gunet5_near 포함)
#   -> outputs.md, learned_results.tex 절, notify, SF1_DONE. 기존 결과 파일은 덮어쓰지 않는다. 아무것도 지우지 않는다.
exec 9>/tmp/sf1_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
C=ccf163a
RL=r2:storage/result/anchorflow/lbours; RG=$RL/sf1_gnn; RU=$RL/sf1_gunet; RNI=$RL/noinertial; R5=$RL/gunet5_near
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; T=/tmp/sf1_dl; mkdir -p $T $D
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
fail(){ echo "[실패] $1 $(date)"; notify "sf1 실패" "$1" --priority high --tags x; }
# psi 3칸 + 색막대 + 분포 패널 (gunet_local.sh psi3 와 같은 식): psi3 <디렉토리> <라벨2> <라벨3> <출력> <제목 접두>
psi3(){ P=$1; OUT=$4; X=$5
  printf '정답 (MPM)' > $T/l_gt.txt; printf '%s' "$2" > $T/l_2.txt; printf '%s' "$3" > $T/l_3.txt
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $P/psi_gt_lego.mp4)
  NS=""; for n in gt g1 u1; do NS="$NS $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $P/psi_${n}_lego.mp4)"; done
  NG=$(echo $NS | cut -d' ' -f1); [ $(echo $NS | tr ' ' '\n' | sort -u | wc -l) = 1 ] || { fail "psi 프레임 수 다름 ($NS) $P"; return 1; }
  TW=$((W*3)); [ -e $P/frames ] && mv $P/frames $P/frames_prev_$(date +%H%M%S)
  ( cd $REPO && python3 exe/psi_dist_vid.py --hist $P/psi_hist_g1.npz --hist2 $P/psi_hist_u1.npz --range $P/range.json --out_dir $P/frames --cbar $P/cbar.png \
    --w $TW --h 560 --lab_gt "정답 (MPM)" --lab_ov "${2%% ·*}" --lab2 "${3%% ·*}" ) || { fail "분포 패널 $P"; return 1; }
  DT="fontfile=$F:fontsize=26:fontcolor=black:box=1:boxcolor=white@0.8:boxborderw=6:expansion=none"
  ffmpeg -y -v error -r 15 -i $P/psi_gt_lego.mp4 -r 15 -i $P/psi_g1_lego.mp4 -r 15 -i $P/psi_u1_lego.mp4 -loop 1 -r 15 -i $P/cbar.png -framerate 15 -i $P/frames/%04d.png \
   -filter_complex "[0:v]setpts=N/15/TB,drawtext=textfile=$T/l_gt.txt:x=20:y=20:$DT[a];[1:v]setpts=N/15/TB,drawtext=textfile=$T/l_2.txt:x=20:y=20:$DT[b];\
[2:v]setpts=N/15/TB,drawtext=textfile=$T/l_3.txt:x=20:y=20:$DT[c];[a][b][c]hstack=inputs=3[top];[3:v]scale=$TW:-2,setsar=1[cb];[4:v]scale=$TW:-2,setsar=1[d];[top][cb][d]vstack=inputs=3:shortest=1,\
drawtext=fontfile=$F:text='$X 프레임 %{frame_num}':x=w-tw-20:y=h-th-20:fontsize=44:fontcolor=red:box=1:boxcolor=white@0.8:boxborderw=6,format=yuv420p" \
   -frames:v $NG -c:v libx264 -crf 18 -r 15 $OUT || { fail "psi ffmpeg $OUT"; return 1; }
  ffprobe -v error -count_frames -show_entries stream=nb_read_frames,width,height -of csv=p=0 $OUT; }

post(){ echo "[후처리] $(date)"
  for m in gnn gunet; do X=$T/$m; mkdir -p $X
    rclone copy $RL/sf1_$m $X --max-depth 1 --include "*.txt" --include "*.json" --include "*.jsonl" --include "*.mp4" --include "SF1_*" -q || { fail "내려받기 $m"; return 1; }
    rclone copy $RL/sf1_$m/psi $X/psi --include "*/psi_*_lego.mp4" --include "*/*.npz" --include "*/*.json" --include "*/*.txt" --include "*/*.log" -q || { fail "psi 내려받기 $m"; return 1; }
  done
  N=$T/ni; G5=$T/g5; mkdir -p $N $G5
  for t in L000 0004; do
    get $RNI/psi/$t/defstats_$t.txt $N/defstats_$t.txt; get $R5/psi/$t/defstats_$t.txt $G5/defstats_$t.txt
    get $RNI/inv_noinertial_multi_$t.jsonl $N/inv_$t.jsonl; get $R5/inv_gunet5_multi_$t.jsonl $G5/inv_$t.jsonl
  done
  get $RNI/errors_multi.txt $N/errors.txt; get $R5/errors_multi.txt $G5/errors.txt
  ER(){ grep -a "^$2 " $1 | grep -o '[0-9.]*%'; }
  EG_L=$(ER $T/gnn/errors.txt sf1_gnn_L000); EG_4=$(ER $T/gnn/errors.txt sf1_gnn_0004)
  EU_L=$(ER $T/gunet/errors.txt sf1_gunet_L000); EU_4=$(ER $T/gunet/errors.txt sf1_gunet_0004)
  EN_L=$(ER $N/errors.txt noinertial_multi_L000); EN_4=$(ER $N/errors.txt noinertial_multi_0004)
  E5_L=$(ER $G5/errors.txt gunet5_multi_L000); E5_4=$(ER $G5/errors.txt gunet5_multi_0004)
  [ -n "$EG_L" ] && [ -n "$EG_4" ] && [ -n "$EU_L" ] && [ -n "$EU_4" ] || { fail "sf1 오차 값 없음"; return 1; }
  echo "[오차] 한 궤적 GNN $EG_L / $EG_4, U-Net $EU_L / $EU_4 | 다섯 궤적 GNN $EN_L / $EN_4, U-Net ${E5_L:-?} / ${E5_4:-?}"
  # ---- 3DGS 3칸 영상 ----
  for x in L000 0004; do OUT=$D/sf1_$x.mp4; [ -e $OUT ] && { echo "[건너뜀] $OUT"; continue; }
    [ $x = L000 ] && { A=$EG_L; B_=$EU_L; TT="L000 (학습 궤적)"; } || { A=$EG_4; B_=$EU_4; TT="0004 (학습 안 쓴 궤적)"; }
    cell $T/gnn/gt_$x.mp4 a "정답"; cell $T/gnn/sf1_gnn_$x.mp4 b "일반 GNN · L000 한 궤적 학습 · $A"; cell $T/gunet/sf1_gunet_$x.mp4 c "Graph U-Net · L000 한 궤적 학습 · $B_"
    stack3 $OUT "$TT 롤아웃 80프레임 · 한 궤적 학습 · 관성 예측 없음 · 바닥 softplus + 미끄러짐 방지 · % = 위치 오차" || { fail "3칸 영상 $x"; return 1; }
  done
  # ---- 탄성 에너지 색 영상 (공통 고정 범위) ----
  for x in L000 0004; do P=$T/psi_$x; mkdir -p $P; OUT=$D/lego_psi_sf1_$x.mp4; [ -e $OUT ] && continue
    for f in psi_gt_lego.mp4 range.json; do cp $T/gnn/psi/$x/$f $P/; done
    cp $T/gnn/psi/$x/psi_g1_lego.mp4 $T/gnn/psi/$x/psi_hist_g1.npz $P/; cp $T/gunet/psi/$x/psi_u1_lego.mp4 $T/gunet/psi/$x/psi_hist_u1.npz $P/
    python3 -c "
import json
r=[json.load(open(p)) for p in ('$T/gnn/psi/$x/range.json','$T/gunet/psi/$x/range.json','$T/gnn/psi/$x/range_noinertial.json')]
assert all(abs(a['lo']-r[0]['lo'])<1e-9 and abs(a['hi']-r[0]['hi'])<1e-9 for a in r); print('[범위 동일]', r[0]['lo'], r[0]['hi'])" || { fail "psi 범위 다름 $x"; return 1; }
    [ $x = L000 ] && { A=$EG_L; B_=$EU_L; } || { A=$EG_4; B_=$EU_4; }
    psi3 $P "일반 GNN 한 궤적 · 위치 오차 $A" "Graph U-Net 한 궤적 · 위치 오차 $B_" $OUT $x || return 1
  done
  # ---- 학습 곡선 ----
  CV=$D/sf1_curves.png; [ -e $CV ] || python3 $REPO/exe/sf1_plot.py $T/gnn/curves.json $T/gunet/curves.json $CV || { fail "곡선"; return 1; }
  # ---- 표 ----
  TB=$D/sf1_table.txt
  [ -e $TB ] || python3 - $T $TB "$EN_L" "$EN_4" "${E5_L:-?}" "${E5_4:-?}" "$EG_L" "$EG_4" "$EU_L" "$EU_4" <<'PY' || { fail "표"; return 1; }
import json, sys
T, out = sys.argv[1], sys.argv[2]
E = sys.argv[3:]
rows = [("일반 GNN · 다섯 궤적 (noinertial)", "ni", f"{T}/ni", E[0], E[1]), ("Graph U-Net · 다섯 궤적 (gunet5_near)", "gu", f"{T}/g5", E[2], E[3]),
        ("일반 GNN · L000 한 궤적 (sf1_gnn)", "g1", f"{T}/gnn/psi", E[4], E[5]), ("Graph U-Net · L000 한 궤적 (sf1_gunet)", "u1", f"{T}/gunet/psi", E[6], E[7])]
def ds(d, k, t):
    p = f"{d}/{t}/defstats_{t}.txt" if d.endswith("/psi") else f"{d}/defstats_{t}.txt"
    r = {}
    for ln in open(p):
        w = ln.split()
        if len(w) == 7 and w[0] == k:
            r[int(w[1])] = (float(w[2]), float(w[3]), int(w[4]), float(w[5]), float(w[6]))
        if len(w) == 7 and w[0] == "gt":
            r[("gt", int(w[1]))] = (float(w[2]), float(w[3]), int(w[4]), float(w[5]), float(w[6]))
    return r
def inv(d, k, t):
    import glob
    fs = [f"{d}/inv_{t}.jsonl"] if not d.endswith("/psi") else glob.glob(f"{d[:-4]}/inv_sf1_*_{t}.jsonl")
    L = [json.loads(l) for l in open(fs[0]) if l.strip()]
    return max(q["inv_tets"] for q in L), max(q.get("inv_particles_cum", 0) for q in L), min(q["det_min"] for q in L), len(L)
lines = ["# lego 롤아웃 80 프레임, 정답 시작. 위치 오차 = 평균 |x-x_gt| / EXT (롤아웃 로그 평균). ψ = 고정 FCR 탄성 에너지 밀도 [Pa] (psi_color_lb, 정답 ψ 평균 40: GT40, 80: GT80)",
         "# 뒤집힘 = AF_INV_LOG: 프레임별 뒤집힌 사면체 수 최대, 누적 뒤집힌 입자 수, det 최소; det<0 = 프레임 80 입자 수 (psi_color_lb)",
         f"{'실행':<40} {'위치오차 L000':>12} {'위치오차 0004':>12} | {'ψ평균40 L000':>12} {'ψ평균80 L000':>12} {'ψ평균80 0004':>12} | {'뒤집힌사면체최대 L000/0004':>22} {'det최소 L000/0004':>18} {'det<0(80) L000/0004':>18}"]
gt = None
for nm, k, d, eL, e4 in rows:
    try:
        sL, s4 = ds(d, k, "L000"), ds(d, k, "0004"); gt = (sL.get(("gt", 40)), sL.get(("gt", 80)), s4.get(("gt", 80)))
        iL, i4 = inv(d, k, "L000"), inv(d, k, "0004")
        lines.append(f"{nm:<40} {eL:>12} {e4:>12} | {sL[40][3]:>12.3e} {sL[80][3]:>12.3e} {s4[80][3]:>12.3e} | {iL[0]:>10d}/{i4[0]:<10d} {iL[2]:>8.3f}/{i4[2]:<8.3f} {sL[80][2]:>8d}/{s4[80][2]:<8d}")
    except Exception as ex:
        lines.append(f"{nm:<40} {eL:>12} {e4:>12} | 통계 없음 ({ex})")
if gt and all(gt):
    lines.append(f"{'정답 (MPM)':<40} {'-':>12} {'-':>12} | {gt[0][3]:>12.3e} {gt[1][3]:>12.3e} {gt[2][3]:>12.3e} |")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
  cat $TB
  for f in $D/sf1_L000.mp4 $D/sf1_0004.mp4 $D/lego_psi_sf1_L000.mp4 $D/lego_psi_sf1_0004.mp4 $CV $TB; do
    rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $RG/final -q; rclone copy $f $RU/final -q; done
  rclone copy $REPO/exe/sf1_local.sh $RG/final -q; rclone copy $REPO/exe/sf1_plot.py $RG/final -q
  VAL=$(python3 -c "
import json; M={}; [M.update(json.load(open(p))) for p in ('$T/gnn/curves.json','$T/gunet/curves.json')]
for r,k in (('sf1_gnn','상태적합_롤아웃/위치오차_평균%'),('sf1_gunet','상태적합_롤아웃/위치오차_평균%')):
    s=M[r][k]; b=min(s,key=lambda p:p[1]); print(f'{r} 마지막 {s[-1][1]:.2f}% 최저 {b[1]:.2f}% (반복 {b[0]+1})')")
  RT=$(for m in gnn gunet; do head -1 $T/$m/resume_test.txt | cut -c1-260; done | tr '\n' ' ')
  cat >> $RR/outputs.md <<EOM

## $(date +%F) sf1: lego L000 한 궤적 학습 (일반 GNN · Graph U-Net), 바닥 softplus 0.01 + 미끄러짐 방지, 관성 예측 없음
- Path: $D/sf1_L000.mp4, $D/sf1_0004.mp4 (정답 | 일반 GNN 한 궤적 | Graph U-Net 한 궤적, 3DGS), $D/lego_psi_sf1_L000.mp4, $D/lego_psi_sf1_0004.mp4 (탄성 에너지 색, 공통 고정 범위 lbours/lego_psi/range.json), 학습 곡선 $CV, 표 $TB
- R2 원본: $RG/, $RU/ (sf/ 체크포인트·tb·run.log, roll_sf1_*_*.pt F32 덤프, inv_/tun_*.jsonl, h5_*, psi/{L000,0004}/, resume_test.txt, base_equiv.txt, cmdline.txt, curves.json, final/)
- Commit: $C (exe/sf1.sh gnn|gunet; 부트 /home/dkta/work/sf1_{gnn,gunet}_run_boot.sh, 다시 띄우기 /home/dkta/work/sf1_relaunch.txt)
- Config: 다섯 궤적 실행(noinertial / gunet5_near)이 실제 실행한 학습 명령줄 인자 그대로 (--bc_softplus 0.01 --bc_stick 0.01 --bc_stick_xn, --net_inertial 없음; U-Net 은 + --gnn_layer unet --gunet_levels 3 --bc_feat near --bc_near_h 1.0); 바꾼 것 = AF_STATE_FIT_MULTI(다섯 궤적) -> AF_STATE_FIT=lbours/spstick/roll_ov_L000.pt:15000:8, --data multi5 -> one/L000, --hold_traj -> L000
- 위치 오차: 일반 GNN L000 $EG_L, 0004 $EG_4 | Graph U-Net L000 $EU_L, 0004 $EU_4 (다섯 궤적: 일반 GNN $EN_L / $EN_4, Graph U-Net ${E5_L:-?} / ${E5_4:-?})
- 학습 중 검증 (L000 80 프레임): $(echo "$VAL" | tr '\n' ';')
- 이어 돌리기 시험: $RT
EOM
  TBT=$(sed 's/%/\\%/g; s/&/\\&/g' $TB | grep -v '^#')
  texadd "
\\subsection*{L000 한 궤적 학습 · 일반 GNN 과 Graph U-Net (2026-10-24, $C)}
관성 예측 없이 학습한 다섯 궤적 두 실행(일반 GNN: noinertial, Graph U-Net: gunet5\\_near)의 학습 명령을 그대로 쓰고 학습 데이터만 L000 한 궤적(같은 출력만 최적화 상태 덤프)으로 바꿨다. 바닥은 softplus 0.01 + 미끄러짐 방지(\\texttt{--bc\\_stick 0.01 --bc\\_stick\\_xn}), 15000 반복, 배치 8, 손실 증분 포텐셜.
\\begin{center}
\\begin{tabular}{lcccc}
\\hline
위치 오차 & GNN 다섯 궤적 & U-Net 다섯 궤적 & GNN 한 궤적 & U-Net 한 궤적 \\\\
\\hline
L000 (학습 궤적) & $(q $EN_L) & $(q ${E5_L:-?}) & $(q $EG_L) & $(q $EU_L) \\\\
0004 (학습 안 쓴 궤적) & $(q $EN_4) & $(q ${E5_4:-?}) & $(q $EG_4) & $(q $EU_4) \\\\
\\hline
\\end{tabular}
\\end{center}
학습 중 검증(L000 정답 시작 80 프레임): $(q "$(echo "$VAL" | sed 's/^sf1_gnn/일반 GNN/; s/^sf1_gunet/Graph U-Net/' | tr '\n' ';')").
탄성 에너지·뒤집힘 (프레임 40·80 \$\\psi\$ 평균 [Pa], 뒤집힌 사면체 최대, det 최소, 프레임 80 det\$<0\$ 입자):
{\\scriptsize\\begin{verbatim}
$(grep -v '^#' $TB)
\\end{verbatim}}
"
  touch $D/SF1_DONE; echo SF1_DONE
  notify "sf1 한 궤적 학습 끝" "일반 GNN L000 $EG_L / 0004 $EG_4, Graph U-Net L000 $EU_L / 0004 $EU_4 (다섯 궤적 GNN $EN_L / $EN_4)" --tags white_check_mark; }

relaunch(){ local m=$1 LG=/home/dkta/work/sf1_${1}_run_$(date +%m%d_%H%M).log
  echo "[재기동 $m] $(date '+%F %T') -> $LG"
  ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/sf1_${m}_run_boot.sh $LG" 2>&1 | tail -3
  date +%s > /tmp/sf1_last_launch_$m; }
echo "[시작] $(date) 커밋 $C"
for m in gnn gunet; do [ -f /tmp/sf1_last_launch_$m ] || date +%s > /tmp/sf1_last_launch_$m; done
while true; do
  [ -f $D/SF1_DONE ] && { echo "SF1_DONE (이미) $(date)"; break; }
  ST=$(ssh -o ConnectTimeout=20 $LOGIN 'W=/home/dkta/work; now=$(date +%s)
    /opt/pbs/bin/qstat -u dkta >/dev/null 2>&1 || echo PBSERR
    /opt/pbs/bin/qstat -answ1 -u dkta 2>/dev/null | grep -q " R " || echo NOJOB
    for m in gnn gunet; do U=$W/lbours/sf1_$m; M=$(echo $m | tr a-z A-Z)
      [ -f $U/SF1_${M}_CLUSTER_DONE ] && echo "${m}DONE"; [ -f $U/SF1_${M}_FAIL ] && echo "${m}FAIL $(head -c 200 $U/SF1_${M}_FAIL)"
      echo "${m}HB $(( now - $(cat $U/driver.hb 2>/dev/null || echo 0) ))"
      echo "${m}IT $(grep -a "상태 적합\] [0-9]*/15000" $U/sf/run.log 2>/dev/null | tail -1 | cut -c1-80)"; done' 2>&1)
  echo "[점검] $(date '+%F %T') $(echo "$ST" | tr '\n' ' ' | cut -c1-500)"
  for m in gnn gunet; do
    grep -q "^${m}FAIL" <<<"$ST" && [ ! -f /tmp/sf1_fail_seen_$m ] && { touch /tmp/sf1_fail_seen_$m; fail "$m 드라이버: $(grep "^${m}FAIL" <<<"$ST")"; }
  done
  if ! grep -q 'NOJOB\|PBSERR\|Connection\|timed out' <<<"$ST"; then
    for m in gnn gunet; do
      grep -q "^${m}DONE\|^${m}FAIL" <<<"$ST" && continue
      AGE=$(grep -o "^${m}HB [0-9]*" <<<"$ST" | awk '{print $2}'); SINCE=$(( $(date +%s) - $(cat /tmp/sf1_last_launch_$m 2>/dev/null || echo 0) ))
      [ -n "$AGE" ] && [ "$AGE" -gt 900 ] && [ "$SINCE" -gt 1500 ] && relaunch $m
    done
  fi
  if [ -n "$(rclone lsf $RG/SF1_GNN_CLUSTER_DONE 2>/dev/null)" ] && [ -n "$(rclone lsf $RU/SF1_GUNET_CLUSTER_DONE 2>/dev/null)" ]; then
    NP=$(( $(cat /tmp/sf1_post_tries 2>/dev/null || echo 0) + 1 )); echo $NP > /tmp/sf1_post_tries
    post && { echo "SF1_DONE $(date)"; break; }
    [ $NP -ge 3 ] && { echo "SF1_POST_FAILED 후처리 3 번 실패 $(date)"; notify "sf1 후처리 3 번 실패" "로그 /tmp/sf1_local.log" --priority high --tags x; break; }
  fi
  sleep 600
done
