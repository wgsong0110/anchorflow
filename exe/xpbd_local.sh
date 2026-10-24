#!/bin/bash
# XPBD 실행 (사용자 10-24) 로컬 감시·재기동·후처리. 로그 /tmp/xpbd_local.log (tmux xpbd)
#   10 분마다 클러스터 lbours/xpbd 의 하트비트·표식 확인 -> 하트비트 15 분 넘게 멈췄고 안 끝났으면 부트를 job_run.sh(AF_DETACH=1)로 다시 (멱등)
#   끝나면 (XPBD_CLUSTER_DONE): 영상 v20261009/xpbd_{lego_L000,ficus}.mp4 (정답 | 출력만 최적화 | XPBD 다섯, 3DGS, 프레임 번호 빨간색),
#   탄성 색 v20261009/xpbd_psi_{lego_L000,ficus}.mp4 (공통 고정 범위), ovnet/xpbd_cmp.{png,md}, outputs.md, learned_results.tex 절, notify
exec 9>/tmp/xpbd_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
C=${1:?커밋}
RL=r2:storage/result/anchorflow/lbours; RX=$RL/xpbd
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; O=$RR/ovnet; T=/tmp/xpbd_dl; mkdir -p $T $D $O
TEX=/home/wgsong/workspace/writing/anchorflow/learned_results.tex
LOGIN=dkta@147.46.121.39
FONT=$(fc-list :lang=ko file | head -1 | cut -d: -f1)
KEYS="n1k1 n1k10 n4k1 n16k1 n64k1"
declare -A LB=([n1k1]="XPBD N=1 K=1" [n1k10]="XPBD N=1 K=10" [n4k1]="XPBD N=4 K=1" [n16k1]="XPBD N=16 K=1" [n64k1]="XPBD N=64 K=1")
fail(){ echo "[실패] $1 $(date)"; notify "xpbd 실패" "$1" --priority high --tags x; }
nfr(){ ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $1; }
cell(){ # 입력 이름 라벨 목표프레임수 -> $T/c_이름.mp4 (가로 480, 위에 라벨, 짧으면 마지막 프레임 반복)
  local n=$(nfr $1) pad=0; [ $n -lt $4 ] && pad=$(( $4 - n ))
  printf '%s' "$3" > $T/l_$2.txt
  ffmpeg -loglevel error -y -i $1 -vf "scale=480:-2,tpad=stop_mode=clone:stop=$pad,pad=iw:ih+36:0:36:white,drawtext=fontfile=$FONT:textfile=$T/l_$2.txt:expansion=none:x=8:y=7:fontsize=18:fontcolor=black" \
    -frames:v $4 -c:v libx264 -pix_fmt yuv420p $T/c_$2.mp4; }
grid8(){ # 출력 제목 c1..c8 (2 x 4, 8 번째 빈칸 허용 'blank')
  local out=$1 title=$2; shift 2; local ins="" i=0
  for c in "$@"; do ins="$ins -i $T/c_$c.mp4"; done
  printf '%s' "$title" > $T/l_title.txt
  ffmpeg -loglevel error -y $ins -filter_complex "xstack=inputs=8:layout=0_0|w0_0|w0+w1_0|w0+w1+w2_0|0_h0|w0_h0|w0+w1_h0|w0+w1+w2_h0:fill=white,pad=iw:ih+40:0:40:white,drawtext=fontfile=$FONT:textfile=$T/l_title.txt:expansion=none:x=10:y=8:fontsize=20:fontcolor=black,drawtext=fontfile=$FONT:text='프레임 %{frame_num}':x=w-tw-14:y=h-th-12:fontsize=52:fontcolor=red" \
    -c:v libx264 -pix_fmt yuv420p $out; }
blank(){ ffmpeg -loglevel error -y -f lavfi -i color=white:s=${1}x${2}:r=30 -frames:v $3 -c:v libx264 -pix_fmt yuv420p $T/c_blank.mp4; }
post(){ echo "[후처리] $(date)"
  rclone copy $RX/post $T/post -q || { fail "post 내려받기"; return 1; }
  [ -f $T/post/xpbd_cmp.md ] || { fail "표 없음"; return 1; }
  for sc in lego_L000 ficus_v0; do
    if [ $sc = lego_L000 ]; then GTV=$RL/spstick/gt_L000.mp4; OVV=$RL/spstick/ov_L000.mp4; PG=$RX/lego_L000/ov_psi/psi_gt_lego.mp4; PO=$RX/lego_L000/ov_psi/psi_ov_lego.mp4; NF=81; SH=lego; NM=lego_L000
    else GTV=$RL/ficus_v0/gt_ficus.mp4; OVV=$RL/ficus_v0/ov_ficus.mp4; PG=$RL/ficus_v0/psi/psi_gt_ficus.mp4; PO=$RL/ficus_v0/psi/psi_ov_ficus.mp4; NF=126; SH=ficus; NM=ficus; fi
    S=$T/$sc; mkdir -p $S
    rclone copyto $GTV $S/gt.mp4 -q; rclone copyto $OVV $S/ov.mp4 -q; rclone copyto $PG $S/psi_gt.mp4 -q; rclone copyto $PO $S/psi_ov.mp4 -q
    for k in $KEYS; do rclone copyto $RX/$sc/$k/${sc}_$k.mp4 $S/$k.mp4 -q; rclone copyto $RX/$sc/$k/psi/psi_${k}_$SH.mp4 $S/psi_$k.mp4 -q; done
    E=$(python3 - $T/post/$sc.json <<'PY'
import json, sys
R = json.load(open(sys.argv[1]))["runs"]
for k, r in R.items():
    if k == "gt":
        continue
    s = "%.2f%%" % (100 * r["err_mean"]) + (" · 발산 %d" % r["diverged_at"] if r.get("diverged_at") else "")
    print(f"{k}={s}")
PY
)
    declare -A ER=(); while IFS='=' read -r kk vv; do ER[$kk]="$vv"; done <<<"$E"
    for mode in rgb psi; do
      if [ $mode = rgb ]; then OUT=$D/xpbd_$NM.mp4; pre=""; TT="$sc · 3DGS · 정답 | 출력만 최적화 | XPBD (서브스텝 N, 반복 K) · % = 위치 오차 평균"
      else OUT=$D/xpbd_psi_$NM.mp4; pre="psi_"; TT="$sc · 탄성 에너지 밀도 ψ_FCR 색 (모든 칸 공통 고정 범위) · 정답 | 출력만 최적화 | XPBD"; fi
      [ -e $OUT ] && continue
      cell $S/${pre}gt.mp4 g "정답 (PG)" $NF || { fail "$sc 칸 정답"; return 1; }
      cell $S/${pre}ov.mp4 o "출력만 최적화 · ${ER[ov]}" $NF || { fail "$sc 칸 ov"; return 1; }
      read -r WW HH <<<"$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=' ' $T/c_g.mp4)"; blank $WW $HH $NF
      for k in $KEYS; do if [ -f $S/${pre}$k.mp4 ]; then cell $S/${pre}$k.mp4 $k "${LB[$k]} · ${ER[$k]}" $NF
        else printf '%s' "${LB[$k]} · 영상 없음 (${ER[$k]:-덤프 없음})" > $T/l_$k.txt
          ffmpeg -loglevel error -y -i $T/c_blank.mp4 -vf "drawtext=fontfile=$FONT:textfile=$T/l_$k.txt:expansion=none:x=8:y=7:fontsize=18:fontcolor=black" -c:v libx264 -pix_fmt yuv420p $T/c_$k.mp4; fi; done
      grid8 $OUT "$TT" g o n1k1 n1k10 n4k1 n16k1 n64k1 blank || { fail "$sc $mode 영상"; return 1; }
      ls -la $OUT; rclone copy $OUT r2:storage/result/anchorflow/v20261009 -q 2>/dev/null; done
  done
  cp $T/post/xpbd_cmp.png $O/xpbd_cmp.png; cp $T/post/xpbd_cmp.md $O/xpbd_cmp.md
  python3 - "$TEX" $T/post/xpbd_cmp.tex <<'PY'
import sys
p, f = sys.argv[1:]
s = open(p).read(); t = open(f).read()
if "한 스텝을 XPBD 로" not in s:
    s = s.replace(r"\end{document}", t + "\n" + r"\end{document}"); open(p, "w").write(s); print("[tex] 절 추가")
PY
  SUM=$(python3 - $T/post/lego_L000.json $T/post/ficus_v0.json <<'PY'
import json, sys
out = []
for p in sys.argv[1:]:
    R = json.load(open(p)); rr = R["runs"]
    out.append(R["scene"] + " " + ", ".join(f"{k} {100 * r['err_mean']:.2f}%" + (f"(발산 {r['diverged_at']})" if r.get("diverged_at") else "") for k, r in rr.items() if k != "gt"))
print("; ".join(out))
PY
)
  cat >> $RR/outputs.md <<EOM

## $(date +%F) xpbd: 한 스텝을 XPBD 로 (우리 사면체 격자·입자 상태 그대로) -- lego L000 80 프레임, ficus_v0 125 프레임, N∈{1,4,16,64} x K=1 + N=1 K=10
- Path: $D/xpbd_lego_L000.mp4, $D/xpbd_ficus.mp4 (정답 | 출력만 최적화 | XPBD 다섯, 3DGS), $D/xpbd_psi_lego_L000.mp4, $D/xpbd_psi_ficus.mp4 (탄성 색, 공통 고정 범위 lego lbours/lego_psi/range.json, ficus -3.0~4.93), $O/xpbd_cmp.png, $O/xpbd_cmp.md
- R2 원본: $RX/ (씬/키/roll_xpbd.pt (AF_ROLL_DUMP 형식, F float32), stats_xpbd.jsonl, summary_xpbd.json, h5/, psi/, 단위 시험 unit_$C.log, post/)
- Commit: $C (lib/anchorflow/xpbd.py, exe/xpbd_roll.py, exe/xpbd.sh, exe/test_xpbd.py, exe/xpbd_post.py)
- Config: fixed corotated 셀 구속 (C_R = ‖F−R‖, α=1/(2μV); C_V = det F−1, α=1/(λV); 셀 안 입자 F 모음), 결합 2x2, 48 색 GS, 바닥 = 사면체별 최심 입자 비침투 + 정적 마찰 μ_s=∞, ficus = cuboid 노드 고정 + PG 영역 밀도 + grid_v_damping (서브스텝마다)
- 결과: $SUM
EOM
  touch $D/XPBD_POST_DONE; echo "XPBD_DONE $(date)"
  notify "XPBD 끝" "$SUM" --tags white_check_mark; }
relaunch(){ local LG=/home/dkta/work/xpbd_boot_$(date +%m%d_%H%M).log
  echo "[재기동] $(date '+%F %T') -> $LG"
  ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/xpbd_boot.sh $LG" 2>&1 | tail -3
  date +%s > /tmp/xpbd_last_launch; }
echo "[시작] $(date) 커밋 $C"
[ -f /tmp/xpbd_last_launch ] || date +%s > /tmp/xpbd_last_launch
while true; do
  ST=$(ssh -o ConnectTimeout=20 $LOGIN 'O=/home/dkta/work/lbours/xpbd; echo HB $(( $(date +%s) - $(cat $O/driver.hb 2>/dev/null || echo 0) )); ls $O | grep -E "XPBD_|unit_" | tr "\n" " "; echo; ls $O/*/*/DONE 2>/dev/null | wc -l; tail -c 300 $O/XPBD_FAIL 2>/dev/null' 2>&1)
  echo "[점검] $(date '+%F %T') $(echo "$ST" | tr '\n' ' ' | cut -c1-300)"
  if grep -q "kex_exchange\|Connection\|timed out" <<<"$ST"; then sleep 600; continue; fi
  if grep -q "XPBD_FAIL" <<<"$ST"; then fail "클러스터: $(echo "$ST" | tail -1)"; break; fi
  if grep -q "XPBD_CLUSTER_DONE" <<<"$ST"; then [ -f $D/XPBD_POST_DONE ] || post; break; fi
  AGE=$(grep -o "^HB [0-9]*" <<<"$ST" | awk '{print $2}'); SINCE=$(( $(date +%s) - $(cat /tmp/xpbd_last_launch) ))
  [ -n "$AGE" ] && [ "$AGE" -gt 900 ] && [ "$SINCE" -gt 1500 ] && relaunch
  sleep 600
done
echo "XPBD_LOCAL_END $(date)"
