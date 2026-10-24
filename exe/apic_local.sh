#!/bin/bash
# APIC 두 실행 (사용자 10-24) 로컬 감시·재기동·후처리. 로그 /tmp/apic_local.log (tmux apic)
#   A = lbours/pfip_apic/L000 (P2G + APIC C 상태·입력 + 응력 + near 바닥), B = lbours/pfip_p2gnear/L000 (P2G + near 바닥), 커밋 c38a519
#   10 분마다 apic_hb.sh 로 하트비트 -> 죽었고(run.log 15 분 넘게 멈춤) 안 끝났으면 apic_relaunch.sh 를 job_run.sh(AF_DETACH=1)로 (멱등, 이어 돈다)
#   둘 다 끝나면: 영상 pfip_apic_L000.mp4 (정답 | 출력만 최적화 | B | A, 프레임 번호 빨간색), 탄성 색 lego_psi_pfip_apic_L000.mp4 (정답 | B | A, 공통 고정 범위),
#   프레임별 위치 오차·E/E* 그림 pfip_apic_L000_curves.png, outputs.md, learned_results.tex 절 -> APIC_DONE / P2GNEAR_DONE, notify
exec 9>/tmp/apic_local.lock
flock -n 9 || { echo "[중단] 다른 감시가 돌고 있다 $(date)"; exit 0; }
C=c38a519
RL=r2:storage/result/anchorflow/lbours; RA=$RL/pfip_apic/L000; RB=$RL/pfip_p2gnear/L000; RSP=$RL/spstick; RS=$RL/apic_smoke_$C
RR=/home/wgsong/workspace/result/anchorflow; D=$RR/v20261009; T=/tmp/apic_dl; mkdir -p $T $D
TEX=/home/wgsong/workspace/writing/anchorflow/learned_results.tex
REPO=/home/wgsong/workspace/execute/anchorflow/master
LOGIN=dkta@147.46.121.39
F=$(fc-list :lang=ko file | head -1 | cut -d: -f1)
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
fail(){ echo "[실패] $1 $(date)"; notify "apic 실패" "$1" --priority high --tags x; }
summ(){ python3 - $1 <<'PY'
import json, sys, numpy as np
r = json.load(open(sys.argv[1])); assert len(r) == 80, len(r)
print(f"{100 * np.mean([q['pos_err_model'] for q in r]):.2f}%")
print(f"{np.median([q['ratio_E_Estar'] for q in r if q.get('ratio_E_Estar') is not None]):.3f}")
print(f"{min(q['F_stats']['det_min'] for q in r):.3f}")
PY
}
post(){ echo "[후처리] $(date)"; XD=$T/x; mkdir -p $XD/A $XD/B
  for k in A B; do eval R=\$R$k
    rclone copy $R $XD/$k --max-depth 1 --include "records.json" --include "net.mp4" --include "cmd*.txt" -q
    rclone copy $R/psi $XD/$k/psi --include "psi_*_lego.mp4" --include "*.npz" --include "*.json" --include "*.txt" --include "*.log" -q; done
  get $RL/pfip_noin/L000/records.json $XD/noin_records.json
  [ -f $T/ov_L000.mp4 ] || get $RSP/ov_L000.mp4 $T/ov_L000.mp4; [ -f $T/gt_L000.mp4 ] || get $RSP/gt_L000.mp4 $T/gt_L000.mp4
  SA=($(summ $XD/A/records.json)) && SB=($(summ $XD/B/records.json)) && SN=($(summ $XD/noin_records.json)) || { fail "기록 요약"; return 1; }
  echo "A ${SA[*]} | B ${SB[*]} | noin ${SN[*]}"
  OUT=$D/pfip_apic_L000.mp4
  if [ ! -e $OUT ]; then
    cell $T/gt_L000.mp4 a "정답"; cell $T/ov_L000.mp4 b "출력만 최적화 · 11.47%"; cell $XD/B/net.mp4 c "B: P2G + 바닥 near · ${SB[0]}"; cell $XD/A/net.mp4 d "A: P2G + APIC + 응력 + 바닥 near · ${SA[0]}"
    ffmpeg -loglevel error -y -i $T/c_a.mp4 -i $T/c_b.mp4 -i $T/c_c.mp4 -i $T/c_d.mp4 -filter_complex "hstack=inputs=4:shortest=1" -c:v libx264 -pix_fmt yuv420p $T/stack.mp4 \
    && ffmpeg -loglevel error -y -i $T/stack.mp4 -vf "pad=iw:ih+44:0:44:white,$(lab "L000 80프레임 · 증분 포텐셜 프레임별 망 최적화 (Adam 2000/프레임, 관성 예측 없음, sgnn 8 층) · % = 위치 오차 평균" title),drawtext=fontfile=$F:text='프레임 %{frame_num}':x=w-tw-14:y=h-th-12:fontsize=48:fontcolor=red" -c:v libx264 -pix_fmt yuv420p $OUT \
    || { fail "4칸 영상"; return 1; }; ls -la $OUT; fi
  # 탄성 색: A 의 psi 디렉토리에 B 의 것을 합쳐 정답 | B | A (범위 = 공통 고정 lbours/lego_psi/range.json, 둘 다 같은지 확인)
  P=$XD/A/psi; cp $XD/B/psi/psi_p2gnear_lego.mp4 $XD/B/psi/psi_hist_p2gnear.npz $P/
  python3 -c "import json;a=json.load(open('$P/range.json'));b=json.load(open('$XD/B/psi/range.json'));assert abs(a['lo']-b['lo'])<1e-9 and abs(a['hi']-b['hi'])<1e-9;print('[범위 동일]',a['lo'],a['hi'])" || { fail "psi 범위 다름"; return 1; }
  OUT=$D/lego_psi_pfip_apic_L000.mp4
  if [ ! -e $OUT ]; then
    source <(sed -n '/^psi3(){/,/^  ffprobe.*nb_read_frames,width,height.*; }$/p' $REPO/exe/gunet_local.sh)
    psi3 $P p2gnear apic "B: P2G + 바닥 near · ${SB[0]}" "A: P2G + APIC + 응력 + 바닥 near · ${SA[0]}" $OUT L000 || { fail "psi 영상"; return 1; }; fi
  CV=$D/pfip_apic_L000_curves.png
  [ -e $CV ] || python3 $REPO/exe/apic_plot.py $XD/noin_records.json $XD/B/records.json $XD/A/records.json $CV || { fail "그림"; return 1; }
  for f in $D/pfip_apic_L000.mp4 $D/lego_psi_pfip_apic_L000.mp4 $CV; do rclone copy $f r2:storage/result/anchorflow/v20261009 -q; rclone copy $f $RA/final -q; done
  PA=$(awk '$1=="apic" && ($2==40||$2==80){printf "프레임 %s ψ 평균 %s det<0 %s; ", $2, $6, $5}' $XD/A/psi/defstats_L000.txt)
  PB=$(awk '$1=="p2gnear" && ($2==40||$2==80){printf "프레임 %s ψ 평균 %s det<0 %s; ", $2, $6, $5}' $XD/B/psi/defstats_L000.txt)
  cat >> $RR/outputs.md <<EOM

## $(date +%F) pfip_apic / pfip_p2gnear: 증분 포텐셜 프레임별 망 최적화 · P2G 입력 + APIC 아핀 속도 C + 응력 (비교 B = P2G + near 바닥)
- Path: $D/pfip_apic_L000.mp4 (정답 | 출력만 최적화 | B | A, 3DGS), $D/lego_psi_pfip_apic_L000.mp4 (탄성 에너지 색 정답 | B | A, 공통 고정 범위 lbours/lego_psi/range.json), $CV (프레임별 위치 오차·E/E*)
- R2 원본: $RA/, $RB/ (frames/frame_XX.pt -- A 는 state_in/out 에 C, records.json, roll_net.pt -- A 는 C [80,N,3,3], net.mp4, psi/), 짧은 시험 $RS/
- Commit: $C (exe/pfnew.sh apic / p2gnear; 부트 /home/dkta/work/apic_go_boot.sh, 재기동 apic_relaunch.sh)
- Config: 기준판 pfip_noin 명령 (Adam 2000/프레임, --ov_vars net --ov_obj ip, sgnn 8 층 128) + A: --node_feat p2g --apic --p2g_stress --bc_feat near --bc_near_h 1.0 / B: --node_feat p2g --bc_feat near --bc_near_h 1.0
- 결과: A 위치 오차 평균 ${SA[0]}, E/E* 중앙 ${SA[1]}, det 최소 ${SA[2]}; B ${SB[0]}, ${SB[1]}, ${SB[2]}; 기준판 noin ${SN[0]}, ${SN[1]}, ${SN[2]}; A $PA B $PB
EOM
  texadd "
\\subsection*{증분 포텐셜 프레임별 망 최적화 · P2G 입력에 APIC 아핀 속도와 응력 (2026-10-24, $C)}
입자 상태를 \$(x, v, F)\$ 에서 \$(x, v, F, C)\$ 로 늘렸다. \$C_p = (J_p - I)/h\$ (\$J_p\$ = 이번 스텝 셀 사상 야코비안, \$F \\leftarrow J F\$ 에 쓰는 것과 같은 것), 시작 \$C = 0\$ (L000 초기 속도 균일). 노드 입력(A)은 2 차 B-스플라인 P2G 로 \$m_i v_i = \\sum_p w_{ip} m_p (v_p + C_p (x_i - x_p))\$, \$C_i = \\sum_p w_{ip} m_p C_p / m_i\$ (\$C_i h / v_s\$ 9 칸), Kirchhoff 응력 \$\\tau_p = P(F_p) F_p^\\top\$ 의 내부력 \$f_i = -\\sum_p V_p^0 \\tau_p \\nabla w_{ip}\$ (\$\\div E h^2\$, 3 칸)과 질량 가중 평균 \$\\tau_i/E\$ (대칭 6 칸)를 기존 P2G (\$\\log(1+10^3 m_i)\$, \$v_i/v_s\$, \$F_i - I\$) 에 더했다. 바닥 입력은 near. 관성 예측은 \$x + h v\$ 그대로. 비교 B 는 P2G + near 바닥 (APIC·응력 없음).
\\begin{center}
\\begin{tabular}{lccc}
\\hline
 & 위치 오차 평균 & \$E/E^*\$ 중앙 & det 최소 \\\\
\\hline
기준판 (입자 모멘트 입력) & $(q ${SN[0]}) & ${SN[1]} & ${SN[2]} \\\\
B: P2G + 바닥 near & $(q ${SB[0]}) & ${SB[1]} & ${SB[2]} \\\\
A: P2G + APIC + 응력 + 바닥 near & $(q ${SA[0]}) & ${SA[1]} & ${SA[2]} \\\\
\\hline
\\end{tabular}
\\end{center}
출력만 최적화는 11.47\\%. 탄성 에너지 (A): $(q "$PA") (B): $(q "$PB")
"
  touch $D/APIC_POST_DONE; echo "APIC_DONE $(date)"; echo "P2GNEAR_DONE $(date)"
  notify "APIC 두 실행 끝" "L000 위치 오차 A(APIC+응력) ${SA[0]}, B(P2G) ${SB[0]}, 기준판 ${SN[0]}; E/E* 중앙 A ${SA[1]} B ${SB[1]}" --tags white_check_mark; }

relaunch(){ local k=$1 LG=/home/dkta/work/apic_${1}_relaunch_$(date +%m%d_%H%M).log
  echo "[재기동 $k] $(date '+%F %T') -> $LG"
  ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && AF_DETACH=1 bash exe/job_run.sh '/home/dkta/work/apic_relaunch.sh $k' $LG" 2>&1 | tail -3
  date +%s > /tmp/apic_last_launch_$k; }
echo "[시작] $(date) 커밋 $C"
date +%s > /tmp/apic_last_launch_apic; date +%s > /tmp/apic_last_launch_p2gnear
ANOTE=0; BNOTE=0
while true; do
  ST=$(ssh -o ConnectTimeout=20 $LOGIN "bash /home/dkta/work/apic_hb.sh" 2>&1)
  echo "[점검] $(date '+%F %T') $(echo "$ST" | tr '\n' ' ' | cut -c1-300)"
  grep -q "APIC_SMOKE_FAIL\|APIC_NO_GPU" <<<"$ST" && { fail "짧은 시험/띄우기: $(grep 'APIC_' <<<"$ST")"; break; }
  [ $ANOTE = 0 ] && grep -q "^ADONE" <<<"$ST" && { ANOTE=1; echo "[A 클러스터 끝] $(date)"; }
  [ $BNOTE = 0 ] && grep -q "^BDONE" <<<"$ST" && { BNOTE=1; echo "[B 클러스터 끝] $(date)"; }
  if grep -q "APIC_FULL_LAUNCHED" <<<"$ST" && ! grep -q "NOJOB\|PBSERR\|kex_exchange\|Connection" <<<"$ST"; then
    for kk in A:apic B:p2gnear; do n=${kk%%:*}; k=${kk#*:}
      grep -q "^${n}DONE" <<<"$ST" && continue
      AGE=$(grep -o "^${n}HB [0-9]*" <<<"$ST" | awk '{print $2}'); SINCE=$(( $(date +%s) - $(cat /tmp/apic_last_launch_$k 2>/dev/null || echo 0) ))
      [ -n "$AGE" ] && [ "$AGE" -gt 900 ] && [ "$SINCE" -gt 1500 ] && relaunch $k
    done
  fi
  if [ $ANOTE = 1 ] && [ $BNOTE = 1 ] && [ ! -f $D/APIC_POST_DONE ]; then post || true; fi
  [ -f $D/APIC_POST_DONE ] && { echo "APIC_ALL_DONE $(date)"; break; }
  sleep 600
done
