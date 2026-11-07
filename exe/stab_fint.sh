#!/bin/bash
# 절 CM20 묶음 fint: 입자 F 를 프레임 끝 가우스 점 F 보간으로 교체 (--fp_interp, 사용자 승인 11-07 정의 변경) 끔 | 켬 (잡 안, 커밋 디렉토리에서; 출력 $1).
# 레고 (데이터 /dev/shm/dkta/anerr_data/L000.pt, 모두 결정적 --det; 원래 명령 = exe/stab_gqm.sh cmd_of 그대로):
#   (B) 장벽 K 2 (채택 판)·4·8·16, (A) 원래 고착 fa, 노드 −z 클램프 nc, −z 저항 강 vs (k 9) -- 끔 off / 켬 on
#   --gq_mass 함께: (B) K 2·4 -- gq (--gq_mass 만) / gqon (--gq_mass --fp_interp)
#   모든 판 + 진단 --pdiag --qjd --fhash (결과 불변: 프레임 끝 입자 det F·가우스 점 J), K 2 는 영상용 --dump --dump_F --dump_part (fa 는 원래 명령에 있음)
# 수박: 절 CM13 s_a 명령 (SNH, 옛 재질 실행기, 비결정) 끔 / 켬 두 번씩 (wm_off_a·wm_on_a·wm_off_b·wm_on_b)
# 카드: 공용 잡기 규약 (/dev/shm/dkta/anerr_claim/<카드>, 그 카드 계산 프로세스 0 일 때만), 카드 하나에 판 하나. 끝나고 K 2 단독 시간 (비결정 = --det 뺌, 끔·켬 번갈아 두 번), 렌더.
# 판 결과 (summary.json) 있으면 건너뜀. 표: exe/fint_tab.py -> $1/fint_table.md, 끝 표식 $1/fint.END (go 스크립트가 R2 확인 뒤 올림).
O=$1; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
SD=/dev/shm/dkta/anerr_data; mkdir -p $SD
[ -s $SD/L000.pt ] || { cp /home/dkta/work/one/L000/L000.pt $SD/L000.pt.$$ && mv $SD/L000.pt.$$ $SD/L000.pt; }
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
D="--data $SD/L000.pt --frames 80"
BB="--floor vel --fl_bar --fl_bst 1 --y0eb --cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
BA="--floor vel --cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
TL="--tolN 1e-2 --tolP 1e-1 --det"
WMC="--data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --floor_slip --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --K 3 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --gq_chk --dump --nan_step --jdiag --elx snh --qjd"
cmd_of() { case $1 in
  K2) echo "$D $BB --K 2 --newton 2 --cg 4 $TL --dump --dump_F --dump_part";;
  K4|K8|K16) echo "$D $BB --K ${1#K} --newton 2 --cg 4 $TL";;
  fa)  echo "$D $BA --K 2 --newton 2 --cg 6 $TL --pen --hgt --pdiag --dump --dump_F --dump_part";;
  vs)  echo "$D $BA --K 2 --newton 2 --cg 4 $TL --hgt --pen --dump --dump_F --dump_part --fl_vres 9";;
  nc)  echo "$D $BA --K 2 --newton 2 --cg 4 $TL --hgt --pen --dump --dump_F --dump_part --fl_nclamp";;
  wm)  echo "$WMC";;
esac; }
DG="--pdiag --qjd --fhash"
add_of() { [ "$2" = wm ] && { [ $1 = on ] && echo "--fp_interp"; return; }; case $1 in off) echo "$DG";; on) echo "--fp_interp $DG";; gq) echo "--gq_mass $DG";; gqon) echo "--gq_mass --fp_interp $DG";; esac; }
sim_of() { [ $1 = wm ] && echo exe/fem_ao_rt_sim.py || echo exe/fem_an_rt_sim.py; }
# ---- 카드 잡기 (공용 규약): 잡기 파일 없음 (또는 주인 죽음) + 그 카드 계산 프로세스 0
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
run() {    # run 판 모드 카드 [이름]
  local r=$1 m=$2 g=$3 n=${4:-$1_$2} c a s
  [ -f $O/$n/summary.json ] && return
  c=$(cmd_of $r); a=$(add_of $m $r); s=$(sim_of $r)
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $s $c --tag $r --out $O/$n $a" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  echo "[$(date '+%F %T')] $n 카드 $g 시작" >> $O/fint.log
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $s $c --tag $r --out $O/$n $a < /dev/null > $O/$n/log.txt 2>&1
  echo "[$(date '+%F %T')] $n rc=$? $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')" >> $O/fint.log
}
# ---- 연기 (레고 K 2 끔·켬 4 프레임, 수박 켬 3 프레임): 켬 컴파일·실행, 끔 해시 = 기록 (gqm K2_off_w 앞 4 프레임), 켬은 프레임 2 부터 달라야 함
SM=$O/smoke; mkdir -p $SM
smk() { local r=$1 m=$2 nf=$3 g S=$SM/$1_$2 c a s me=$BASHPID; mkdir -p $S
  g=$(claim_wait $me)
  c=$(cmd_of $r | sed "s/--frames [0-9]*/--frames $nf/"); a=$(add_of $m $r); s=$(sim_of $r)
  echo "$PY -u $s $c --tag $r --out $S $a" > $S/cmd.txt
  CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u $s $c --tag $r --out $S $a < /dev/null > $S/log.txt 2>&1; echo "rc=$? 카드 $g" > $S/rc.txt; rm -f $CL/$g; }
if [ ! -f $SM/ok ]; then
  smk K2 off 4 & sleep 5; smk K2 on 4 & sleep 5; smk wm on 3 & wait
  HO=$(grep -o '"xh": "[0-9a-f]*"' $SM/K2_off/fem.jsonl 2>/dev/null | tr '\n' ' '); HN=$(grep -o '"xh": "[0-9a-f]*"' $SM/K2_on/fem.jsonl 2>/dev/null | tr '\n' ' ')
  HR=$(grep -o '"xh": "[0-9a-f]*"' $O/ref/K2_off_w/fem.jsonl 2>/dev/null | head -4 | tr '\n' ' ')
  OK=1; MSG="끔 [$HO] 기록 [$HR] 켬 [$HN] 수박 $(cat $SM/wm_on/rc.txt 2>/dev/null) $(grep -ah '롤아웃\]' $SM/wm_on/log.txt | tail -1 | cut -c1-160)"
  [ $(echo "$HO" | grep -o xh | wc -l) -eq 4 ] && [ $(echo "$HN" | grep -o xh | wc -l) -eq 4 ] || OK=0
  [ "$HO" = "$HR" ] || OK=0
  [ "$HO" != "$HN" ] || OK=0
  [ -f $SM/wm_on/summary.json ] || OK=0
  echo "[$(date '+%F %T')] [연기] OK=$OK $MSG $(for q in K2_off K2_on wm_on; do grep -a 'rror\|Trace' $SM/$q/log.txt | tail -2 | cut -c1-300 | tr '\n' ' '; done)" >> $O/fint.log
  [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/fint.FAIL; exit 1; }
  touch $SM/ok
fi
# ---- 본 묶음: 공용 대기열 (긴 판 먼저), 줄마다 카드를 잡고 판 하나
Q=$O/queue.txt; [ -f $Q ] || {
  for r in K2 nc K4 K16 K8 vs fa; do echo "$r on"; echo "$r off"; done
  echo "wm on wm_on_a"; echo "wm off wm_off_a"; echo "wm on wm_on_b"; echo "wm off wm_off_b"
  for r in K2 K4; do echo "$r gqon"; echo "$r gq"; done; } > $Q
echo 0 > $O/qpos
lane() { local me=$BASHPID i ln g
  while :; do
    i=$(flock $O/qpos.lock bash -c "i=\$(cat $O/qpos); echo \$((i + 1)) > $O/qpos; echo \$i")
    ln=$(sed -n "$((i + 1))p" $Q); [ -n "$ln" ] || break
    set -- $ln; [ -f $O/${3:-$1_$2}/summary.json ] && continue
    g=$(claim_wait $me)
    run $1 $2 $g $3
    rm -f $CL/$g
  done; }
for k in 1 2 3 4 5 6 7; do lane & sleep 2; done
wait
echo "[$(date '+%F %T')] 판정 판 끝" >> $O/fint.log
$PY exe/fint_tab.py $O > $O/fint_table.md 2> $O/fint_table.err
# ---- K 2 단독 시간 (비결정 판 = 판정 명령에서 --det·덤프 뺌; 카드 하나 잡고 그 카드 다른 계산 프로세스 0, 판마다 그 카드·노드 프로세스 수 10 초마다 기록)
zrun() { local n=$1 a=$2 g=$3 c
  [ -f $O/$n/summary.json ] && return
  c=$(cmd_of K2 | sed 's/ --det//; s/ --dump --dump_F --dump_part//'); mkdir -p $O/$n
  echo "$PY -u exe/fem_an_rt_sim.py $c --tag L000 --out $O/$n $a" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
  ( while sleep 10; do echo "$(date '+%T') 카드 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .) 노드 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/gpu_procs.txt; done ) & local wp=$!
  CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u exe/fem_an_rt_sim.py $c --tag L000 --out $O/$n $a < /dev/null > $O/$n/log.txt 2>&1
  kill $wp 2>/dev/null
  echo "[$(date '+%F %T')] $n 카드 $g $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200)" >> $O/fint.log; }
ZG=$(claim_wait $$); echo "[$(date '+%F %T')] 단독 시간 카드 $ZG" >> $O/fint.log
zrun z_off_a "" $ZG; zrun z_on_a "--fp_interp" $ZG; zrun z_off_b "" $ZG; zrun z_on_b "--fp_interp" $ZG
rm -f $CL/$ZG
$PY exe/fint_tab.py $O > $O/fint_table.md 2> $O/fint_table.err
# ---- 렌더 (3DGS): 레고 (A) fa·(B) K2 끔·켬 (덤프 -> h5 -> exe/lb_render.py), 수박 끔·켬 (완주한 a 먼저, 없으면 b, 둘 다 발산이면 더 오래 간 판; 발산 판은 마지막 유한 프레임에서 정지)
W=/home/dkta/work; CD=$(pwd -P)
RG=$(claim_wait $$); echo "[$(date '+%F %T')] 렌더 카드 $RG" >> $O/fint.log
for n in fa_off fa_on K2_off K2_on; do Od=$O/$n
  [ -f $Od/vj/fem_lego.mp4 ] && continue
  [ -f $Od/pred.pt ] || { echo "[영상 없음] $n 덤프 없음" >> $O/fint.log; continue; }
  $PY -u exe/lb_roll_to_h5.py --dump $Od/pred.pt --traj $W/lbdata/test_low/L000 --out $Od/h5 > $Od/h5.log 2>&1
  mkdir -p $Od/vj
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/lb_render.py --sim $Od/h5 --config $W/lbdata/config.json --shape lego --out $Od/vj/fem_lego.mp4 --frame_label 0 --caption " " < /dev/null > $Od/vj/r.log 2>&1
  echo "[$(date '+%F %T')] [렌더] $n $(ls -la $Od/vj/fem_lego.mp4 2>&1 | cut -c1-160) $(tail -1 $Od/vj/r.log | cut -c1-200)" >> $O/fint.log
done
pick_wm() { local m=$1 b best=""; bn=-1
  for b in a b; do local s=$O/wm_${m}_$b/summary.json; [ -f $s ] || continue
    local dv=$($PY -c "import json; s=json.load(open('$s')); print(s.get('diverged') or 0)"); local nd=$($PY -c "import json; print(json.load(open('$s'))['frames_done'])")
    [ "$dv" = 0 ] && { echo wm_${m}_$b; return; }
    [ $nd -gt $bn ] && { bn=$nd; best=wm_${m}_$b; }; done
  echo $best; }
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json
GTD=/tmp/dkta/ag_data/gt_wm; [ -f $GTD/meta.json ] || GTD=/dev/shm/dkta/ag_data/gt_wm
if [ ! -f $GTD/meta.json ]; then mkdir -p /dev/shm/dkta/ag_data
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/pg_h5_to_pt.py --h5_dir $W/wm2/gf/simulation_ply --config $CFG --out_pt /dev/shm/dkta/ag_data/wm_rend_tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1
  rm -f /dev/shm/dkta/ag_data/wm_rend_tmp.pt; fi
for m in off on; do n=$(pick_wm $m); [ -n "$n" ] || continue; Od=$O/$n
  echo $n > $O/wm_vid_$m.txt
  [ -f $Od/vj/fem_watermelon.mp4 ] && continue
  mkdir -p $Od/vj
  $PY - $Od/pred.pt $Od/pred_v.pt <<'PYX' > $Od/vj/pad.log 2>&1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
ok = [i for i in range(P.shape[0]) if bool(torch.isfinite(P[i]).all()) and bool(torch.isfinite(F[i]).all())]
n = 0
while n in ok: n += 1
L = 100
P, F = P[:n], F[:n]
if n < L:
    P = torch.cat([P, P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F, F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("유한 프레임", n, "->", P.shape[0])
PYX
  $PY -u exe/lb_roll_to_h5.py --dump $Od/pred_v.pt --traj $GTD --out $Od/h5_fem > $Od/h5_fem.log 2>&1
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$RG $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --sim $Od/h5_fem --out $Od/vj/fem_watermelon.mp4 < /dev/null > $Od/vj/r_fem.log 2>&1)
  rm -f $Od/pred_v.pt; rm -rf $Od/h5_fem
  echo "[$(date '+%F %T')] [렌더] $n $(tail -1 $Od/vj/pad.log) $(ls -la $Od/vj/fem_watermelon.mp4 2>&1 | cut -c1-160) $(tail -1 $Od/vj/r_fem.log | cut -c1-200)" >> $O/fint.log
done
rm -f $CL/$RG
$PY exe/fint_tab.py $O > $O/fint_table.md 2> $O/fint_table.err
echo "[$(date '+%F %T')] FINT_END" >> $O/fint.log
