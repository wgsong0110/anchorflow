#!/bin/bash
# 절 CM27 회귀: wolf 를 무너뜨린 플래그를 레고 L000·ficus·bread (새 실행기, 절 CM23 k1new 명령 K 1 그대로, 결정적 판정 + 단독 시간) 와
#   수박 (옛 실행기, 절 CM24 (2) 판 = k1new 수박 명령 + --ue be --uf_res, 비결정 a·b + 단독 시간) 에 같은 플래그로. 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2, 판 이름 $3, 플래그 = 나머지.
#   씬마다 off (새 플래그 없음; 새 실행기는 끝 해시 = k1new <씬>_K1 이어야) 와 판. 플래그가 할 일 없는 씬 (바닥 없는 ficus·bread 의 --y0loc, 구속 없는 씬의 --pbc_phi) 은 그대로 넘김 (무시됨).
O=$1; RO=$2; VN=$3; VF="${*:4}"; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
source exe/fem_cm2_scn.sh
for t in L000 ficus bread; do cm_data $t; done
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
BA0="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
FBB="--floor vel --fl_bar --fl_bst 1 --y0eb"
TL="--tolN 1e-2 --tolP 1e-1"
WMB="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1 --ue be --uf_res"
base_of() { case $1 in                                                                         # 절 CM23 stab_k1new.sh base_of 글자 그대로 (+ 수박 절 CM24 (2))
  L) echo "exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 $FBB $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000 --K 1";;
  F) echo "exe/fem_an_rt_sim.py --data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp $BA0 --newton 2 --cg 24 $TL --fp_interp --tag ficus --K 1";;
  B) echo "exe/fem_an_rt_sim.py --data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 $BA0 --newton 2 --cg 2 $TL --fp_interp --tag bread --K 1";;
  M) echo "$WMB";;
esac; }
dg_of() { case $1 in L) echo "--det --pdiag --qjd --fhash";; F) echo "--det --pdiag --qjd --fhash --dump";; B) echo "--det --pdiag --qjd --fhash";; M) echo "--bar_cnt";; esac; }
# 수박 (옛 실행기) 에 없는 플래그 빼기, 이미 명령에 있는 플래그 겹치지 않게
vf_of() { local s=$1 x="$VF"
  if [ $s = M ]; then x=$(echo " $x " | sed 's/ --pbc_phi / /g; s/ --ue be / /g; s/ --uf_res / /g'); fi
  echo $x; }
log() { echo "[$(date '+%F %T')] $*" >> $O/reg_$VN.log; rclone copyto $O/reg_$VN.log $RO/reg_$VN.log -q; }
up() { rclone copy $O/$1 $RO/$1 -q --transfers 8 --s3-upload-concurrency 16 --s3-chunk-size 64M; }
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
run() {   # run 이름 씬 추가플래그 카드 진단
  local n=$1 s=$2 x=$3 g=$4 dg=$5 c
  [ -f $O/$n/summary.json ] && return
  c="$(base_of $s) $dg $x"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
# ---- 판정 판 (카드마다 하나): 새 실행기 결정적 off·판, 수박 비결정 off·판 a·b
for s in B F L; do for v in off $VN; do x=""; [ $v = $VN ] && x="$(vf_of $s)"
  ( g=$(claim_wait $BASHPID); run r_${s}_$v $s "$x" $g "$(dg_of $s)"; rm -f $CL/$g ) & sleep 3; done; done
for r in a b; do for v in off $VN; do x=""; [ $v = $VN ] && x="$(vf_of M)"
  ( g=$(claim_wait $BASHPID); run r_M_${v}_$r M "$x" $g "$(dg_of M)"; rm -f $CL/$g ) & sleep 3; done; done
wait
for s in L F B; do
  rclone copyto r2:storage/result/anchorflow/lbours/k1new/${s}_K1/summary.json $O/r_${s}_off/k1new_summary.json -q
  log "[끔 대조] $s off $($PY -c "import json; print(json.load(open('$O/r_${s}_off/summary.json')).get('x_sha1'))" 2>&1) k1new $($PY -c "import json; print(json.load(open('$O/r_${s}_off/k1new_summary.json')).get('x_sha1'))" 2>&1)"
  up r_${s}_off; done
log "판정 판 끝 $VN"
# ---- 단독 시간 (카드 하나, 차례로; 비결정·진단 없음, off·판 번갈아 a·b)
ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
zrun() { local n=$1 s=$2 x=$3 c
  [ -f $O/$n/summary.json ] && return
  c="$(base_of $s) $x"; rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
  CUDA_VISIBLE_DEVICES=$ZG timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  log "$n 카드 $ZG $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200)"; up $n; }
for s in L F B M; do for r in a b; do zrun z_${s}_off${VN}_$r $s ""; zrun z_${s}_${VN}_$r $s "$(vf_of $s)"; done; done
rm -f $CL/$ZG
log "단계 끝 reg $VN"
