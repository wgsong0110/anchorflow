#!/bin/bash
# 절 CM27 묶음 wolffix: wolf 기본 판 (장벽 바닥 + 보간 교체, K 1, 결정적) 무너짐 고치기 -- (a) --ue be --uf_res, (b) --ue a0 --uf_res, (c) --y0loc 3 (국소 시작점),
#   (d) (a)+(c), (e) --pbc_phi (순차 방출 구속 이동도 정답 서브스텝 닫힌 꼴) 단독·(a)·(c)·(a)+(c) 조합. 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2.
#   판정 판 = 결정적 --det + 결과 불변 진단 (--hgt --fhash --bar_cnt --nrec --prec 10:50 판정 커널 --ufrec) + --dump --dump_F (퍼짐 폭·영상), 카드마다 하나 동시에.
#   base = 새 플래그 없음 -> 프레임 해시가 절 CM25 base 와 같아야 (끄면 비트 같음). 단독 시간 = 비결정·진단 없음, 카드 하나에 차례로 (a·b 두 번).
O=$1; RO=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WD=/tmp/dkta/ag_data; [ -s $WD/wolf.pt ] || { WD=/dev/shm/dkta/ag_data; mkdir -p $WD; [ -s $WD/wolf.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wolf.pt $WD/wolf.pt.$$ -q && mv $WD/wolf.pt.$$ $WD/wolf.pt; }; }
FB="--floor vel --fl_bar --fl_bst 1 --y0eb"
REST="--pbc --mlow 1e-3 --cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8 --cell 2"
IT="--newton 2 --cg 5"; TL="--tolN 1e-2 --tolP 1e-1"; TAIL="--fp_interp --tag wolf --K 1"
VARS="base a_ube b_ua0 c_loc d_locube e_phi e_phiube e_philoc e_all ${WF_EXTRA:-f_ks f_ksloc f_ksphi f_ksall}"
vf() { case $1 in
  base) echo "";;
  a_ube) echo "--ue be --uf_res";;
  b_ua0) echo "--ue a0 --uf_res";;
  c_loc) echo "--y0loc 3";;
  d_locube) echo "--y0loc 3 --ue be --uf_res";;
  e_phi) echo "--pbc_phi";;
  e_phiube) echo "--pbc_phi --ue be --uf_res";;
  e_philoc) echo "--pbc_phi --y0loc 3";;
  e_all) echo "--pbc_phi --y0loc 3 --ue be --uf_res";;
  f_ks) echo "--ksplit 3";;
  f_ksloc) echo "--ksplit 3 --y0loc 3";;
  f_ksphi) echo "--ksplit 3 --pbc_phi";;
  f_ksall) echo "--ksplit 3 --y0loc 3 --pbc_phi";;
  g_dks) echo "--ksplit 3 --y0loc 3 --ue be --uf_res";;
esac; }
dgof() { local x="--det --hgt --fhash --bar_cnt --nrec --prec 10:$2"; case "$(vf $1)" in *--ue*|*--ksplit*) ;; *) x="$x --ufrec --uf_res";; esac; echo "$x"; }
log() { echo "[$(date '+%F %T')] $*" >> $O/wolffix.log; rclone copyto $O/wolffix.log $RO/wolffix.log -q; }
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
run() {    # run 판 카드 프레임 출력 덤프
  local n=$1 g=$2 fr=$3 od=$4 dp=$5
  [ -f $od/summary.json ] && return
  rm -rf $od; mkdir -p $od
  local c="exe/fem_an_rt_sim.py --data $WD/wolf.pt --frames $fr $FB $REST $IT $TL $TAIL $(dgof $n $fr) $dp $(vf $n) --out $od"
  echo "$PY -u $c" > $od/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $od/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작 (프레임 $fr)"
  CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u $c < /dev/null > $od/log.txt 2>&1
  local rc=$?
  log "$n rc=$rc $(grep -ah '롤아웃\]' $od/log.txt | tail -1 | cut -c1-200) 진단 줄 $(wc -l < $od/prec.jsonl 2>/dev/null) 실패 $(grep -c '"err"' $od/prec.jsonl 2>/dev/null) $(grep -a 'rror' $od/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
}
# ---- 연기 (12 프레임, 덤프 없음): 모든 판 완주 + base 프레임 해시 = 절 CM25 base 앞 12
SM=$O/smoke; mkdir -p $SM
if [ ! -f $SM/ok ]; then
  for n in $VARS; do ( g=$(claim_wait $BASHPID); run $n $g 12 $SM/$n ""; rm -f $CL/$g ) & sleep 3; done; wait
  [ -s $O/ref_base_fem.jsonl ] || rclone copyto r2:storage/result/anchorflow/lbours/wolfwhy/base/fem.jsonl $O/ref_base_fem.jsonl -q
  OK=1; MSG=""
  for n in $VARS; do
    v=$($PY -c "import json; s=json.load(open('$SM/$n/summary.json')); print(int(s['frames_done'] == 12 and not s.get('diverged')))" 2>/dev/null || echo 0)
    [ "$v" = 1 ] || OK=0; MSG="$MSG $n:$v $(grep -a 'rror\|Trace' $SM/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"; done
  HS=$($PY -c "
import json
a=[json.loads(l).get('xh') for l in open('$O/ref_base_fem.jsonl')][:12]; b=[json.loads(l).get('xh') for l in open('$SM/base/fem.jsonl')]
print(sum(1 for x,y in zip(a,b) if x and x==y))" 2>&1)
  [ "$HS" = 12 ] || OK=0
  log "[연기] OK=$OK base 해시 같음 $HS/12;$MSG"
  rclone copy $SM $RO/smoke -q
  [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/wolffix.FAIL; exit 1; }
  touch $SM/ok
fi
# ---- 판정 판 (결정적, 50 프레임, 덤프 F 포함; 카드마다 하나씩 동시에, 끝나면 바로 R2)
for n in $VARS; do ( g=$(claim_wait $BASHPID); run $n $g 50 $O/$n "--dump --dump_F"; rm -f $CL/$g; up $n ) & sleep 3; done
wait
log "판정 판 끝"
# ---- 단독 시간 (카드 하나, 차례로; 비결정·진단 없음, a·b)
zrun() { local n=$1 v=$2 g=$3 od=$O/z_$1
  [ -f $od/summary.json ] && return
  local c="exe/fem_an_rt_sim.py --data $WD/wolf.pt --frames 50 $FB $REST $IT $TL $TAIL $(vf $v) --out $od"
  rm -rf $od; mkdir -p $od; echo "$PY -u $c" > $od/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $od/gpu_before.txt 2>&1
  CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u $c < /dev/null > $od/log.txt 2>&1
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $od/gpu_after.txt 2>&1
  log "z_$n 카드 $g $(grep -ah '롤아웃\]' $od/log.txt | tail -1 | cut -c1-200)"; up z_$n; }
ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
for r in a b; do for v in $VARS; do zrun ${v}_$r $v $ZG; done; done
rm -f $CL/$ZG
log "단독 시간 끝"
