#!/bin/bash
# 파레토 비교 묶음 (doc/pareto.md; 커밋 $1, 잡 130567): lego L000·ficus (탄성), 오차 = 롤아웃 평균 |x−x_gt|/EXT, 시간 = A6000 카드 0 단독 (gpu0.lock, 판마다 카드 0 의 다른 계산 프로세스 0).
#  A (비측정, 빈 카드 1-6 -- anerr_claim 규약): VR-GS 설정별 E 맞춤 (exe/pareto_vrgs.py --mode fit), 설정 묶음마다 프로세스 하나.
#  B (단독 카드 0, 순서대로): VR-GS 규모 대조 (scale) -> 우리 (절 AN.22 채택 판, K x PCG 상한) -> 정답 MPM (서브스텝 수) -> VR-GS (맞춘 E, 시간).
#  마감 PA_UNTIL (새 판 안 띄움) · PA_HARD (표·끝 표식). 출력 /dev/shm/dkta/lbours/pareto -> R2 lbours/pareto/, 표 pareto_table.md, 그림 pareto.png·pdf, 끝 표식 pareto.END.
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pareto XDG_CACHE_HOME=/dev/shm/dkta/cache; mkdir -p $CUPY_CACHE_DIR
PY=/home/dkta/.conda/envs/af/bin/python
R=/dev/shm/dkta/lbours/pareto; RR=r2:storage/result/anchorflow/lbours/pareto; CL=/dev/shm/dkta/anerr_claim
mkdir -p $R $CL
UNTIL=${PA_UNTIL:-$(date -d '2026-11-05 12:12 KST' +%s)}; HARD=${PA_HARD:-$(date -d '2026-11-05 12:24 KST' +%s)}
LG=$R/pareto.log; echo "[pareto 시작] $C $(date '+%F %T')" >> $LG
SD=/dev/shm/dkta/anerr_data; mkdir -p $SD
L0=$SD/L000.pt; [ -s $L0 ] || cp /home/dkta/work/one/L000/L000.pt $L0
FI=$SD/F_pg_v0.pt; [ -s $FI ] || cp /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt $FI
DL="--data $L0 --tag L000 --frames 80 --floor vel"
DF="--data $FI --tag ficus --frames 125 --floor none --cfg_bc --region_density --pg_vdamp"
# 설정 격자 (VR-GS: 서브스텝 N x 반복; N 167 = dt 1e-4 s (논문 PG 비교 설정, 반복 11) -- ficus N 400 이 같은 dt)
VL="8x1 8x5 8x11 16x1 16x5 16x11 32x1 32x5 32x11 64x1 64x5 64x11 167x1 167x5 167x11"
VF="25x1 25x5 25x11 50x1 50x5 50x11 100x1 100x5 100x11 200x1 200x5 200x11 400x1 400x5 400x11"
sync_r2() { rclone copy $R $RR -q --exclude "*.pt"; }
( while true; do sleep 300; sync_r2; done ) & SY=$!

# ---------- A: E 맞춤 (빈 카드) ----------
free_card() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || return
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r i u; do
    [ "$i" = 0 ] && continue
    echo "$apps" | grep -q "$u" && continue
    if [ -f $CL/$i ] && kill -0 "$(cat $CL/$i)" 2>/dev/null; then continue; fi
    echo $i; break
  done
}
fit_one() {   # <이름> <씬 인자> <설정...>
  local NM=$1 DA=$2; shift 2
  local G=""
  while [ -z "$G" ]; do G=$(free_card); [ -z "$G" ] && sleep 30; [ "$(date +%s)" -ge "$UNTIL" ] && { echo "마감 -- 맞춤 안 띄움 $NM" >> $LG; return; }; done
  echo $BASHPID > $CL/$G
  local O=$R/fit_$NM; mkdir -p $O
  echo "카드 $G $(date '+%F %T')" > $O/card.txt; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $O/card.txt
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/pareto_vrgs.py --mode fit --out $O $DA --set "$@" < /dev/null > $O/log.txt 2>&1
  echo "rc=$? $(date '+%F %T')" >> $O/card.txt
  rm -f $CL/$G
  echo "맞춤 끝 $NM $(date +%T)" >> $LG; rclone copy $O $RR/fit_$NM -q
}
# 비싼 설정 (반복 11, 큰 N) 을 고르게 나눔: 씬마다 반복 수별 3 묶음
FP=""
for it in 1 5 11; do
  SL=$(for q in $VL; do case $q in *x$it) echo -n "$q ";; esac; done)
  SF=$(for q in $VF; do case $q in *x$it) echo -n "$q ";; esac; done)
  fit_one L000_i$it "$DL" $SL & FP="$FP $!"; sleep 20
  fit_one ficus_i$it "$DF" $SF & FP="$FP $!"; sleep 20
done

# ---------- B: 단독 카드 0 ----------
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 9>&- | grep -c "$U0"; }
excl() {   # <이름> <명령...> (카드 0)
  local NM=$1; shift
  local O=$R/t_$NM; mkdir -p $O
  if [ -f $O/card.txt ] && grep -q rc= $O/card.txt; then return; fi
  [ "$(date +%s)" -ge "$UNTIL" ] && { echo "마감 -- 안 띄움 $NM" | tee -a $R/skipped.txt >> $LG; return; }
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others0) 개 $(date '+%F %T') $*" > $O/card.txt
  ( "$@" ) 9>&- < /dev/null > $O/log.txt 2>&1 &
  local PP=$!
  while kill -0 $PP 2>/dev/null; do nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 9>&- | awk -F', ' -v u=$U0 -v t=$(date +%s) '$1==u{print t, $2}' >> $O/apps_during.txt; sleep 5; done
  wait $PP; local RC=$?
  local NF=$(( $(awk '{print $2}' $O/apps_during.txt 2>/dev/null | sort -u | wc -l) - 1 )); [ $NF -lt 0 ] && NF=0
  echo "rc=$RC 끝 $(date '+%F %T') 실행 중 카드 0 의 다른 계산 프로세스 $NF 개" >> $O/card.txt
  echo "$NM | $(grep -a '롤아웃\|결과\|rror\|발산\|\[실행\]\|\[규모\]' $O/log.txt | tail -3 | cut -c1-220 | tr '\n' ' ') | 다른 $NF" >> $LG
  rclone copy $O $RR/t_$NM -q --exclude "*.pt"
}
exec 9>/home/dkta/work/gpu0.lock; flock 9
echo "잠금 $(date +%T)" >> $LG
export CUDA_VISIBLE_DEVICES=0
excl vrgs_scale $PY -u exe/pareto_vrgs.py --mode scale --out $R/t_vrgs_scale $DL
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga hht --rho 0.8 --tolN 1e-2 --tolP 1e-1 --newton 2"
for K in 2 1 4; do for c in 3 1 2 5 10; do
  excl ours_L000_K${K}c$c $PY -u exe/fem_an_rt_sim.py --out $R/t_ours_L000_K${K}c$c --tag L000 --data $L0 --frames 80 --floor vel --fl_bar --fl_bst 1 $B --K $K --cg $c
done; done
for K in 2 1 4; do for c in 20 5 10 40; do
  excl ours_ficus_K${K}c$c $PY -u exe/fem_an_rt_sim.py --out $R/t_ours_ficus_K${K}c$c --tag ficus --data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp $B --K $K --cg $c
done; done
W=/home/dkta/work
export WARP_CACHE_PATH=/dev/shm/dkta/warp_cache_0; mkdir -p $WARP_CACHE_PATH
for n in 833 416 208 104 52 26; do
  excl mpm_L000_n$n bash -c "cd $W/i-physgaussian && $PY -u $D/exe/pareto_mpm.py --out $R/t_mpm_L000_n$n --frames 80 --no_dump --rpic_cfg --sub_chk --time --nsub $n"
done
for n in 400 200 100 50 25; do
  excl mpm_ficus_n$n bash -c "cd $W/i-physgaussian && $PY -u $D/exe/pareto_mpm.py --out $R/t_mpm_ficus_n$n --gt $W/lbours/ficus_v0/gt --config $W/lbours/ficus_v0/gt/config.json --pt $FI --frames 125 --pg_cfg --no_dump --rpic_cfg --sub_chk --time --nsub $n"
done
flock -u 9
# VR-GS 시간: 맞춤이 끝나기를 기다림 (그동안 카드 0 잠금은 놓음)
for p in $FP; do while kill -0 $p 2>/dev/null && [ "$(date +%s)" -lt "$UNTIL" ]; do sleep 30; done; done
flock 9
FJL=$(ls $R/fit_L000_i*/fit.json 2>/dev/null | tr '\n' ',' | sed 's/,$//'); FJF=$(ls $R/fit_ficus_i*/fit.json 2>/dev/null | tr '\n' ',' | sed 's/,$//')
[ -n "$FJL" ] && excl vrgs_L000 $PY -u exe/pareto_vrgs.py --mode run --out $R/t_vrgs_L000 $DL --fit_json $FJL --set 16x5 $VL
[ -n "$FJF" ] && excl vrgs_ficus $PY -u exe/pareto_vrgs.py --mode run --out $R/t_vrgs_ficus $DF --fit_json $FJF --set 50x5 $VF
flock -u 9; exec 9>&-
$PY exe/pareto_table.py $R > $R/pareto_table.md 2> $R/table_err.txt
echo "[pareto 끝] $C $(date '+%F %T') 안 띄운 판 $(cat $R/skipped.txt 2>/dev/null | wc -l)" | tee -a $LG > $R/pareto.END
kill $SY; sync_r2; rclone copy $R/pareto.END $RR -q
