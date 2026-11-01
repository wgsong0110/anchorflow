#!/bin/bash
# 절 AN MG 고정 예산 발산 진단 (비측정): 발산 궤적을 --diag 로 다시 (발산 프레임까지). 카드마다 일꾼 하나 (빈 카드, 없으면 사용률 낮은 넷).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
NM=${1:-div1}; R=/tmp/dkta/lbours/fem_an/$NM; M=/tmp/dkta/lbours/fem_an/marks; Q=$R/queue; mkdir -p $R $M
MG1="--cell 1 --mg_levels 5 --newton 1 --cg 10"; MG15="--cell 1.5 --mg_levels 5 --newton 2 --cg 5"
: > $Q; for t in L000 0002 0003 0004 0006 0008 0013 0016; do echo "$t m1mn1c10 $MG1" >> $Q; done
echo "ficus m1mn1c10 $MG1" >> $Q; echo "ficus m15mn2c5 $MG15" >> $Q
free_cards() { for i in 1 2 3 4 5 6; do u=$(nvidia-smi -i $i --query-gpu=uuid --format=csv,noheader); n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u"); [ $n -eq 0 ] && echo $i; done; }
GS=($(free_cards)); [ ${#GS[@]} -eq 0 ] && GS=($(for i in $(seq 5); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{for (k in s) print s[k], k}' | sort -n | head -4 | awk '{print $2}'))
echo "카드 ${GS[*]} $(cat .done) $(date +%T)" > $M/${NM}_cards.txt
worker() { G=$1
  while true; do L=$(flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"); [ -z "$L" ] && break
    set -- $L; t=$1; v=$2; shift 2; O=$R/${t}_$v; mkdir -p $O
    case $t in ficus) DA="--data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --frames 125 --floor none --cfg_bc --region_density --pg_vdamp";;
      L000) DA="--data /home/dkta/work/one/L000/L000.pt --frames 80 --floor vel";; *) DA="--data /tmp/dkta/af_data/$t.pt --frames 80 --floor vel";; esac
    CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_rt_sim.py --out $O --tag $t $DA --K 1 --integ mid $* --diag < /dev/null > $O/log.txt 2>&1
    rclone copy $O r2:storage/result/anchorflow/lbours/fem_an/$NM/${t}_$v -q; done; }
for G in ${GS[@]}; do worker $G & done; wait
echo "[끝] $(date +%T)" > $M/${NM}_done; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
