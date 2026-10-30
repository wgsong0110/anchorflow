#!/bin/bash
# 절 AH.9: 경합 없는 단독 측정 -- 회마다 카드 1~6 이 모두 10 MiB 이하가 될 때까지 (상한 없이) 기다린 뒤 GPU0 + gpu0.lock. 판 AH_FN (태그 ah_<FN>_<장면>_c<회>), 플래그 AH_FIN.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
W=/home/dkta/work; FN=${AH_FN:?}; FL=${AH_FIN:?}
DD=/tmp/dkta/af_data; OL=/tmp/dkta/lbours/fem_ah_lego; OF=/tmp/dkta/lbours/fem_ah_ficus; RB=r2:storage/result/anchorflow/lbours
LEGO="--floor vel --node_h 0.028284 --frames 80 --no_dump --N 267"
FIC="--data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --no_dump --N 640"
idle() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1>=1 && $2+0>10{b=1} END{exit b}'; }
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T') 판 $FN: $FL"
for r in 1 2 3; do for t in $(seq -f "%04g" 0 19) L000 F; do
  w=0; until idle; do sleep 20; w=$((w + 20)); done
  if [ $t = F ]; then TG=ah_${FN}_F_c$r; EX_TIMED=1 EX_ROOT=fem_ah_ficus bash exe/fem_ab_run.sh $TG 0 $FIC $FL < /dev/null > /dev/null 2>&1; O=$OF/$TG
  else TG=ah_${FN}_${t}_c$r; A="--data $DD/$t.pt --tag T$t"; [ $t = L000 ] && A="--data $W/one/L000/L000.pt --tag TL000"
    EX_TIMED=1 EX_ROOT=fem_ah_lego bash exe/fem_ab_run.sh $TG 0 $A $LEGO $FL < /dev/null > /dev/null 2>&1; O=$OL/$TG; fi
  echo "$(date +%T) $TG (대기 $w s) $(grep -a '롤아웃\]' $O/v_fem.log | grep -o '평균 [0-9.]*%  프레임당 [0-9.]* ms') | 카드 1~6 시작 $(awk -F, '$1!=0{print $2}' $O/gpu_at_start.txt | tr -d ' \n') 끝 $(awk -F, '$1!=0{print $2}' $O/gpu_at_end.txt | tr -d ' \n')"
done; done
echo "AHC_DONE $(date +%T)"; touch $OL/AHC_${FN}_DONE; rclone copy $OL/AHC_${FN}_DONE $RB/fem_ah_lego -q
