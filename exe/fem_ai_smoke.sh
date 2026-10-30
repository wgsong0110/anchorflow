#!/bin/bash
# 절 AI 연기 시험 (비측정): lego L000 5 프레임 x 판 8 개, 빈 카드 하나 (1~6 중 10 MiB 이하 첫 카드) 에서 차례로. 출력 /tmp/dkta/lbours/fem_ai_smoke (R2 같은 경로).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
OD=/tmp/dkta/lbours/fem_ai_smoke
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1>=1 && $2+0<=10 {print $1; exit}')
[ -z "$G" ] && { echo "[빈 카드 없음]"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 1; }
echo "[커밋] $(git log -1 --oneline) 카드 $G $(date +%T)"
REC="--mode lag --opt --mass tet --rstat --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5 --floor vel --N 267 --frames ${FR:-5} --no_dump --ai_stat 1"
while read -r TG ARGS; do
  EX_ROOT=fem_ai_smoke bash exe/fem_ab_run.sh $TG $G $REC $ARGS < /dev/null > /dev/null 2>&1
  echo "$(date +%T) $TG $(grep -a '롤아웃\]' $OD/$TG/v_fem.log | cut -c1-200) $(cat $OD/$TG/EX_FAIL 2>/dev/null | tail -2)"
done <<LST
s_base --lim_sr 5
s_off --lim_sr 1e30
s_a1 --lim_sr 5 --ai_p2g rs
s_a2 --lim_sr 5 --ai_stress
s_a12 --lim_sr 5 --ai_p2g rs --ai_stress
s_bs --lim_sr 5 --ai_g2p bs
s_dev --lim_sr 5 --ai_g2p dev
s_devnavg --lim_sr 5 --ai_g2p devnavg
s_devbs --lim_sr 5 --ai_g2p devbs
LST
echo "[끝] $(date +%T)"
