#!/bin/bash
# 절 AD 2 차 스윕 (비측정, 카드 3·4 x 일꾼 2 -- 1 차 일꾼 6 과 함께): 묶음 판을 V_t 가중 수정본 (45921fe~) 으로 다시 (태그 끝 v) + 점유율 기준 묶음.
# 1 차 목록의 mg*/cb* 는 같은 몫 (V_g/n) 판이라 프레임 2 발산 -- 표에서 _bug_v1 로 뺀다.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OL=$W/lbours/fem_ad_lego; OF=$W/lbours/fem_ad_ficus
export AD_SET=2
C='--sp_chk 10,30,60'
export AD_V2="('mg1tv','--sp merge --sp_a tet --sp_k 1 $C');('mg2tv','--sp merge --sp_a tet --sp_k 2 $C');('cb2tv','--sp cube --sp_a tet --sp_k 2 $C');('mg2av','--sp merge --sp_a avg --sp_k 2 $C');('mg2h1v','--sp merge --sp_a avg --sp_hg 1 --sp_k 2 $C');('mgo3tv','--sp merge --sp_a tet --sp_k 0 --sp_occ 0.3 $C')"
for k in 0 1 2 3; do bash exe/fem_ad_gpu.sh $((3 + k % 2)) $k 4 exe/fem_ad_jobs.py > $OL/w2_$k.log 2>&1 < /dev/null & done
wait
echo AD2_DONE $(date +%T)
