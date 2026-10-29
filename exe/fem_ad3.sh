#!/bin/bash
# 절 AD 3 차 스윕 (비측정, 카드 3·4 x 일꾼 2): 점유율 응력 가중 (stiff) 변형 + 1·2 차에서 클램프 켬 완주가 좋았던 판을 절 AC 빠른 경로 (--cfuse --cfuse_mode 2, 같은 식) 로, N 192·224·256 x 5 회 (클램프 5 만),
# 클램프 끔 stiff 변형 N192 x 2 회 (발산 여부만, 기존 경로).
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OL=$W/lbours/fem_ad_lego; OF=$W/lbours/fem_ad_ficus
C='--sp_chk 10,30,60'; F='--cfuse --cfuse_mode 2'
J=$W/lbours/fem_ad_lego/jobs3.txt
( export AD_SET=2 AD_CL=c5 AD_N="192 224 256"
  export AD_V2="('reff','$F');('st2f','--sp stiff --sp_k 2 $C $F');('st2w2f','--sp stiff --sp_k 2 --sp_wocc 2 $C $F');('st5f','--sp stiff --sp_k 5 $C $F');('staf','--sp stiff --sp_k 1000000 $C $F');('mgo3tf','--sp merge --sp_a tet --sp_k 0 --sp_occ 0.3 $C $F');('ml01f','--sp mlow --sp_mlow 0.1 $C $F')"
  python exe/fem_ad_jobs.py
  export AD_CL= AD_N=192 AD_R=2 AD_V2="('st2w2','--sp stiff --sp_k 2 --sp_wocc 2 $C');('st5','--sp stiff --sp_k 5 $C');('sta','--sp stiff --sp_k 1000000 $C')"
  python exe/fem_ad_jobs.py | grep "_c0_" ) > $J
wc -l $J
for k in 0 1 2 3; do bash exe/fem_ad_gpu.sh $((3 + k % 2)) $k 4 "cat_$J" > $OL/w3_$k.log 2>&1 < /dev/null & done
wait
echo AD3_DONE $(date +%T)
