#!/bin/bash
# 절 AD 4 차 (비측정, 카드 3·4 x 일꾼 2): 3 차 중 /home 100 % 로 실패한 실행 다시 (같은 목록 -- EX_DONE 은 건너뜀) + 점유율 응력 가중을 더 세게
# (모든 점유 사면체 w = min(점유율/θ, 1), θ 2·3; 입자 ≤ 5 개 사면체 θ 2) -- 절 AC 빠른 경로, N 192·224·256 x 5 (클램프 5), 클램프 끔 N192 x 2.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"; df -h /home | tail -1
OL=$W/lbours/fem_ad_lego
C='--sp_chk 10,30,60'; F='--cfuse --cfuse_mode 2'
J=$OL/jobs4.txt
( cat $OL/jobs3.txt
  export AD_SET=2 AD_CL=c5 AD_N="192 224 256"
  export AD_V2="('staw2f','--sp stiff --sp_k 1000000 --sp_wocc 2 $C $F');('staw3f','--sp stiff --sp_k 1000000 --sp_wocc 3 $C $F');('st5w2f','--sp stiff --sp_k 5 --sp_wocc 2 $C $F')"
  python exe/fem_ad_jobs.py
  export AD_CL= AD_N=192 AD_R=2 AD_V2="('staw2','--sp stiff --sp_k 1000000 --sp_wocc 2 $C');('staw3','--sp stiff --sp_k 1000000 --sp_wocc 3 $C')"
  python exe/fem_ad_jobs.py | grep "_c0_" ) > $J
wc -l $J
for k in 0 1 2 3; do bash exe/fem_ad_gpu.sh $((3 + k % 2)) $k 4 "cat_$J" > $OL/w4_$k.log 2>&1 < /dev/null & done
wait
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
P=$(ls $W/lbours/fem_ad_ficus/ad_f_*/pred.pt 2>/dev/null); [ -n "$P" ] && python exe/fem_p_period.py --data $FD --pred $P > $W/lbours/fem_ad_ficus/period4.txt 2>&1
for r in fem_ad_lego fem_ad_ficus; do rclone copy $W/lbours/$r r2:storage/result/anchorflow/lbours/$r -q; done
for f in $W/lbours/fem_ad_ficus/ad_f_*/pred.pt; do [ -f $(dirname $f)/period.json ] && rclone check $W/lbours/fem_ad_ficus r2:storage/result/anchorflow/lbours/fem_ad_ficus --one-way --size-only --include "/${f#$W/lbours/fem_ad_ficus/}" -q && rm -f $f; done
df -h /home | tail -1
echo AD4_DONE $(date +%T)
