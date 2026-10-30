#!/bin/bash
# 절 AD 5 차 (비측정, 카드 3·4 x 일꾼 2): 4 차에서 모든 점유 사면체 응력 가중 (θ 2·3) 이 lego 오차 < 10 % 와 ficus 주기 0.78 -> 0.90·1.03 s (정답 1.398) 를 냄
# -> 원인 가설 (입자 부피 합 V_T 가 사면체 기하 부피를 넘어 (점유율 중앙 1.3, 90 % 3.6) 탄성이 과하다) 을 직접: 탄성 부피 상한 min(V_T, θ V_기하) θ 1·0.75·0.5,
#    그리고 부드러워진 만큼 N 을 낮춰 (128·160) -- 절 AC 빠른 경로, 클램프 5 x 5 회, 클램프 끔 N 128·192 x 2 회.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"; df -h /home | tail -1
OL=$W/lbours/fem_ad_lego; OF=$W/lbours/fem_ad_ficus
C='--sp_chk 10,30,60'; F='--cfuse --cfuse_mode 2'
J=$OL/jobs5.txt
( export AD_SET=2 AD_CL=c5 AD_N="128 160 192 224 256"
  export AD_V2="('vc1f','--sp vcap --sp_vcap 1 $C $F');('vc075f','--sp vcap --sp_vcap 0.75 $C $F');('vc05f','--sp vcap --sp_vcap 0.5 $C $F')"
  python exe/fem_ad_jobs.py
  export AD_N="128 160" AD_V2="('staw2f','--sp stiff --sp_k 1000000 --sp_wocc 2 $C $F');('staw3f','--sp stiff --sp_k 1000000 --sp_wocc 3 $C $F')"
  python exe/fem_ad_jobs.py | grep ad_l_
  export AD_CL= AD_N="128 192" AD_R=2 AD_V2="('vc1','--sp vcap --sp_vcap 1 $C');('vc05','--sp vcap --sp_vcap 0.5 $C')"
  python exe/fem_ad_jobs.py | grep "_c0_" ) > $J
wc -l $J
for k in 0 1 2 3; do bash exe/fem_ad_gpu.sh $((3 + k % 2)) $k 4 "cat_$J" > $OL/w5_$k.log 2>&1 < /dev/null & done
wait
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
P=$(ls $OF/ad_f_*/pred.pt 2>/dev/null); [ -n "$P" ] && python exe/fem_p_period.py --data $FD --pred $P > $OF/period5.txt 2>&1
for r in fem_ad_lego fem_ad_ficus; do rclone copy $W/lbours/$r r2:storage/result/anchorflow/lbours/$r -q; done
for f in $OF/ad_f_*/pred.pt; do [ -f $(dirname $f)/period.json ] && rclone check $OF r2:storage/result/anchorflow/lbours/fem_ad_ficus --one-way --size-only --include "/${f#$OF/}" -q && rm -f $f; done
python exe/fem_ad_table.py $OL/ad_l_* > $OL/table5.md; python exe/fem_ad_table.py $OF/ad_f_* > $OF/table5.md
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_ad_lego -q; rclone copy $OF r2:storage/result/anchorflow/lbours/fem_ad_ficus -q
df -h /home | tail -1
echo AD5_DONE $(date +%T)
