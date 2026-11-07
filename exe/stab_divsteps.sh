#!/bin/bash
# 레고 (A) 원래 FCR (divwhy fa) 폭증 프레임 21·22 의 걸음·뉴턴 시점별 위치 덤프 (잡 안, 커밋 디렉토리에서; exe/stab_launch.sh 로 띄움).
# 명령 = R2 lbours/divwhy/fao/cmd.txt 의 exe/fem_an_rt_sim.py 인자 그대로 + --wdiag $W0 --wdiag_end $W1 --wdiag_pos --fhash (진단만, 결과 불변 -> 해시를 fao 와 대조).
# 빈 카드 하나에서 혼자, 출력은 /dev/shm 에서 R2 lbours/divwhy_steps/fa 로 1 분마다·끝에 복사.
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
R=r2:storage/result/anchorflow; W0=${DS_W0:-27}; W1=${DS_W1:-29}; RR=$R/lbours/divwhy_steps/fa_f${W0}_${W1}
SD=/dev/shm/dkta/anerr_data; O=/dev/shm/dkta/lbours/divwhy_steps/fa_f${W0}_${W1}; mkdir -p $SD $O
[ -s $SD/L000.pt ] || rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q
rclone copyto $R/lbours/divwhy/fao/cmd.txt $O/ref_cmd.txt -q
c=$(head -1 $O/ref_cmd.txt | sed 's/^.*exe\/fem_an_rt_sim.py/exe\/fem_an_rt_sim.py/; s/ --out [^ ]*//; s/ --fhash//')
G=""
for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  u=$(nvidia-smi -i $i --query-gpu=uuid --format=csv,noheader)
  if [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $u)" = 0 ]; then G=$i; break; fi
done
[ -n "$G" ] || { echo "[오류] 빈 카드 없음"; nvidia-smi; exit 1; }
echo "카드 $G (빈 카드), 커밋 $(git rev-parse --short HEAD), $(hostname) $(date '+%F %T')" | tee $O/card.txt
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader >> $O/card.txt
echo "$PY -u $c --out $O --wdiag $W0 --wdiag_end $W1 --wdiag_pos --fhash" | tee $O/cmd.txt
( while true; do sleep 60; rclone copy $O $RR -q; done ) & SY=$!
CUDA_VISIBLE_DEVICES=$G timeout 3600 $PY -u $c --out $O --wdiag $W0 --wdiag_end $W1 --wdiag_pos --fhash < /dev/null > $O/log.txt 2>&1
echo "rc=$? $(date '+%F %T')" > $O/rc.txt
kill $SY; rclone copy $O $RR -q; echo 끝 > $O/END; rclone copy $O $RR -q
