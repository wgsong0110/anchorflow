#!/bin/bash
# 절 AN.16 큰 dt 실험 (논문 표): 프레임당 걸음 수 K = 1, 2, 4, 8 (걸음 간격 dt = frame_dt / K).
# 우리 암시적 판: 판정 j1 의 bej (후퇴 오일러) · hhj (HHT ρ 0.7) 설정 그대로 (공통 허용치 εN 1e-2 · εP 1e-1, 씬별 상한 lego 1 x 5 · ficus 2 x 10, 결정적 --det) 에서 --K 만 바꿈.
#   궤적 lego L000·0005·0013·0016, ficus.  (K 가 달라도 걸음당 상한은 j1 그대로)
# 비교: 같은 dt 의 명시적 기준선 = bnv (exe/fem_bsel_cuda_sim.py --theta 0 --vfl_out, 같은 B-스플라인 요소, 서브스텝 N = K) 와
#       정답 MPM 솔버 (exe/gt_sens.py --nsub K), lego L000 · ficus. 발산 프레임만 기록.
# 카드: 0 은 쓰지 않음 (fem_an_bigdt_pool.sh 가 비어 있고 안 잡힌 카드만). 출력 /dev/shm/dkta/lbours/fem_an/bigdt -> R2 lbours/fem_an/bigdt/, 표 bigdt_table.md, 끝 표식 bigdt.END
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:$PATH
R=/dev/shm/dkta/lbours/fem_an/bigdt; RR=r2:storage/result/anchorflow/lbours/fem_an/bigdt
mkdir -p $R /dev/shm/dkta/an_lists; df -h /tmp /dev/shm | tail -2
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse"
T="--tolN 1e-2 --tolP 1e-1"
L=/dev/shm/dkta/an_lists/bigdt.txt; : > $L
for K in 1 2 4 8; do
  # 암시적 (우리 판): 큰 dt 에서 발산이 날 만한 K 1 먼저가 아니라 K 큰 순서는 상관 없음
  for vv in "be:--integ be" "hh:--ga hht --rho 0.7"; do v=${vv%%:*}; I=${vv#*:}
    echo "${v}K$K ficus $B --K $K --det $I $T --newton 2 --cg 10" >> $L
    for t in L000 0005 0013 0016; do echo "${v}K$K $t $B --K $K --det $I $T --newton 1 --cg 5" >> $L; done
  done
  # 명시적 기준선과 정답 MPM
  for t in L000 ficus; do echo "exK$K $t --N $K" >> $L; echo "mpmK$K $t --nsub $K" >> $L; done
done
wc -l $L
AN_CARDS="${AN_CARDS:-1 2 3 4 5 6}" AN_PAR=${AN_PAR:-6} bash exe/fem_an_bigdt_pool.sh bigdt $L > $R/pool.log 2>&1
PY=/home/dkta/.conda/envs/af/bin/python
$PY exe/fem_an_bigdt_table.py $R > $R/bigdt_table.md 2>&1
cat $R/bigdt_table.md
echo "[bigdt 끝] $(cat .done) $(date '+%F %T')" > $R/bigdt.END
rclone copy $R/bigdt_table.md $RR -q; rclone copy $R/bigdt.END $RR -q; rclone copy $R/pool.log $RR -q
