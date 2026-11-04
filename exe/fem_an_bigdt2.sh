#!/bin/bash
# 절 BD 큰 dt 실험 둘째 판 (bigdt2; 커밋 $1, 잡 130567): 프레임당 걸음 수 K = 1, 2, 4, 8 (dt = frame_dt / K), 결정적 판정 (비측정, 빈 카드 1-6).
#  0. 정답 MPM 실행기 확인: 원래 서브스텝 (config substep_dt, K0) 으로 같은 실행기 (exe/gt_sens.py) -- L000·ficus·wolf·bread
#  1. 후퇴 오일러 발산 원인 (절 AN.16 bej 설정 K 8, 0005·0013; --diag): 재현 / 허용치 0 / ω 재사용 끔 / 블록 야코비 / 뉴턴 2 / PCG 10 / 비결정
#  2. 표: 우리 채택 판 (절 AN.22: HHT ρ 0.8 + 띠 고착 장벽 바닥, lego 2x3 · ficus 2x20; 같은 플래그 적분기만 be 인 행도),
#     재질 씬 wolf·bread (절 CM3 공통 묶음, HHT ρ 0.8, wolf 2x4 · bread 2x1 = cm3 판정 상한), 명시적 B-스플라인 bnv (L000·ficus),
#     정답 MPM (--nsub K, 공식 config 그대로 + 서브스텝마다 격자 안 확인 --sub_chk)
# 출력 /dev/shm/dkta/lbours/fem_an/bigdt2 -> R2 lbours/fem_an/bigdt2/, 표 bigdt2_table.md, 끝 표식 bigdt2.END
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1
PY=/home/dkta/.conda/envs/af/bin/python
R=/dev/shm/dkta/lbours/fem_an/bigdt2; RR=r2:storage/result/anchorflow/lbours/fem_an/bigdt2
mkdir -p $R /dev/shm/dkta/an_lists; df -h /tmp /dev/shm | tail -2
echo "[bigdt2 시작] $C $(date '+%F %T')" > $R/bigdt2.log; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $R/bigdt2.log
rclone copy $R/bigdt2.log $RR -q
OLD="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse"            # 절 AN.16 bej 설정
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2"   # 절 AN.22 최종
T="--tolN 1e-2 --tolP 1e-1"
FL="--fl_bar --fl_bst 1"
L=/dev/shm/dkta/an_lists/bigdt2.txt; : > $L
# 0 (원래 dt; 서브스텝 많아 오래 걸리므로 먼저)
for t in L000 ficus wolf bread; do echo "mpmK0 $t --sub_chk --rpic_cfg" >> $L; done
# 1
Q="--K 8 --det --integ be $T --newton 1 --cg 5 --diag"
for t in 0005 0013; do
  echo "dq0 $t $OLD $Q" >> $L
  echo "dq1 $t $OLD ${Q/$T/--tolN 0 --tolP 0}" >> $L
  echo "dq2 $t ${OLD/ --mg_wreuse --mg_lreuse/} $Q" >> $L
  echo "dq3 $t ${OLD/--mg_levels 4 --mg_nc 4/--mg_levels 1} $Q" >> $L
  echo "dq4 $t $OLD ${Q/--newton 1/--newton 2}" >> $L
  echo "dq5 $t $OLD ${Q/--cg 5/--cg 10}" >> $L
  echo "dq6 $t $OLD ${Q/--det /}" >> $L
done
# 2
for K in 1 2 4 8; do
  for t in L000 0005 0013 0016; do
    echo "hhK$K $t $B --K $K --det --ga hht --rho 0.8 $FL $T --newton 2 --cg 3" >> $L
    echo "beK$K $t $B --K $K --det --integ be $FL $T --newton 2 --cg 3" >> $L
  done
  echo "hhK$K ficus $B --K $K --det --ga hht --rho 0.8 $T --newton 2 --cg 20" >> $L
  echo "beK$K ficus $B --K $K --det --integ be $T --newton 2 --cg 20" >> $L
  echo "hhK$K wolf --ga hht --rho 0.8 --K $K --newton 2 --cg 4 $T --det" >> $L
  echo "hhK$K bread --ga hht --rho 0.8 --K $K --newton 2 --cg 1 $T --det" >> $L
  for t in L000 ficus; do echo "exK$K $t --N $K" >> $L; done
  for t in L000 ficus wolf bread; do echo "mpmK$K $t --nsub $K --sub_chk --rpic_cfg" >> $L; done
done
wc -l $L; cp $L $R/list.txt; rclone copy $R/list.txt $RR -q
# 마감: 잡 만료 (11-05 12:34 KST) 25 분 전부터 새 판 안 띄움; 12:20 KST 에는 돌고 있는 판이 있어도 표·끝 표식을 냄 (그 판은 표에 "미완")
AN_UNTIL=$(date -d '2026-11-05 12:09 KST' +%s) AN_CARDS="${AN_CARDS:-1 2 3 4 5 6}" AN_PAR=${AN_PAR:-5} AN_GAP=4 bash exe/fem_an_bigdt_pool.sh bigdt2 $L > $R/pool.log 2>&1 &
PP=$!
HARD=$(date -d '2026-11-05 12:20 KST' +%s); n=0
while kill -0 $PP 2>/dev/null && [ "$(date +%s)" -lt "$HARD" ]; do
  sleep 60; n=$((n + 1))
  if [ $((n % 10)) = 0 ]; then $PY exe/fem_an_bigdt2_table.py $R > $R/bigdt2_table.md 2>&1; rclone copy $R/bigdt2_table.md $RR -q; rclone copy $R/pool.log $RR -q; fi
done
kill -0 $PP 2>/dev/null && echo "[12:20 마감] 풀이 아직 돎 -- 지금까지로 표" >> $R/bigdt2.log
$PY exe/fem_an_bigdt2_table.py $R > $R/bigdt2_table.md 2>&1
echo "[bigdt2 끝] $C $(date '+%F %T') 안 띄운 판 $(cat $R/skipped.txt 2>/dev/null | wc -l)" | tee -a $R/bigdt2.log > $R/bigdt2.END
rclone copy $R/bigdt2_table.md $RR -q; rclone copy $R/pool.log $RR -q; rclone copy $R/bigdt2.log $RR -q; rclone copy $R/skipped.txt $RR -q 2>/dev/null
rclone copy $R/bigdt2.END $RR -q
