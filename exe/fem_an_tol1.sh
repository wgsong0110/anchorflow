#!/bin/bash
# 절 AN.14 1 차 묶음: (1) 스모크 -- 허용치 0 (조기 종료 없음) 판이 고정 판과 비트 단위로 같은지 (결정적, L000 10 프레임, 후퇴·HHT),
# (2) 통과하면 카드 0 단독 시간 (예산 상한별, 허용치 0 = 상한까지 다 씀) 과 허용치 훑기 (결정적, 비측정, 카드 1·2·4·6) 를 함께.
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:$PATH
M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $M /dev/shm/dkta/an_lists
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse"
BE="--integ be"; HH="--ga hht --rho 0.7"
L=/dev/shm/dkta/an_lists/smoke1.txt
cat > $L <<EOT
fixbe L000 $B --det $BE --newton 1 --cg 5 --frames 10
t0be L000 $B --det $BE --newton 1 --cg 5 --frames 10 --tolN 0 --tolP 0
fixhh L000 $B --det $HH --newton 1 --cg 5 --frames 10
t0hh L000 $B --det $HH --newton 1 --cg 5 --frames 10 --tolN 0 --tolP 0
adbe L000 $B --det $BE --newton 4 --cg 20 --tolN 1e-2 --tolP 1e-1 --frames 10
EOT
bash exe/fem_an_pool.sh smoke1 $L > $M/smoke1.log 2>&1
R=/dev/shm/dkta/lbours/fem_an/smoke1
h() { /home/dkta/.conda/envs/af/bin/python -c "import json;print(json.load(open('$R/$1_L000/summary.json'))['x_sha1'])" 2>/dev/null; }
echo "후퇴 고정 $(h fixbe) 허용치0 $(h t0be) / HHT 고정 $(h fixhh) 허용치0 $(h t0hh)" > $M/smoke1_hash.txt
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
if [ -z "$(h fixbe)" ] || [ "$(h fixbe)" != "$(h t0be)" ] || [ "$(h fixhh)" != "$(h t0hh)" ]; then echo "[스모크 실패] 멈춤" >> $M/smoke1_hash.txt; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q; exit 1; fi
# (2a) 단독 시간 (백그라운드)
S="$(echo $B | tr ' ' '+')"
X=""
for t in L000 ficus; do p=${t:0:1}; fr=10; [ $t = ficus ] && fr=20
  X="$X ${p}fix:$t:$fr:$S+--integ+be+--newton+1+--cg+5"
  for nc in 1:5 1:10 2:5 2:10 3:10 2:20 4:20; do IFS=: read a b <<< "$nc"; X="$X ${p}$a$b:$t:$fr:$S+--integ+be+--newton+$a+--cg+$b+--tolN+0+--tolP+0"; done
done
AN_LIST="$X" bash exe/fem_an_tolx.sh tolx1 > $M/tolx1.log 2>&1 &
# (2b) 허용치 훑기 (결정적, 상한 6 x 30; 기준 = 8 x 40 허용치 1e-6 / 1e-3)
L=/dev/shm/dkta/an_lists/tol1.txt; : > $L
for t in ficus L000 0016 0001; do for ig in be hh; do [ $ig = be ] && I="$BE" || I="$HH"
  echo "${ig}ref $t $B --det $I --newton 8 --cg 40 --tolN 1e-6 --tolP 1e-3" >> $L
  for n in 1e-1 1e-2 1e-3; do for p in 1e-1 1e-2; do echo "${ig}n${n#1e-}p${p#1e-} $t $B --det $I --newton 6 --cg 30 --tolN $n --tolP $p" >> $L; done; done
done; done
bash exe/fem_an_pool.sh tol1 $L > $M/tol1.log 2>&1
wait
echo "[tol1 끝] $(date '+%F %T')" > $M/tol1.END; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
