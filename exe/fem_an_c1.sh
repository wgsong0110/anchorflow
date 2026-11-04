#!/bin/bash
# 절 AN.17 묶음 c1 (잡 안, 비측정, 결정적): (1) lego 0016 접촉 분석 (j1 덤프, exe/fem_an_contact_ana.py, CPU),
# (2) 점별 고착 (--fl_pt, 정의 변경) x 적분기 후보 (후퇴 K 2, HHT ρ 0.7·0.8·0.9, generalized-α ρ 0.8) 판정 묶음 -- j1 과 같은 공통 허용치 εN 1e-2 · εP 1e-1 + 씬별 상한
#     (lego·낙하·블록 1 x 5, ficus 2 x 10), 궤적 lego L000 + 0000 에서 0019, ficus, 블록 신축·전단, 낙하; 0016·0003 덤프 (접촉 분석용).
# 결과: R2 lbours/fem_an/c1/<판>_<궤적>/, 표 marks/c1_table.md, 분석 marks/c1_diag.txt·c1_diag2.txt, 끝 표식 marks/c1.END
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1
PY=/home/dkta/.conda/envs/af/bin/python
M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $M /dev/shm/dkta/an_lists
J1=/dev/shm/dkta/lbours/fem_an/j1; TJ=/tmp/dkta/af_data/0016.pt
[ -s $TJ ] || TJ=/dev/shm/dkta/anerr_data/0016.pt
echo "[c1 시작] $(cat .done) $(date '+%F %T')" > $M/c1.log
# (1) 접촉 분석 (CPU, 카드 안 씀)
( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 $PY -u exe/fem_an_contact_ana.py $TJ bej:$J1/bej_0016/pred.pt hhj:$J1/hhj_0016/pred.pt beref:$J1/beref_0016/pred.pt hhref:$J1/hhref_0016/pred.pt --pic > $M/c1_diag.txt 2>&1
  rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q ) &
# (2) 판정 묶음
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --det"
T="--tolN 1e-2 --tolP 1e-1"
L=/dev/shm/dkta/an_lists/c1.txt; : > $L
VS=("bejp:--integ be --fl_pt" "hhjp:--ga hht --rho 0.7 --fl_pt" "hh8:--ga hht --rho 0.8" "hh8p:--ga hht --rho 0.8 --fl_pt" "hh9:--ga hht --rho 0.9" "hh9p:--ga hht --rho 0.9 --fl_pt" "ga8:--ga ga --rho 0.8" "ga8p:--ga ga --rho 0.8 --fl_pt")
for vv in "${VS[@]:0:2}"; do v=${vv%%:*}; I=${vv#*:}                         # 먼저: 점별 고착 두 판의 0016·0003 덤프·낙하
  for t in 0016 0003; do echo "$v $t $B $I $T --newton 1 --cg 5 --dump" >> $L; done
  echo "$v drop1 $B $I $T --newton 1 --cg 5" >> $L; done
echo "benv 0016 $B --integ be --no_vfl_out $T --newton 1 --cg 5 --dump" >> $L     # 대조: 이탈 해제 끔 (정답 sticky 처럼 세 성분 모두)
echo "benv drop1 $B --integ be --no_vfl_out $T --newton 1 --cg 5" >> $L
for vv in "${VS[@]}"; do v=${vv%%:*}; I=${vv#*:}
  echo "$v ficus $B $I $T --newton 2 --cg 10" >> $L
  for k in stretch shear; do echo "$v $k $B $I $T --newton 1 --cg 5" >> $L; done
  grep -q "^$v drop1 " $L || echo "$v drop1 $B $I $T --newton 1 --cg 5" >> $L
  for t in L000 $(seq -f %04g 0 19); do grep -q "^$v $t " $L && continue; D=""; { [ $t = 0016 ] || [ $t = 0003 ]; } && D="--dump"; echo "$v $t $B $I $T --newton 1 --cg 5 $D" >> $L; done
done
wc -l $L >> $M/c1.log
AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh c1 $L > $M/c1_pool.log 2>&1
wait
R=/dev/shm/dkta/lbours/fem_an/c1
$PY exe/fem_an_tol_table.py $R > $M/c1_table.md 2>&1
{ for v in bejp hhjp benv hh8 hh8p hh9 hh9p ga8 ga8p; do echo "## $v drop"; cat $R/${v}_drop1/drop_ana.txt 2>/dev/null | tail -1; done; } >> $M/c1_table.md
A=""; for v in bejp hhjp benv hh8p hh9p ga8p; do [ -s $R/${v}_0016/pred.pt ] && A="$A $v:$R/${v}_0016/pred.pt"; done
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 $PY -u exe/fem_an_contact_ana.py $TJ bej:$J1/bej_0016/pred.pt $A > $M/c1_diag2.txt 2>&1
echo "[c1 끝] $(date '+%F %T')" > $M/c1.END; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
