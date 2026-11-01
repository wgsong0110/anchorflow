#!/bin/bash
# 절 AK.33 (비측정): B-스플라인 요소 상한 없음 -- 입자 적분점 (bn) 과 칸 2x2x2 가우스 적분점 (gq), 바닥 고착 이탈 제외 (vfl_out).
# 블록 아핀 (신축·전단) · ficus 주기 · 떨어지는 블록 -> lego 21 (두 판).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
df -h /tmp | tail -1
B=/tmp/dkta/ak_block; D=/tmp/dkta/ak_drop2; R=/tmp/dkta/lbours/fem_ak_bselc; SC=L000,$(seq -f %04g 0 19 | paste -sd,)
[ -f $D/drop_f1.pt ] || rclone copyto r2:storage/result/anchorflow/lbours/ak_drop2/drop_f1.pt $D/drop_f1.pt -q
bash exe/fem_bsel_cuda_run.sh bnv stretch,shear,ficus,drop1 --theta 0 --vfl_out
bash exe/fem_bsel_cuda_run.sh gqv stretch,shear,ficus,drop1 --theta 0 --vfl_out --gq
for k in stretch shear; do python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt nocap:/tmp/dkta/lbours/fem_ak_block/${k}_nocap/roll_fem.pt bn:$R/${k}_bnv/pred.pt gq:$R/${k}_gqv/pred.pt 2>&1 | grep -v "^ \|Traceback\|Warning"; done
python -u exe/fem_p_period.py --data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --pred $R/ficus_bnv/pred.pt $R/ficus_gqv/pred.pt 2>&1 | grep -v Warning
python exe/fem_ak_drop_ana.py $D/init.pt mpm:$D/mpm/x240.pt:0.0041666667 bn:$R/drop1_bnv/pred.pt:0.0166666667 gq:$R/drop1_gqv/pred.pt:0.0166666667 2>&1 | grep -v Warning
for t in stretch shear ficus drop1; do for v in bnv gqv; do echo "${t}_$v $(grep -a '롤아웃' $R/${t}_$v/log.txt | tail -1 | cut -c1-170) $(grep -a '가우스 점' $R/${t}_$v/log.txt | tail -1 | grep -o '가우스 점.*')"; done; done
echo "[시험 끝] $(date +%T)"
bash exe/fem_bsel_cuda_run.sh bnv $SC --theta 0 --vfl_out
bash exe/fem_bsel_cuda_run.sh gqv $SC --theta 0 --vfl_out --gq
for v in bnv gqv; do for t in L000 $(seq -f %04g 0 19); do echo "$t $v $(grep -a '롤아웃' $R/${t}_$v/log.txt | tail -1 | cut -c1-170)"; done; done
echo "[끝] $(date +%T)"
