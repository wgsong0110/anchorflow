#!/bin/bash
# 절 AN 검증 사다리 (비측정, 고른 조합 하나): 블록 아핀 (신축·전단) · 떨어지는 블록 · ficus 잎 끝 주기 -> lego L000 + 홀드아웃 20. 분석 표를 marks 로.
# 사용: AN_GPUS="1 2 3" bash exe/fem_an_ladder.sh <판 이름> <플래그...>
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
VK=$1; shift; FL="$*"
B=/tmp/dkta/ak_block; D=/tmp/dkta/ak_drop2; R=/tmp/dkta/lbours/fem_an; M=$R/marks; mkdir -p $M
df -h /tmp | tail -1
bash exe/fem_an_run.sh $VK stretch,shear,drop1,ficus $FL > $M/ladder_${VK}_unit.out 2>&1
{
echo "## $VK ($FL) $(cat .done)"
for k in stretch shear; do python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt $VK:$R/${k}_$VK/pred.pt 2>&1 | grep -v "^ \|Traceback\|Warning"; done
python -u exe/fem_p_period.py --data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --pred $R/ficus_$VK/pred.pt 2>&1 | grep -v Warning
python exe/fem_ak_drop_ana.py $D/init.pt mpm:$D/mpm/x240.pt:0.0041666667 $VK:$R/drop1_$VK/pred.pt:0.0166666667 2>&1 | grep -v Warning
for t in stretch shear drop1 ficus; do echo "${t} $(grep -a '롤아웃' $R/${t}_$VK/log.txt | tail -1 | cut -c1-200)"; done
} > $M/ladder_${VK}_unit.md 2>&1
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
SC=L000,$(seq -f %04g 0 19 | paste -sd,)
bash exe/fem_an_run.sh $VK $SC $FL > $M/ladder_${VK}_lego.out 2>&1
python exe/fem_an_table.py $R $VK $SC > $M/ladder_${VK}_lego.md 2>&1
echo "[끝 $VK] $(date +%T)" > $M/ladder_${VK}_done; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
