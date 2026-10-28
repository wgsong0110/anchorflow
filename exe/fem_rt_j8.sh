#!/bin/bash
# doc/fem_realtime.md 절 K 최종 판: 결과 불변 항목 조합 (기준 i_tmix_asm_r3 설정 + J4 호스트 쪽 (--rawptr --lowsync --gridk2 --p2gk) + E·기울기 float32 대수
# (--eg32 both) + 헤시안 양정치 문 (--hgate)) 80 프레임 3 번 + 구간 계측 1 번 + F 덤프 판 (영상) 렌더, 궤적 대조·퍼짐. 한 GPU 순차.
# 사용: bash exe/fem_rt_j8.sh <GPU>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${1:?GPU}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
L=$W/lbours/fem_rt_lego/k_final_log_$C; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/k_final_log_$C; mkdir -p $L
BASE="--floor particle --fast --cuda --graph --lite --warp --near32 --sym --true_res --asm_k --pcg2 6 --mix_delta 0.1"
FIN="$BASE --rawptr --lowsync --gridk2 --p2gk --eg32 both --hgate"
S="bash exe/fem_rt_sim.sh"
$S k_final_r1 $G $FIN --no_dump --dump_pred
$S k_final_r2 $G $FIN --no_dump --dump_pred
$S k_final_r3 $G $FIN --no_dump --dump_pred
$S k_final_prof $G $FIN --no_dump --dump_pred --prof
$S k_base_r3 $G $BASE --no_dump --dump_pred
bash exe/fem_rt_jfinal.sh k_final_F $G $FIN > $L/final_F.log 2>&1
RT=$W/lbours/fem_rt_lego
{
for p in "j_base_r1 j_base_r2" "j_base_r2 k_base_r3" "j_base_r1 k_base_r3" "j_base_r2 k_final_r1" "k_base_r3 k_final_r1" "k_final_r1 k_final_r2" "k_final_r2 k_final_r3" \
         "j_base_r2 j4_raw_ls_g2_p2g" "j_base_r2 j4_eg32both" "j_base_r2 j4_hgate" "j_base_r2 j_nonear32"; do
  set -- $p; echo "## $1 vs $2"; python exe/fem_rt_cmp.py $RT/$1 $RT/$2 | tail -1
done
python exe/fem_rt_spread.py $RT/j_base_r1 $RT/j_base_r2 $RT/k_base_r3 $RT/k_final_r1 $RT/k_final_r2 $RT/k_final_r3 $RT/k_final_F $RT/j_nonear32 $RT/j4_eg32both $RT/j4_hgate $RT/j4_raw_ls_g2_p2g
} > $L/cmp.log 2>&1
touch $L/J8_DONE; rclone copy $L $R -q
