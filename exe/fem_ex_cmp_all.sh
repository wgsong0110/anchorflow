#!/bin/bash
# 절 L 결과 불변 대조 (exe/fem_rt_cmp.py, 위치 궤적 프레임 평균 |Δx|/EXT) -- 잡 안 (계산 노드) 에서만. 출력 lbours/fem_ex_lego/cmp_all.txt
export PYTHONUTF8=1 OMP_NUM_THREADS=8
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
L=/home/dkta/work/lbours/fem_ex_lego; O=$L/cmp_all.txt; : > $O
pair() { [ -f $L/$1/pred.pt ] && [ -f $L/$2/pred.pt ] && { echo "== $1 vs $2 ($3)" >> $O; python -u exe/fem_rt_cmp.py $L/$1 $L/$2 2>&1 | tail -1 >> $O; }; }
pair s5_fu_n128_p1 s5_fu_n128_p2 "같은 설정 반복 (융합)"
pair s6_sort_n128_p1 s6_sort_n128_p2 "같은 설정 반복 (정렬)"
pair s6_sort_n96_p1 s6_sort_n96_p2 "같은 설정 반복 (정렬, N 96)"
pair s5_fu_n96_p1 s6_fu_n96_p2 "같은 설정 반복 (융합, N 96)"
pair s5_base_n128_p s5_nofu_f64n_n128_p "기준 판 vs 최적화 (융합 전, 바닥 후보 f64)"
pair s5_nofu_f64n_n128_p s5_nofu_n128_p "바닥 후보 f64 vs f32"
pair s5_nofu_n128_p s5_nofu_nog_n128_p "그래프 vs 그래프 없음"
pair s5_nofu_n128_p s5_fu_n128_p1 "융합 전 vs 융합"
pair s5_fu_n128_p1 s6_sort_n128_p1 "융합 vs 정렬·구간 합"
pair s5_base_n128_p s6_sort_n128_p1 "기준 판 vs 최종"
pair s5_fu_n96_p1 s6_sort_n96_p1 "융합 vs 정렬 (N 96)"
cat $O; rclone copy $O r2:storage/result/anchorflow/lbours/fem_ex_lego/ -q
echo CMP_DONE
