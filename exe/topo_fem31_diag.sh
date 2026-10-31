# 시험 진단: bread 1 프레임 -- fem 이 느리게 줄어드는 원인 가르기 (격자 크기 vs 입자 det 장벽 계량)
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; C=f5b2369; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/topo_$C; T=$W/topo; O=$T/fem31/test/diag; mkdir -p $O; cd $WT
B="--flow $T/flow_bread_v2.npz --aux $T/aux_bread_v2.npz --opt riem --kappa 0 --tb none --riem_energy logbarrier --frames 1"
r() { n=$1; g=$2; shift 2; CUDA_VISIBLE_DEVICES=$g python -u exe/rep_track2.py $B "$@" --out $O/$n.npz > $O/$n.log 2>&1; echo "$n: $(grep -a '요약\|rror' $O/$n.log | tail -1 | cut -c1-200)"; }
r fem_plain 5 --method fem --mpm_dx 0.013333333333333334 &
r tet_plain 5 --method tet &
r tet_partF 6 --method tet --riem_partF --feas &
r fem_partF_nofeas 6 --method fem --mpm_dx 0.013333333333333334 --riem_partF &
wait; echo DIAG_DONE
