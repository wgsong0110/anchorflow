#!/bin/bash
# 절 CM178: wolf·ficus·bread 판 (렌더 시간용 --rr 사슬) 자료를 인스턴스에: 3DGS 모델 · 정답 궤적 (wolf = topo pg_wolf_sand -> exe/pg_h5_to_pt.py, bread = topo pg_bread_plastic 같은 변환, ficus = R2 lbours/ficus_v0/gt)
export PATH=/home/dkta/bin:$PATH HOME=/root
R=r2:storage/result/anchorflow; MI=$R/_vastmirror; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; TC=/dev/shm/dkta/t2chain; mkdir -p $AG $TC $W/lbours/ficus_v0 $W/topo $W/PhysGaussian/config
rclone copy $MI/pgmodel/wolf_whitebg-trained $W/pgmodel/wolf_whitebg-trained --transfers 16 -q &
rclone copy $MI/pgmodel/ficus_whitebg-trained $W/pgmodel/ficus_whitebg-trained --transfers 16 -q &
rclone copy $MI/pgmodel/bread-trained $W/pgmodel/bread-trained --transfers 16 -q &
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
rclone copy $MI/topo/pg_bread_plastic $W/topo/pg_bread_plastic --transfers 16 -q &
rclone copyto $MI/topo/bread_plastic_config.json $W/topo/bread_plastic_config.json -q &
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q &
rclone copy $R/lbours/ficus_v0/gt $W/lbours/ficus_v0/gt --transfers 16 -q &
wait
cd ${1:-/root/tf/cur}
PY=/home/dkta/.conda/envs/af/bin/python
[ -s $AG/gt_wolf/config.json ] || CUDA_VISIBLE_DEVICES="" $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/gtw.log 2>&1
[ -s $TC/gt_bread/config.json ] || CUDA_VISIBLE_DEVICES="" $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/gtbread_unused.pt --out_dir $TC/gt_bread > /root/gtb.log 2>&1
echo "wolf $(ls $AG/gt_wolf | wc -l) bread $(ls $TC/gt_bread | wc -l) ficus $(ls $W/lbours/ficus_v0/gt | wc -l) models $(ls $W/pgmodel | tr '\n' ' ')"
