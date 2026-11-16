#!/bin/bash
# 절 CM194 (gq_justify) 남의 인스턴스에 얹는 부팅 (설치·빌드 없음, 기존 af 환경 그대로): 코드 tar (R2 lbours/gq_justify/code/gq_<커밋>.tar) -> /root/gq/code/<커밋>,
#   장면 자료 (없을 때만; vast_ffx.sh 의 받기 부분) -> 카드 GQ_GPU 에 일꾼 GQ_NW 개 (exe/gqj_worker.sh). env: GQ_H GQ_Q GQ_SC GQ_NW GQ_TIME GQ_TONLY
set -u
H=${GQ_H:?}; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH HOME=/root
R=r2:storage/result/anchorflow; RX=$R/lbours/paper_remeasure2; MI=$R/_vastmirror; RG=$R/lbours/gq_justify
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; SD=/dev/shm/dkta/anerr_data; mkdir -p /root/gq $AG $SD $W
say() { echo "[gqboot $(date '+%m-%d %T')] $*"; }
D=/root/gq/code/$H; mkdir -p $D; [ -f $D/exe/gqj_worker.sh ] || rclone cat $RG/code/gq_$H.tar | tar -x -C $D
SC=${GQ_SC:-L000,pillow2sofa,bread,wolf}
has() { case ",$SC," in *",$1,"*) return 0;; esac; return 1; }
[ -d $W/i-physgaussian ] || rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q
[ -d $W/pgmodel ] || rclone copy $MI/pgmodel $W/pgmodel --transfers 16 -q
if has L000; then [ -s $SD/L000.pt ] || rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q; [ -d $W/lbdata/test_low/L000 ] || rclone copy $MI/lbdata $W/lbdata --transfers 16 -q; [ -d $W/stab/lego/init ] || rclone copy $MI/stab/lego/init $W/stab/lego/init -q; fi
if has bread; then [ -s $AG/bread.pt ] || rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q; [ -d $W/topo/pg_bread_plastic ] || rclone copy $MI/topo/pg_bread_plastic $W/topo/pg_bread_plastic --transfers 16 -q
  [ -s $W/topo/bread_plastic_config.json ] || rclone copyto $MI/topo/bread_plastic_config.json $W/topo/bread_plastic_config.json -q; fi
if has wolf; then [ -s $AG/wolf.pt ] || rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q; [ -d $W/topo/pg_wolf_sand ] || rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q
  [ -s $W/PhysGaussian/config/wolf_config.json ] || rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q; fi
cd $D
export PYTHONUTF8=1 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_gq XDG_CACHE_HOME=/dev/shm/dkta/cache; mkdir -p $CUPY_CACHE_DIR
G=${GQ_GPU:-0}
has bread && { [ -s $AG/gt_bread/config.json ] || CUDA_VISIBLE_DEVICES=$G python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/gq/gtbread_unused.pt --out_dir $AG/gt_bread --out_dir_n 1 > /root/gq/gt_bread.log 2>&1
  [ -s $AG/gt_bread_full/config.json ] || CUDA_VISIBLE_DEVICES=$G python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/gq/gtbread_unused2.pt --out_dir $AG/gt_bread_full > /root/gq/gt_bread_full.log 2>&1; }
has wolf && { [ -s $AG/gt_wolf/config.json ] || CUDA_VISIBLE_DEVICES=$G python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/gq/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/gq/gt_wolf.log 2>&1; }
if has pillow2sofa; then [ -s $W/newsc/pillow2sofa.pt ] || { export PR_SC=pillow2sofa RX; source exe/pr2_scene_fetch.sh; }; fi
python -c "import pyiqa" 2>/dev/null || pip install -q pyiqa -c <(pip freeze | grep -iE "^(torch|torchvision|numpy|scipy|scikit-image|opencv-python-headless)==") 2>&1 | tail -1
say "자료 L000 $(stat -c %s $SD/L000.pt 2>&1 | cut -c1-20) bread $(stat -c %s $AG/bread.pt 2>&1 | cut -c1-20) wolf $(stat -c %s $AG/wolf.pt 2>&1 | cut -c1-20) p2s $(stat -c %s $W/newsc/pillow2sofa.pt 2>&1 | cut -c1-20) gt_bread_full $(ls $AG/gt_bread_full 2>/dev/null | wc -l)"
[ "${GQ_PREP_ONLY:-0}" = 1 ] && exit 0
WP=""; for i in $(seq 0 $(( ${GQ_NW:-1} - 1 ))); do WID=${GQ_WID0:-0}$i GQ_GPU=$G bash exe/gqj_worker.sh $H > /root/gq/w${GQ_WID0:-0}$i.log 2>&1 & WP="$WP $!"; sleep 20; done
wait $WP
say "끝"
