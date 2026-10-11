# lbdata -> PGND sub_episodes_v (ONLY=k 면 그 episode 하나)
set -u
W=/home/dkta/work
cd $W/anchorflow && git fetch -q origin && git merge -q --ff-only FETCH_HEAD; git log --oneline -1
source /tools/anaconda3/etc/profile.d/conda.sh; conda activate /home/dkta/.conda/envs/pgnd
python -u exe/lb_to_pgnd.py --src $W/lbdata --out $W/lb/pgnd_data/lego_lb/sub_episodes_v --scale ${SCALE:-0.45} \
  --only ${ONLY:--1} --workers ${WORKERS:-12}
echo CONVERT_DONE rc=$?
