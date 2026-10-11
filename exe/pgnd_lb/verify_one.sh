source /tools/anaconda3/etc/profile.d/conda.sh; conda activate /home/dkta/.conda/envs/pgnd
cd /home/dkta/work/anchorflow && git fetch -q origin && git merge -q --ff-only FETCH_HEAD
python -u exe/pgnd_lb/verify_one.py; echo VERIFY_DONE rc=$?
