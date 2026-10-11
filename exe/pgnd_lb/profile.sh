source /tools/anaconda3/etc/profile.d/conda.sh; conda activate /home/dkta/.conda/envs/pgnd
cd /home/dkta/work/anchorflow && git fetch -q origin && git merge -q --ff-only FETCH_HEAD
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
CUDA_VISIBLE_DEVICES=5 python -u exe/pgnd_lb/profile.py 2>&1 | grep -v Warning; echo PROFILE_DONE
