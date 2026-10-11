# PGND 공식 코드 사본 + 전용 env (클러스터 잡 안에서: bash exe/job_run.sh .../setup.sh .../pgnd_setup.out)
set -u
W=/home/dkta/work
cd $W/anchorflow && git fetch -q origin && git merge -q --ff-only FETCH_HEAD; git log --oneline -1
P=$W/lb/pgnd
if [ ! -d $P/.git ]; then
  git clone -q https://github.com/kywind/pgnd $P && git -C $P checkout -q 06ef5db
  git -C $P apply $W/anchorflow/exe/pgnd_lb/anchorflow.patch && echo "[패치] 적용"
fi
cp $W/anchorflow/exe/pgnd_lb/PATCHES_anchorflow.md $P/
git -C $P log --oneline -1; git -C $P status --short
mkdir -p $W/lb/pgnd_data/lego_lb $P/experiments/log/data
ln -sfn $W/lb/pgnd_data/lego_lb $P/experiments/log/data/lego_lb
source /tools/anaconda3/etc/profile.d/conda.sh
E=/home/dkta/.conda/envs/pgnd
[ -x $E/bin/python ] || conda create -y -q -p $E python=3.10
conda activate $E
python -m pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -q dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
python -m pip install -q warp-lang==1.1.0 numpy==1.26.4 scipy scikit-learn scikit-image open3d kornia omegaconf \
  hydra-core wandb lpips pyvista opencv-python-headless matplotlib h5py pyyaml tqdm ipdb psutil
cd $P
CUDA_VISIBLE_DEVICES=5 python - <<'PY'
import torch, warp as wp, dgl, kornia, lpips, numpy, scipy, sklearn, open3d, pyvista, hydra
from dgl.geometry import farthest_point_sampler
print("torch", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), "warp", wp.__version__, "dgl", dgl.__version__, "numpy", numpy.__version__)
wp.init()
x = torch.rand(1, 5000, 3, device="cuda"); print("fps", farthest_point_sampler(x, 10, start_idx=3)[0][:5].tolist())
lpips.LPIPS(net="alex"); print("lpips ok")
import sys; sys.path += ["experiments", "."]
from train import train_eval; print("train_eval import ok")
PY
echo SETUP_DONE
