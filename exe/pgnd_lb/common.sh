# PGND lbdata 공통 인자 (sloth 공식 train_sloth.sh 에서 데이터·배율·프레임 간격·분할만 바꿈)
W=/home/dkta/work
P=$W/lb/pgnd
source /tools/anaconda3/etc/profile.d/conda.sh; conda activate /home/dkta/.conda/envs/pgnd
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 2026-10-11 첫 실행이 단편화 OOM (할당 14.5G, 예약-미사용 16G) -- 계산 불변, 배치 32 유지
PGND_ARGS=(
 'gpus=[0]' overwrite=False resume=True debug=True
 train.source_dataset_name=data/lego_lb/sub_episodes_v
 sim.preprocess_scale=0.45
 train.dataset_load_skip_frame=6 train.dataset_skip_frame=1
 train.num_workers=8 train.batch_size=${BATCH:-32}
 train.training_start_episode=0 train.training_end_episode=100
 train.eval_start_episode=100 train.eval_end_episode=120
 train.iteration_log_interval=10
 train.resume_iteration=-1
)
