# PGND (kywind/pgnd @ 06ef5db) -- anchorflow 패치 기록

클러스터 사본: /home/dkta/work/lb/pgnd (setup.sh 가 clone 후 `git apply anchorflow.patch`).
학습 결과(손실·모델·물리)에 영향을 주는 변경은 없다. 바뀐 것은 체크포인트 저장/재개뿐이다.

## experiments/train/train_eval.py
1. 체크포인트에 Adam 상태·스케줄러 상태·iteration·total_step_count 를 함께 저장 (원본은 material 만).
   `.tmp` 에 쓰고 `os.replace` (72 h 잡 한도에서 잘려도 깨진 파일이 남지 않게).
   eval 코드가 읽는 `ckpt['material']` 키는 그대로.
2. 재개 시 체크포인트에 optimizer/scheduler 가 있으면 복원 (원본은 Adam 모멘트를 잃고 스케줄러만 last_epoch 로 맞춤).
3. `train.resume_iteration=-1` 이면 ckpt/ 에서 가장 큰 번호로 자동 재개 (없으면 0 = 처음부터). 이미 num_iterations 면 종료.
4. 스케줄러 step 을 저장 앞으로 옮김 (같은 반복 안의 순서만 바뀜 -- optimizer.step 뒤 scheduler.step 은 원본과 동일).

## 데이터 (코드 변경 아님, exe/lb_to_pgnd.py)
- 그리퍼 없음 = 영역 밖 먼 곳의 열린 가짜 그리퍼 1 개 (sloth 설정 num_grippers=1, gripper_forcing=True 그대로).
- 축: 파일에 (x,-y,-z) 로 저장 -> 공식 R 뒤 +y 가 위.

## 설정 (sloth 공식 스크립트 대비 오버라이드)
- train.source_dataset_name=data/lego_lb/sub_episodes_v, train.dataset_name=lego/dataset, train.name=lego/train
- sim.preprocess_scale=0.45 (공식 손잡이; 2 m 상자 -> [0,1] 영역)
- train.dataset_load_skip_frame=6 (60 fps -> 공식 물리 스텝 0.1 s)
- train.training_start/end_episode=0/100, eval_start/end_episode=100/120
- train.resume_iteration=-1 (패치 3)
5. `torch.autograd.set_detect_anomaly(True)` (Trainer.__init__) 를 환경변수로: 기본은 공식대로 켬,
   `PGND_NO_ANOMALY=1` 이면 끔. 디버그 검사라 계산은 같다. 실측(lbdata, GPU 5 공유): 켬 2.49 s/it, 끔 1.21 s/it.
   재개가 Adam 상태까지 복원하므로 중간에 바꿔도 이어진다. 끌지는 사용자 결정.
