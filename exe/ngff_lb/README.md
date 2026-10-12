# NGFF 발산 진단 패치 (2026-10-12)

`diag.patch` 를 클러스터 NGFF_lb (기존 anchorflow 수정 위) 에 `git apply` 한다. AF_DIAG 가 없으면 동작이 원본과 같다.
- AF_DIAG=dir : 배치마다 dir/batches.jsonl (에폭, 전역 스텝, 학습률, 손실, 기울기 노름·최대값, 갱신 크기, 가중치 노름, 샘플별 손실, 샘플 번호)
- AF_DIAG_SAVE="40-65,175-195" : 그 에폭마다 전체 체크포인트 dir/epNNNN.pt (모델·Adam·스케줄러·난수)
- AF_DIAG_STOP=에폭 : 그 에폭 끝나고 멈춤
