# i-PhysGaussian Newmark 가속도 패치 (클러스터 /home/dkta/work/i-physgaussian 사본에 적용된 것)

- `ipg_an_patch.py` + `af_newmark.py` (2026-10-11): 원본의 "BUG 4 FIX"(aⁿ=0) 때문에 Newmark(β 0.25, γ 0.5) 에서
  일정 가속도가 절반만 들어가 자유낙하가 정확히 절반이 되던 것을, 입자 가속도를 저장해 P2G 로 되살려 고침.
- 2026-10-12 추가: 첫 스텝(저장된 가속도 없음)의 aⁿ 을 0 대신 중력 g 로 (정지·무응력 상태 M a0 = M g).
  0 이면 N1 에서 1 프레임 뒤 운동량이 PG 의 절반. 이 부분은 패치 스크립트가 아니라 해 놓은 파일에 직접 넣었다.
- `solver_vs_original.diff`: 원본(implicit_mpm_solver.py.bak_before_an) 대비 지금 파일 전체 차이.
