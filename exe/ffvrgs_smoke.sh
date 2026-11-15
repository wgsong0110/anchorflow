#!/bin/bash
# [final_fix_vrgs] 충돌 넣은 VR-GS 짧은 확인 (인스턴스 코드 폴더에서): (1) 레고 = 예전 측정 명령 그대로 (E 245525.5, 167x11, 80 프레임) -> 오차 7.5375% 그대로인지 (케이지 성분 하나 -> 충돌 커널 없음)
#   (2) pillow2sofa 100 프레임 (E 10000·3371.8, 200x11) 방석 평균 z 정답 대조 (fpabl_check_release.py)
W=/home/dkta/work; O=/dev/shm/dkta/ffv; mkdir -p $O
export CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1
python -u exe/pareto_vrgs.py --mode run --out $O/lego --E 245525.5146265931 --data /dev/shm/dkta/anerr_data/L000.pt --tag L000 --frames 80 --floor vel --set 167x11 > $O/lego.log 2>&1; tail -3 $O/lego.log
for E in 10000 3371.817380817165; do
  python -u exe/pareto_vrgs.py --mode run --out $O/p2s_$E --E $E --data $W/newsc/pillow2sofa.pt --tag pillow2sofa --frames ${FF_FR:-100} --floor none --cfg_bc --pg_vdamp --set 200x11 --dump_F > $O/p2s_$E.log 2>&1; tail -3 $O/p2s_$E.log
  python exe/fpabl_check_release.py --data $W/newsc/pillow2sofa.pt --pred $O/p2s_$E/pred.pt 2>&1 | tail -20
done
