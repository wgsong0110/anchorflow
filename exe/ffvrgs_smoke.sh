#!/bin/bash
# [final_fix_vrgs] 짧은 확인 (인스턴스 코드 폴더에서): 반복 FF_IT (기본 1, 논문 6.1), 접촉 마찰 FF_FRIC (기본 none)
#   (1) 레고 80 프레임 E 245525.5 (반복 11 에서 맞춘 값; 성분 하나라 충돌 경로 없음 -- 돌기만 확인)  (2) pillow2sofa FF_FR 프레임 E FF_E 방석 평균 z 정답 대조
W=/home/dkta/work; O=/dev/shm/dkta/ffv2; mkdir -p $O; IT=${FF_IT:-1}; FR=${FF_FRIC:-none}
export CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1
python -u exe/pareto_vrgs.py --mode run --out $O/lego --E 245525.5146265931 --data /dev/shm/dkta/anerr_data/L000.pt --tag L000 --frames 80 --floor vel --set 167x$IT > $O/lego.log 2>&1; tail -2 $O/lego.log
for E in ${FF_E:-6071.540272375929}; do
  python -u exe/pareto_vrgs.py --mode run --out $O/p2s_$E --E $E --data $W/newsc/pillow2sofa.pt --tag pillow2sofa --frames ${FF_FR:-100} --floor none --cfg_bc --pg_vdamp --set 200x$IT --contact_fric $FR --dump_F > $O/p2s_$E.log 2>&1; tail -2 $O/p2s_$E.log
  [ -f $O/p2s_$E/pred.pt ] && python exe/fpabl_check_release.py --data $W/newsc/pillow2sofa.pt --pred $O/p2s_$E/pred.pt 2>&1 | grep -E "상자|프레임 +(1|100|300) "
done
