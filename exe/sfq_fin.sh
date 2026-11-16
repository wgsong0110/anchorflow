#!/bin/bash
# [CM192] 대기열 작업: 장면 마무리 (렌더 vis -> 분석 an -> 추가 지표 -> 비교 영상). 모든 방법 계획이 끝난 뒤. bash exe/sfq_fin.sh <장면> <우리 접미> <드라이버 추가 인자...>
s=$1; ht=$2; shift 2
python -u exe/stab1115_drv.py --scene $s --par ${SF_PAR:-4} --stages vis --methods pg,ipg,omni,ours --hextag $ht "$@" || exit 1
python -u exe/stab1115_an.py --scene $s --out /home/dkta/work/stab/$s/an.json --hextag $ht || exit 1
bash exe/sfq_extra.sh $s $ht || exit 1
python -u exe/stabfinal_vid.py --root /home/dkta/work/stab --scene $s --font /home/dkta/work/font.otf --tag $ht --out /home/dkta/work/stab/$s/vis/${s}_stab_cmp.mp4
