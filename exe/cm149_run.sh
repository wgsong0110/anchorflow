#!/bin/bash
# 절 CM149 인스턴스 쪽: 정답 h5 (vast 세팅이 받은 /home/dkta/work/topo/pg_wolf_sand/simulation_ply) 와 R2 우리 판 pred.pt 로 꼭짓점 몫 표. 사용 go_cm140.sh 가 판 목록 APEX 일 때 부름.
export PATH=/home/dkta/bin:$PATH
R=r2:storage/result/anchorflow/lbours/wolfcv/cm140; O=/home/dkta/work/cm149; mkdir -p $O; PY=/home/dkta/.conda/envs/af/bin/python
H5=/home/dkta/work/topo/pg_wolf_sand/simulation_ply; ls $H5 | head -2
ARGS=""; for n in TB "TWS+sv+n3" "TWS+nf"; do mkdir -p "$O/$n"; rclone copyto "$R/$n/pred.pt" "$O/$n/pred.pt" -q && ARGS="$ARGS $n=$O/$n/pred.pt"; done
$PY -u exe/cm149_apex.py $H5 $ARGS > $O/cm149_apex.log 2>&1; rclone copyto $O/cm149_apex.log $R/cm149_apex.log -q
echo "끝 $(date)" | rclone rcat $R/cm149.END
