#!/bin/bash
# 절 CM2 공통 설정: 모든 씬 (lego L000·0000..0019, 낙하 drop1, ficus, wolf 모래, bread plasticine) 을 한 실행기 exe/fem_an_rt_sim.py 로 (재질 판은 lib/anchorflow/fem_mat_rt.py 가 cfg 재질로 고름).
# source 해서 cm_scene <씬> <공통 인자...> 를 부르면 SIM·DA·XA 를 채운다. 공통 묶음 BA 는 모든 씬 같은 글자; 바닥 있는 씬 (lego·낙하·wolf) 은 장벽 바닥 FB 같은 글자.
# 씬 인자 (자료·프레임·셀 규칙 값·바닥 유무·config 구속·감쇠) 만 씬별. 재질 씬의 --pbc --mlow 1e-3 (--visc_rpic) 는 재질 몫 (탄성 씬에는 해당 없음).
SD=/dev/shm/dkta/anerr_data; mkdir -p $SD
LD=/tmp/dkta/af_data; [ -f $LD/0019.pt ] || LD=$SD
AG=/dev/shm/dkta/ag_data; mkdir -p $AG 2>/dev/null
[ -s /tmp/dkta/ag_data/wolf.pt ] && AG=/tmp/dkta/ag_data
BA="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut"
FB="--floor vel --fl_bar --fl_bst 1"
FI=$SD/F_pg_v0.pt
DD=/tmp/dkta/ak_drop2; [ -s $DD/drop_f1.pt ] && [ -s $DD/mpm/x240.pt ] || DD=$SD/ak_drop2
cm_data() { local t=$1
  case $t in
    L000) [ -s $SD/L000.pt ] || cp /home/dkta/work/one/L000/L000.pt $SD/L000.pt;;
    ficus) [ -s $FI ] || cp /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt $FI;;
    drop1) [ -s $DD/mpm/x240.pt ] || rclone copy r2:storage/result/anchorflow/lbours/ak_drop2 $DD -q --include "drop_f1.pt" --include "init.pt" --include "mpm/x240.pt";;
    wolf|bread) [ -s $AG/$t.pt ] || rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/$t.pt $AG/$t.pt -q;;
    *) [ -s $LD/$t.pt ] || { cp /home/dkta/work/lbpt/test/$t.pt $LD/$t.pt.$$ && mv $LD/$t.pt.$$ $LD/$t.pt; };;
  esac; }
cm_scene() { local t=$1; shift
  cm_data $t
  SIM=exe/fem_an_rt_sim.py
  case $t in
    wolf)  DA="--data $AG/wolf.pt --frames 50 $FB --pbc --mlow 1e-3 $BA --cell 2";;              # 셀 규칙 값 2 (BA 의 --cell 1 뒤라 덮음)
    bread) DA="--data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 $BA";;
    ficus) DA="--data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp --dump $BA";;
    drop1) DA="--data $DD/drop_f1.pt --frames 80 $FB --dump $BA";;
    L000)  DA="--data $SD/L000.pt --frames 80 $FB $BA";;
    *)     DA="--data $LD/$t.pt --frames 80 $FB $BA";;
  esac
  [ -n "${CM_NOBA:-}" ] && DA=${DA//$BA/}                                                     # 확인용 (옛 판 비교): 공통 묶음 빼기
  XA="$*"; }
