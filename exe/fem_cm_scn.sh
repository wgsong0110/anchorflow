#!/bin/bash
# 절 CM 공통 설정: 씬 이름 -> 실행기 (SIM) 와 씬 인자 (DA). source 해서 cm_scene <씬> <공통 인자...> 를 부르면 SIM·DA·XA 를 채운다.
# 탄성 (lego L000·0000..0019, ficus) = exe/fem_an_rt_sim.py (절 AN 판; 장벽 바닥·MG 장벽 몫·--mv2 등), 재질 (wolf 모래, bread plasticine) = exe/fem_ao_rt_sim.py (절 AO·AQ 판).
# 공통 인자 (적분기·걸음 수·반복 상한·허용치) 는 두 실행기에 같은 글자 그대로; 실행기 한쪽에만 있는 인자는 아래 판별 묶음 (BA·BO) 에만.
SD=/dev/shm/dkta/anerr_data; mkdir -p $SD
LD=/tmp/dkta/af_data; [ -f $LD/0019.pt ] || LD=$SD
AG=/tmp/dkta/ag_data; mkdir -p $AG 2>/dev/null
BA="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2"   # 절 AN.22 판 (탄성 실행기 전용)
BO="--hspec --mlow 1e-3 --integ mid"                                                                          # 절 AO·AQ 판 (재질 실행기 전용)
FI=$SD/F_pg_v0.pt
cm_data() { local t=$1
  case $t in
    L000) [ -s $SD/L000.pt ] || cp /home/dkta/work/one/L000/L000.pt $SD/L000.pt;;
    ficus) [ -s $FI ] || cp /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt $FI;;
    wolf|bread) [ -s $AG/$t.pt ] || rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/$t.pt $AG/$t.pt -q;;
    *) [ -s $LD/$t.pt ] || { cp /home/dkta/work/lbpt/test/$t.pt $LD/$t.pt.$$ && mv $LD/$t.pt.$$ $LD/$t.pt; };;
  esac; }
cm_scene() { local t=$1; shift; local A="$*"
  cm_data $t
  case $t in
    wolf)  SIM=exe/fem_ao_rt_sim.py; DA="--data $AG/wolf.pt --frames 50 --cell 2 --floor vel --pbc $BO";;
    bread) SIM=exe/fem_ao_rt_sim.py; DA="--data $AG/bread.pt --frames 300 --cell 1 --floor none --pbc --pg_vdamp --visc_rpic $BO";;
    ficus) SIM=exe/fem_an_rt_sim.py; DA="--data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp --dump $BA";;
    L000)  SIM=exe/fem_an_rt_sim.py; DA="--data $SD/L000.pt --frames 80 --floor vel --fl_bar --fl_bst 1 $BA";;
    *)     SIM=exe/fem_an_rt_sim.py; DA="--data $LD/$t.pt --frames 80 --floor vel --fl_bar --fl_bst 1 $BA";;
  esac
  if [ $SIM = exe/fem_an_rt_sim.py ]; then A=${A//--ga_y0ut/}; fi                                             # 처치 (a) 끔은 재질 실행기에만 있는 인자 (탄성 판 HHT 는 늘 켬)
  XA="$A"; }
