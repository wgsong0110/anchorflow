#!/bin/bash
# [final_fix_mpmbl] 재질 값 확인 (전 구간) + OmniPhysGS bread·wolf 다시 (필요할 때). 이미 세팅된 mpmbl 인스턴스 안에서 (코드 /root/mb/code, 자료 그대로).
#   1) PG 명시 MPM (정답 경로) 을 재질 키 안 넘김 (base: PG 기본 항복 0·마찰각 25) vs 넘김 (extra: config 항복 200·마찰각 30) 으로 전 구간 -> 정답과의 오차
#      둘이 다르면 정답과 맞는 쪽의 값, 같으면 (PG 가 그 키를 안 씀) config 값을 OmniPhysGS 에 줌.
#   2) 판 boot 에서 쓴 값과 다르면 omni_<장면>_m<값> 폴더에 판 a (덤프) b c (시간, 카드 단독) + 렌더·물리·시각. 결과 R2 lbours/final_fix/bl_mpm/
set -u
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUDA_VISIBLE_DEVICES=0
R=r2:storage/result/anchorflow; RB=$R/lbours/final_fix/bl_mpm; X=/workspace/mb; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data
LOG=/root/mb/fix.log; say() { echo "[fix $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $RB/marks/fix_${MB_TAG:-x}.log -q; }
for s in bread wolf; do
  case $s in
    bread) SIM="--gt $W/topo/pg_bread_plastic/simulation_ply --config $W/topo/bread_plastic_config.json --pt $AG/bread.pt --frames 300 --pg_cfg"; NPG=100; NO=34; GTD=$AG/gt_bread_full; FR=300; FLOOR=none
           KEY=yield_stress; ARG=--omni_yield; DEF=0.0; CF=$W/topo/bread_plastic_config.json;;
    wolf) SIM="--gt $W/topo/pg_wolf_sand/simulation_ply --config $W/PhysGaussian/config/wolf_config.json --pt $AG/wolf.pt --frames 50 --pg_cfg"; NPG=1999; NO=134; GTD=$AG/gt_wolf; FR=50; FLOOR=vel
          KEY=friction_angle; ARG=--omni_fric; DEF=25.0; CF=$W/PhysGaussian/config/wolf_config.json;;
  esac
  [ -d /root/mb/ref/$s ] || rclone copy $R/lbours/final_fix/phys/mpm_$s/png /root/mb/ref/$s -q --transfers 16
  [ -s $GTD/config.json ] || python -u exe/pg_h5_to_pt.py --h5_dir $(echo $SIM | grep -oP -- '--gt \K\S+' | sed 's|/simulation_ply||') --config $CF --out_pt /root/mb/gt${s}_unused.pt --out_dir $GTD > /root/mb/gt_$s.log 2>&1
  O=$X/matfull_$s; mkdir -p $O
  for v in base extra; do
    [ -s $O/$v/err.json ] && continue
    EXF=""; [ $v = extra ] && EXF="--mat_extra yield_stress,friction_angle"
    python -u exe/pareto_mpmbl.py --method pg --nsub $NPG $SIM $EXF --out $O/$v > $O/$v.log 2>&1
  done
  VAL=$(python - $O $CF $KEY $DEF <<'PY'
import json, sys
O, cf, key, de = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
e = {v: json.load(open(f"{O}/{v}/err.json"))["err_mean"] for v in ("base", "extra")}
cv = float(json.load(open(cf))[key])
val = cv if abs(e["base"] - e["extra"]) < 1e-6 else (de if e["base"] < e["extra"] else cv)
json.dump(dict(err=e, key=key, config=cv, pg_default=de, chosen=val), open(f"{O}/choice.json", "w"))
print(val)
PY
)
  say "[재질 전 구간] $s $(cat $O/choice.json)"
  rclone copy $O $RB/matfull_$s -q
  USED=$(rclone cat $RB/omni_$s/chosen.txt 2>/dev/null | grep -oP -- "$ARG \K\S+" || echo $DEF)
  python -c "import sys; sys.exit(0 if abs(float('$USED') - float('$VAL')) < 1e-9 else 1)" && { say "[같음] $s omni 값 $USED -- 다시 안 함"; continue; }
  J=omni_${s}_m$VAL; O=$X/$J; mkdir -p $O
  for rep in a b c; do
    [ -s $O/N${NO}_$rep/err.json ] && continue
    DU=""; [ $rep = a ] && DU=--dump
    python -u exe/pareto_mpmbl.py --method omni --nsub $NO $SIM $ARG $VAL --time $DU --out $O/N${NO}_$rep > $O/N${NO}_$rep.log 2>&1
    say "[판 $rep] $J $(tail -1 $O/N${NO}_$rep.log | cut -c1-200)"
  done
  [ -s $O/N${NO}_a/pred.pt ] && rclone copyto $O/N${NO}_a/pred.pt $RB/$J/N${NO}_a/pred.pt -q
  echo "a:1" > $O/tried.txt; echo "$NO $NO 1 $ARG $VAL" > $O/chosen.txt
  [ -f $O/png/$(printf %04d $FR).png ] || { python -u exe/lb_roll_to_h5.py --dump $O/N${NO}_a/pred.pt --traj $GTD --out $O/h5 > $O/roll_h5.log 2>&1
    python -u exe/lb_render.py --sim $O/h5 --config $GTD/config.json --shape $s --out $O/$J.mp4 --png $O/png --frames $FR < /dev/null > $O/render.log 2>&1; }
  rm -rf $O/h5
  [ -s $O/phys.json ] || python -u exe/ffx_phys.py --method omni --scene $s --N $NO --data $AG/$s.pt --pred $O/N${NO}_a/pred.pt --floor $FLOOR --frames $FR --out $O/phys.json \
      --note "대표 설정 덤프 판 (omni N $NO, $ARG $VAL)" > $O/phys.log 2>&1
  [ -s $O/vis.json ] || python -u exe/ffx_vis.py --png $O/png --ref /root/mb/ref/$s --out $O/vis.json > $O/vis.log 2>&1
  [ -d /root/mb/ref/$s ] || say "[경고] 정답 렌더 없음 $s"
  rclone copy $O $RB/$J -q --transfers 8 --exclude "h5/**"
  [ -s $O/phys.json ] && [ -s $O/vis.json ] && touch $O/DONE && rclone copyto $O/DONE $RB/$J/DONE -q
  say "[끝] $J $(tail -1 $O/phys.log | cut -c1-120)"
done
say "FIX_DONE"
