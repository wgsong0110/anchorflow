#!/bin/bash
# 절 AN 후보 판정 (비측정, 전체 프레임): 판 목록 x (lego L000 + 홀드아웃 20, 80 프레임 + ficus 125 프레임 덤프).
# GPU0 를 뺀 빈 카드 (계산 프로세스 0) 마다 일꾼 하나 -- 한 카드에 한 실행씩 순서대로. 끝에 표 (10 % 미만 개수·평균·ficus 주기 비).
# 사용: bash exe/fem_an_cand.sh <이름> "판:플래그(+ 는 빈칸) ..."
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
NM=$1; VS="$2"; R=/tmp/dkta/lbours/fem_an; M=$R/marks; Q=$R/$NM.queue; mkdir -p $M; df -h /tmp | tail -1
SC="L000 $(seq -f %04g 0 19 | tr '\n' ' ')"; UT=${AN_UNIT:-}
: > $Q; for e in $VS; do v=${e%%:*}; for t in $UT ficus $SC; do echo "$t $v $(echo ${e#*:} | tr '+' ' ')" >> $Q; done; done
free_cards() { for i in 1 2 3 4 5 6; do u=$(nvidia-smi -i $i --query-gpu=uuid --format=csv,noheader); n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u"); [ $n -eq 0 ] && echo $i; done; }
for k in $(seq 10); do GS=($(free_cards)); [ ${#GS[@]} -gt 0 ] && break; sleep 60; done
if [ ${#GS[@]} -gt 0 ]; then echo "빈 카드 ${GS[*]} $(date +%T)" | tee $M/${NM}_cards.txt
else                                                                                  # 빈 카드가 10 분 동안 없으면 사용률 낮은 카드 넷에 일꾼 하나씩 (내 실행은 카드당 하나)
  GS=($(for i in $(seq 10); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{for (k in s) print s[k], k}' | sort -n | head -${AN_NW:-4} | awk '{print $2}'))
  echo "빈 카드 없음 -> 사용률 낮은 카드 ${GS[*]} (다른 실행과 함께, 내 실행은 카드당 하나) $(date +%T)" | tee $M/${NM}_cards.txt
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $M/${NM}_cards.txt
fi
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
worker() { G=$1
  while true; do
    L=$(flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"); [ -z "$L" ] && break
    set -- $L; t=$1; v=$2; shift 2; A="$*"; O=$R/${t}_$v; mkdir -p $O
    case $t in ficus) DA="--data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --frames 125 --floor none --cfg_bc --region_density --pg_vdamp --dump";;
      L000) DA="--data /home/dkta/work/one/L000/L000.pt --frames 80 --floor vel";;
      stretch|shear) DA="--data /tmp/dkta/ak_block/$t.pt --frames 80 --floor none --dump";; drop1) DA="--data /tmp/dkta/ak_drop2/drop_f1.pt --frames 80 --floor vel --dump";; *) DA="--data /tmp/dkta/af_data/$t.pt --frames 80 --floor vel";; esac
    echo "카드 $G $(cat .done) $(date +%T)" > $O/run_info.txt
    CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_rt_sim.py --out $O --tag $t $DA --K 1 --integ mid $A ${AN_X:-} < /dev/null > $O/log.txt 2>&1
    rclone copy $O r2:storage/result/anchorflow/lbours/fem_an/${t}_$v -q
  done; }
for G in ${GS[@]}; do worker $G & done; wait
VL=$(for e in $VS; do echo -n "${e%%:*},"; done); VL=${VL%,}
{ python exe/fem_an_table.py $R $VL $(echo L000 $(seq -f %04g 0 19) | tr ' ' ',')
  for e in $VS; do v=${e%%:*}; echo "## ficus $v $(grep -a '롤아웃' $R/ficus_$v/log.txt | tail -1 | cut -c1-160)"
    [ -f $R/ficus_$v/pred.pt ] && python -u exe/fem_p_period.py --data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --pred $R/ficus_$v/pred.pt 2>&1 | grep -v Warning; done
  if [ -n "$UT" ]; then B=/tmp/dkta/ak_block; D=/tmp/dkta/ak_drop2
    for k in stretch shear; do python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt $(for e in $VS; do v=${e%%:*}; [ -f $R/${k}_$v/pred.pt ] && echo "$v:$R/${k}_$v/pred.pt"; done) 2>&1 | grep -v "^ \|Traceback\|Warning" | sed 's/, 처음 10 프레임.*//'; done
    python exe/fem_ak_drop_ana.py $D/init.pt mpm:$D/mpm/x240.pt:0.0041666667 $(for e in $VS; do v=${e%%:*}; [ -f $R/drop1_$v/pred.pt ] && echo "$v:$R/drop1_$v/pred.pt:0.0166666667"; done) 2>&1 | grep -v Warning | cut -c1-220
    for e in $VS; do v=${e%%:*}; for t in $UT; do echo "${t}_$v $(grep -a '롤아웃\|rror' $R/${t}_$v/log.txt | tail -1 | cut -c1-160)"; done; done; fi
  for e in $VS; do v=${e%%:*}; python - <<PY
import json,glob,numpy as np
ms=[json.load(open(p)).get('ms_mean') for p in glob.glob('$R/*_$v/summary.json')]; ms=[m for m in ms if m]
print('$v 프레임당 ms (비측정) 중앙값', round(float(np.median(ms)),2) if ms else None, '판 수', len(ms))
PY
  done; } > $M/${NM}_table.md 2>&1
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
