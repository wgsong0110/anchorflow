#!/bin/bash
# [blx] 로컬 마무리 (tmux blxfinish): (1) vast 크레딧이 돌아오고 final_blx_sgb 가 멈춰 있으면 start (onstart 가 q4 로 sg_resume 저장점부터 이어 학습 -> 단독 시간 1 회·덤프·지표),
#   (2) R2 marks_SGB/MEAS_DONE 이 생기면 R2 pareto_blx.json 에 Spring-Gaus bread 행 추가, 표식 BREAD_SG_READY, notify (표는 final_fix 의 표 생성기가 다시 만듦 -- 표 파일 직접 안 고침).
R=r2:storage/result/anchorflow/lbours/final_pa/blx; WR=${PAPER:-/home/wgsong/workspace/FusedGS/paper}; ID=60860713
log() { echo "[finish $(date -u '+%F %T')] $*"; }
last=0
until rclone lsf $R/marks_SGB/ 2>/dev/null | grep -q "^MEAS_DONE"; do
  st=$(vastai show instance $ID --raw 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('actual_status'))")
  cr=$(vastai show user --raw 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(1 if (d.get('credit') or 0) > 0.5 and not d.get('billing_failed') else 0)")
  if [ -z "${NOSTART:-}" ] && [ "$st" != running ] && [ "$cr" = 1 ] && [ $(( $(date +%s) - last )) -gt 1800 ]; then log "크레딧 회복, $ID 상태 $st -> start"; vastai start instance $ID 2>&1 | tail -1; last=$(date +%s); fi
  rclone lsf $R/marks_SGB/ 2>/dev/null | grep -q "^MEAS_CHECK_FAIL" && log "MEAS_CHECK_FAIL 표식 -- 결과 확인 필요"
  sleep 120
done
log "MEAS_DONE"
mkdir -p /tmp/blx/res; rclone copyto $R/SGB/grid.json /tmp/blx/res/SGB_grid.json
python3 - <<'PY'
import json, re
g = [r for r in json.load(open('/tmp/blx/res/SGB_grid.json')) if r['method'] == 'springgaus' and r['scene'] == 'bread']
assert g and g[0]['ok'] >= 1, g
r = g[0]
P = '/tmp/blx/res/pareto_blx.json'
out = [o for o in json.load(open(P)) if (o['method'], o['scene']) != ('springgaus', 'bread')]
out.append(dict(method='springgaus', scene='bread', N=r['N'], setting='', reps=r['reps'], ok=r['ok'], err=r['err'], err_all=r['err_all'], ms=r['ms'], ms_all=r['ms_all'],
                div=r['div'], others=[0], E=None, note='재질 확장 (원 논문은 탄성 전용): 스프링 소성 휴지 길이; 시간 1 회 (RTX 6000 Ada 단독, 조정자 11-16)', src='final_pa/blx/SGB/grid.json', ext='material'))
json.dump(out, open(P, 'w'), ensure_ascii=False, indent=1)
print(r['N'], r['err'], r['ms'])
PY
rclone copyto /tmp/blx/res/pareto_blx.json $R/pareto_blx.json
echo "$(date -u) $(python3 -c "import json;r=[q for q in json.load(open('/tmp/blx/res/pareto_blx.json')) if q['method']=='springgaus' and q['scene']=='bread'][0];print(r['N'],round(r['err'],3),round(r['ms'],1))")" | rclone rcat $R/BREAD_SG_READY
notify "blx Spring-Gaus bread 측정 끝" "R2 blx/pareto_blx.json·SGB/g/springgaus_bread_*_dump (phys.json·vis_ffx.json) -- 표 생성기 (fpa_table·ffx_pareto_final) 다시 돌릴 것"
