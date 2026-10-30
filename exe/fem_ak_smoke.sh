#!/bin/bash
# 절 AK 연기 시험 (비측정, 카드 1 하나): 기준·APIC·재시작 20 프레임 -- 진단 (--ak_en) 이 도는지, 아핀 전달이 컴파일·실행되는지
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${AK_G:-1}
python exe/fem_ak_jobs.py "v:base,aff,cfm:,g:1:--frames@20:_smk" 0003 > /tmp/dkta/ak_smoke.txt
python exe/fem_ak_jobs.py "v:base:g:1:--frames@20@--ak_t0@14:_smks" 0003 >> /tmp/dkta/ak_smoke.txt
while read -r ROOT TG ARGS; do
  EX_ROOT=$ROOT bash exe/fem_ab_run.sh "$TG" "$G" $ARGS < /dev/null > /dev/null 2>&1 &
done < /tmp/dkta/ak_smoke.txt
wait
while read -r ROOT TG ARGS; do
  O=/tmp/dkta/lbours/$ROOT/$TG
  echo "== $TG $( [ -f $O/EX_DONE ] && echo 끝 || echo 실패)"; grep -a "Error\|error\|롤아웃\]\|발산\|절 AK" $O/v_fem.log | tail -4 | cut -c1-300
  python -c "import json;r=[json.loads(l) for l in open('$O/fem.jsonl')];a=r[-1].get('ak');print(type(a)==dict and {k:a[k] for k in ['ke_p0','ke_n0','ke_n1','ke_p1','ec0','ec1','ep1','ep_c','ep_gt','ke_gt']} or a);print(r[-1].get('rigid'))" 2>&1 | tail -3
done < /tmp/dkta/ak_smoke.txt
echo AK_SMOKE_DONE
