#!/bin/bash
# elastic_new 진단 (EN_BASE 로 기준 명령 파일 지정, 감싸개는 맨 앞): 기준 명령 ($W/ours/<SC>_K1/cmd.txt = 공통 수정 판) 에 변형을 얹어 덤프 없이 다시 돌리고 프레임별 뉴턴 요약 (K·덜 풀림·보폭 최소·‖g‖²/F²·det F) 을 남김.
#   en_try.sh SC NAME "SSF 추가 인자" "앞 감싸개 (ssfix 다음)" "실행기 추가 플래그"
set -u
export PYTHONUTF8=1 OMP_NUM_THREADS=8 TMPDIR=/dev/shm/dkta/tmp HOME=/root; mkdir -p /dev/shm/dkta/tmp
SC=$1; NAME=$2; SSF=${3:-}; WR=${4:-}; EX=${5:-}
W=/home/dkta/work; AF=${EN_AF:-/root/sv/af_$SC}; D=$W/try/${SC}_$NAME; mkdir -p $D; RX=r2:storage/result/anchorflow/lbours/elastic_new/try
C=$(cat ${EN_BASE:-$W/ours/${SC}_K1/cmd.txt})
C=${C/ --dump --dump_F/}
C=$(sed -E "s# --out \S+\$##" <<< "$C")
[ -n "$SSF" ] && C=${C/exe\/ssfix_wrap.py /exe\/ssfix_wrap.py $SSF }
[ -n "$WR" ] && C=$(sed "s#python -u exe/#python -u $WR exe/#" <<< "$C")      # 감싸개 맨 앞 (ptan·bloc 은 import 전에 env)
C="$C $EX --out $D"
echo "$C" > $D/cmd.txt
cd $AF; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 timeout 3600 bash -c "$C" < /dev/null > $D/run.log 2>&1; rc=$?
python - $D $rc $(( $(date +%s) - t0 )) <<'PY' > $D/sum.txt
import json, sys, re
D, rc, sec = sys.argv[1:4]
s = {}
try: s = json.load(open(f"{D}/summary.json"))
except Exception: pass
J = {}
for l in open(f"{D}/run.log", errors="ignore"):
    m = re.search(r"\[pdiag\] 프레임 (\d+): J 최소 ([0-9.e+-]+)", l)
    if m: J[int(m.group(1))] = float(m.group(2))
print(f"[{D.split('/')[-1]}] rc {rc} {sec}s 완주 {s.get('frames_done')}/{s.get('args',{}).get('frames')} 발산 {s.get('diverged')} 오차 {None if s.get('err_mean_done') is None else round(100*s['err_mean_done'],2)}% ms {s.get('ms_mean')}")
try:
    for l in open(f"{D}/fem.jsonl"):
        r = json.loads(l); u = r.get("uf", {}); f = u.get("fail", []) or []
        print(f"  t{r['t']:>2} K{r.get('K')} err {100*(r.get('err') or 0):.2f}% ms {r['ms']:.0f} 덜풀림 {sum(f)}/{len(f)} 보폭최소 {min(u.get('smin',[1]) or [1]):.4g} g2최대 {max(u.get('g2',[0]) or [0]):.3f} J최소 {J.get(r['t']+1)}")
except Exception as e: print("  fem.jsonl 없음", e)
PY
cat $D/sum.txt
rclone copy $D $RX/${SC}_$NAME -q --exclude "*.npy" --exclude "pred.pt" --exclude "*.pt"
