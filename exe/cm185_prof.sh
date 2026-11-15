#!/bin/bash
# 사용: prof.sh 장면 실행디렉토리(cmd.txt 있는 곳) -- 12 프레임, 프레임 4·8 에서 커널별 시간 (그래프 밖), 결과 R2 lbours/gfscenes/prof/<장면>
export PYTHONUTF8=1 OMP_NUM_THREADS=8
n=$1; O=$2; P=/root/prof/$n; mkdir -p $P
until ! pgrep -f "gfs_render.py|fem_an_rt_sim.py|gfs_post.py" >/dev/null; do sleep 10; done
C=$(cat $O/cmd.txt | sed -e 's/ --dump_part//; s/ --dump_F//; s/ --dump / /' -e "s#--frames [0-9]*#--frames 12#" -e "s#--out [^ ]*\$#--prof 4 --out $P#")
echo "$C" > $P/cmd.txt
cd /root/sv/af && CUDA_VISIBLE_DEVICES=0 timeout 1200 bash -c "$C" > $P/log.txt 2>&1
python3 - $P/fem.jsonl > $P/prof.txt <<'PY'
import json,sys
r=[json.loads(l) for l in open(sys.argv[1])]
for x in r:
    kp=x.get('kernel_ms')
    if not kp: continue
    tot=sum(v[0] for v in kp.values()); print('frame',x['t'],'ms',round(x['ms'],2),'prof sum',round(tot,2))
    for k,v in sorted(kp.items(), key=lambda q:-q[1][0])[:25]: print('  %-22s %8.3f ms %4d' % (k, v[0], v[1]))
PY
rclone copy $P r2:storage/result/anchorflow/lbours/gfscenes/prof/$n -q
