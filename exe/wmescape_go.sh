#!/bin/bash
# 수박 입자 날아감 비교 영상 진입점 (잡 133077 안, exe/job_run.sh 와 같은 pbs_attach 떼어냄으로 `rclone cat .../wmescape/code/wmescape_go.sh | bash`).
# 코드 = R2 lbours/wmescape/code/launch.txt 의 커밋 bundle -> /tmp, 결과 -> R2 lbours/wmescape/run2 (판 끝날 때마다 + 5 분마다). 끝 표식 R2 wmescape.END.
export PATH=/home/dkta/bin:$PATH
RC=r2:storage/result/anchorflow/lbours/wmescape/code; R=r2:storage/result/anchorflow/lbours/wmescape/run2   # run2 = 두 번째 시도 (첫 시도 결과는 lbours/wmescape/ 바로 밑에 그대로)
H=$(rclone cat $RC/launch.txt 2>/dev/null | head -1 | awk '{print $1}')
CB=/tmp/dkta/wmescape_code; C=$CB/$H
# 출력 자리: 덤프 6 판 x 6.6 GB + 렌더 h5 6.6 GB -> /tmp 남은 자리 ≥ 70 GB 면 /tmp, 아니면 /home/dkta/work (≥ 200 GB), 둘 다 아니면 멈춤 (첫 시도 11-08 15:56 은 /tmp 13 GB 로 덤프 쓰기 실패)
O0=/tmp/dkta/lbours/wmescape
av() { df -BG --output=avail $1 | tail -1 | tr -dc 0-9; }
if [ -d $O0 ] && [ ! -f $O0/run2.txt ]; then                                               # 첫 시도 자리 (덤프 쓰기 실패로 /tmp 를 채움): 끝난 E_a 덤프는 R2 첫 시도 자리에 올린 뒤 비움
  [ "$(grep -c 'E_a rc=0' $O0/vid.log 2>/dev/null)" = 1 ] && rclone copyto $O0/E_a/pred.pt r2:storage/result/anchorflow/lbours/wmescape/E_a/pred.pt -q
  rm -rf $O0
fi
if [ $(av /tmp) -ge 70 ]; then O=$O0; elif [ $(av /home/dkta/work) -ge 200 ]; then O=/home/dkta/work/wmescape_out; else O=""; fi
mkdir -p $CB; [ -n "$O" ] && mkdir -p $O && echo run2 > $O/run2.txt
L=$O/go.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go.log -q; }
say "시작 $(hostname) 커밋 '$H' 출력 '$O' (/tmp $(av /tmp) G · /home $(av /home/dkta/work) G 남음)"
[ -n "$O" ] || { say "자리 없음 -- 멈춤"; echo "자리 없음" | rclone rcat $R/wmescape.END; exit 1; }
rm -f $O/wmescape.END
[ -n "$H" ] || { say "launch.txt 없음"; echo "코드 실패" | rclone rcat $R/wmescape.END; exit 1; }
[ -s $CB/wmescape_$H.bundle ] || rclone copyto $RC/wmescape_$H.bundle $CB/wmescape_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wmescape_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/wmescape.END; exit 1; }
say "코드 $(git log --oneline -1 | cut -c1-100); /tmp $(df -h /tmp | tail -1 | awk '{print $4}') · /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr '\n' ' ')"
mkdir -p $O/cache; export CUPY_CACHE_DIR=$O/cache/cupy XDG_CACHE_HOME=$O/cache   # /dev/shm 가 다 참 (첫 시도 0 남음)
( while [ ! -f $O/wmescape.END ]; do sleep 300; [ -f $O/wmescape.END ] && break; rclone copy $O $R --exclude "pred.pt" --exclude "h5_fem/**" -q; done ) &
bash exe/wmescape_vid.sh $O $R < /dev/null > $O/vid.out 2>&1
say "vid 끝 rc=$?"
rclone copy $O $R --exclude "pred.pt" --exclude "h5_fem/**" -q
echo "끝 $(date '+%F %T')" > $O/wmescape.END; rclone copyto $O/wmescape.END $R/wmescape.END -q
say "끝"
