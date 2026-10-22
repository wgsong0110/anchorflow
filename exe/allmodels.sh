#!/bin/bash
# lego 학습 모델 열 개를 궤적 L000·0000(+0004) 에서 굴린 비교 (사용자 10-22). 학습 없음 (--iters 0).
#   칸: 정답 | 장벽·벌점 바닥 다섯 궤적 | softplus 바닥 다섯 궤적 | softplus+미끄러짐 방지 다섯 궤적(기준) | 같은 바닥 풀 학습
#       | 에너지 미분 51채널 | 에너지 미분 18채널만 | 접촉 셀 바닥 정보 | 공중/충돌 정규화+균형 샘플링 | 주파수 인코딩 | 주파수+접촉 바닥
#   각 모델은 **자기 학습 커밋 워크트리**에서 **자기 원래 롤아웃 명령**(BASE 는 원래 스크립트 원문의 변수 줄을 그대로 평가해 다시 만들고,
#   학습/롤아웃 기록에 남은 명령줄과 글자 대조)으로 굴린다. 바꾸는 것은 AF_ROLL_DUMP_F=32 (F float32 저장; 계산 경로는 같다)와 출력 경로뿐.
#   이미 같은 명령으로 만든 F32 덤프가 있으면 재사용하고(오차는 그 로그에서), half 덤프만 있던 것은 F32 로 다시 굴려 기존 오차와 대조한다.
#   체크포인트는 사본(lbours/allmodels/ck/)에서 읽는다 (풀 모델 MAIN_last.pt 는 학습이 끝난 최종 파일을 그대로).
# 순서: 명령줄 대조 -> 체크포인트 사본·반복 -> 롤아웃(빈 GPU 나눠 동시에) -> 0000 정답 디렉토리 판정 -> h5·3DGS 렌더 + 실패 양상 지표
#       -> 격자 영상 두 개·오차 표 (ALLMODELS_CLUSTER_DONE) -> 실패 양상 표·그림 (FAILMODE_CLUSTER_DONE)
# 출력은 처음부터 R2 lbours/allmodels 와 통째 동기화 (제외 없음, 5분 루프), 큰 덤프는 생기는 즉시 백그라운드 copyto. 삭제 없음.
# 잡 안에서만: AF_DETACH=1 bash exe/job_run.sh $W/allmodels_boot.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LC_ALL=C.UTF-8
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
ME=$(pwd); C=$(git rev-parse --short HEAD)
P=$W/lbpool; L=$W/lbours; U=$L/allmodels; R=r2:storage/result/anchorflow/lbours/allmodels
RV=r2:storage/result/anchorflow/v20261009; RO=r2:storage/result/anchorflow/ovnet
FONT=/home/dkta/.fonts/NotoSansCJKkr-Regular.otf
mkdir -p $U/ck $U/roll $U/h5 $U/vid $U/fail $U/log
if [ -f $U/driver.hb ] && [ $(( $(date +%s) - $(cat $U/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $U/driver.pid
( while true; do date +%s > $U/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $U $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
echo "[시작] $(date) 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?}"
up(){ rclone copy $U $R -q --include "*.mp4" --include "*.txt" --include "*.json" --include "*.jsonl" --include "*.log" --include "*.md" --include "*.png" --include "*.tsv" --include "ALLMODELS_*" --include "FAILMODE_*"; }
bgup(){ ( rclone copyto $1 $R/${1#$U/} -q && echo "[R2 덤프 올림] ${1#$U/} $(date +%T)" ) & }
fail(){ echo "[오류] $1 $(date)"; echo "$1" >> $U/ALLMODELS_FAIL; echo "$1" >> $U/FAILMODE_FAIL; up; exit 1; }
ffail(){ echo "[오류] $1 $(date)"; echo "$1" >> $U/FAILMODE_FAIL; up; }
ck_k(){ python -c "import torch;c=torch.load('$1',map_location='cpu',weights_only=False);print(c.get('k',c.get('it',c.get('step','?'))))" 2>/dev/null; }
gt_of(){ case $1 in L000) echo $W/lbdata/test_low/L000;; 0000) cat $U/gt_dir_0000.txt 2>/dev/null;; *) echo $W/lbdata/test/$1;; esac; }
pt_of(){ [ $1 = L000 ] && echo $W/lbpt/test_low/L000.pt || echo $W/lbpt/test/$1.pt; }
errof(){ grep -a '롤아웃\] ' $1 2>/dev/null | tail -1 | grep -o '평균 [0-9.]*%' | grep -o '[0-9.]*'; }
iterof(){ grep -a -m1 '^\[재개\]' $1 2>/dev/null | grep -o 'step [0-9]*' | grep -o '[0-9]*'; }
resof(){ grep -a -m1 '^\[재개\]' $1 2>/dev/null | awk '{print $2}'; }

# ---------------- 모델 표 ----------------
NAMES=(old_multi sp_multi spstick spstick_pool ipfeat ipfeatonly bccontact relexcess pfreq pfreq_bcc)
declare -A LAB WT CKO B REF REFSRC OLDERR
LAB[old_multi]="장벽·벌점 바닥 다섯 궤적";            WT[old_multi]=$W/wt/spf_93957f3;        CKO[old_multi]=$L/spfloor/ck_old_multi.pt
LAB[sp_multi]="softplus 바닥 다섯 궤적";               WT[sp_multi]=$W/wt/spf_93957f3;         CKO[sp_multi]=$L/spfloor/multi/ck.pt
LAB[spstick]="softplus+미끄러짐 방지 다섯 궤적 (기준)"; WT[spstick]=$W/wt/spstick_13471f4;      CKO[spstick]=$L/spstick/multi/ck.pt
LAB[spstick_pool]="softplus+미끄러짐 방지 풀 학습";     WT[spstick_pool]=$W/wt/spstick_13471f4; CKO[spstick_pool]=$L/spstick/lgvpool/pool/MAIN_last.pt
LAB[ipfeat]="에너지 미분 51채널";                       WT[ipfeat]=$W/wt/ipfeat_e831c7b;        CKO[ipfeat]=$L/ipfeat/multi/ck.pt
LAB[ipfeatonly]="에너지 미분 18채널만";                 WT[ipfeatonly]=$W/wt/ipfeatonly_90064ba; CKO[ipfeatonly]=$L/ipfeatonly/multi/ck.pt
LAB[bccontact]="접촉 셀 바닥 정보";                     WT[bccontact]=$W/wt/bccontact_1bf78b7;  CKO[bccontact]=$L/bccontact/multi/ck.pt
LAB[relexcess]="공중/충돌 정규화 손실+균형 샘플링";     WT[relexcess]=$W/wt/relexcess_53bebb1;  CKO[relexcess]=$L/relexcess/multi/ck.pt
LAB[pfreq]="입자 물리량 주파수 인코딩";                 WT[pfreq]=$W/wt/pfreq_824be7c;          CKO[pfreq]=$L/pfreq/multi/ck.pt
LAB[pfreq_bcc]="주파수 인코딩+접촉 바닥";               WT[pfreq_bcc]=$W/wt/pfreqbcc_f5f427c;   CKO[pfreq_bcc]=$L/pfreq_bcc/multi/ck.pt
# 원래 스크립트 원문의 변수 줄을 그대로 평가해 BASE 를 다시 만든다 (pfreq 계열 KF 는 학습 기록대로 6)
mkb(){ P=$P W=$W KF=6 bash -c "$(grep -E "^($2)=" $1)"$'\n'"echo \"\$$3\""; }   # 줄바꿈: 마지막 줄 주석이 echo 를 삼키지 않게
SPV='COMMON|REST_OLD|FLOOR_SP|REST|BASE_SP|BASE_NOXN|BASE'
B[old_multi]=$(mkb $W/spfloor.sh 'COMMON|REST_OLD|FLOOR_OLD|FLOOR_IPG|FLOOR_SP|REST|BASE_OLD' BASE_OLD)
B[sp_multi]=$(mkb $W/spfloor.sh 'COMMON|REST_OLD|FLOOR_OLD|FLOOR_IPG|FLOOR_SP|REST|BASE_OLD' BASE)
B[spstick]=$(mkb $W/spstick.sh "$SPV" BASE)
B[spstick_pool]=$(mkb $W/spstick2.sh 'GATE|COMMON|PCOMMON_SP|PC' PC)
B[ipfeat]=$(mkb $W/ipfeat.sh "$SPV|BASE_IP" BASE_IP)
B[ipfeatonly]=$(mkb $W/ipfeatonly.sh "$SPV|BASE_IP" BASE_IP)
B[bccontact]=$(mkb $W/bccontact.sh "$SPV|BASE_BC" BASE_BC)
B[relexcess]=$(mkb ${WT[relexcess]}/exe/relexcess.sh "$SPV" BASE)
B[pfreq]=$(mkb ${WT[pfreq]}/exe/pfreq.sh "$SPV|BASE_PF" BASE_PF)
B[pfreq_bcc]=$(mkb ${WT[pfreq_bcc]}/exe/pfreq_bcc.sh "$SPV|BASE_PF" BASE_PF)
# 학습·롤아웃 기록에 남은 명령줄 (대조 기준)
rcmd(){ sed -E 's/.* --resume [^ ]+ (.*) --iters 0 .*/\1/' $1 | head -1; }
REF[old_multi]=$(grep -a -m1 '^\[BASE_OLD\] ' $W/spfloor.out | sed 's/^\[BASE_OLD\] //');  REFSRC[old_multi]="spfloor.out [BASE_OLD]"
REF[sp_multi]=$(grep -a -m1 '^\[BASE\] ' $W/spfloor.out | sed 's/^\[BASE\] //');           REFSRC[sp_multi]="spfloor.out [BASE]"
REF[spstick]=$(grep -a -m1 '^\[BASE\] ' $W/spstick.out | sed 's/^\[BASE\] //');            REFSRC[spstick]="spstick.out [BASE]"
REF[spstick_pool]=$(grep -a -m1 '^\[pool\] ' $W/spstick2.out | sed 's/^\[pool\] //');      REFSRC[spstick_pool]="spstick2.out [pool]"
REF[ipfeat]=$(grep -a '^\[학습 명령줄 BASE\] ' $L/ipfeat/IPFEAT_LAUNCHED | tail -1 | sed 's/^\[학습 명령줄 BASE\] //');             REFSRC[ipfeat]="ipfeat/IPFEAT_LAUNCHED"
REF[ipfeatonly]=$(grep -a '^\[학습 명령줄 BASE\] ' $L/ipfeatonly/IPFEATONLY_LAUNCHED | tail -1 | sed 's/^\[학습 명령줄 BASE\] //'); REFSRC[ipfeatonly]="ipfeatonly/IPFEATONLY_LAUNCHED"
REF[bccontact]=$(grep -a '^\[학습 명령줄 BASE\] ' $L/bccontact/BCCONTACT_LAUNCHED | tail -1 | sed 's/^\[학습 명령줄 BASE\] //');    REFSRC[bccontact]="bccontact/BCCONTACT_LAUNCHED"
REF[relexcess]=$(rcmd $L/relexcess/cmd_roll_L000.txt);  REFSRC[relexcess]="relexcess/cmd_roll_L000.txt"
REF[pfreq]=$(rcmd $L/pfreq/cmd_roll_L000.txt);          REFSRC[pfreq]="pfreq/cmd_roll_L000.txt"
REF[pfreq_bcc]=$(rcmd $L/pfreq_bcc/cmd_roll_L000.txt);  REFSRC[pfreq_bcc]="pfreq_bcc/cmd_roll_L000.txt"
# 기존 오차 (half 덤프 또는 원래 롤아웃)
for T in L000 0004; do
  OLDERR[old_multi:$T]=$(grep -o '[0-9.]*%' $L/spfloor/err_old_multi_$T.txt 2>/dev/null)
  OLDERR[sp_multi:$T]=$(grep -o '[0-9.]*%' $L/spfloor/err_m_multi_$T.txt 2>/dev/null)
  OLDERR[spstick:$T]=$(grep -o '[0-9.]*%' $L/spstick/err_m_multi_$T.txt 2>/dev/null)
  OLDERR[spstick_pool:$T]=$(grep -o '[0-9.]*%' $L/spstick/lgvpool/err_st_pool_$T.txt 2>/dev/null)
  for m in relexcess pfreq pfreq_bcc; do OLDERR[$m:$T]=$(grep -o '[0-9.]*%' $L/$m/err_${m}_multi_$T.txt 2>/dev/null); done
done
# 같은 명령으로 이미 만든 F32 덤프 (재사용): 덤프 로그 [명령 기록]
declare -A REUSE RCMD
for T in L000 0004; do
  REUSE[spstick:$T]="$L/lego_psi_spstick/roll_mm_$T.pt $L/lego_psi_spstick/v_mm_$T.log"
  RCMD[spstick:$T]=$(mkb $W/lego_psi_spstick.sh 'COMMON|REST_OLD|FLOOR_SP|REST|BASE' BASE)
  REUSE[spstick_pool:$T]="$L/lego_psi_pool/roll_pool_$T.pt $L/lego_psi_pool/v_pool_$T.log"
  RCMD[spstick_pool:$T]=$(sed -E 's/.* --resume [^ ]+ //' $L/lego_psi_pool/cmd_$T.txt | head -1)
  for m in relexcess pfreq pfreq_bcc; do
    REUSE[$m:$T]="$L/$m/roll_${m}_multi_$T.pt $L/$m/v_${m}_multi_$T.log"; RCMD[$m:$T]=$(rcmd $L/$m/cmd_roll_$T.txt); done
done

# ---------------- 0) 명령줄 대조 ----------------
{ echo "# 명령줄 대조 (커밋 $C, $(date '+%F %T')): 원래 스크립트 원문 변수 줄 평가 = 기록된 명령줄"
  ok=1
  for m in "${NAMES[@]}"; do
    h=$(git -C ${WT[$m]} rev-parse --short HEAD 2>/dev/null); dirty=$(git -C ${WT[$m]} status --porcelain -uno 2>/dev/null | wc -l)
    if [ "${B[$m]}" = "${REF[$m]}" ] && [ -n "${B[$m]}" ]; then r=일치; else r=불일치; ok=0; fi
    echo "[$m] ${LAB[$m]} | 워크트리 ${WT[$m]} 커밋 $h (추적 파일 변경 $dirty) | 체크포인트 ${CKO[$m]} | 기준 ${REFSRC[$m]} -> $r"
    echo "    BASE: ${B[$m]}"
    [ "$r" = 불일치 ] && echo "    기록: ${REF[$m]}"
    [ "$dirty" = 0 ] || ok=0
  done
  for k in "${!RCMD[@]}"; do m=${k%%:*}
    if [ "${RCMD[$k]}" = "${B[$m]}" ]; then r=일치; else r=불일치; ok=0; fi
    echo "[재사용 덤프 $k] $(echo ${REUSE[$k]} | awk '{print $1}') 의 명령 BASE = 모델 BASE -> $r"
    [ "$r" = 불일치 ] && echo "    덤프 명령: ${RCMD[$k]}"
  done
  echo "OK=$ok"; } > $U/cmdline_check.txt
cat $U/cmdline_check.txt | cut -c1-260
grep -q '^OK=1' $U/cmdline_check.txt || fail "명령줄 대조 실패 (cmdline_check.txt)"
up

# ---------------- 1) 체크포인트 사본·반복 ----------------
for d in ipfeat ipfeatonly bccontact; do   # 중지된 학습: 쓰는 프로세스가 없어야 한다
  pgrep -f "lbours/$d/multi" > /dev/null && fail "$d 학습 프로세스가 아직 돈다"
done
if [ ! -f $U/ck/ck_list.tsv ]; then
  for m in "${NAMES[@]}"; do
    if [ $m = spstick_pool ]; then CK=${CKO[$m]}
    else CK=$U/ck/$m.pt; cp -p ${CKO[$m]} $CK && cmp -s ${CKO[$m]} $CK || fail "체크포인트 사본 실패 $m"; fi
    printf "%s\t%s\t%s\t%s\n" $m $CK ${CKO[$m]} "$(ck_k $CK)" >> $U/ck/ck_list.tmp
  done
  mv $U/ck/ck_list.tmp $U/ck/ck_list.tsv
fi
declare -A CK KK
while IFS=$'\t' read -r m ck orig k; do CK[$m]=$ck; KK[$m]=$k; done < $U/ck/ck_list.tsv
cat $U/ck/ck_list.tsv; up

# ---------------- GPU ----------------
free_gpu(){ # 빈 GPU (0 번 = noinertial 제외, 메모리 1GB 미만, 연산 프로세스 없음, 인자 카드 제외)
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  for c in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$1!=0 && $2<1024{print $1}'); do
    [[ " $* " == *" $c "* ]] && continue
    u=$(nvidia-smi -i $c --query-gpu=uuid --format=csv,noheader); grep -q "$u" <<<"$apps" && continue
    echo $c; return; done; }
runpar(){ # 작업들을 빈 GPU 에 하나씩 배정해 동시에 돌리고 모두 끝날 때까지 기다린다
  local Q=("$@"); declare -A BZ; local g t
  while [ ${#Q[@]} -gt 0 ] || [ ${#BZ[@]} -gt 0 ]; do
    for g in "${!BZ[@]}"; do kill -0 ${BZ[$g]} 2>/dev/null || unset "BZ[$g]"; done
    if [ ${#Q[@]} -gt 0 ]; then g=$(free_gpu "${!BZ[@]}")
      if [ -n "$g" ]; then t=${Q[0]}; Q=("${Q[@]:1}")
        echo "[GPU 배정] $t -> GPU$g $(date '+%F %T') | $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' ')" | tee -a $U/gpu.log
        task $t $g > $U/log/task_${t//:/_}.log 2>&1 & BZ[$g]=$!; sleep 20; continue; fi; fi
    sleep 30; done; }

# ---------------- 2) 롤아웃 ----------------
TRJ=(L000 0000 0004)
roll(){ m=$1; T=$2; g=$3; N=${m}_$T; O=$U/roll
  [ -f $O/.done_$N ] && return 0
  rm -f $O/inv_$N.jsonl $O/tun_$N.jsonl
  echo "[롤아웃 명령] (cd ${WT[$m]}) AF_INV_LOG=$O/inv_$N.jsonl AF_TUNNEL_LOG=$O/tun_$N.jsonl AF_ROLL_DUMP=$O/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $O/v_$N --tag EV --resume ${CK[$m]} ${B[$m]} --iters 0 --eval_t0 0 --eval_len 80 --tb none" > $O/cmd_$N.txt
  ( cd ${WT[$m]} && env AF_INV_LOG=$O/inv_$N.jsonl AF_TUNNEL_LOG=$O/tun_$N.jsonl AF_ROLL_DUMP=$O/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g \
      python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $O/v_$N --tag EV --resume ${CK[$m]} ${B[$m]} --iters 0 --eval_t0 0 --eval_len 80 --tb none > $O/v_$N.log 2>&1 < /dev/null )
  grep -a "롤아웃\] $T\|Traceback\|Error" $O/v_$N.log | tail -2 | cut -c1-200
  [ -f $O/roll_$N.pt ] && [ -n "$(errof $O/v_$N.log)" ] || { echo "[오류] $N 롤아웃 실패"; return 1; }
  bgup $O/roll_$N.pt; touch $O/.done_$N; up; }
task(){ local k=$1 g=$2; IFS=: read -r kind a b <<<"$k"; echo "[시작] $k GPU$g $(date '+%F %T')"
  case $kind in
    roll) roll $a $b $g ;;
    rend) rend $a $b $g ;;
    fm) fm $a $g ;;
  esac; echo "[끝] $k GPU$g $(date '+%F %T')"; }
RQ=()
for m in "${NAMES[@]}"; do for T in "${TRJ[@]}"; do
  [ -n "${REUSE[$m:$T]}" ] && continue
  [ -f $U/roll/.done_${m}_$T ] || RQ+=(roll:$m:$T); done; done
echo "[롤아웃 작업] ${#RQ[@]} 개: ${RQ[*]}"
runpar "${RQ[@]}"
# 표 (manifest): 모델 라벨 궤적 덤프 로그 오차 반복(로그) 반복(사본) 커밋 체크포인트 출처
{ for m in "${NAMES[@]}"; do for T in "${TRJ[@]}"; do
    if [ -n "${REUSE[$m:$T]}" ]; then read -r D LG <<<"${REUSE[$m:$T]}"; SRC=재사용; else D=$U/roll/roll_${m}_$T.pt; LG=$U/roll/v_${m}_$T.log; SRC=새로; fi
    [ -f $D ] || fail "덤프 없음 $m $T ($D)"
    [ "$(resof $LG)" = "${CKO[$m]}" ] || [ "$(resof $LG)" = "${CK[$m]}" ] || fail "덤프 로그의 [재개] 체크포인트가 다르다 $m $T: $(resof $LG)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $m "${LAB[$m]}" $T $D $LG "$(errof $LG)" "$(iterof $LG)" "${KK[$m]}" \
      "$(git -C ${WT[$m]} rev-parse --short HEAD)" ${CKO[$m]} $SRC
  done; done; } > $U/manifest.tsv
cat $U/manifest.tsv | cut -f1,3,6,7,8,9,11
{ echo "# 기존 오차와 대조 (같은 명령, F 저장 형식만 F32)"; while IFS=$'\t' read -r m lab T D LG e it k c ck src; do
    o=${OLDERR[$m:$T]}; [ -n "$o" ] && echo "$m $T 이번 $e% ($src) | 기존 $o | 차 $(python -c "print(f'{abs($e-${o%\%}):.3f}')")"; done < $U/manifest.tsv; } | tee $U/err_compare.txt
up

# ---------------- 3) 0000 정답 디렉토리 판정 (덤프 첫 프레임과 맞는 쪽) ----------------
if [ ! -s $U/gt_dir_0000.txt ]; then
  python -u $ME/exe/allmodels_gtcheck.py --dump $U/roll/roll_old_multi_0000.pt --cand test=$W/lbdata/test/0000 --cand train=$W/lbdata/train/0000 \
    --out $U/gt_check_0000.json > $U/gt_check_0000.log 2>&1
  cat $U/gt_check_0000.log; grep -a '^GTDIR ' $U/gt_check_0000.log | awk '{print $2}' > $U/gt_dir_0000.txt
  [ -s $U/gt_dir_0000.txt ] || fail "0000 정답 디렉토리 판정 실패"
  for T in L000 0004; do python -u $ME/exe/allmodels_gtcheck.py --dump $U/roll/roll_old_multi_$T.pt --cand x=$(gt_of $T) --out $U/gt_check_$T.json > $U/gt_check_$T.log 2>&1
    grep -a -q '^GTDIR ' $U/gt_check_$T.log || fail "$T 정답 디렉토리가 덤프와 맞지 않는다"; done
  up
fi
echo "[0000 정답] $(cat $U/gt_dir_0000.txt)"

# ---------------- 4) h5·렌더 (L000·0000) + 실패 양상 지표 (L000·0000·0004) ----------------
rend(){ m=$1; T=$2; g=$3; N=${m}_$T
  [ -f $U/vid/$N.mp4 ] && return 0
  if [ $m = gt ]; then SIM=$(gt_of $T)
  else D=$(awk -F'\t' -v m=$m -v T=$T '$1==m && $3==T{print $4}' $U/manifest.tsv); SIM=$U/h5/$N
    [ -f $SIM/meta.json ] && [ -f $SIM/sim_0000000080.h5 ] || python -u $ME/exe/lb_roll_to_h5.py --dump $D --traj $(gt_of $T) --out $SIM > $U/h5/$N.log 2>&1; tail -1 $U/h5/$N.log; fi
  echo "[렌더 명령] CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $SIM --config $W/lbdata/config.json --shape lego --out $U/vid/$N.mp4 (커밋 $C)" > $U/vid/cmd_$N.txt
  CUDA_VISIBLE_DEVICES=$g python -u $ME/exe/lb_render.py --sim $SIM --config $W/lbdata/config.json --shape lego --out $U/vid/$N.mp4 > $U/vid/r_$N.log 2>&1 < /dev/null
  tail -1 $U/vid/r_$N.log; [ -f $U/vid/$N.mp4 ] || { echo "[오류] 렌더 $N"; return 1; }; up; }
fm(){ T=$1; g=$2; [ -f $U/fail/fm_$T.json ] && return 0
  A=""; while IFS=$'\t' read -r m lab TT D rest; do [ "$TT" = $T ] && A="$A --dump $m=$D"; done < $U/manifest.tsv
  CUDA_VISIBLE_DEVICES=$g python -u $ME/exe/allmodels_failmode.py --traj $T --gt_dir $(gt_of $T) --pt $(pt_of $T) --floor_cfg $P/cfg/lego_jelly.json $A \
    --out $U/fail/fm_$T.json > $U/fail/fm_$T.log 2>&1 < /dev/null
  grep -a "^\[" $U/fail/fm_$T.log | tail -15 | cut -c1-220; [ -f $U/fail/fm_$T.json ] || { echo "[오류] 실패 양상 지표 $T"; return 1; }; up; }
VQ=()
for T in L000 0000; do for m in gt "${NAMES[@]}"; do [ -f $U/vid/${m}_$T.mp4 ] || VQ+=(rend:$m:$T); done; done
for T in "${TRJ[@]}"; do [ -f $U/fail/fm_$T.json ] || VQ+=(fm:$T); done
echo "[렌더·지표 작업] ${#VQ[@]} 개"
runpar "${VQ[@]}"

# ---------------- 5) 격자 영상·오차 표 ----------------
for T in L000 0000; do
  [ -f $U/allmodels_$T.mp4 ] && continue
  CELLS=(--cell "정답 (MPM)||$U/vid/gt_$T.mp4")
  while IFS=$'\t' read -r m lab TT D LG e it k c ck src; do [ "$TT" = $T ] || continue
    CELLS+=(--cell "$lab|오차 ${e}% · 반복 $it|$U/vid/${m}_$T.mp4"); done < $U/manifest.tsv
  python -u $ME/exe/allmodels_grid.py "${CELLS[@]}" --cols 4 --font $FONT --head "궤적 $T · 같은 프레임 · lb_render 3DGS" --out $U/allmodels_$T.mp4 > $U/grid_$T.log 2>&1
  cat $U/grid_$T.log; [ -f $U/allmodels_$T.mp4 ] || fail "격자 영상 $T"
done
python -u $ME/exe/allmodels_report.py err --manifest $U/manifest.tsv --gtdir0000 "$(cat $U/gt_dir_0000.txt)" --cmdcheck $U/cmdline_check.txt \
  --errcmp $U/err_compare.txt --commit $C --out $U/allmodels_err.md > $U/report_err.log 2>&1 || fail "오차 표"
cat $U/allmodels_err.md
rclone copyto $U/allmodels_L000.mp4 $RV/allmodels_L000.mp4 && rclone copyto $U/allmodels_0000.mp4 $RV/allmodels_0000.mp4 \
  && rclone copyto $U/allmodels_err.md $RO/allmodels_err.md || fail "R2 v20261009/ovnet 복사 실패"
up; touch $U/ALLMODELS_CLUSTER_DONE; up
echo ALLMODELS_CLUSTER_DONE

# ---------------- 6) 실패 양상 표·그림 ----------------
for T in "${TRJ[@]}"; do [ -f $U/fail/fm_$T.json ] || { ffail "실패 양상 지표 없음 $T"; exit 1; }; done
python -u $ME/exe/allmodels_report.py fail --manifest $U/manifest.tsv --fm $U/fail/fm_L000.json $U/fail/fm_0000.json $U/fail/fm_0004.json \
  --font $FONT --commit $C --out $U/allmodels_failmode.md --png $U/allmodels_failmode.png > $U/report_fail.log 2>&1 || { cat $U/report_fail.log; ffail "실패 양상 표"; exit 1; }
cat $U/allmodels_failmode.md
rclone copyto $U/allmodels_failmode.md $RO/allmodels_failmode.md && rclone copyto $U/allmodels_failmode.png $RO/allmodels_failmode.png || { ffail "R2 ovnet 복사 실패"; exit 1; }
touch $U/FAILMODE_CLUSTER_DONE
kill $SYNC 2>/dev/null; for p in $(jobs -p); do [ "$p" = "$HB" ] || wait $p 2>/dev/null; done
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)" && touch $U/ALL_DONE && rclone copy $U $R -q
rclone check $U $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo FAILMODE_CLUSTER_DONE
