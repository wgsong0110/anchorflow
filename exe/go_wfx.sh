#!/bin/bash
# wolffix (11-15): wolf 마지막 무너짐·튐 진단과 수정 판. 코드 디렉토리에서.
# 사용: bash exe/go_wfx.sh <태그> <카드> <명세 파일>
#   명세 한 줄 = NAME|SCENE|ENV|PREWRAP|RUNNER_EXTRA|SED|RENDER   (# 줄 무시)
#     SCENE = exe/pr_ours_tmpl.json 키 (wolf · L000 · ficus · bread) -- 기본 판 명령에서 --out 만 바꾼 뒤
#     ENV 는 명령 앞에, PREWRAP 은 'python -u ' 바로 뒤 (가장 바깥 감싸개), RUNNER_EXTRA 는 --out 앞 (실행기 인자), SED 는 sed 식 (선택), RENDER 1 이면 3DGS 렌더
#   카드 G 에서 차례로 (단독). 결과 /home/dkta/work/wfx/<태그>/<NAME> -> R2 lbours/wolffix1115/<태그>/<NAME>
export PATH=/home/dkta/bin:$PATH
TAG=$1; G=$2; SPEC=$3
R=r2:storage/result/anchorflow/lbours/wolffix1115/$TAG
O=/home/dkta/work/wfx/$TAG; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_g$G.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_g$G.log -q; }
mk() {
  local NAME=$1 SC=$2 EN=$3 PW=$4 RX=$5 SD=$6 C
  C=$($PY -c "import json,sys;print(json.load(open('exe/pr_ours_tmpl.json'))[sys.argv[1]]['cmd'])" $SC | tr -s ' \n' ' ' | sed -e 's/ *$//')
  C=$(echo "$C" | sed -e "s#--out [^ ]*\$#--out $O/$NAME#")
  [ -n "$PW" ] && C=$(echo "$C" | sed -e "s#python -u #python -u $PW #")
  [ -n "$RX" ] && C=$(echo "$C" | sed -e "s# --out $O/$NAME\$# $RX --out $O/$NAME#")
  [ -n "$SD" ] && C=$(echo "$C" | sed -e "$SD")
  [ -n "$EN" ] && C="$EN $C"
  echo "$C"; }
render() {   # 3DGS 렌더 (exe/fem_ao_render.sh 와 같은 렌더러·config): wolf 만
  local NAME=$1 D=$O/$1 GTD=/dev/shm/dkta/ag_data/gt_wolf
  [ -f $D/pred.pt ] || return
  $PY -u exe/lb_roll_to_h5.py --dump $D/pred.pt --traj $GTD --out $D/h5_fem > $D/h5_fem.log 2>&1
  mkdir -p $D/vj
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $D/h5_fem --config $GTD/config.json --shape wolf --out $D/vj/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $D/vj/r_fem.log 2>&1
  rm -rf $D/h5_fem
  say "$NAME 렌더 $(ls -la $D/vj/fem_wolf.mp4 2>&1 | cut -c1-120)"; }
while IFS='|' read -r NAME SC EN PW RX SD RD; do
  [ -z "$NAME" ] || [[ $NAME == \#* ]] && continue
  mkdir -p $O/$NAME
  mk "$NAME" "$SC" "$EN" "$PW" "$RX" "$SD" > $O/$NAME/cmd.txt
  grep -q -- "--out $O/$NAME\$" $O/$NAME/cmd.txt || { say "$NAME 명령 생성 실패"; continue; }
  rclone copyto $O/$NAME/cmd.txt $R/$NAME/cmd.txt -q
  say "$NAME 시작 카드 $G (계산 프로세스 $(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  env CUDA_VISIBLE_DEVICES=$G timeout 5400 bash -c "$(cat $O/$NAME/cmd.txt)" < /dev/null > $O/$NAME/log.txt 2>&1
  rc=$?
  say "$NAME rc=$rc $(grep -ah '롤아웃\]' $O/$NAME/log.txt | tail -1 | cut -c1-220) $(grep -a 'rror' $O/$NAME/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  [ "$SC" = wolf ] && $PY -u exe/wfx_an.py $O/$NAME /dev/shm/dkta/ag_data/gt_wolf > $O/$NAME/wfx_an.txt 2>&1
  [ "$RD" = 1 ] && [ "$SC" = wolf ] && render $NAME
  rclone copy $O/$NAME $R/$NAME -q --transfers 8; say "$NAME R2 올림 $(head -c 600 $O/$NAME/wfx_an.txt 2>/dev/null | tr '\n' ' ')"
done < $SPEC
say "끝 카드 $G"
echo "끝 $(date)" | rclone rcat $R/END_g$G
