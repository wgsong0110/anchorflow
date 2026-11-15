#!/bin/bash
# 절 CM184: GF 장면 바닥 접촉 뒤집힘 원인 분리 실험 (짧은 프레임, 지표·영상 없음). vast_gfs.sh XS_STAGE=prep 뒤 인스턴스에서 tmux 로.
#   사용: bash cm184_exp.sh 목록파일   (줄: 이름 모델 설정 프레임수 꼬리표 추가인자(+로 이음, 없으면 -) 뺄플래그(+로 이음, 없으면 -))
#   결과: R2 lbours/gfscenes/ours/<이름>_<꼬리표>/ (run.log, nl.jsonl, fem.jsonl ...), 요약 줄은 로그 cm184x.log
set -u
export W=/root/gfs AF=${XS_AF:-/root/sv/af} R=r2:storage/result/anchorflow; export RX=$R/lbours/gfscenes DS=r2:storage/datasets/anchorflow/gf_scenes GF=/root/gfs/GaussianFluent
export TAG=cm184x LOG=/root/sv/cm184x.log
say() { echo "[$TAG $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $RX/$TAG.log -q 2>/dev/null; }
export -f say
[ -n "${XS_NOSYNC:-}" ] || rclone copy ${XS_CODE:-$RX/code_cm184}/exe $AF/exe -q
[ -f ${XS_LIBTAR:-/nonexist} ] && tar -xf $XS_LIBTAR -C $AF
while read n m c nf lab ex dr; do
  [ -z "$n" ] || [[ "$n" == \#* ]] && continue
  [ "$ex" = - ] && ex=""; [ "$dr" = - ] && dr=""
  say "[실험] $n $lab 프레임 $nf 추가 '${ex//+/ }' 뺌 '${dr//+/ }'"
  GFS_DROP="${dr//+/ }" XS_SFX=_$lab bash $AF/exe/gfs_ours.sh $n $m $c $nf "ABL0${ex:++$ex}"
done < $1
say "[실험 끝] $1"
