#!/bin/bash
# allmodels 로컬 감시 (tmux allmodels_local 에서): 10 분마다 로그인 노드 한 번 접속(ControlMaster 재사용)으로 표식·하트비트를 읽는다.
#   드라이버가 죽었고(하트비트 15 분 넘음) 안 끝났으면 지금 돌고 있는 잡에 exe/job_run.sh(AF_DETACH=1)로 다시 띄운다 (최대 3 번; 드라이버는 멱등).
#   ALLMODELS_CLUSTER_DONE -> R2 v20261009/allmodels_{L000,0000}.mp4·ovnet/allmodels_err.md 확인, outputs.md 기록, 로컬 표식 ALLMODELS_DONE
#   FAILMODE_CLUSTER_DONE  -> R2 ovnet/allmodels_failmode.{md,png} 확인, outputs.md 기록, 로컬 표식 FAILMODE_DONE
#   *_FAIL -> 로컬 표식 ALLMODELS_FAIL / FAILMODE_FAIL (내용 포함)
LOG=/tmp/allmodels_local.log; LOGIN=dkta@147.46.121.39; U=/home/dkta/work/lbours/allmodels
BOOT=${1:?부트 스크립트}; C=${2:?커밋}
RA=r2:storage/result/anchorflow; OUT=/home/wgsong/workspace/result/anchorflow/outputs.md
exec 9>/tmp/allmodels_local.lock; flock -n 9 || { echo "[중단] 다른 감시가 돈다 $(date)" >> $LOG; exit 0; }
log(){ echo "[$(date '+%F %T')] $*" >> $LOG; }
log "감시 시작 (부트 $BOOT, 커밋 $C)"
NRE=0; AD=0; FD=0
while :; do
  S=$(ssh -o ConnectTimeout=20 $LOGIN "cd $U 2>/dev/null || exit 0; for f in ALLMODELS_CLUSTER_DONE FAILMODE_CLUSTER_DONE ALL_DONE; do [ -f \$f ] && echo MARK \$f; done
    for f in ALLMODELS_FAIL FAILMODE_FAIL; do [ -f \$f ] && [ \"\$(tail -1 \$f | cut -c1-8)\" != \"[정정]\" ] && echo MARK \$f; done
    echo HB \$(( \$(date +%s) - \$(cat driver.hb 2>/dev/null || echo 0) ))
    echo NROLL \$(ls roll/.done_* 2>/dev/null | wc -l) NVID \$(ls vid/*.mp4 2>/dev/null | wc -l) NFM \$(ls fail/fm_*.json 2>/dev/null | wc -l)
    for f in ALLMODELS_FAIL FAILMODE_FAIL; do [ -f \$f ] && echo \"TXT \$f: \$(tail -2 \$f | tr '\n' ' ')\"; done; true" 2>&1)
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q '^HB ' <<<"$S"; then log "접속 실패 (rc $rc) -- 이번 틱 건너뜀: $(echo "$S" | tail -1)"; sleep 600; continue; fi
  log "상태: $(grep -E '^(HB|NROLL)' <<<"$S" | tr '\n' ' ') $(grep '^MARK' <<<"$S" | tr '\n' ' ')"
  if [ $AD = 0 ] && grep -q 'MARK ALLMODELS_CLUSTER_DONE' <<<"$S"; then
    if rclone lsf $RA/v20261009/ --include "allmodels_*.mp4" | grep -q allmodels_0000.mp4 && rclone lsf $RA/ovnet/ --include allmodels_err.md | grep -q .; then
      M=$(rclone cat $RA/ovnet/allmodels_err.md)
      { echo; echo "## $(date +%F) lego 학습 모델 열 개 L000·0000 롤아웃 비교 영상 (allmodels)"
        echo "- Path: ~/workspace/result/anchorflow/v20261009/allmodels_L000.mp4, allmodels_0000.mp4 (4열×3행: 정답 MPM + 모델 10 개, 칸 제목 = 모델·위치 오차·반복), 오차 표 ~/workspace/result/anchorflow/ovnet/allmodels_err.md"
        echo "- R2 원본: r2:storage/result/anchorflow/lbours/allmodels/ (roll/ F32 덤프·로그·명령, h5/, vid/ 칸별 렌더, ck/ 체크포인트 사본, manifest.tsv, cmdline_check.txt, err_compare.txt, gt_check_*.log)"
        echo "- Commit: $C (드라이버 exe/allmodels.sh); 각 모델 롤아웃은 자기 학습 커밋 워크트리·원래 명령 (BASE 원문 재구성·기록 글자 대조), --iters 0, AF_ROLL_DUMP_F=32"
        echo "- 오차 표:"; grep '^| ' <<<"$M"; grep '^- 0000 정답' <<<"$M"; } >> $OUT
      log "ALLMODELS_DONE"; AD=1
    else log "클러스터 끝 표식은 있는데 R2 v20261009/ovnet 파일이 아직 안 보임 -- 다음 틱"; fi
  fi
  if [ $FD = 0 ] && grep -q 'MARK FAILMODE_CLUSTER_DONE' <<<"$S"; then
    if rclone lsf $RA/ovnet/ --include "allmodels_failmode.*" | grep -q png; then
      { echo; echo "## $(date +%F) lego 학습 모델 바닥 충돌 이후 실패 양상 판정 (allmodels failmode)"
        echo "- Path: ~/workspace/result/anchorflow/ovnet/allmodels_failmode.md (표: 모델×궤적 L000·0000·0004 -> 판정·시작 프레임·핵심 수치), allmodels_failmode.png (프레임 곡선·판정 구간)"
        echo "- R2 원본: r2:storage/result/anchorflow/lbours/allmodels/fail/fm_{L000,0000,0004}.json (프레임별 지표), allmodels_failmode.json"
        echo "- Commit: $C (exe/allmodels_failmode.py, exe/allmodels_report.py); 덤프는 allmodels 롤아웃(F float32)과 같다"
        echo "- 문턱: 멈춤 이동량·무게중심 비 < 0.2 가 10 프레임 연속 & 뒤집힘 0; 늘어남 max|σ−1| 99% > 0.5 또는 σ1/σ3 99% > 3 & 뒤집힘 0 & 멈춤 아님; 망가짐 det F≤0 > 0.1% 또는 ψ > 정답×1e3 또는 비유한"; } >> $OUT
      log "FAILMODE_DONE"; FD=1
    else log "FAILMODE 클러스터 표식은 있는데 R2 ovnet 파일이 아직 안 보임 -- 다음 틱"; fi
  fi
  if grep -q 'MARK ALLMODELS_FAIL' <<<"$S" && [ $AD = 0 ]; then log "ALLMODELS_FAIL $(grep 'TXT ALLMODELS_FAIL' <<<"$S")"; AD=2; fi
  if grep -q 'MARK FAILMODE_FAIL' <<<"$S" && [ $FD = 0 ]; then log "FAILMODE_FAIL $(grep 'TXT FAILMODE_FAIL' <<<"$S")"; FD=2; fi
  [ $AD != 0 ] && [ $FD != 0 ] && { log "감시 끝 (ALLMODELS $AD, FAILMODE $FD)"; exit 0; }
  HBA=$(grep '^HB ' <<<"$S" | awk '{print $2}')
  if [ "$HBA" -gt 900 ]; then
    if [ $NRE -ge 3 ]; then log "드라이버 죽음 (하트비트 ${HBA}s), 재기동 3 번 다 씀 -- ALLMODELS_FAIL/FAILMODE_FAIL"; [ $AD = 0 ] && log "ALLMODELS_FAIL 드라이버 재기동 한도"; [ $FD = 0 ] && log "FAILMODE_FAIL 드라이버 재기동 한도"; exit 1; fi
    NRE=$((NRE + 1)); LG=/home/dkta/work/allmodels_driver_$(date +%m%d_%H%M).log
    log "드라이버 죽음 (하트비트 ${HBA}s) -- 재기동 $NRE: $LG"
    ssh -o ConnectTimeout=20 $LOGIN "cd /home/dkta/work/anchorflow && AF_DETACH=1 bash exe/job_run.sh $BOOT $LG" >> $LOG 2>&1
  fi
  sleep 600
done
