#!/bin/bash
# 절 CM4 묶음 (커밋 $1, 잡 130567): 절 CM3 채택 공통 설정 그대로 (정의 변경 없음) --
#   HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, 묶음 BA = exe/fem_cm3_scn.sh (cm2 BA + --hspec --ncomp: 시작점 ŷ --ga_y0ut, 장벽 바닥 띠 고착 --fl_bar --fl_bst 1,
#   점유 가중 --gq_oct 0.1, --mgbar 2 --mv2 --lsfast --fk), PCG 상한 = 묶음 cm3b 예산 규칙 고른 값 (marks/cm3b_sel.txt: lego 6 · ficus 24 · wolf 5 · bread 2; 낙하 drop1 은 lego 값).
#  J. lego 21 궤적 (L000 + 0000..0019) + 낙하 drop1 판정 (결정적, 빈 카드 1-6, exe/fem_cm_pool.sh)
#  V. 영상 덤프 (결정적, --dump --dump_F): lego L000·0016, ficus, wolf, bread -> h5 (exe/lb_roll_to_h5.py) -> 3DGS 알파블렌딩 (exe/lb_render.py), 빈 카드
# 결과 R2 lbours/fem_cm/cm4 (판별 log·summary·fem.jsonl, v_* 는 pred.pt·h5·vj 까지), 표 marks/cm4_table.md, 끝 표식 marks/cm4.END
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_cm XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
W=/home/dkta/work; B=/dev/shm/dkta/lbours/fem_cm; R=$B/cm4; RR=r2:storage/result/anchorflow/lbours/fem_cm/cm4
M=$B/marks; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; L=/dev/shm/dkta/cm_lists/cm4; CL=/dev/shm/dkta/anerr_claim
mkdir -p $M $L $R $CL $CUPY_CACHE_DIR
LOG=$M/cm4.log; say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
say "[cm4 시작] $C $(date '+%F %T')"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $LOG
export CM_SCN=exe/fem_cm3_scn.sh
source exe/fem_cm3_scn.sh
LG="L000 $(seq -f %04g 0 19 | tr '\n' ' ')"
for t in $LG drop1 ficus wolf bread; do cm_data $t; done
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
cap() { case $1 in ficus) echo 24;; wolf) echo 5;; bread) echo 2;; *) echo 6;; esac; }     # marks/cm3b_sel.txt (L000 값을 lego 전체·낙하에)
: > $L/j.txt
for t in L000 0016 ficus wolf bread; do echo "v $t $G --cg $(cap $t) $T --det --dump --dump_F" >> $L/j.txt; done
for t in $LG drop1; do echo "j $t $G --cg $(cap $t) $T --det" >> $L/j.txt; done
cat $L/j.txt >> $LOG; rclone copy $LOG $MR -q
CM_CARDS="1 2 3 4 5 6" CM_PAR=6 CM_GAP=5 bash exe/fem_cm_pool.sh cm4 $L/j.txt > $M/cm4_pool.log 2>&1
rclone copy $M/cm4_pool.log $MR -q
say "[판정·덤프 끝] $(date '+%T')"
# V 렌더 (빈 카드 하나에 하나; 덤프·h5 는 지우지 않고 R2 로)
free_card() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || return
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r i u; do
    [ "$i" = 0 ] && continue
    echo "$apps" | grep -q "$u" && continue
    if [ -f $CL/$i ] && kill -0 "$(cat $CL/$i)" 2>/dev/null; then continue; fi
    echo $i; break
  done
}
AGD=/tmp/dkta/ag_data
gtd() { case $1 in
  L000) echo "$W/lbdata/test_low/L000 $W/lbdata/config.json lego";;
  ficus) echo "$W/lbours/ficus_v0/gt $W/lbours/ficus_v0/gt/config.json ficus";;
  wolf|bread) local g=$AGD/gt_$1; [ -f $g/meta.json ] || g=/dev/shm/dkta/ag_data/gt_$1; echo "$g $g/config.json $1";;
  *) echo "$W/lbdata/test/$1 $W/lbdata/config.json lego";;
esac; }
mkgt() {   # 재질 씬 정답 궤적 디렉토리 (렌더용 meta·config) 가 없으면 절 AO 렌더와 같은 변환으로 /dev/shm 에
  local t=$1 h c; [ -f $AGD/gt_$t/meta.json ] && return; [ -f /dev/shm/dkta/ag_data/gt_$t/meta.json ] && return
  case $t in wolf) h=$W/topo/pg_wolf_sand; c=$W/PhysGaussian/config/wolf_config.json;; bread) h=$W/topo/pg_bread_plastic; c=$W/topo/bread_plastic_config.json;; esac
  mkdir -p /dev/shm/dkta/ag_data
  CUDA_VISIBLE_DEVICES=$2 $PY -u exe/pg_h5_to_pt.py --h5_dir $h --config $c --out_pt /dev/shm/dkta/ag_data/${t}_rend_tmp.pt --out_dir /dev/shm/dkta/ag_data/gt_$t > /dev/shm/dkta/ag_data/cv_${t}_rend.log 2>&1; }
rend() { local Gc=$1 t=$2; local O=$R/v_$t
  [ -f $O/pred.pt ] || { say "[덤프 없음] v_$t (발산 또는 실패)"; return; }
  rclone copy $O/pred.pt $RR/v_$t -q
  case $t in wolf|bread) mkgt $t $Gc;; esac
  read -r GTD CFG SH <<< "$(gtd $t)"
  [ -f $O/h5/meta.json ] || $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5 > $O/h5.log 2>&1
  mkdir -p $O/vj
  CUDA_VISIBLE_DEVICES=$Gc $PY -u exe/lb_render.py --sim $O/h5 --config $CFG --shape $SH --out $O/vj/fem_$SH.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r.log 2>&1
  rclone copy $O $RR/v_$t -q
  say "[렌더] v_$t 카드 $Gc: $(ls $O/vj/*.mp4 2>/dev/null || tail -3 $O/vj/r.log | tr '\n' ' ')"; }
for t in L000 0016 ficus wolf bread; do
  while true; do Gc=$(free_card); [ -n "$Gc" ] && break; sleep 20; done
  rend $Gc $t & echo $! > $CL/$Gc; sleep 10
done
wait
say "[렌더 끝] $(date '+%T')"
# 표
{ echo "# cm4 ($C) -- 절 CM3 채택 공통 설정 그대로: HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, BA = exe/fem_cm3_scn.sh, 상한 lego·낙하 2x6 · ficus 2x24 · wolf 2x5 · bread 2x2 (cm3b_sel), 결정적"; echo
  $PY - $R <<'PYE'
import json, os, sys
R = sys.argv[1]
def ld(d):
    p = os.path.join(R, d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None
LG = ["L000"] + ["%04d" % i for i in range(20)]
print("## lego 21 궤적 (판정 판 j_)")
print("| 궤적 | 오차 % | 발산 | <10 % | 프레임당 N · P (최대 P) | 끝 해시 |")
print("|---|---|---|---|---|---|")
es, ok, nd, n = [], 0, 0, 0
for t in LG:
    s = ld("j_" + t)
    if s is None:
        print("| %s | 미완 | | | | |" % t); continue
    n += 1
    dv = s.get("diverged")
    e = 100 * (s["err_mean"] if not dv else s["err_mean_done"])
    good = (not dv) and e < 10
    ok += good; nd += bool(dv)
    if not dv: es.append(e)
    it = "N %.1f · P %.1f (%d)" % (s["it_nN"]["mean"], s["it_nP"]["mean"], s["it_nP"]["max"]) if "it_nN" in s else ""
    print("| %s | %.2f%s | %s | %s | %s | %s |" % (t, e, " (발산 전)" if dv else "", ("x%d" % dv) if dv else "-", "O" if good else "**X**", it, s.get("x_sha1")))
print()
print("- 끝난 궤적 %d / 21, **<10 %% %d / %d**, 발산 %d, 평균 오차 (발산 없는 %d 궤적) %.2f %%" % (n, ok, n, nd, len(es), sum(es) / max(len(es), 1)))
s = ld("j_drop1")
print()
print("## 낙하 drop1 (판정 판)")
if s:
    print("- 오차 %s %%, 발산 %s" % ("%.2f" % (100 * (s["err_mean"] or s["err_mean_done"])), s.get("diverged") or "없음"))
p = os.path.join(R, "j_drop1", "drop_ana.txt")
if os.path.exists(p):
    for l in open(p).read().splitlines():
        if l.startswith("mpm:") or l.startswith("j:"): print("- " + l.split(", ε_zz 1/60")[0])
print()
print("## 영상 덤프 판 v_ (결정적, --dump --dump_F; L000·0016 은 판정 판 해시와 같아야 함)")
print("| 씬 | 오차 % | 발산 | 끝 해시 | 판정 판 해시 | 덤프 | 영상 |")
print("|---|---|---|---|---|---|---|")
for t in ["L000", "0016", "ficus", "wolf", "bread"]:
    s = ld("v_" + t); j = ld("j_" + t)
    if s is None:
        print("| %s | 미완 | | | | | |" % t); continue
    dv = s.get("diverged"); e = 100 * (s["err_mean"] if not dv else s["err_mean_done"])
    per = ""
    q = os.path.join(R, "v_" + t, "period.txt")
    if os.path.exists(q):
        import re
        m = re.findall(r"정답 대비 ([0-9.]+)\)[^\n]*?진폭 [0-9.]+ m \(비 ([0-9.]+)\)", open(q).read())
        if m: per = " (주기 비 %s 진폭 비 %s)" % m[-1]
    vj = os.path.join(R, "v_" + t, "vj"); mp = [f for f in os.listdir(vj) if f.endswith(".mp4")] if os.path.isdir(vj) else []
    print("| %s | %.2f%s | %s | %s | %s | %s | %s |" % (t, e, per, ("x%d" % dv) if dv else "-", s.get("x_sha1"), j.get("x_sha1") if j else "", "있음" if os.path.exists(os.path.join(R, "v_" + t, "pred.pt")) else "없음", ",".join(mp) or "없음"))
PYE
} > $M/cm4_table.md 2>&1
rclone copy $M $MR -q --include "cm4*"
rclone copy $R $RR -q --exclude "*/pred.pt"
echo "[cm4 끝] $(date '+%F %T')" > $M/cm4.END; rclone copy $M/cm4.END $MR -q
say "[cm4 끝] $(date '+%F %T')"
