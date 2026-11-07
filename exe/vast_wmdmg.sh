#!/bin/bash
# 절 CM19 묶음 wmdmg -- 수박 손상 모델 (vast.ai A6000 2 장, onstart -> tmux wmdmg; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, CONTAINER_ID·CONTAINER_API_KEY):
#   판 = 절 CM13 s_a 명령 (exe/fem_cm9_scn.sh wm + --gq_mass --gq_chk --dump --nan_step --jdiag --elx snh) + --dump_lj (결과 불변) 에
#     s  : 그대로 (SNH 완주 판)
#     ct·cu : --dmg cdm --dmg_psic tens|uni (연속체 손상, ψ_c = NACC 정수압 인장 꼭짓점 / 단축 인장 항복의 ψ⁺)
#     pt·pu : --dmg pf  --dmg_psic tens|uni (위상장 AT2, G_c = 512/27 ψ_c l0, l0 = 칸)
#   셀 1.5 (각 2 회) + 2·2.5·3 (각 1 회 먼저, 그 뒤 2 회째) -- 카드 0 차례로. 카드 1 은 단독 시간 z_<판>_<셀> (진단·덤프 없이, 그 카드에 판 하나).
#   영상용 렌더 (3DGS, GF wm_render): 셀 1.5 와 "가장 빠르면서 발산 안 한 셀" 에서 SNH·CDM(정수압)·위상장(정수압) 판의 logJp 색 (−0.2~0.4), d 색 (0~1), SNH 일반 렌더.
#   합성은 로컬 (exe/wmdmg_compose.py, 정답 칸 = R2 wmJd/gt_lj.mp4·gt_new_plain.mp4).
# 출력 R2 lbours/fem_cm/wmdmg/<판>, 표 marks/wmdmg_table.md, 로그 marks/wmdmg.log, 끝 표식 marks/wmdmg.END. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 marks/wmdmg_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/wg
LOG=/root/wg/wmdmg.log
say() { echo "[wmdmg $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/wmdmg.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/wmdmg.END; exit 1; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $C $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG
free -g | head -2 | tee -a $LOG; df -h /workspace | tail -1 | tee -a $LOG
# ---- 클러스터 경로 그대로 (exe/vast_wmd.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/wg/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib scikit-learn vastai 2>&1 | tail -2 | tee -a $LOG; pip install -q --only-binary=:all: PyMCubes==0.1.6 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/wg/pip.done
fi
[ -f /root/wg/pip.done ] || fail "[세팅 실패] pip"
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C
[ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1 | cut -c1-120)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
# ---- 자료 (GPU 일 전에 모두)
W=/home/dkta/work; MI=$R/_vastmirror
export CM_SCN=exe/fem_cm9_scn.sh
set +u
source exe/fem_cm9_scn.sh
cm_data wm
rclone copy $MI/gfmodel/model/watermelon $W/gfmodel/model/watermelon --transfers 16 -q
rclone copy $MI/GaussianFluent $W/GaussianFluent --transfers 32 -q
rclone copyto $MI/wm2/gf/config.json $W/wm2/gf/config.json -q
rclone copy $MI/wm2/gf/simulation_ply $W/wm2/gf/simulation_ply --transfers 16 -q
B=/dev/shm/dkta/lbours/fem_cm/wmdmg; RR=$R/lbours/fem_cm/wmdmg; M=/dev/shm/dkta/lbours/fem_cm/marks; mkdir -p $B $M
rclone copyto $R/lbours/fem_cm/wmJd/wmd_stats.json $B/gt_wmd_stats.json -q
CFG=$W/wm2/gf/config.json; WMM=$W/gfmodel/model/watermelon; GTD=/dev/shm/dkta/ag_data/gt_wm; mkdir -p /dev/shm/dkta/ag_data
[ -f $GTD/meta.json ] || $PY -u exe/pg_h5_to_pt.py --h5_dir $W/wm2/gf/simulation_ply --config $CFG --out_pt /dev/shm/dkta/ag_data/wm_rend_tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1
rm -f /dev/shm/dkta/ag_data/wm_rend_tmp.pt
rm -rf $W/wm2/gf/simulation_ply                                                         # 렌더 처음 상태 (GTD) 만 쓰면 됨 -- 디스크
say "[자료] wm.pt $(ls -la $WMD/wm.pt 2>&1 | awk '{print $5}') 3DGS $(du -sh $WMM $W/GaussianFluent 2>&1 | awk '{print $1}' | tr '\n' ' ') GTD $(ls $GTD | wc -l) 정답 통계 $(ls -la $B/gt_wmd_stats.json 2>&1 | awk '{print $5}')"
[ -s $WMD/wm.pt ] && [ -f $GTD/meta.json ] && [ -s $B/gt_wmd_stats.json ] || fail "[자료 실패] $(tail -3 /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1 | tr '\n' ' ')"
NG=$(nvidia-smi -L | wc -l); C1=$(( NG > 1 ? 1 : 0 ))
# 이어 돌리기 (11-07: 첫 인스턴스가 렌더 단계에서 정지 -> 새 인스턴스): R2 에 이미 있는 판 디렉토리 (pred.pt 빼고) 를 받아 run1 이 건너뛰게
rclone copy $RR $B -q --transfers 16 --exclude "*/pred.pt" --exclude "*/vid/**" --exclude "smoke_*/**"
say "[이어 돌리기] R2 에서 받은 판 $(ls $B | grep -c _) 개 (summary.json $(ls $B/*/summary.json 2>/dev/null | wc -l))"
# ---- 연기: 새 커널 컴파일·3 프레임 (cdm·pf, 셀 1.5) -- 실패면 멈춤
for v in cdm pf; do O=/root/wg/smoke_$v; grep -q "롤아웃" $O/log.txt 2>/dev/null && continue; mkdir -p $O; cm_scene wm --gq_mass --elx snh --dmg $v --frames 3
  CUDA_VISIBLE_DEVICES=0 timeout 900 $PY -u $SIM --out $O --tag wm $DA $XA < /dev/null > $O/log.txt 2>&1
  say "  [연기 $v] rc=$? $(grep -a '절 CM19\|롤아웃\|rror' $O/log.txt | tail -3 | cut -c1-400 | tr '\n' ' ')"
  grep -q "롤아웃" $O/log.txt || { rclone copy $O $RR/smoke_$v -q; fail "[연기 실패] $v $(tail -5 $O/log.txt | tr '\n' ' ' | cut -c1-900)"; }
  rclone copy $O $RR/smoke_$v -q; done
say "[준비 끝] GPU 일 시작 (카드 $NG 장, 판정 카드 0, 단독 시간 카드 $C1)"
table() { $PY exe/fem_wmdmg_table.py $B $B/gt_wmd_stats.json > $M/wmdmg_table.md.tmp 2>$M/wmdmg_table.err
  { echo "# wmdmg ($C) -- 절 CM19 수박 손상 모델 (CDM·위상장) x 셀 ($(date '+%F %T') UTC 기준)"; echo; cat $M/wmdmg_table.md.tmp; } > $M/wmdmg_table.md; rclone copy $M $MR -q --include "wmdmg_table.*"; }
( while [ ! -f /root/wg/all.done ]; do sleep 300; table; done ) &
PT=$!
declare -A VA=([s]="--elx snh" [ct]="--elx snh --dmg cdm --dmg_psic tens" [cu]="--elx snh --dmg cdm --dmg_psic uni" [pt]="--elx snh --dmg pf --dmg_psic tens" [pu]="--elx snh --dmg pf --dmg_psic uni")
declare -A CV=([c15]=1.5 [c2]=2 [c25]=2.5 [c3]=3)
X="--gq_mass --gq_chk --dump --nan_step --jdiag --dump_lj"
# 판 하나: $1 이름 $2 카드 나머지 = 목록 인자. 끝나면 손상 통계·R2 올림, pred.pt 는 올린 뒤 지움 (렌더 때 다시 받음)
run1() { local v=$1 cd=$2; shift 2; local O=$B/$v; mkdir -p $O
  [ -s $O/summary.json ] && { say "  $v 이미 있음 (이어 돌리기)"; return; }
  export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wg_$cd; mkdir -p $CUPY_CACHE_DIR
  cm_scene wm "$@"
  local UU=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v c=$cd '$1==c {print $2}')
  echo "카드 $cd ($UU) $(cat .done) $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -c "$UU" > $O/apps_at_start.txt
  CUDA_VISIBLE_DEVICES=$cd timeout 5400 $PY -u $SIM --out $O --tag wm $DA $XA < /dev/null > $O/log.txt 2>&1 &
  local PP=$! mx=0
  while kill -0 $PP 2>/dev/null; do sleep 10
    local me=$(pgrep -P $PP | head -1); local n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$UU" | grep -vc ", ${me:-x}$"); [ "$n" -gt "$mx" ] && mx=$n; done
  wait $PP; echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt; echo $mx > $O/apps_during.txt
  [ -s $O/lj.pt ] && $PY exe/wmdmg_stats.py --run $O > $O/dmg_stats.log 2>&1
  say "  $v 카드 $cd $(tail -1 $O/run_info.txt) 다른 프로세스 최대 $mx $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-260 | tr '\n' ' ') $(tail -1 $O/dmg_stats.log 2>/dev/null | cut -c1-200)"
  rclone copy $O $RR/$v -q --transfers 8 && [ -s $O/pred.pt ] && rclone check $O $RR/$v --one-way --size-only -q --include pred.pt && rm -f $O/pred.pt; }
# 카드 1: 단독 시간 20 판 (셀 x 판)
( for c in c15 c2 c25 c3; do for v in s ct cu pt pu; do run1 z_${v}_$c $C1 --gq_mass ${VA[$v]} --cell ${CV[$c]}; done; done; touch /root/wg/z.done ) &
PZ=$!
[ $C1 = 0 ] && wait $PZ                                                                  # 카드 하나면 차례로
# 카드 0: 판정 -- 셀 1.5 a, 다른 셀 a, 셀 1.5 b, 다른 셀 b
for v in s ct cu pt pu; do run1 ${v}_c15_a 0 $X ${VA[$v]}; done
for c in c2 c25 c3; do for v in s ct cu pt pu; do run1 ${v}_${c}_a 0 $X ${VA[$v]} --cell ${CV[$c]}; done; done
for v in s ct cu pt pu; do run1 ${v}_c15_b 0 $X ${VA[$v]}; done
table
for c in c2 c25 c3; do for v in s ct cu pt pu; do run1 ${v}_${c}_b 0 $X ${VA[$v]} --cell ${CV[$c]}; done; done
wait $PZ
table
# ---- 영상용 렌더: 셀 1.5 + 가장 빠르면서 발산 안 한 셀 (판 s·ct·pt 모두 a 또는 b 완주인 셀 중 단독 시간 (셋 중 가장 느린 판의 평균) 이 가장 작은 셀, 1.5 빼고)
pick() { $PY - $B $1 $2 <<'PYX'
import json, os, sys
B, v, c = sys.argv[1:4]
def sm(r):
    try: return json.load(open(f"{B}/{v}_{c}_{r}/summary.json"))
    except Exception: return None
ok = [r for r in "ab" if sm(r) and sm(r).get("diverged") is None and sm(r).get("frames_done") == 100]
if ok: print(f"{v}_{c}_{ok[0]} 1")
else:
    rs = [r for r in "ab" if sm(r)]
    print((f"{v}_{c}_" + max(rs, key=lambda r: sm(r).get("frames_done", 0)) + " 0") if rs else "- 0")
PYX
}
FAST=$($PY - $B <<'PYX'
import json, os, sys
B = sys.argv[1]; best = None
for c in ("c2", "c25", "c3"):
    ok = True; ms = []
    for v in ("s", "ct", "pt"):
        fin = False
        for r in "ab":
            try:
                s = json.load(open(f"{B}/{v}_{c}_{r}/summary.json")); fin = fin or (s.get("diverged") is None and s.get("frames_done") == 100)
            except Exception: pass
        try: ms.append(json.load(open(f"{B}/z_{v}_{c}/summary.json"))["ms_mean"])
        except Exception: ok = False
        ok = ok and fin
    if ok and ms and (best is None or max(ms) < best[1]): best = (c, max(ms))
print(best[0] if best else "")
PYX
)
say "[영상 셀] c15 ${FAST:-(다른 셀 없음 -- 셋 다 완주·시간 있는 셀 없음)}"
echo "c15 $FAST" > $B/video_cells.txt
rend() { local r=$1 kind=$2 cd=$3; local O=$B/$r; local V=$O/vid; mkdir -p $V
  [ -s $V/$kind.mp4 ] && { say "  [렌더 있음] $r $kind"; return; }
  [ -s $O/pred.pt ] || rclone copyto $RR/$r/pred.pt $O/pred.pt -q
  [ -d $O/h5_fem ] || { $PY - $O/pred.pt $O/pred_v.pt <<'PYX' > $V/pad.log 2>&1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
ok = [i for i in range(P.shape[0]) if bool(torch.isfinite(P[i]).all()) and bool(torch.isfinite(F[i]).all())]
n = 0
while n < P.shape[0] and n in ok: n += 1
L = 100
if n < L:
    P = torch.cat([P[:n], P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F[:n], F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("유한 프레임", n, "->", P.shape[0])
PYX
    $PY -u exe/lb_roll_to_h5.py --dump $O/pred_v.pt --traj $GTD --out $O/h5_fem > $V/h5.log 2>&1; rm -f $O/pred_v.pt; }
  [ -d $O/sc_lj ] || $PY exe/wmdmg_stats.py --run $O --sc_lj $O/sc_lj --sc_d $O/sc_d > $V/sc.log 2>&1
  case $kind in
    lj) SA="--scalar_dir $O/sc_lj --vmin -0.2 --vmax 0.4 --cmap viridis";;
    d) SA="--scalar_dir $O/sc_d --vmin 0 --vmax 1 --cmap viridis";;
    *) SA="";;
  esac
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$cd $PY -u $D/exe/wm_render.py --model_path $WMM --config $CFG --sim $O/h5_fem $SA --out $V/$kind.mp4 < /dev/null > $V/r_$kind.log 2>&1)
  say "  [렌더] $r $kind $(tail -1 $V/pad.log 2>/dev/null) $(ls -la $V/$kind.mp4 2>&1 | awk '{print $5}') $(tail -1 $V/r_$kind.log | cut -c1-160)"
  rclone copy $V $RR/$r/vid -q; }
for c in c15 $FAST; do
  read RS FS < <(pick s $c); read RC FC < <(pick ct $c); read RP FP < <(pick pt $c)
  echo "$c $RS $FS $RC $FC $RP $FP" >> $B/video_runs.txt
  say "[렌더 판] 셀 $c: SNH $RS (완주 $FS) CDM $RC ($FC) 위상장 $RP ($FP)"
  # (고침: 인자 없는 wait 가 표 고리 (PT) 까지 기다려 멈춤 -> 이 단계 pid 만 기다림)
  PD=""; for r in $RS $RC $RP; do [ "$r" = - ] || [ -s $B/$r/pred.pt ] || { rclone copyto $RR/$r/pred.pt $B/$r/pred.pt -q & PD="$PD $!"; }; done; [ -n "$PD" ] && wait $PD
  # 셋 동시 (카드 0·1 나눔)
  ( [ "$RS" = - ] || { rend $RS lj 0; rend $RS plain 0; } ) & P1=$!
  ( [ "$RC" = - ] || { rend $RC lj $C1; rend $RC d $C1; } ) & P2=$!
  ( [ "$RP" = - ] || { rend $RP lj 0; rend $RP d $C1; } ) & P3=$!
  wait $P1 $P2 $P3
  for r in $RS $RC $RP; do rm -rf $B/$r/h5_fem $B/$r/sc_lj $B/$r/sc_d $B/$r/pred.pt; done
done
# ---- 사용자 추가 (11-07): 셀 2·3 영상 -- 판 s·ct·cu·pt·pu 각각 a·b 중 마지막 프레임 손상 몫 (SNH = logJp > 0, 손상 판 = d > 0.5 또는 logJp > 0) 이 큰 실행
pickd() { $PY - $B $1 $2 <<'PYX'
import json, sys
B, v, c = sys.argv[1:4]; best = None
for r in "ab":
    try:
        st = json.load(open(f"{B}/{v}_{c}_{r}/dmg_stats.json")); sm = json.load(open(f"{B}/{v}_{c}_{r}/summary.json"))
    except Exception: continue
    fr = [q for q in st["frames"] if q["t"] <= max(st["n_fin_frames"], 1)] or st["frames"]
    x = fr[-1]["lj0"] if v == "s" else fr[-1].get("uni", 0.0)
    if best is None or x > best[1]: best = (f"{v}_{c}_{r}", x, sm.get("diverged") or 0)
print(f"{best[0]} {best[1]:.6f} {best[2]}" if best else "- 0 0")
PYX
}
for c in c2 c3; do
  PL=""; for v in s ct cu pt pu; do read RR_ XX_ DV_ < <(pickd $v $c); echo "$c $v $RR_ $XX_ $DV_" >> $B/video6_runs.txt; PL="$PL $RR_"; done
  say "[렌더 판 6] 셀 $c: $(grep "^$c " $B/video6_runs.txt | awk '{print $3"("$4","$5")"}' | tr '\n' ' ')"
  set -- $PL
  ( [ "$1" = - ] || rend $1 lj 0; [ "$2" = - ] || rend $2 d 0; [ "$3" = - ] || rend $3 d 0 ) & P1=$!
  ( [ "$4" = - ] || rend $4 d $C1; [ "$5" = - ] || rend $5 d $C1 ) & P2=$!
  wait $P1 $P2
  for r in $PL; do rm -rf $B/$r/h5_fem $B/$r/sc_lj $B/$r/sc_d $B/$r/pred.pt; done
done
touch /root/wg/all.done; kill $PT 2>/dev/null
table
rclone copy $B $RR -q --transfers 8 --exclude "*/h5_fem/**" --exclude "*/sc_*/**"
echo "[wmdmg 끝] $(date '+%F %T')" > $M/wmdmg.END; rclone copy $M/wmdmg.END $MR -q
say "[wmdmg 끝]"
if rclone check $B $RR --one-way --size-only -q --exclude "*/h5_fem/**" --exclude "*/sc_*/**" 2>/root/wg/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/wg/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/wmdmg_CHECK_FAIL
fi
