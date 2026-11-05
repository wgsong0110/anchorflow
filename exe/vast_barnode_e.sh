#!/bin/bash
# 절 CM14.3 묶음 barnode_e (vast.ai A6000 1 장 인스턴스 onstart -> tmux bne; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   절 CM14 barnode 의 lego L000·wolf 판 ((B) o, (가) na5·na7, (나) no5·no7) 을 같은 명령 + 진단 --bar_ecell (결과 불변) 로 다시 -> 칸별 장벽 에너지 덤프 bener.pt, 끝 해시가 CM14 와 같은지 대조.
#   영상: 덤프 -> h5 -> 3DGS 알파블렌딩 + 장벽 걸린 칸 반투명 상자 (exe/lb_render.py --cells, 색 = log10 프레임 안 걸음 끝 최대 에너지, 모든 판·칸·두 씬 공통 고정 범위)
#     -> 두 줄 격자 + 색 막대 (exe/fem_bener_compose.py, 가로 <= 4096, yuv420p, faststart) -> result/anchorflow/v20261009/fem_barnode_{lego_L000,wolf}_energy_cmp.mp4.
#   정답·(A) 칸 = CM14 와 같은 영상 (장벽 없음 -- 겹침 없음).
# 출력 R2 lbours/fem_cm/barnode_e/<판>_<씬>, 대조 marks/barnode_e_check.md, 로그 marks/barnode_e.log, 끝 표식 marks/barnode_e.END.
# 준비를 다 끝낸 뒤 GPU 일을 이어서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 marks/barnode_e_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/bn
LOG=/root/bn/barnode_e.log
say() { echo "[bne $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/barnode_e.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/barnode_e.END
  [ -d /dev/shm/dkta/lbours/fem_cm/barnode_e ] && rclone copy /dev/shm/dkta/lbours/fem_cm/barnode_e $R/lbours/fem_cm/barnode_e -q --exclude pred.pt
  say "[실패 -> destroy] 자료는 R2 에"; vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG; exit 1; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $C $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,power.draw,clocks.max.sm,memory.total --format=csv | tee -a $LOG
say "[GPU] $(nvidia-smi --query-gpu=name,power.limit,power.default_limit --format=csv,noheader | tr '\n' ' ')"
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/bn/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    for k in 1 2 3 4 5; do curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; unzip -tq $f >/dev/null 2>&1 && break; sleep 20; done; done   # 절 CM14.2: HTML 이 받아진 적 있음 -> zip 확인·재시도
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, cv2, imageio, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/bn/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror
rclone copyto $R/vastpkg/L000.pt $W/one/L000/L000.pt -q
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q
rclone copy $MI/pgmodel/wolf_whitebg-trained $W/pgmodel/wolf_whitebg-trained --transfers 16 -q
rclone copy $MI/pgmodel/lego_whitebg-trained $W/pgmodel/lego_whitebg-trained --transfers 16 -q
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q
rclone copy $MI/lbdata $W/lbdata --transfers 16 -q
rclone copyto $MI/stab/lego/init/sim_0000000000.h5 $W/stab/lego/init/sim_0000000000.h5 -q
BSL=/dev/shm/dkta/lbours/fem_cm/barstick
rclone copy $R/lbours/fem_cm/barstick/a_wolf $BSL/a_wolf -q --exclude "h5/**" --exclude h5.log --exclude pred.pt
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C; [ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1 | cut -c1-120)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_bne XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/barnode_e; RR=$R/lbours/fem_cm/barnode_e; M=/dev/shm/dkta/lbours/fem_cm/marks
mkdir -p $B $M $CUPY_CACHE_DIR
TF=/dev/shm/dkta/bnecmp; mkdir -p $TF
rclone copyto $R/lbours/fem_ag_wolf/gt/gt_wolf.mp4 $TF/gt_wolf.mp4 -q
rclone copyto $R/lbours/fem2_lego/gt_lego.mp4 $TF/gt_L000.mp4 -q
rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Bold.otf $TF/fb.otf -q; rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $TF/fr.otf -q
export CM_SCN=exe/fem_cm8_scn.sh
set +u
source exe/fem_cm8_scn.sh
for t in L000 wolf; do cm_data $t; done
say "[자료] $(du -sh $W/* 2>&1 | tr '\n' ' ') 정답 h5 $(ls $W/topo/pg_wolf_sand/simulation_ply | wc -l) 개, lego 궤적 $(ls $W/lbdata/test_low/L000 | wc -l) 개, a_wolf $(ls $BSL/a_wolf/vj | tr '\n' ' ')"
[ -f /root/bn/pip.done ] && [ -s $W/one/L000/L000.pt ] && [ -s $AG/wolf.pt ] && [ -s $BSL/a_wolf/vj/fem_wolf.mp4 ] && [ -s $TF/fb.otf ] \
  && [ -s $TF/gt_wolf.mp4 ] && [ -s $TF/gt_L000.mp4 ] && [ -f $W/lbdata/test_low/L000/meta.json ] && [ -f $W/lbdata/config.json ] && [ -s $W/stab/lego/init/sim_0000000000.h5 ] && [ -d $W/pgmodel/lego_whitebg-trained ] \
  || fail "[세팅 실패] pip $(ls /root/bn/pip.done 2>&1) wolf $(ls $AG 2>&1 | tr '\n' ' ') a $(ls $BSL/a_wolf/vj 2>&1 | tr '\n' ' ') lego $(ls $W/lbdata 2>&1 | tr '\n' ' ')"
GT=/dev/shm/dkta/ag_data/gt_wolf; mkdir -p /dev/shm/dkta/ag_data
say "[준비 끝] GPU 일 시작"
[ -f $GT/meta.json ] || CUDA_VISIBLE_DEVICES=0 $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /dev/shm/dkta/ag_data/wolf_rend_tmp.pt --out_dir $GT > /dev/shm/dkta/ag_data/cv_wolf_rend.log 2>&1
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
declare -A NV=([na5]="--bar_node add --bar_ndh 0.5" [na7]="--bar_node add --bar_ndh 0.75" [no5]="--bar_node only --bar_ndh 0.5" [no7]="--bar_node only --bar_ndh 0.75" [o]="")
# 절 CM14.2 끝 해시 (같은 명령, --bar_ecell 없음)
declare -A H14=([o_L000]=c43a42c7d0 [na5_L000]=8fe29f5418 [na7_L000]=3ebad75624 [no5_L000]=3e9a81c639 [no7_L000]=2c66cf6816 [o_wolf]=9b119d8fd3 [na5_wolf]=5e8d9b835c [na7_wolf]=097361f982 [no5_wolf]=5e8d9b835c [no7_wolf]=097361f982)
DW="--det --hgt --bar_cnt --dump --dump_F --dump_part --bar_ecell"
cg() { [ $1 = wolf ] && echo 5 || echo 4; }
run_one() { local V=$1 t=$2 O=$B/$1_$2; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene $t $G --cg $(cg $t) $T $DW ${NV[$V]} --v=b && echo "카드 0 $C $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt \
    && CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1; echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt )
  local h=$(grep -o '"x_sha1": "[0-9a-f]*"' $O/summary.json 2>/dev/null | grep -o '[0-9a-f]\{10\}$')
  echo "| ${V}_$t | $h | ${H14[${V}_$t]} | $([ "$h" = "${H14[${V}_$t]}" ] && echo 같음 || echo **다름**) | $(ls -la $O/bener.pt 2>&1 | awk '{print $5}') |" >> $M/barnode_e_check.rows
  say "  $V $t $(tail -1 $O/run_info.txt) 해시 $h (CM14 ${H14[${V}_$t]}) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ')"
  rclone copy $O $RR/${V}_$t -q; }
rend1() { local V=$1 t=$2 O=$B/$1_$2 sh F=${3:-0}
  [ -f $O/pred.pt ] || { say "[영상 없음] ${V}_$t 덤프 없음"; return 1; }
  if [ $t = wolf ]; then sh=wolf; $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GT --out $O/h5 > $O/h5.log 2>&1; CF=$GT/config.json
  else sh=lego; $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $W/lbdata/test_low/L000 --out $O/h5 > $O/h5.log 2>&1; CF=$W/lbdata/config.json; fi
  mkdir -p $O/vj
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/lb_render.py --sim $O/h5 --config $CF --shape $sh --out $O/vj/fem_${sh}_e.mp4 --frame_label 0 --caption " " --frames $F \
    --cells $O/bener.pt --erange $ER --cell_key em < /dev/null > $O/vj/r.log 2>&1
  local rc=$?
  say "  [렌더] ${V}_$t rc=$rc $(ls -la $O/vj/fem_${sh}_e.mp4 2>&1 | awk '{print $5}') $(grep -a '칸 겹치기\|rror' $O/vj/r.log | tail -2 | cut -c1-200 | tr '\n' ' ')"
  rclone copy $O $RR/${V}_$t -q --exclude pred.pt --exclude "h5/**"; return $rc; }
LEGL="칸 = B-스플라인 칸 L ∈ [b+½, b+1½)³|색 = 그 칸 장벽 에너지의|  프레임 안 최대 (걸음 끝 K=2 중)|  = 표본점 몫 + 노드 몫|표본점 장벽 -> 그 점이 속한 칸|노드 장벽 -> 노드가 가운데인 칸|상자 = 프레임 끝 꼭짓점 8 개|색 범위 고정: 두 씬·모든 판·|  모든 칸 공통 (정규화 없음)|정답·(A) = 장벽 없음 (겹침 없음)"
# ---- 연기 (3 프레임): 새 진단 커널 컴파일·실행 + 렌더 겹치기 + 합성 확인. 실패하면 기록 올리고 destroy
smk() { local V=$1 t=$2 O=$B/smoke_$1_$2; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene $t $G --cg $(cg $t) $T $DW ${NV[$V]} --v=b && CUDA_VISIBLE_DEVICES=0 timeout 900 $PY -u $SIM --out $O --tag $t $DA $XA --frames 3 < /dev/null > $O/log.txt 2>&1; echo "rc=$?" > $O/rc.txt )
  grep -q '"be"' $O/fem.jsonl 2>/dev/null && [ "$(cat $O/rc.txt)" = rc=0 ] && [ -s $O/bener.pt ]; }
ER=-6,0
smk na5 L000 && smk no7 wolf && smk o L000 && rend1 smoke_na5 L000 3 && rend1 smoke_no7 wolf 3 \
  && $PY exe/fem_bener_compose.py $TF/smoke.mp4 "연기" $TF/fb.otf $TF/fr.otf $ER "$LEGL" 2 "가" $B/smoke_na5_L000/vj/fem_lego_e.mp4 - "나" $B/smoke_no7_wolf/vj/fem_wolf_e.mp4 - "정답" $TF/gt_L000.mp4 - > $TF/smoke_c.log 2>&1 \
  && rclone copyto $TF/smoke.mp4 $RR/vid/smoke.mp4 -q \
  && say "[연기 통과] $(for q in $B/smoke_*; do echo "$(basename $q) $(tail -1 $q/fem.jsonl | grep -o '"be": {[^}]*}' | cut -c1-200)"; done | tr '\n' ' ') $(tail -1 $TF/smoke_c.log)" \
  || fail "[연기 실패] $(for q in $B/smoke_*; do echo "$(basename $q) $(cat $q/rc.txt 2>/dev/null) $(grep -a 'rror\|Trace' $q/log.txt $q/vj/r.log 2>/dev/null | tail -3 | cut -c1-300)"; done | tr '\n' ' ') $(tail -3 $TF/smoke_c.log 2>/dev/null | tr '\n' ' ')"
: > $M/barnode_e_check.rows
lane1() { for V in o na5 na7 no5 no7; do run_one $V wolf; done; }
lane2() { for V in o na5 na7 no5 no7; do run_one $V L000; done; }
lane1 & sleep 5; lane2 & wait
{ echo "# barnode_e ($C) -- 절 CM14.3 같은 명령 + --bar_ecell, 끝 해시 대 CM14.2 ($(date '+%F %T'))"; echo; echo "| 판 | 해시 (barnode_e) | CM14.2 | 대조 | bener.pt 바이트 |"; echo "|---|---|---|---|---|"; sort $M/barnode_e_check.rows; } > $M/barnode_e_check.md
rclone copy $M/barnode_e_check.md $MR -q
say "[판 끝] $(grep -c 같음 $M/barnode_e_check.rows) / 10 같음"
# ---- 색 범위: 모든 판·모든 칸·두 씬의 프레임 안 최대 에너지 em > 0 -- 상한 = 최대의 10 거듭제곱 올림, 하한 = 1 백분위의 내림 (상한 − 8 아래면 상한 − 8)
ER=$($PY - $B/*_L000/bener.pt $B/*_wolf/bener.pt <<'PYX'
import sys, math, torch
v = []
for p in sys.argv[1:]:
    if "/smoke_" in p:
        continue
    for c in torch.load(p, weights_only=False)["frames"]:
        e = c["em"]; v.append(e[e > 0].double())
v = torch.cat(v)
hi = math.ceil(math.log10(float(v.max()))); lo = math.floor(math.log10(float(torch.quantile(v[torch.randperm(v.numel())[:1000000]], 0.01))))
print(f"{max(lo, hi - 8)},{hi}")
PYX
)
say "[색 범위 log10] $ER"; echo "$ER" > $M/barnode_e_erange.txt; rclone copy $M/barnode_e_erange.txt $MR -q
for V in o na5 na7 no5 no7; do rend1 $V wolf & sleep 3; rend1 $V L000 & wait; done
CAP="HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, 결정적 · (B) = 장벽 + --y0eb (표본점 장벽) · (가) 더함 = 표본점 + 노드 장벽, (나) 노드만 · d̂_n = ½ 또는 ¾ dx_정답 · 칸 겹침 = 장벽 에너지 (색 막대)"
NM() { case $1 in o) echo "(B) 장벽 (표본점)";; na5) echo "(가) 표본점 + 노드 장벽 d̂ ½dx";; na7) echo "(가) 표본점 + 노드 장벽 d̂ ¾dx";; no5) echo "(나) 노드 장벽만 d̂ ½dx";; no7) echo "(나) 노드 장벽만 d̂ ¾dx";; esac; }
ARGL=("정답 (MPM)" $TF/gt_L000.mp4 -)
ARGW=("정답 (PG 모래)" $TF/gt_wolf.mp4 - "(A) 가우스 점 고착" $BSL/a_wolf/vj/fem_wolf.mp4 $BSL/a_wolf/summary.json)
for V in o na5 na7 no5 no7; do ARGL+=("$(NM $V)" $B/${V}_L000/vj/fem_lego_e.mp4 $B/${V}_L000/summary.json); ARGW+=("$(NM $V)" $B/${V}_wolf/vj/fem_wolf_e.mp4 $B/${V}_wolf/summary.json); done
for t in L000 wolf; do
  if [ $t = L000 ]; then NAME=fem_barnode_lego_L000_energy_cmp.mp4; TT="lego L000 80 프레임 · PCG 상한 4 · $CAP"; A=("${ARGL[@]}"); NC=3; else NAME=fem_barnode_wolf_energy_cmp.mp4; TT="wolf 50 프레임 · PCG 상한 5 ((A) 3) · $CAP"; A=("${ARGW[@]}"); NC=4; fi
  if $PY exe/fem_bener_compose.py $TF/$NAME "$TT" $TF/fb.otf $TF/fr.otf $ER "$LEGL" $NC "${A[@]}" > $TF/compose_$t.log 2>&1; then
    rclone copyto $TF/$NAME $R/v20261009/$NAME -q && rclone check $TF $R/v20261009 --include $NAME --one-way --size-only -q \
      && say "[영상] result/anchorflow/v20261009/$NAME (R2 확인) $(tail -1 $TF/compose_$t.log)" || say "[영상 올리기 확인 실패] $NAME"
    rclone copyto $TF/$NAME $RR/vid/$NAME -q
  else say "[영상 실패] $t $(tail -5 $TF/compose_$t.log | tr '\n' ' ' | cut -c1-600)"; fi
done
echo "[barnode_e 끝] $(date '+%F %T')" > $M/barnode_e.END
say "[barnode_e 끝]"
rclone copy $B $RR -q --transfers 8 --exclude "*/h5/**"
if rclone check $B $RR --one-way --size-only -q --exclude "*/h5/**" 2>/root/bn/check.txt; then
  rclone copy $M/barnode_e.END $MR -q
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/bn/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/barnode_e_CHECK_FAIL
fi
