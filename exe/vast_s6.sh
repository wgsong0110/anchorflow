#!/bin/bash
# PBS 서버 장애 동안 vast.ai A6000 인스턴스에서 클러스터 잡 안 묶음을 그대로 돌린다 (doc/pareto.md 절 S.6).
# 인스턴스 onstart 가 tmux 세션 s6 안에서 부름: env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT.
# 클러스터 경로를 그대로 만든다 (스크립트 무수정): /home/dkta/work (자료), /dev/shm/dkta -> /workspace/shm (디스크; 컨테이너 /dev/shm 이 작음), /tmp/dkta -> /workspace/tmpd,
#   /home/dkta/.conda/envs/{af,simp}/bin/python -> 이미지 python (pytorch 2.5.1 cu121 py3.11, 바이너리 휠만 -- 인스턴스 빌드 금지라 Spring-Gaus 는 ST_NO_BUILD 로 건너뜀).
# 순서: (a) exe/fem_cm9.sh  ||  (b) stab 연기 3 프레임 (L000·ficus; wolf·bread 는 정답 h5 가 R2 에 없어 PBS 몫) -> 칸이 다 찬 방법으로 본 묶음 stab1
#       (c) cm9 끝난 뒤 exe/fem_cm8a_vid.sh.  표식 R2 lbours/vast_s6/marks/*.  /dev/shm/dkta/lbours 전체를 10 분마다 R2 lbours/ 로 (필터 없음).
# 모두 끝나면 rclone check 차이 0 일 때 스스로 destroy (CONTAINER_API_KEY).
set -u
C=${AF_COMMIT:?}
PART=${S6_PART:-}   # 빈 값 = 전부, cm = (a)(c) 만, stab = (b) 만 (2 장짜리 인스턴스 둘로 나눌 때; 표식·로그 이름에 붙음)
SFX=${PART:+_$PART}
MK=r2:storage/result/anchorflow/lbours/vast_s6/marks
mkdir -p /workspace/shm /workspace/tmpd /root/s6
LOG=/root/s6/s6$SFX.log
say() { echo "[s6$SFX $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MK/s6$SFX.log -q 2>/dev/null; }
# ---- rclone
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
ldconfig -p | grep -q libEGL.so.1 || (apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libgomp1) >/dev/null 2>&1   # open3d (PhysTwin·Spring-Gaus)
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $C $(hostname)"
nvidia-smi --query-gpu=index,name,power.limit,power.draw,clocks.max.sm,memory.total --format=csv | tee -a $LOG
# ---- 경로
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/.conda/envs/simp/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python)
for e in af simp; do ln -sf $PYS /home/dkta/.conda/envs/$e/bin/python; done
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
# ---- 파이썬 (바이너리 휠만)
if [ ! -f /root/s6/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, warp, cupy, h5py, lpips, diff_gaussian_rasterization, simple_knn; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'warp', warp.config.version, 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/s6/pip.done
fi
# ---- 공개 코드 베이스라인 의존성 (절 S.3; 버리는 인스턴스라 이미지 python 에 의존성째 -- 바이너리 휠만, numpy<2 고정 (warp 0.10.1·taichi))
if [ ! -f /root/s6/pipbl.done ]; then
  pip install -q --only-binary=:all: "numpy<2" open3d scikit-learn cma kornia termcolor trimesh yacs pygltflib usd-core dataclasses-json plotly iopath 2>&1 | grep -v WARNING | tail -3 | tee -a $LOG
  python -c "import numpy, open3d, sklearn, pygltflib, dataclasses_json, pxr, kornia, cma, yacs, iopath; assert int(numpy.__version__.split('.')[0]) < 2; print('[pip bl] ok numpy', numpy.__version__, 'open3d', open3d.__version__)" 2>&1 | tail -1 | tee -a $LOG
  python -c "import numpy, open3d, sklearn, pygltflib, dataclasses_json, pxr; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null && touch /root/s6/pipbl.done
fi
# ---- 자료 (R2 -> 클러스터 경로)
W=/home/dkta/work; R=r2:storage/result/anchorflow
if [ ! -f /root/s6/data.done ]; then
  rclone copy $R/_vastmirror/i-physgaussian $W/i-physgaussian --transfers 16 -q
  rclone copy $R/_vastmirror/lbdata $W/lbdata --transfers 16 -q
  rclone copy $R/lbdata/test/0016 $W/lbdata/test/0016 --transfers 16 -q
  rclone copyto $R/_vastmirror/lbpt/test_low/L000.pt $W/lbpt/test_low/L000.pt -q
  rclone copyto $R/_vastmirror/lbpt/test/0016.pt $W/lbpt/test/0016.pt -q
  rclone copyto $R/vastpkg/L000.pt $W/one/L000/L000.pt -q
  rclone copy $R/lbours/ficus_v0/gt $W/lbours/ficus_v0/gt --transfers 16 -q
  rclone copyto $R/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt -q
  rclone copy $R/_vastmirror/pgmodel $W/pgmodel --transfers 16 -q
  du -sh $W/* | tee -a $LOG
  [ -s $W/one/L000/L000.pt ] && [ -s $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt ] && [ -f $W/lbdata/test_low/L000/meta.json ] && touch /root/s6/data.done
fi
[ -f /root/s6/data.done ] && [ -f /root/s6/pip.done ] || { say "[세팅 실패] pip $(ls /root/s6/pip.done 2>&1) data $(ls /root/s6/data.done 2>&1)"; exit 1; }
# ---- 클러스터에서 R2 _vastmirror 로 옮긴 wolf·bread 정답·3DGS, 수박 3DGS·정답·GaussianFluent (로컬 ~/s6x/pull.sh, 끝 표식 pull.log PULL_END) -> 같은 클러스터 경로 (배경)
MI=$R/_vastmirror
( for i in $(seq 480); do rclone cat $MI/pull.log 2>/dev/null | grep -q PULL_END && break; sleep 30; done
  rclone cat $MI/pull.log > /root/s6/pull.log 2>&1; grep -q PULL_END /root/s6/pull.log || { say "[미러 없음] 4 h 안에 pull.log PULL_END 없음 -> wolf·bread·수박 영상 보류"; exit 0; }
  grep -q "rc=[^0]" /root/s6/pull.log && say "[미러 경고] $(grep 'rc=[^0]' /root/s6/pull.log | tr '\n' ' ')"
  for p in topo/bread_plastic_config.json PhysGaussian/config/wolf_config.json wm2/gf/config.json; do rclone copyto $MI/$p $W/$p -q; done
  for p in pgmodel/wolf_whitebg-trained pgmodel/bread-trained topo/pg_wolf_sand topo/pg_bread_plastic; do rclone copy $MI/$p $W/$p --transfers 16 -q; done
  grep -q "check topo/pg_wolf_sand rc=0" /root/s6/pull.log && grep -q "check topo/pg_bread_plastic rc=0" /root/s6/pull.log && touch /root/s6/wb.done
  say "[미러 wolf·bread] $(du -sh $W/topo/pg_wolf_sand $W/topo/pg_bread_plastic 2>&1 | tr '\n' ' ') 표식 $(ls /root/s6/wb.done 2>&1)"
  for p in gfmodel/model/watermelon GaussianFluent wm2/gf/simulation_ply; do rclone copy $MI/$p $W/$p --transfers 16 -q; done
  grep -q "check wm2/gf/simulation_ply rc=0" /root/s6/pull.log && touch /root/s6/wm.done
  say "[미러 수박] $(du -sh $W/gfmodel/model/watermelon $W/wm2/gf/simulation_ply $W/GaussianFluent 2>&1 | tr '\n' ' ') 표식 $(ls /root/s6/wm.done 2>&1)" ) &
export S6_WM_MARK=/root/s6/wm.done
# ---- 코드 (github 공개 레포에서 clone/pull -> 커밋별 디렉토리; 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline && git -C /root/af pull -q origin implicit-bspline 2>/dev/null; git -C /root/af checkout -q $C || { say "[코드 실패] $C"; exit 1; }
D=/dev/shm/dkta/impwt/$C
[ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
echo "[ok] $(date)" | rclone rcat $MK/setup$SFX.done
say "[세팅 끝] $(git -C $D log --oneline -1)"
# ---- 결과 동기화 (필터 없음)
( while true; do sleep 600; rclone copy /dev/shm/dkta/lbours $R/lbours -q --transfers 8; done ) & SY=$!
cd $D
# ---- (a) cm9
PA=""
[ "$PART" = stab ] || { ( if [ -n "$(rclone lsf $MK/cm9.done 2>/dev/null)" ]; then say "[cm9] 이미 끝 (표식)"; else
    say "[cm9 띄움]"; bash exe/fem_cm9.sh $C > /root/s6/cm9.out 2>&1; say "[cm9 끝] rc=$?"; rclone copy /dev/shm/dkta/lbours/fem_cm $R/lbours/fem_cm -q
    echo "$(date)" | rclone rcat $MK/cm9.done; fi
  # ---- (c) A 판 lego 영상 (cm9 뒤)
  if [ -z "$(rclone lsf $MK/cm8av.done 2>/dev/null)" ]; then say "[cm8a 영상 띄움]"; bash exe/fem_cm8a_vid.sh $C > /root/s6/cm8av.out 2>&1; say "[cm8a 영상 끝] rc=$?"
    echo "$(date)" | rclone rcat $MK/cm8av.done; fi ) &
PA=$!; }
# ---- (b) stab 연기 -> 본 묶음
if [ "$PART" != cm ]; then
export ST_NO_BUILD=1
MS="mpm,vrgs,bnv,ours,phystwin,simplicits"; SC="L000,ficus"
if [ -z "$(rclone lsf $MK/stab_smoke.done 2>/dev/null)" ]; then
  say "[stab 연기 띄움] 방법 $MS 씬 $SC"
  ST_OUT=/dev/shm/dkta/lbours/stab_smoke ST_UNTIL=$(( $(date +%s) + 8 * 3600 )) ST_HARD=$(( $(date +%s) + 9 * 3600 )) ST_ARGS="--smoke 3 --par 3 --methods $MS --scenes $SC" bash exe/stab.sh $C
  say "[stab 연기 끝] $(tail -1 /dev/shm/dkta/lbours/stab_smoke/bl_setup.log)"
fi
# 칸이 다 찬 방법 (모든 씬에서 최소 N·시간·PSNR 이 있음) 만 본 묶음으로
OK=$(python - /dev/shm/dkta/lbours/stab_smoke/stab.json $MS $SC <<'PYX'
import json, sys
R = json.load(open(sys.argv[1])); ok = []
for m in sys.argv[2].split(","):
    good = True
    for t in sys.argv[3].split(","):
        q = R.get(f"{m}_{t}") or {}
        ns = q.get("nmin")
        if ns is None or (q.get("time") or {}).get("ms") is None or (q.get("vis") or {}).get("psnr") is None:
            good = False
    if good: ok.append(m)
print(",".join(ok))
PYX
)
echo "방법 $OK" | rclone rcat $MK/stab_smoke.done
say "[stab 연기 판정] 칸이 다 찬 방법: ${OK:-없음} (전체 $MS)"
# ---- Spring-Gaus: pytorch3d 휠 (GitHub Actions build-pytorch3d-torch251 릴리스, 인스턴스 빌드 없음) -> R2 vastpkg 경유 -> 연기 (L000·ficus) -> 통과하면 본 묶음에
P3D=""
for i in $(seq 240); do
  P3D=$(rclone lsf $R/vastpkg --include "pytorch3d-*.whl" 2>/dev/null | head -1)
  if [ -z "$P3D" ]; then
    U=$(curl -s https://api.github.com/repos/wgsong0110/anchorflow/releases/tags/pytorch3d-torch251 | python -c "import json,sys; a=json.load(sys.stdin).get('assets') or []; print(next((x['browser_download_url'] for x in a if x['name'].endswith('.whl')), ''))" 2>/dev/null)
    [ -n "$U" ] && curl -sfL -o /tmp/${U##*/} "$U" && rclone copy /tmp/${U##*/} $R/vastpkg -q && P3D=${U##*/}
  fi
  [ -n "$P3D" ] && break; sleep 60
done
if [ -n "$P3D" ]; then
  rclone copy $R/vastpkg/$P3D /root/s6/ -q; export ST_P3D_WHL=/root/s6/$P3D; say "[pytorch3d 휠] $P3D"
  if [ -z "$(rclone lsf $MK/stab_smoke_sg.done 2>/dev/null)" ]; then
    say "[Spring-Gaus 연기 띄움]"
    ST_OUT=/dev/shm/dkta/lbours/stab_smoke_sg ST_UNTIL=$(( $(date +%s) + 8 * 3600 )) ST_HARD=$(( $(date +%s) + 9 * 3600 )) ST_ARGS="--smoke 3 --par 3 --methods springgaus --scenes $SC" bash exe/stab.sh $C
    SGOK=$(python - /dev/shm/dkta/lbours/stab_smoke_sg/stab.json $SC <<'PYX'
import json, sys
R = json.load(open(sys.argv[1]))
ok = all(((R.get(f"springgaus_{t}") or {}).get("nmin") is not None and ((R.get(f"springgaus_{t}") or {}).get("time") or {}).get("ms") is not None
          and ((R.get(f"springgaus_{t}") or {}).get("vis") or {}).get("psnr") is not None) for t in sys.argv[2].split(","))
print("springgaus" if ok else "")
PYX
)
    echo "${SGOK:-실패} $(tail -1 /dev/shm/dkta/lbours/stab_smoke_sg/bl_setup.log)" | rclone rcat $MK/stab_smoke_sg.done
    say "[Spring-Gaus 연기 판정] ${SGOK:-실패} $(tail -1 /dev/shm/dkta/lbours/stab_smoke_sg/bl_setup.log)"
  else SGOK=$(rclone cat $MK/stab_smoke_sg.done | awk '{print $1}'); [ "$SGOK" = springgaus ] || SGOK=""; fi
  [ -n "$SGOK" ] && OK=${OK:+$OK,}springgaus
else say "[pytorch3d 휠 없음] 4 h 안에 릴리스·R2 에 휠 없음 -> Spring-Gaus 보류"; fi
# wolf·bread (정답 MPM·우리; 탄성 전용은 stab.py 가 해당 없음) -- 미러가 내려왔으면
for i in $(seq 120); do [ -f /root/s6/wb.done ] && break; [ -n "$(jobs -rp)" ] || break; sleep 30; done
[ -f /root/s6/wb.done ] && SC="$SC,wolf,bread" || say "[wolf·bread 보류] 미러 표식 없음"
if [ -n "$OK" ]; then
  say "[stab1 띄움] 방법 $OK 씬 $SC"
  ST_ARGS="--par 3 --methods $OK --scenes $SC" bash exe/stab.sh $C
  say "[stab1 끝] $(cat /dev/shm/dkta/lbours/stab/stab.END 2>/dev/null)"
  echo "$(date) 방법 $OK" | rclone rcat $MK/stab1.done
fi
fi
[ -n "$PA" ] && wait $PA
# ---- 끝: 전부 R2 로, 차이 0 이면 destroy
kill $SY 2>/dev/null
rclone copy /dev/shm/dkta/lbours $R/lbours -q --transfers 8
if rclone check /dev/shm/dkta/lbours $R/lbours --one-way --size-only -q 2>/root/s6/check.txt; then
  say "[R2 확인 차이 0] 모두 끝 -> destroy"; echo "$(date)" | rclone rcat $MK/ALL_DONE$SFX
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG || vastai stop instance ${CONTAINER_ID:-0} --api-key ${CONTAINER_API_KEY:-x}
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/s6/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MK/ALL_DONE_CHECK_FAIL$SFX
fi
