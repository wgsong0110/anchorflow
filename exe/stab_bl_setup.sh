#!/bin/bash
# 안정 최소 서브스텝 표의 공개 코드 베이스라인 (PhysTwin · Simplicits · Spring-Gaus, doc/pareto.md 절 S.3) 환경 확인·설치 -- **잡 안 컨테이너에서만** (exe/stab.sh 가 부름).
# 공유 환경 (af · simp) 은 건드리지 않는다: 모자란 패키지는 /dev/shm/dkta/pylib/<이름> 에 pip --target (--no-deps) 으로 넣고 그 실행기만 PYTHONPATH 로 앞에 둠.
# 공식 코드는 고정 커밋 (PhysTwin 81c7187 · Spring-Gaus 62a1bb5 · kaolin v0.18.0) -- 공유 사본이 다른 커밋이면 /dev/shm/dkta/ext/ 에 따로 받음.
# 출력: $ENVF (source 하면 ST_PT_REPO · ST_PT_PYPATH · ST_SG_REPO · ST_SG_PYPATH · ST_SIMP_PY · ST_SIMP_PYPATH), 끝 줄 BLSETUP rc=<0|1> (실패 항목 목록).
set -u
W=/home/dkta/work
AF=/home/dkta/.conda/envs/af/bin/python
SIMP=${ST_SIMP_PY:-/home/dkta/.conda/envs/simp/bin/python}
LIB=/dev/shm/dkta/pylib; EXT=/dev/shm/dkta/ext; mkdir -p $LIB $EXT
ENVF=${1:-$LIB/bl_env.sh}
BAD=""
say() { echo "[bl_setup $(date '+%T')] $*"; }

repo() {   # repo <이름> <공유 경로> <원격> <커밋> -> 쓸 경로 (stdout)
  local n=$1 p=$2 u=$3 c=$4
  if [ -d $p/.git ] && [ "$(git -C $p rev-parse --short=7 HEAD 2>/dev/null)" = "$c" ]; then echo $p; return; fi
  local q=$EXT/$n
  if [ ! -d $q/.git ]; then git clone -q $u $q >&2 || { echo FAIL; return; }; fi
  git -C $q checkout -q $c >&2 && git -C $q submodule update -q --init --recursive >&2; echo $q
}
need() {   # need <python> <target 이름> <import 이름> <pip 이름...>: import 가 안 되면 target 에 --no-deps 설치
  local py=$1 tg=$2 mod=$3; shift 3
  if PYTHONPATH=$LIB/$tg:${PYTHONPATH:-} $py -c "import $mod" 2>/dev/null; then say "있음 $tg: $mod"; return 0; fi
  say "설치 $tg: $*"; mkdir -p $LIB/$tg
  $py -m pip install -q --no-deps --target $LIB/$tg "$@" || { BAD="$BAD $tg:$mod"; return 1; }
  PYTHONPATH=$LIB/$tg:${PYTHONPATH:-} $py -c "import $mod" || { BAD="$BAD $tg:$mod"; return 1; }
}

# ---- PhysTwin (af 환경 + warp 1.x · cma · kornia; 렌더·대화형 모듈은 실행기가 빈 모듈로 막음)
PT=$(repo PhysTwin $W/PhysTwin https://github.com/Jianghanxiao/PhysTwin.git 81c7187); [ "$PT" = FAIL ] && BAD="$BAD PhysTwin_repo"
wv=$($AF -c "import warp; print(warp.config.version)" 2>/dev/null || echo 0)
if [ "${wv%%.*}" -lt 1 ] 2>/dev/null; then say "af warp $wv < 1 -> pt 에 warp-lang 1.x"; need $AF pt "warp; assert int(warp.config.version.split('.')[0]) >= 1" "warp-lang>=1.5,<1.10"; fi
need $AF pt cma cma
need $AF pt kornia kornia kornia_rs
need $AF pt termcolor termcolor
need $AF pt open3d open3d
need $AF pt trimesh trimesh
need $AF pt sklearn scikit-learn
need $AF pt cv2 opencv-python-headless
need $AF pt yaml pyyaml
PTP=$LIB/pt
PYTHONPATH=$PTP $AF -c "import sys; sys.path.insert(0, '$PT'); import warp; warp.init(); print('[bl_setup] PhysTwin warp', warp.config.version)" || BAD="$BAD PhysTwin_warp"

# ---- Spring-Gaus (af 환경: 예전 벤치와 같음 -- pytorch3d · 그쪽 서브모듈 래스터라이저 · simple_knn)
SGR=$(repo Spring-Gaus $W/Spring-Gaus https://github.com/Colmar-zlicheng/Spring-Gaus.git 62a1bb5); [ "$SGR" = FAIL ] && BAD="$BAD SG_repo"
for m in yacs termcolor plyfile trimesh imageio open3d; do need $AF sg $m $m || true; done
if ! PYTHONPATH=$LIB/sg $AF -c "import pytorch3d.ops" 2>/dev/null; then
  say "빌드 sg: pytorch3d (잡 안 빌드)"; $AF -m pip install -q --no-deps --no-build-isolation --target $LIB/sg "git+https://github.com/facebookresearch/pytorch3d.git@stable" || BAD="$BAD SG_pytorch3d"
fi
for sm in diff-gaussian-rasterization simple-knn; do
  mod=$([ $sm = simple-knn ] && echo simple_knn._C || echo diff_gaussian_rasterization)
  if ! PYTHONPATH=$LIB/sg $AF -c "import $mod" 2>/dev/null; then
    say "빌드 sg: $sm (Spring-Gaus 서브모듈)"; $AF -m pip install -q --no-deps --no-build-isolation --target $LIB/sg $SGR/submodules/$sm || BAD="$BAD SG_$sm"
  fi
done
PYTHONPATH=$LIB/sg $AF -c "import sys; sys.path.insert(0, '$SGR'); import lib.models; from lib.models.gaus import render; print('[bl_setup] Spring-Gaus 모듈 적재')" || BAD="$BAD SG_import"

# ---- Simplicits (simp 환경의 kaolin 이 0.18.0 이 아니면 kaolin 0.18.0 휠을 simp 의 torch·CUDA 에 맞춰)
SMP=""
if [ -x $SIMP ]; then
  kv=$($SIMP -c "import kaolin; print(kaolin.__version__)" 2>/dev/null || echo none)
  if [ "$kv" != "0.18.0" ]; then
    tv=$($SIMP -c "import torch; print(torch.__version__.split('+')[0])"); cu=$($SIMP -c "import torch; print(torch.version.cuda.replace('.', ''))")
    say "simp kaolin $kv -> kaolin 0.18.0 (torch $tv cu$cu) 를 simp18 에"
    mkdir -p $LIB/simp18
    $SIMP -m pip install -q --no-deps --target $LIB/simp18 kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${tv}_cu${cu}.html || BAD="$BAD simp_kaolin"
    PYTHONPATH=$LIB/simp18 $SIMP -c "import warp; assert int(warp.config.version.split('.')[0]) >= 1" 2>/dev/null || need $SIMP simp18 warp "warp-lang>=1.5,<1.10"
    SMP=$LIB/simp18
  fi
  PYTHONPATH=$SMP $SIMP -c "import kaolin; from kaolin.physics.simplicits import SimplicitsObject, SimplicitsScene; print('[bl_setup] kaolin', kaolin.__version__)" || BAD="$BAD simp_import"
else
  BAD="$BAD simp_env_missing($SIMP)"
fi

cat > $ENVF <<EOF
export ST_PT_REPO=$PT ST_PT_PYPATH=$PTP ST_SG_REPO=$SGR ST_SG_PYPATH=$LIB/sg ST_SIMP_PY=$SIMP ST_SIMP_PYPATH=$SMP
EOF
say "환경 파일 $ENVF: $(cat $ENVF)"
if [ -n "$BAD" ]; then echo "BLSETUP rc=1 실패:$BAD"; exit 1; fi
echo "BLSETUP rc=0"
