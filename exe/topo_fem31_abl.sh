# 원인 가르기 (11-01 조정자 지시): wolf·bread 우리 표현, 앞 30 프레임. 바뀐 것 셋을 하나씩 되돌린다
#  N   = 새 판 그대로 (fem, 씬 node_h, 시뮬 원점, 입자 det 장벽 + 가능성 검사)
#  Ah  = node_h 만 기존 0.028284 (--mpm_dx 0.02)
#  Ao  = 격자 원점만 기존 정규화 원점 (--lat_origin norm)
#  AJ  = 입자 det 장벽·가능성 검사만 끔
#  OLD = 셋 다 되돌림 (= 기존 tet)
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; C=f09c9ad; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/topo_$C; [ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add --detach $WT $C >/dev/null 2>&1)
T=$W/topo; O=$T/fem31/abl; mkdir -p $O; cd $WT; git log -1 --oneline
R2=r2:storage/result/anchorflow/topo/fem31/abl
dxof() { case $1 in wolf) echo 0.01;; bread) echo 0.013333333333333334;; esac; }
COM="--opt riem --kappa 0 --tb none --save_traj --riem_energy logbarrier --frames 30 --ckpt_every 5"
run() { S=$1; V=$2; G=$3; F="--riem_partF --feas"
  case $V in N) X="--method fem --mpm_dx $(dxof $S) $F";; Ah) X="--method fem --mpm_dx 0.02 $F";;
    Ao) X="--method fem --mpm_dx $(dxof $S) --lat_origin norm $F";; AJ) X="--method fem --mpm_dx $(dxof $S)";; OLD) X="--method tet";; esac
  [ -f $O/${S}_$V.npz ] && { echo "$S $V: 이미 있음"; return; }
  RES=""; [ -f $O/${S}_$V.npz.ckpt.pt ] && RES=--resume
  echo "$(date +%H:%M) [명령 GPU $G] exe/rep_track2.py --flow $T/flow_${S}_v2.npz --aux $T/aux_${S}_v2.npz $COM $X $RES --out $O/${S}_$V.npz"
  CUDA_VISIBLE_DEVICES=$G python -u exe/rep_track2.py --flow $T/flow_${S}_v2.npz --aux $T/aux_${S}_v2.npz $COM $X $RES --out $O/${S}_$V.npz >> $O/${S}_$V.log 2>&1
  echo "$(date +%H:%M) $S $V: $(grep -a '요약\|rror' $O/${S}_$V.log | tail -1 | cut -c1-230)"; }
G4=${G4:-4}; G5=${G5:-5}; G6=${G6:-6}
run wolf N $G4 & run wolf Ah $G4 & run wolf Ao $G5 & run wolf AJ $G5 & run wolf OLD $G6 &
run bread N $G6 & run bread Ah $G4 & run bread Ao $G5 & run bread AJ $G6 & run bread OLD $G6 &
wait
~/bin/rclone copy $O $R2 --transfers 8 2>&1 | tail -1
echo FEM31_ABL_DONE
