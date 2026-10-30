#!/bin/bash
# 절 AD 6 차: 단독 측정 (GPU0, gpu0.lock flock -- exe/fem_c_run.sh EX_TIMED=1) + 최종 F 덤프 3DGS 렌더 (렌더 카드 3).
# 5 차 최고: 탄성 부피 상한 min(V_T, θ V_기하) (--sp vcap) + 클램프 5 + 절 AC 빠른 경로 (--cfuse --cfuse_mode 2). 두 형상 같은 코드·같은 θ.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"; df -h /home | tail -1
OL=$W/lbours/fem_ad_lego; OF=$W/lbours/fem_ad_ficus
LB="--mode lag --opt --mass tet --floor vel --rstat --lim_sr 5 --cfuse --cfuse_mode 2"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --lim_sr 5 --cfuse --cfuse_mode 2"
T="env EX_TIMED=1 bash exe/fem_c_run.sh"
for r in 1 2 3; do
  EX_ROOT=fem_ad_lego $T t_l_vc05_n256_r$r 0 $LB --no_dump --sp vcap --sp_vcap 0.5 --N 256 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_ad_lego $T t_l_vc1_n256_r$r 0 $LB --no_dump --sp vcap --sp_vcap 1 --N 256 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_ad_lego $T t_l_vc1_n224_r$r 0 $LB --no_dump --sp vcap --sp_vcap 1 --N 224 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_ad_lego $T t_l_vc05_n192_r$r 0 $LB --no_dump --sp vcap --sp_vcap 0.5 --N 192 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_ad_lego $T t_l_ref_n256_r$r 0 $LB --no_dump --N 256 < /dev/null > /dev/null 2>&1
done
for r in 1 2; do
  EX_ROOT=fem_ad_ficus $T t_f_vc05_n640_r$r 0 $FB --sp vcap --sp_vcap 0.5 --N 640 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_ad_ficus $T t_f_vc1_n640_r$r 0 $FB --sp vcap --sp_vcap 1 --N 640 < /dev/null > /dev/null 2>&1
done
python exe/fem_p_period.py --data $FD --pred $OF/t_f_*/pred.pt > $OF/period6t.txt 2>&1
python exe/fem_ad_table.py $OL/t_l_* > $OL/table6t.md; python exe/fem_ad_table.py $OF/t_f_* > $OF/table6t.md
cat $OL/table6t.md $OF/table6t.md
for d in $OL/t_l_* $OF/t_f_*; do echo "$(basename $d) 시작 $(tr '\n' ' ' < $d/gpu_at_start.txt) 끝 $(tr '\n' ' ' < $d/gpu_at_end.txt)"; done > $OL/gpu6t.txt
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_ad_lego -q; rclone copy $OF r2:storage/result/anchorflow/lbours/fem_ad_ficus -q
echo "[단독 측정 끝] $(date +%T)"
fin() {  # fin <형상> <태그> <인자...>
  S=$1; TG=$2; shift 2
  for a in 1 2 3; do
    SHAPE=$S EX_ROOT=fem_ad_$S RG=3 bash exe/fem_ad_final.sh ${TG}_F$a "$@" < /dev/null > /dev/null 2>&1
    [ -f $W/lbours/fem_ad_$S/${TG}_F$a/vj/RENDER_DONE ] && { echo "[최종] $S $TG: 시도 $a 에서 완주·렌더"; return; }
    echo "[최종] $S $TG: 시도 $a 발산/실패 $(cat $W/lbours/fem_ad_$S/${TG}_F$a/err_fem.txt 2>/dev/null | tail -1 | cut -c1-120)"
  done
}
fin lego f_l_vc05_n256 $LB --dump_f32 --sp vcap --sp_vcap 0.5 --N 256
fin ficus f_f_vc05_n640 $FB --dump_f32 --sp vcap --sp_vcap 0.5 --N 640
python exe/fem_p_period.py --data $FD --pred $OF/f_f_*/pred.pt > $OF/period6f.txt 2>&1
python exe/fem_ad_table.py $OL/f_l_* > $OL/table6f.md; python exe/fem_ad_table.py $OF/f_f_* > $OF/table6f.md
cat $OL/table6f.md $OF/table6f.md $OF/period6f.txt
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_ad_lego -q; rclone copy $OF r2:storage/result/anchorflow/lbours/fem_ad_ficus -q
df -h /home | tail -1
echo AD6_DONE $(date +%T)
