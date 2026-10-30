#!/bin/bash
# 절 AG 씬 정의 (source 해서 쓴다): 씬 이름 -> 정답 .pt·정답 config·렌더 형상·씬 인자 (물리 조건은 정답 config 그대로; 출처는 doc/fem_realtime.md 절 AG.1)
#   hotdog : PG 점소성 foam (wmats/hotdog_viscoplastic.json, topo/pg_hotdog_visco 180 프레임) -- 바닥 sticky z 0.48, g -9.8, 격자 100 (dx 0.02)
#   wolf   : PG 공식 wolf_config.json (모래, topo/pg_wolf_sand 50 프레임) -- 바닥 sticky z 0.48, release_particles_sequentially, 격자 200 (dx 0.01)
#   bread  : PG tear_bread 에서 재질만 plasticine 항복 200 (topo/bread_plastic_config.json, topo/pg_bread_plastic 300 프레임) -- g 0, 입자 속도 구속 둘, 감쇠 0.9995/1e-4 s, 격자 150
#   wm     : GF 공식 watermelon (wm2/gf/config.json, wm2/gf 100 프레임, NACC) -- 바닥 = bounding_box padding 3 칸 (z = 3 dx = 0.02, 한쪽 방향), 격자 300 (dx 2/300)
# node_h = 정답 MPM dx·√2 (lego·ficus 와 같은 규칙: lego dx 0.02 -> 0.028284, ficus dx 0.04 -> 0.056569)
W=/home/dkta/work; AGD=${AGD:-/tmp/dkta/ag_data}
# 공통 서브스텝 간격 (사용자 10-30): h_s = 6.25e-5 s 고정, N = round(frame_dt / h_s) (lego 267·ficus 640 과 같은 h_s) -- 씬마다 N 을 찾지 않는다
HS=6.25e-5
REC="--mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5"      # 절 AD 확정 플래그 그대로
ag_scene() {
  case $1 in
    hotdog) SHAPE=hotdog; FR=180; NC=267; GTH5=$W/topo/pg_hotdog_visco; CFG=$W/wmats/hotdog_viscoplastic.json
            SARGS="--data $AGD/hotdog.pt --tag hotdog --frames 180 --node_h 0.028284 --floor vel --plast" ;;
    wolf)   SHAPE=wolf; FR=50; NC=640; GTH5=$W/topo/pg_wolf_sand; CFG=$W/PhysGaussian/config/wolf_config.json
            SARGS="--data $AGD/wolf.pt --tag wolf --frames 50 --node_h 0.014142 --floor vel --pbc --plast" ;;
    bread)  SHAPE=bread; FR=300; NC=160; GTH5=$W/topo/pg_bread_plastic; CFG=$W/topo/bread_plastic_config.json
            SARGS="--data $AGD/bread.pt --tag bread --frames 300 --node_h 0.018856 --floor none --pbc --pg_vdamp --plast" ;;
    wm)     SHAPE=watermelon; FR=100; NC=480; GTH5=$W/wm2/gf/simulation_ply; CFG=$W/wm2/gf/config.json
            SARGS="--data $AGD/wm.pt --tag wm --frames 100 --node_h 0.0094281 --floor vel --vfl_mode slip --floor_plane 1,1,0.0200000,0,0,1 --plast" ;;
    *) echo "[오류] 씬 $1"; return 1 ;;
  esac
}
