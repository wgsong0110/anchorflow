#!/usr/bin/env python3
"""절 CM182: 우리 기본 판 (육면체 B-스플라인 격자 2차 + 가우스점 2x2x2, K=1 HHT, 공통 기본 H9 명령) 을 GF 장면에 맞춘 명령줄.
기준 = R2 lbours/wolfcv/cm157/BA_1/cmd.txt (bread). **풀이에 걸린 플래그는 하나도 빼지 않는다** (진단 기록 감싸개 startpt 만 뺌: 아래) (--tc, --s4*, --s8g 는 렌더 판 (--rr) 위에서만 설치되는 감싸개 사슬이고 --s8g 는 --s4x 2 위: 풀이에 걸린 것이 있을 수 있어 사슬째 둔다).
바꾸는 것: 데이터 (--data), 프레임 수 (--frames, AF_SP_FR), 이름 (--tag), 출력 (--out), 바닥 (--floor none -> LA_1 의 --floor vel --fl_bar --fl_bst 1 --y0eb --hgt --bar_cnt, 바닥 있는 장면만),
렌더 대상 (--rr pg -> --rr gf: 같은 렌더 훅 (풀이 상태를 읽기만 함) 의 GF 갈래, 모델·궤적·설정을 이 장면의 것으로), 덤프 (--dump --dump_F --dump_part, 영상·지표용).
  gfs_cmd.py BA_1_cmd.txt DATA FRAMES TAG OUT FLOOR(0|1) RR_TRAJ RR_CFG RR_MODEL RR_GF [EXTRA ...]"""
import re, sys
src, data, fr, tag, out, floor, rtraj, rcfg, rmodel, rgf = sys.argv[1:11]; extra = sys.argv[11:]
c = " ".join(open(src).read().split())
# startpt 감싸개 (--sp 1, AF_SP_FR) = 걸음 시작 단계·뉴턴 갱신별 칸 r 기록 (진단, 결과 불변: 러너 로그 [절 CM113] 에 명시) -- 정답 격자 노드 수 x 용량 만큼 GPU 통 (acc) 을 잡아 입자 100 만 개 넘는 장면에서 메모리 폭발 (44 GiB 요청) -> 이 진단 감싸개만 뺀다
c = re.sub(r"AF_SP_FR=\S+\s+", "", c); assert "exe/startpt_wrap.py --sp 1 " in c; c = c.replace("exe/startpt_wrap.py --sp 1 ", "")
assert "--rr pg" in c
c = c.replace("--rr pg", "--rr gf")
c = re.sub(r"--rr_shape \S+", f"--rr_shape {tag}", c); c = re.sub(r"--rr_traj \S+", f"--rr_traj {rtraj}", c); c = re.sub(r"--rr_cfg \S+", f"--rr_cfg {rcfg}", c)
c = re.sub(r"(--rr_fast \S+)", rf"\1 --rr_model {rmodel} --rr_gf {rgf}", c, count=1)
c = re.sub(r"--data \S+", f"--data {data}", c); c = re.sub(r"--frames \d+", f"--frames {fr}", c); c = re.sub(r"--tag \S+", f"--tag {tag}", c)
pre = __import__("os").environ.get("GFS_PREFIX", "").strip()
if pre:      # 사용자 11-15: cut·particle_impulse 경계 감싸개를 사슬 맨 앞에 (exe/cutbc_wrap.py --cut_bc 1 [exe/impulse_wrap.py --impulse_bc 1]); 기본값 0 = 안 붙임
    k = c.index("exe/nlwhy_wrap.py"); c = c[:k] + pre + " " + c[k:]
_FIX = __import__("os").environ.get("GFS_FIX", "1") == "1"
if _FIX:      # 절 CM184 (기본 켬): 바닥 접촉 뒤집힘·발산 수정. (2) 뉴턴 시작점 섞기·보폭에 가우스 점 뒤집힘 불가능 (exe/hexff_wrap.py --hff 1, lib/anchorflow/fem_hexff.py). GFS_FIX=0 이면 옛 판 그대로.
    k = c.index("exe/nlwhy_wrap.py"); c = c[:k] + "exe/hexff_wrap.py --hff 1 " + c[k:]
if floor == "1":
    assert "--floor none" in c; c = c.replace("--floor none", "--floor vel --fl_bar --fl_bst 1 --y0eb --hgt --bar_cnt")
    if _FIX:  # 절 CM184 (1): 장벽 바닥 장면은 거의 빈 노드를 관성 예측 Ũ 에 고정하지 않음 (--mlow 0; wolf 장벽 바닥 판이 이미 쓰는 --mlow 1e-20 과 같은 규칙, 바닥 없는 bread 는 1e-3 그대로)
        assert c.count("--mlow 1e-3") == 1; c = c.replace("--mlow 1e-3", "--mlow 0")
import re as _re
_km = [int(m) for m in _re.findall(r"--K=(\d+)", " ".join(extra))]
if _km and _km[-1] > 1:      # 사용자 11-15: 프레임당 걸음 수 K > 1. --ksplit (쪼개기) 은 K 1 에서만 되므로 K > 1 판은 --ksplit·--ks_big 를 뺀다 (모든 장면 같은 규칙)
    c = c.replace(" --ksplit 3", "").replace(" --ks_big", "")
_drop = __import__("os").environ.get("GFS_DROP", "").split()      # 절 CM184 진단: 기본 명령에서 플래그 빼기 (값 있으면 값까지); 기본 빈 값 = 그대로
for _f in _drop:
    c = re.sub(r"(^| )" + re.escape(_f) + r"( (?!--)\S+)?(?= |$)", " ", c)
c = re.sub(r"--out \S+$", "", c).strip() + " --dump --dump_F --dump_part " + " ".join(extra) + f" --out {out}"
print(c)
