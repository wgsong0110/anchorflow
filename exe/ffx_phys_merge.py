#!/usr/bin/env python3
"""절 CM192: 표·그림용 지표 폴더 모음 (심볼릭 링크만, R2 파일은 안 건드림) -> <출력>/<방법>_<장면>_<판|nN>/{phys.json, vis.json}.
  - final_fix/phys: 정답 MPM, 우리 wolf·bread (CM191 판 = 새 판과 계산 같음, 소성 재질), 베이스라인 (PhysTwin·Spring-Gaus)
  - 우리 레고·pillow2sofa = final_fix/phys/ours_<장면>_<판>_nd2 (CM192 새 판 --wrx_el 1), 노이즈용 반복 = ours_<장면>_def_nd2_r2·r3 -> ours_<장면>_def_r2·r3
  - VR-GS 레고·pillow2sofa = vrgs_<장면>_n<N>it1 (반복 1 대표 설정; 옛 반복 11 폴더는 뺌)
  - 재질 확장 wolf·bread (final_pa/blx/<묶음>/g/<방법>_<장면>_n<N>_dump), i-PG·OmniPhysGS (final_fix/bl_mpm/<방법>_<장면>*)
  python3 exe/ffx_phys_merge.py <출력>"""
import glob, json, os, sys

LB = os.environ.get("LB") or os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(f"{OUT}/*"):
    if os.path.islink(f):
        os.unlink(f)                                                                  # 우리 링크만 다시 만듦
L = {}
PH = f"{LB}/final_fix/phys"
for d in sorted(glob.glob(f"{PH}/*/")):
    j = os.path.basename(d.rstrip("/"))
    if not (os.path.exists(f"{d}/phys.json") or os.path.exists(f"{d}/vis.json")):
        continue
    if j.startswith(("ours_L000_", "ours_pillow2sofa_")):
        if j.endswith("_nd2"):
            L[j[:-4]] = d
        elif "_nd2_r" in j:
            L[j.replace("_nd2_r", "_r")] = d
        continue
    if j.startswith(("ours_bread_", "ours_wolf_")) and "_nd2_r" in j:
        L.setdefault(j.replace("_nd2_r", "_r"), d); continue
    if j.startswith("ours_wolf_ss") and j.endswith(".dump_gran"):                     # 절 CM196 규칙 gran (wolf E N > 1) 이 CM193 판을 덮음
        L["_GRAN_" + j[:-5]] = d; continue
    if j.startswith(("ours_bread_", "ours_wolf_")) and ("_bs2" in j):              # 절 CM193 --be_sub 2 판이 옛 판을 덮음
        L["_BS2_" + j.replace("_bs2", "")] = d; continue
    if j.startswith("vrgs_") and j.split("_")[1] in ("L000", "pillow2sofa"):
        if j.endswith("it1"):
            L[j[:-3]] = d
        continue
    L[j] = d
for k in [k for k in L if k.startswith("_BS2_")]:
    L[k[5:]] = L.pop(k)
for k in [k for k in L if k.startswith("_GRAN_")]:
    L[k[6:]] = L.pop(k)
for d in sorted(glob.glob(f"{LB}/final_pa/blx/*/g/*_dump/")):
    if "smoke" in d or not os.path.exists(f"{d}/phys.json"):
        continue
    L[os.path.basename(d.rstrip("/"))[:-5]] = d
for d in sorted(glob.glob(f"{LB}/final_fix/bl_mpm/*/")):
    if os.path.exists(f"{d}/phys.json") or os.path.exists(f"{d}/vis.json"):
        L[os.path.basename(d.rstrip("/"))] = d
# [repro] 대표 판이 하위 폴더에 있는 i-PG·Omni 칸 (예: i-PG wolf = ipg_wolf/post_N25e5_fr) -> repro/data/pareto_mpmbl_sources.json (예전엔 phys·vis 표에 손으로 채움)
_SP = json.load(open(os.environ.get("PARETO_MPMBL_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repro", "data", "pareto_mpmbl_sources.json")))
for _k, _v in _SP.items():
    if isinstance(_v, dict) and _v.get("phys") and os.path.exists(f"{LB}/{_v['phys']}/phys.json"):
        L[_k.replace(":", "_")] = f"{LB}/{_v['phys']}"
for k, d in L.items():
    d = d.rstrip("/")
    if os.path.exists(f"{d}/vis_ffx.json"):                                          # 재질 확장 (final_blx) 폴더: vis.json = 정답 대비 (stab_vis), ffx_vis 결과는 vis_ffx.json -> 파일 단위 링크
        os.makedirs(f"{OUT}/{k}", exist_ok=True)
        for a_, b_ in (("phys.json", "phys.json"), ("vis_ffx.json", "vis.json")):
            t = f"{OUT}/{k}/{b_}"
            if os.path.lexists(t):
                os.unlink(t)
            if os.path.exists(f"{d}/{a_}"):
                os.symlink(f"{d}/{a_}", t)
        continue
    os.symlink(d, f"{OUT}/{k}")
json.dump(L, open(f"{OUT}.map.json", "w"), ensure_ascii=False, indent=1)
print(len(L), "개 링크 ->", OUT)
