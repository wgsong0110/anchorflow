"""절 CM76 할 일 1: 접촉이 퍼질 때 걸음 시작·반복 0·1 에서 늘어나는 시간을 커널 무리별로 프레임 축에 (커널 프로파일 판 s7trace.json.gz + 진단 s9.jsonl).
사용: python exe/speed9_fa.py <출력 접두> <판 디렉토리...>  -> <접두>.png · <접두>.md (stdout 도 md)
프레임 = s7_fm 표식 사이 (프로파일 프레임 3~), 보통 프레임 (무리 (1,1,3)) 만 그림 (X3c·쪼개기 프레임은 빈 칸). s9.jsonl 의 프레임 f 와는 순서로 맞춤 (프로파일 첫 프레임 = f 3).
"""
import collections
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import speed7_kp as K  # noqa: E402

# 커널 무리 (단계 앞·시작·반복 0·반복 1 만; 반복 2 (판정만)·끝은 따로 합)
GRP = [
    ("시작점 장벽 판정 (y0e·yl·kfe·bar_q 시작)", {"rt_bar_y0e", "yl_kq", "yl_chk", "yl_mix", "yl_merge", "yl_fb", "yl_st", "yl4_chk", "rt_y0epick", "rt_ga_y0", "s9_y0b", "s9_rst"}),
    ("W̄ 모으기 (e3_*)", {"e3_wq", "e3_wqn", "e3_wqz"}),
    ("P2G·칸 (rt_p2g·sp_q2n·...)", {"rt_p2g", "sp_q2n", "rt_gauss", "aq_p2g_s", "gq_p2q", "wb_p2q", "sp_csum", "rt_coord", "rt_mark", "aq_gauss_s", "sp_scan_a", "sp_scan_b", "sp_scan_c"}),
    ("선탐색 장벽 판정 (ws_bk·ws_cand·kfe)", {"ws_bk", "ws_cand", "ws_nk", "ws_merge", "ws4_chk", "ws_init", "ws_fb", "ws_sc", "ws_lsup", "rt_ytry", "rt_bar_kfe", "wb_kmin"}),
    ("반전 막기 (ws_inv_q·aw_*)", {"ws_inv_q", "aw_inv_q", "aw_pick", "aw_y0z_a", "aw_y0z_b", "aw_tdir", "ao_tvec"}),
    ("PCG·MG (장벽 몫 rt_bar_mv 포함)", {"ao_mv_q", "gq_mv", "rt_ap_c", "rt_cg1_c", "rt_cg2_c", "rt_bar_mv", "rt_mg_dbar", "rt_bar_mvl", "rt_sc_roll", "rt_dotz", "rt_pinit", "rt_cg1m"}),
    ("요소·장벽 기울기 (aq_elem·gq_force·rt_bar_q)", {"aq_elem", "gq_force", "rt_bar_q", "rt_bar_n", "rt_grad_c"}),
    ("선탐색 에너지 (en_phi_*)", {"en_phi_q", "en_phi_n", "gq_phi", "en_bt_chk", "en_bt_mix", "en_bt_sc", "en_bt_fin", "en_ph01", "en_mo_set", "en_cut"}),
]
MG_BAR = {"rt_bar_mv", "rt_mg_dbar", "rt_bar_mvl"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PH = ("앞", "시작", "반복0", "반복1")


def frames(d):
    ev = json.load(gzip.open(os.path.join(d, "s7trace.json.gz")))["traceEvents"]
    k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    k.sort(key=lambda e: e["ts"])
    sid = collections.Counter(e["args"].get("stream") for e in k if e["name"] in K.ELEM).most_common(1)[0][0]
    fm = [e for e in k if e["name"] == "s7_fm"]
    fr = [[fm[i]["ts"], fm[i + 1]["ts"], []] for i in range(len(fm) - 1)]
    j = 0
    for e in k:
        if e["name"] == "s7_fm" or e["args"].get("stream") != sid:
            continue
        while j + 1 < len(fr) and e["ts"] >= fr[j + 1][0]:
            j += 1
        if fr and fr[j][0] <= e["ts"] < fr[j][1]:
            fr[j][2].append(e)
    out = []
    for i, (t0, t1, E) in enumerate(fr):
        ph, it, nst, nfl, nel = "앞", -1, 0, 0, 0
        A = collections.defaultdict(float)
        for e in E:
            n = K.short(e["name"])
            if n in K.STEP0:
                ph, it = "시작", -1
                nst += 1
            if n == "ao_floor_n":
                nfl += 1
            if n in K.ELEM:
                it += 1
                nel += 1
                ph = f"반복{it}"
            if n in K.STEPEND:
                ph = "끝"
            A[(ph, n)] += e["dur"] / 1000.0
        out.append(dict(i=i, f=3 + i, sig=(nst, nfl, nel), A=A))
    return out


def grp_ms(A):
    g = [0.0] * (len(GRP) + 1)
    mg = 0.0
    for (p, n), v in A.items():
        if p not in PH:
            continue
        for gi, (_, S) in enumerate(GRP):
            if n in S:
                g[gi] += v
                break
        else:
            g[-1] += v
        if n in MG_BAR:
            mg += v
    return g, mg


def main():
    pre, dirs = sys.argv[1], sys.argv[2:]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fmg
    for f in fmg.findSystemFonts():
        if "NotoSansCJK" in f or "Noto Sans CJK" in f:
            fmg.fontManager.addfont(f)
    plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(3, len(dirs), figsize=(7.5 * len(dirs), 10), sharex="col", squeeze=False,
                            gridspec_kw=dict(height_ratios=[3, 1.3, 1.3]))
    md = ["# 절 CM76 할 일 1 -- 프레임 축 커널 무리 시간 (앞 + 걸음 시작 + 반복 0·1, 보통 프레임)\n"]
    for ci, d in enumerate(dirs):
        F = [q for q in frames(d) if q["sig"] == (1, 1, 3)]
        S9 = {q["f"]: q for q in (json.loads(l) for l in open(os.path.join(d, "s9.jsonl"))) if "bar" in q}
        x = [q["f"] for q in F]
        G = [grp_ms(q["A"]) for q in F]
        ax = axs[0][ci]
        for gi, (nm, _) in enumerate(GRP):
            ax.plot(x, [g[0][gi] for g in G], color=COL[gi], lw=1.6, label=nm)
        ax.set_title(f"{os.path.basename(d.rstrip('/'))}: 커널 무리 GPU ms / 프레임 (앞·시작·반복 0·1)", fontsize=11, loc="left")
        ax.set_ylabel("ms / 프레임")
        ax.grid(color="#e6e5e0", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if ci == 0:
            ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
        tot = [sum(g[0]) for g in G]
        ax2 = axs[1][ci]
        ax2.plot(x, tot, color="#2a78d6", lw=1.6)
        ax2.set_ylabel("합 ms")
        ax2.set_title("앞·시작·반복 0·1 커널 합", fontsize=10, loc="left")
        ax2.grid(color="#e6e5e0", lw=0.6)
        ax2.spines[["top", "right"]].set_visible(False)
        ax3 = axs[2][ci]
        xs = sorted(S9)
        ax3.plot(xs, [S9[f]["bar"]["band"] / 1e3 for f in xs], color="#eb6834", lw=1.6, label="띠 안 표본점 (천)")
        ax3.plot(xs, [S9[f]["bar"]["bblk"] / 100.0 for f in xs], color="#1baf7a", lw=1.6, label="띠 블록 (백)")
        ax3.plot(xs, [S9[f]["ip"]["nn"] / 1e4 for f in xs], color="#6250d6", lw=1.6, label="노드 (만)")
        ax3.set_title("진단 셈 (걸음 시작, y0e 부름 합)", fontsize=10, loc="left")
        ax3.set_xlabel("프레임")
        ax3.grid(color="#e6e5e0", lw=0.6)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.legend(fontsize=8, frameon=False, loc="upper left")
        # 표
        def win(a, b):
            W = [(q, g) for q, g in zip(F, G) if a <= q["f"] <= b]
            return W
        A_, B_ = win(3, 12), win(len(x) and max(x) - 19, 999)
        md.append(f"\n## {d}\n\n보통 프레임 {len(F)} 개; 앞 = 프레임 {A_[0][0]['f'] if A_ else '-'}~{A_[-1][0]['f'] if A_ else '-'} 평균, 뒤 = 프레임 {B_[0][0]['f'] if B_ else '-'}~{B_[-1][0]['f'] if B_ else '-'} 평균 (ms / 프레임)\n")
        md.append("| 커널 무리 | 앞 | 뒤 | 는 몫 |\n|---|---|---|---|")
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        for gi, (nm, _) in enumerate(GRP + [("나머지 (상자 비우기 rt_zero_*·제어 등)", None)]):
            a = mean([g[0][gi] for _, g in A_]); b = mean([g[0][gi] for _, g in B_])
            md.append(f"| {nm} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
        a = mean([sum(g[0]) for _, g in A_]); b = mean([sum(g[0]) for _, g in B_])
        md.append(f"| **합** | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
        a = mean([g[1] for _, g in A_]); b = mean([g[1] for _, g in B_])
        md.append(f"| (그중 MG·PCG 장벽 몫 rt_bar_mv·rt_mg_dbar·rt_bar_mvl) | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
        # 커널 하나씩 는 몫 상위
        def kavg(W):
            D = collections.defaultdict(float)
            for q, _ in W:
                for (p, n), v in q["A"].items():
                    if p in PH:
                        D[f"{p}|{n}"] += v / max(len(W), 1)
            return D
        KA, KB = kavg(A_), kavg(B_)
        dd = sorted(((KB.get(k, 0) - KA.get(k, 0), k) for k in set(KA) | set(KB)), reverse=True)[:12]
        md.append("\n| 단계 · 커널 (는 몫 상위) | 앞 | 뒤 | 는 몫 |\n|---|---|---|---|")
        for v, k in dd:
            md.append(f"| {k.replace('|', ' · ')} | {KA.get(k, 0):.3f} | {KB.get(k, 0):.3f} | {v:+.3f} |")
        sa = [S9[f] for f in S9 if 3 <= f <= 12]; sb = [S9[f] for f in S9 if f >= max(S9) - 19]
        m2 = lambda L_, fn: mean([fn(q) for q in L_])
        md.append(f"\n- 진단 (같은 창 평균): 활성 표본점 {m2(sa, lambda q: q['bar']['act']):.0f} -> {m2(sb, lambda q: q['bar']['act']):.0f} · 띠 안 {m2(sa, lambda q: q['bar']['band']):.0f} -> {m2(sb, lambda q: q['bar']['band']):.0f} · 띠 블록 {m2(sa, lambda q: q['bar']['bblk']):.0f} -> {m2(sb, lambda q: q['bar']['bblk']):.0f} (전체 블록 {m2(sb, lambda q: q['ip']['nq'] / 128):.0f}) · 노드 {m2(sa, lambda q: q['ip']['nn']):.0f} -> {m2(sb, lambda q: q['ip']['nn']):.0f} · 표본점 {m2(sa, lambda q: q['ip']['nq']):.0f} -> {m2(sb, lambda q: q['ip']['nq']):.0f}")
    fig.tight_layout()
    fig.savefig(pre + ".png", dpi=110)
    open(pre + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
