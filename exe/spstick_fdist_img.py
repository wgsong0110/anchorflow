"""spstick F 분포·셀 야코비안 영상에 같은 프레임의 3DGS RGB 렌더(정답 | 출력만 최적화 | 다섯 궤적 모델)를 위에 붙인다.
렌더 프레임 0 = 초기 상태(sim_0), 렌더 프레임 i+1 = 덤프 pred[i] (lb_roll_to_h5). 그래프 f{t} = 덤프 t = 렌더 t+1.
기존 렌더 재사용: lbours/spstick/{gt,ov,m_multi}_T.mp4 (색 안 입힌 RGB)."""
import sys, os, numpy as np, imageio.v2 as iio, torch
from PIL import Image, ImageDraw, ImageFont
W = "/home/dkta/work"; U = f"{W}/lbours/spstick"; O = f"{W}/lbours/spstick_Fdist"; V = sys.argv[1] if len(sys.argv) > 1 else ""
font = None
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")]:
    if os.path.exists(p): font = ImageFont.truetype(p, 28); break
if font is None:
    from matplotlib import font_manager as fm
    for f in fm.findSystemFonts():
        if "Noto" in f and ("CJK" in f or "KR" in f): font = ImageFont.truetype(f, 28); break
LAB = ["정답 (PG)", "새 바닥 출력만 최적화", "새 바닥 다섯 궤적 모델"]
for T in ("L000", "0004"):
    # 위치 대응 검산: 렌더한 덤프 vs 그래프에 쓴 덤프
    a = torch.load(f"{U}/roll_m_multi_{T}.pt", map_location="cpu", weights_only=False, mmap=True)["pred"]
    b = torch.load(f"{W}/lbours/lego_psi_spstick/roll_mm_{T}.pt", map_location="cpu", weights_only=False, mmap=True)["pred"]
    print(f"[검산 {T}] 모델 렌더 덤프(half F) vs 그래프 덤프(F32) 위치 차 최대 {float((a.float()-b.float()).abs().max()):.3e}", flush=True)
    R = []
    for n in ("gt", "ov", "m_multi"):
        fr = iio.mimread(f"{U}/{n}_{T}.mp4", memtest=False); R.append(fr)
        print(f"[렌더] {n}_{T}.mp4 {len(fr)} 프레임 {fr[0].shape}", flush=True)
    assert all(len(r) == 81 for r in R), [len(r) for r in R]
    for kind, sub in (("Fdist", f"F_{T}"), ("Jcell", f"J_{T}")):
        fr_dir = f"{O}/{sub}/fr"; nf = len([f for f in os.listdir(fr_dir) if f.endswith(".png")])
        assert nf == 80, nf
        out = f"{O}/spstick_{kind}_img_{T}.mp4"
        wr = iio.get_writer(out, fps=15, codec="libx264", pixelformat="yuv420p", macro_block_size=1, quality=8)
        for t in range(nf):
            g = Image.open(f"{fr_dir}/f{t:04d}.png").convert("RGB"); GW = g.width
            tiles = [Image.fromarray(r[t + 1]) for r in R]
            th = int(tiles[0].height * (GW / 3) / tiles[0].width)
            top = Image.new("RGB", (GW, th + 40), "white"); d = ImageDraw.Draw(top)
            for i, im in enumerate(tiles):
                top.paste(im.resize((GW // 3, th)), (i * (GW // 3), 40))
                d.text((i * (GW // 3) + 10, 4), f"{LAB[i]} · 렌더 프레임 {t+1}", fill="black", font=font)
            c = Image.new("RGB", (GW, top.height + g.height), "white"); c.paste(top, (0, 0)); c.paste(g, (0, top.height))
            if c.height % 2: c = c.crop((0, 0, GW - GW % 2, c.height - 1))
            wr.append_data(np.asarray(c))
        wr.close(); print(f"[영상] {out} {nf} 프레임", flush=True)
print("SPFDISTIMG_OK")
