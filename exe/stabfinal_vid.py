"""[stabfinal] 장면마다 최소 N 비교 영상 (exe/stab4_vid.py 로 합성): 기준 PG N=1024 | PG | i-PG / OmniPhysGS | 우리, 칸 제목 = 방법·확정 N.
입력 = 드라이버 vis 단계 영상 (<root>/<장면>/vis/<이름>.mp4, lb_render 위에서 내려다보는 카메라). 가로 2400, yuv420p, faststart.
  python exe/stabfinal_vid.py --root /home/dkta/work/stab --scene lego --font font.otf --out out.mp4
"""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--scene", required=True)
ap.add_argument("--font", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", default="_f5")
a = ap.parse_args()
SN = dict(lego="Lego", pillow2sofa="pillow2sofa", ship_plastic_lego="ship (plasticine)", mic_metal_lego="mic (metal)")
D = f"{a.root}/{a.scene}"
items = ["pg_N1024=기준 (PG N=1024)"]
for m, nm in (("pg", "PhysGaussian"), ("ipg", "i-PhysGaussian"), ("omni", "OmniPhysGS"), ("ours", "우리")):
    p = f"{D}/plan_{m}{a.tag if m == 'ours' else ''}.json"
    n = json.load(open(p)).get("N_min") if os.path.exists(p) else None
    if n:
        items.append(f"{m}{a.tag if m == 'ours' else ''}_N{n}={nm} N={n}")
    else:
        items.append(f"{m}_none={nm} (안정 N 없음)")
subprocess.run([sys.executable, f"{HERE}/stab4_vid.py", "--vis", f"{D}/vis", "--scene", SN.get(a.scene, a.scene), "--items", *items,
                "--font", a.font, "--out", a.out, "--pw", "800", "--cols", "3"], check=True)
dec = subprocess.run(["ffmpeg", "-v", "error", "-i", a.out, "-f", "null", "-"], capture_output=True, text=True)
pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,pix_fmt,nb_frames", "-of", "csv=p=0", a.out], capture_output=True, text=True).stdout.strip()
print("영상", a.out, pr, "디코드 오류 줄", len([q for q in dec.stderr.splitlines() if q.strip()]))
