"""HF 저장소 (hbpencil01/GaussianFluent) 를 표준 라이브러리만으로 받는다 (huggingface_hub 의 urllib3/brotli 충돌 회피).
  python hfdl_std.py OUT_DIR PREFIX [PREFIX ...] [--skip P ...]    PREFIX = 'model/bullet_0_psnr36/' 같은 경로 접두"""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
REPO = "hbpencil01/GaussianFluent"
out = sys.argv[1]; args = sys.argv[2:]
skip = []
if "--skip" in args:
    i = args.index("--skip"); skip = args[i + 1:]; args = args[:i]
def get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "dl"}), timeout=120)
files = []
for top in ("model", "config"):
    j = json.load(get(f"https://huggingface.co/api/models/{REPO}/tree/main/{top}?recursive=1&expand=0"))
    files += [x for x in j if x["type"] == "file"]
want = [f for f in files if (not args or any(f["path"].startswith(p) for p in args)) and not any(f["path"].startswith(s) for s in skip)]
print("받을 파일", len(want), sum(f.get("size", 0) for f in want) / 1e9, "GB", flush=True)
def dl(f):
    dst = os.path.join(out, f["path"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == f.get("size", -1): return
    for t in range(4):
        try:
            with get(f"https://huggingface.co/{REPO}/resolve/main/{f['path']}") as r, open(dst + ".part", "wb") as w:
                while True:
                    b = r.read(1 << 22)
                    if not b: break
                    w.write(b)
            os.replace(dst + ".part", dst); return
        except Exception as e:
            print("재시도", f["path"], e, flush=True)
    raise SystemExit("실패 " + f["path"])
with ThreadPoolExecutor(8) as ex: list(ex.map(dl, want))
print("완료", flush=True)
