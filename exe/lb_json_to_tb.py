"""이미 나온 평가 JSON 들을 TensorBoard 에 옮긴다 (일회용).

train_deform 의 최종 평가 JSON (<out>/<tag>.json, 예 lbours/main2/MAIN.json), --infer 평가
JSON (lbours/evalall/*/EV.json), lb_rollout.py 의 JSON 을 모두 받는다. 파일마다 따로
실행 폴더(<부모폴더>_<파일이름>) 를 만들어 요약·궤적별 스칼라를 쓴다 -- 이름은
train_deform / lb_rollout 이 쓰는 것과 같다 (eval/err_pct, eval_traj/<tag>_t<t0>/...).

단계(step):
  JSON 에 "step" 이 있으면 그것 (새 train_deform·lb_rollout 이 남긴다)
  --infer 평가면 args.resume 체크포인트의 step (읽을 수 있으면)
  아니면 args.iters (학습을 끝까지 돌린 뒤의 최종 평가)
  --step 을 주면 모두 그 값으로 덮는다

  python exe/lb_json_to_tb.py --tb /home/dkta/work/lbours/tb_json \
      /home/dkta/work/lbours/{main,main2,main3,mainD}/MAIN.json /home/dkta/work/lbours/evalall/*/EV.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

ap = argparse.ArgumentParser()
ap.add_argument("jsons", nargs="+")
ap.add_argument("--tb", required=True, help="TensorBoard 루트 (파일마다 하위 실행 폴더)")
ap.add_argument("--prefix", default="", help="실행 폴더 이름 앞에 붙일 것")
ap.add_argument("--step", type=int, default=None)
ap.add_argument("--dry", action="store_true", help="쓰지 않고 무엇을 쓸지만 출력")
a = ap.parse_args()


def summarize(rows):
    """deform_model.summarize 와 같은 정의 (torch 없이 돌게 여기 둔다)."""
    r_all = [(w["err_mean"], w["still_mean"]) for r in rows.values()
             for w in r["windows"].values()]
    if not r_all:
        return None
    n = len(r_all)
    return dict(n=n, err_mean=sum(x for x, _ in r_all) / n,
                still_mean=sum(y for _, y in r_all) / n,
                ratio=sum(x / max(y, 1e-12) for x, y in r_all) / n)


def step_of(j):
    if a.step is not None:
        return a.step, "--step"
    if j.get("step") is not None:
        return int(j["step"]), "json"
    args = j.get("args") or {}
    if args.get("infer") and args.get("resume") and os.path.exists(args["resume"]):
        try:
            import torch
            ck = torch.load(args["resume"], map_location="cpu", weights_only=False, mmap=True)
            return int(ck["step"]), "resume ckpt"
        except Exception as e:                                   # noqa: BLE001
            print(f"  [경고] {args['resume']} step 읽기 실패 ({e}) -- iters 를 쓴다")
    return int(args.get("iters", 0)), "args.iters"


for p in a.jsons:
    j = json.load(open(p))
    rows = j.get("rollout") or {}
    S = summarize(rows)
    if S is None:
        print(f"[건너뜀] {p}: rollout 없음")
        continue
    step, src = step_of(j)
    ap_ = os.path.abspath(p)
    name = a.prefix + f"{os.path.basename(os.path.dirname(ap_))}_{os.path.splitext(os.path.basename(ap_))[0]}"
    print(f"[{name}] {p} 단계 {step} ({src}) | 창 {S['n']} 평균 {100*S['err_mean']:.3f}% "
          f"정지 {100*S['still_mean']:.3f}% 비 {S['ratio']:.3f}")
    if a.dry:
        continue
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(os.path.join(a.tb, name))
    w.add_scalar("eval/err_pct", 100 * S["err_mean"], step)
    w.add_scalar("eval/still_pct", 100 * S["still_mean"], step)
    w.add_scalar("eval/ratio", S["ratio"], step)
    for tag, r in rows.items():
        for t0, win in r["windows"].items():
            k = f"eval_traj/{tag}_t{t0}"
            w.add_scalar(f"{k}/err_pct", 100 * win["err_mean"], step)
            w.add_scalar(f"{k}/still_pct", 100 * win["still_mean"], step)
            w.add_scalar(f"{k}/ratio", win["err_mean"] / max(win["still_mean"], 1e-12), step)
    w.close()
print("JSON_TO_TB_OK")
