"""풀 상태 탈락 과정 기록 (사용자 10-25, --pool_trace).

풀 상태마다 시작부터 전진할 때마다의 (위치, F) 를 float16 CPU 텐서로 들고 다니다가, 상태가 풀에서 빠질 때(사유 기록)
  - 모든 탈락을 drops.jsonl 에 한 줄씩: 반복, uid, 사유, 나이(전진 프레임), 정체 횟수, 태어난 반복, 이력 시작 프레임
  - 주기(period 반복)마다 그 구간에 탈락한 상태 중 나이가 가장 긴 n_long 개 + 나머지에서 무작위 n_rand 개의 이력 전체를
    drop_it<반복>_uid<uid>_<사유>_age<나이>.pt 로 저장 (saved.jsonl 에 기록)
  - 지금까지 풀에서 도달한 최대 나이를 새로 갱신하면 그 상태의 이력(시작~그 나이)을 best_it<반복>_age<나이>_uid<uid>.pt 로 저장
    (덮어쓰지 않음; 나이 best_every_after 초과부터는 best_step 프레임 늘 때마다만)
저장 파일 = dict(pred=[T,N,3] fp16, F=[T,N,3,3] fp16, frames=[이력 프레임 번호 1..], meta) -- exe/lb_roll_to_h5.py 가 그대로 읽는다
(프레임 0 = 풀 시작 상태 = --pool_init_traj 의 그 프레임, 이력 시작 프레임 trace_from 이 0 일 때만 시작부터 완전).
학습 난수(torch 생성기)는 건드리지 않는다: 무작위 선택은 따로 둔 python random 으로 한다.
체크포인트에는 이력을 넣지 않는다 (용량) -- 이어 돌리면 풀에 남아 있던 상태는 그 나이부터 이력을 잇는다 (trace_from = 그 나이).
"""
import json
import os
import random

import torch


class PoolTracer:
    def __init__(self, out_dir, period=500, n_long=2, n_rand=1, best_every_after=30, best_step=5, max_frames=400):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.period, self.n_long, self.n_rand = int(period), int(n_long), int(n_rand)
        self.best_every_after, self.best_step, self.max_frames = int(best_every_after), int(best_step), int(max_frames)
        self.it = 0
        self.rng = random.Random(12345)
        st = self._load_state()
        self.uid = int(st.get("uid", 0))
        self.best_age = int(st.get("best_age", 0))
        self.best_saved = int(st.get("best_saved", 0))
        self.n_drop_total = int(st.get("n_drop_total", 0))
        self._reset_window()

    # ---- 상태 파일 (이어 돌릴 때 최고 기록·uid 를 잇는다) ----
    def _load_state(self):
        p = os.path.join(self.dir, "tracer_state.json")
        try:
            return json.load(open(p))
        except Exception:
            return {}

    def _save_state(self):
        p = os.path.join(self.dir, "tracer_state.json")
        json.dump(dict(uid=self.uid, best_age=self.best_age, best_saved=self.best_saved, n_drop_total=self.n_drop_total,
                       it=self.it), open(p + ".tmp", "w"))
        os.replace(p + ".tmp", p)

    def _reset_window(self):
        self.win_long = []            # (나이, 기록) 상위 n_long
        self.win_res = []             # 저수지 표본 (무작위 n_rand)
        self.win_n = 0                # 저수지에 들어온 수

    # ---- 풀 쪽 훅 ----
    def on_fresh(self, st):
        self.uid += 1
        st["uid"] = self.uid
        st["it_born"] = int(self.it)
        st["trace"] = []
        st["trace_from"] = 0

    def on_load(self, st):
        """체크포인트에서 이어받은 상태: 이력은 지금 나이부터."""
        self.uid += 1
        st["uid"] = self.uid
        st["it_born"] = -1
        st["trace"] = []
        st["trace_from"] = int(st.get("age", 0))

    def on_advance(self, st):
        tr = st.get("trace")
        if tr is None:
            return
        if len(tr) < self.max_frames:
            tr.append((st["x"].detach().to("cpu", torch.float16),
                       st["F"].detach().reshape(-1, 3, 3).to("cpu", torch.float16)))
        age = int(st["age"])
        if age > self.best_age:
            self.best_age = age
            if age <= self.best_every_after or age >= self.best_saved + self.best_step:
                self._write(st, f"best_it{self.it + 1:06d}_age{age:03d}_uid{st['uid']}.pt", "최고기록", kind="best")
                self.best_saved = age
            self._save_state()

    def on_drop(self, st, reason):
        if st.get("trace") is None:
            return
        self.n_drop_total += 1
        rec = dict(it=int(self.it) + 1, uid=int(st.get("uid", -1)), reason=reason, age=int(st.get("age", 0)),
                   stuck=int(st.get("stuck", 0)), it_born=int(st.get("it_born", -1)),
                   trace_from=int(st.get("trace_from", 0)), n_trace=len(st["trace"]))
        with open(os.path.join(self.dir, "drops.jsonl"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rec["n_trace"] == 0:
            return
        item = (rec, st["trace"])
        # 나이 상위 n_long (같으면 먼저 탈락한 것), 그 밖은 저수지 표본
        self.win_long.append(item)
        self.win_long.sort(key=lambda q: -q[0]["age"])
        if len(self.win_long) > self.n_long:
            out = self.win_long.pop()
            self.win_n += 1
            if len(self.win_res) < self.n_rand:
                self.win_res.append(out)
            else:
                j = self.rng.randrange(self.win_n)
                if j < self.n_rand:
                    self.win_res[j] = out
        st["trace"] = None

    def step_end(self, it):
        """학습 반복 it 이 끝났을 때: 주기면 이번 구간 탈락 이력을 저장."""
        prev = self.it
        self.it = int(it) + 1
        if self.it // self.period == prev // self.period:     # 이번 반복에서 주기 경계를 넘지 않았다 (중간에 건너뛴 반복이 있어도 경계에서 한 번)
            return
        n = 0
        for tag, items in (("긴것", self.win_long), ("무작위", self.win_res)):
            for rec, tr in items:
                fn = f"drop_it{self.it:06d}_uid{rec['uid']}_{rec['reason']}_age{rec['age']:03d}.pt"
                self._write_trace(tr, rec, fn, tag, kind="drop")
                n += 1
        print(f"[풀 이력] 반복 {self.it}: 구간 탈락 이력 {n} 개 저장 (누적 탈락 {self.n_drop_total}, 최대 나이 {self.best_age})", flush=True)
        self._reset_window()
        self._save_state()

    # ---- 저장 ----
    def _write(self, st, fn, tag, kind):
        rec = dict(it=int(self.it) + 1, uid=int(st.get("uid", -1)), reason=tag, age=int(st.get("age", 0)),
                   stuck=int(st.get("stuck", 0)), it_born=int(st.get("it_born", -1)),
                   trace_from=int(st.get("trace_from", 0)), n_trace=len(st["trace"]))
        self._write_trace(st["trace"], rec, fn, tag, kind)

    def _write_trace(self, tr, rec, fn, tag, kind):
        p = os.path.join(self.dir, fn)
        P = torch.stack([q[0] for q in tr])
        Fq = torch.stack([q[1] for q in tr])
        t0 = rec["trace_from"]
        frames = list(range(t0 + 1, t0 + 1 + len(tr)))
        torch.save(dict(pred=P, F=Fq, frames=frames, meta=dict(rec, select=tag, kind=kind)), p + ".tmp")
        os.replace(p + ".tmp", p)
        with open(os.path.join(self.dir, "saved.jsonl"), "a") as f:
            f.write(json.dumps(dict(rec, file=fn, select=tag, kind=kind, bytes=os.path.getsize(p)), ensure_ascii=False) + "\n")
