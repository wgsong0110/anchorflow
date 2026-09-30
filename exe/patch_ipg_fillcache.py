"""i-PG 의 입자 채우기를 **캐시**하게 만든다 (읽기 + 쓰기).

채우기는 i-PG 실행에서 가장 느리고 가장 잘 깨지는 단계다. 한 번 성공하면 npy 로
남겨 두고 다음 실행은 즉시 읽어 쓴다 -- 그래야 중단돼도 손실 없이 재시작된다.

`patch_ipg_scen.py` 가 넣은 래퍼는 읽기만 했다. 여기서 없으면 만들어 두는 쪽을
더한다. 여러 번 돌려도 안전하다.

  python exe/patch_ipg_fillcache.py --ipg /home/dkta/work/i-physgaussian
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ipg", required=True)
a = ap.parse_args()
p = os.path.join(a.ipg, "gs_simulation.py")
s = open(p).read()
if "AF_FILL_DUMP" in s:
    print("[패치] 이미 들어 있다")
    raise SystemExit(0)

old = """        return _af_orig_fill(*args, **kwargs)
"""
new = """        _r = _af_orig_fill(*args, **kwargs)
        # 성공한 채우기를 남겨 둔다 -- 다음 실행은 위 분기에서 즉시 읽는다.
        if _ck:
            try:
                _af_np.save(_ck, _r.detach().cpu().numpy())
                print(f'[PG채움] 캐시 저장 {_ck} ({_r.shape[0]} 개) AF_FILL_DUMP',
                      flush=True)
            except Exception as _e:
                print(f'[PG채움] 캐시 저장 실패: {_e}', flush=True)
        return _r
"""
assert s.count(old) == 1, "patch_ipg_scen 의 채우기 래퍼를 못 찾았다"
open(p, "w").write(s.replace(old, new, 1))
print(f"[패치] 채우기 캐시 쓰기 추가 -> {p}")
