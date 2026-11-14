"""절 CM176 감싸개: --ssn N (기본 0 = 끔). N=1 = 반매끈 뉴턴: 활성 집합 (탄성·꼬리·꼭짓점) 마다 정확한 일반화 접선 (--ctan 2, 반복마다 현재 시험 상태로 판정)
+ 반환 뒤 실제 잔차 (관성 계량, --rls 2) 에 대한 비단조 (Grippo, 기억 NM = AF_RLS_NM 기본 4) Armijo 선탐색.  새 상수 없음 (c1 1e-4 는 기존 rls 의 것, NM 만 새 값).
사용: python exe/ssn_wrap.py --ssn N <다음 감싸개.py> ... <실행기.py> ...   (--ssn 1 이면 ctan_wrap·rls_wrap 을 안에서 부름: 명령줄에 --ctan·--rls 를 따로 쓰지 말 것)"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
assert argv and argv[0] == "--ssn", "ssn_wrap: 첫 인자는 --ssn N"
n = int(argv[1]); argv = argv[2:]
if n == 1:
    os.environ.setdefault("AF_RLS_NM", "4")
    sys.path.insert(0, os.path.join(HERE, "..", "lib"))
    from anchorflow import fem_ssn
    fem_ssn.patch()
    print(f"[절 CM176] ssn 1 -- ctan 2 + rls 2, 비단조 기억 {fem_ssn.NM}", flush=True)
    argv = [os.path.join(HERE, "ctan_wrap.py"), "--ctan", "2", os.path.join(HERE, "rls_wrap.py"), "--rls", "2"] + argv
assert argv and argv[0].endswith(".py")
sys.argv = argv
runpy.run_path(sys.argv[0], run_name="__main__")
