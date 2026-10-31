"""절 AK: 입자 적분점 커널 레지스터·지역 메모리 (넘침) 확인."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import fem_ak_cuda as AKC   # noqa: E402
from anchorflow import fem_ah_cuda as AHC   # noqa: E402
AKC.set_pit(0)
for nm in ("ak_pq_force_p", "ak_pq_force_p2", "ak_tetA", "ak_pq_force"):
    k = AKC._MOD[0].get_function(nm)
    print(nm, "regs", k.num_regs, "local", k.local_size_bytes, "maxthr", k.max_threads_per_block)
for nm in ("ah_fc", "ah_node_step"):
    k = AHC.mod().get_function(nm)
    print(nm, "regs", k.num_regs, "local", k.local_size_bytes)
