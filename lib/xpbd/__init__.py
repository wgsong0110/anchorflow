"""XPBD kernels of the VR-GS baseline (exe/baseline_vrgs.py)."""
import os


def _cu_sources():
    """CUDA sources of this package (compiled at run time by cupy). Each entry of xpbd.cu is
    '//@@ <name> <length>' on its own line followed by exactly <length> characters of source."""
    global _CU
    if _CU is None:
        _CU = {}
        t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xpbd.cu'), encoding='utf-8', newline='').read()
        p = 0
        while p < len(t):
            e = t.index('\n', p)
            _, name, n = t[p:e].split(' ')
            _CU[name] = t[e + 1:e + 1 + int(n)]
            p = e + 1 + int(n) + 1
    return _CU


def cu(name):
    return _cu_sources()[name]


_CU = None
