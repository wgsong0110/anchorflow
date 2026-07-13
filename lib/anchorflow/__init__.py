"""anchorflow — learned GNN dynamics over 3DGS anchor points.

Pipeline: 3DGS -> anchor extraction -> anchor graph -> GNN autoregressive
dynamics -> anchor-driven Gaussian deformation -> render.

Modules
    synth     synthetic anchor-motion sequences (validation data)
    graph     anchor graph construction (knn / radius)
    dynamics  GNS-style graph-network autoregressive dynamics model
    deform    anchor -> Gaussian deformation (translation / local-affine LBS)
"""

import importlib

__all__ = ["synth", "graph", "dynamics", "deform"]


def __getattr__(name):
    # Lazy submodule import so ``anchorflow.synth`` (pure NumPy) is usable
    # without torch installed; graph/dynamics/deform pull torch only on access.
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
