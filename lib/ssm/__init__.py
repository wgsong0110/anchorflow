"""Per-node Selective SSM (Mamba-style) with CUDA kernel.

State update per node i, dim d, state n:
  A       = -exp(A_log[d, n])
  dA      = exp(dt[m,d] * A) = exp(-dt * exp(A_log))
  dB      = dt[m,d] * B[m,n]
  h_new   = dA * h + dB * u[m,d]
  y[m,d]  = sum_n(C[m,n] * h_new) + D_vec[d] * u[m,d]
"""
import os, torch, torch.nn as nn, torch.nn.functional as F

_lib_dir = os.path.dirname(__file__)

def _load():
    so = [f for f in os.listdir(_lib_dir) if f.startswith("_C") and f.endswith(".so")]
    if not so:
        raise RuntimeError(
            "ssm CUDA kernel not found. Run wbuild or download from releases.")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_C", os.path.join(_lib_dir, so[0]))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_C = None

def _get_C():
    global _C
    if _C is None:
        _C = _load()
    return _C


class _SSMStepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, h, dt, B, C, A_log, D_vec):
        h_new, y = _get_C().forward(u, h, dt, B, C, A_log, D_vec)
        ctx.save_for_backward(u, h, dt, B, C, A_log, D_vec, h_new)
        return h_new, y

    @staticmethod
    def backward(ctx, dh_new, dy):
        u, h, dt, B, C, A_log, D_vec, h_new = ctx.saved_tensors
        grads = _get_C().backward(u, h, dt, B, C, A_log, D_vec, h_new,
                                  dh_new.contiguous(), dy.contiguous())
        return tuple(grads)   # du, dh, ddt, dB, dC, dA_log, dD_vec


def ssm_step(u, h, dt, B, C, A_log, D_vec):
    """
    u:     [M, D]
    h:     [M, D, N]
    dt:    [M, D]
    B,C:   [M, N]
    A_log: [D, N]
    D_vec: [D]
    returns h_new [M, D, N], y [M, D]
    """
    return _SSMStepFn.apply(
        u.contiguous(), h.contiguous(), dt.contiguous(),
        B.contiguous(), C.contiguous(),
        A_log.contiguous(), D_vec.contiguous())


class SelectiveSSM(nn.Module):
    """Per-node Mamba-style selective SSM.

    Input:  u [M, D]  (node encodings)
    State:  h [M, D, N]
    Output: y [M, D], h_new [M, D, N]
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        dt_rank = dt_rank or max(1, d_model // 16)

        # Input-dependent projections
        self.dt_proj = nn.Linear(d_model, d_model)          # → dt logit [M, D]
        self.B_proj  = nn.Linear(d_model, d_state, bias=False)  # [M, N]
        self.C_proj  = nn.Linear(d_model, d_state, bias=False)  # [M, N]

        # Learnable parameters
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) + 0.5))
        self.D_vec = nn.Parameter(torch.zeros(d_model))

    def init_state(self, M: int, device, dtype) -> torch.Tensor:
        return torch.zeros(M, self.d_model, self.d_state, device=device, dtype=dtype)

    def forward(self, u: torch.Tensor, h: torch.Tensor):
        """
        u: [M, D], h: [M, D, N]
        returns y [M, D], h_new [M, D, N]
        """
        dt = F.softplus(self.dt_proj(u))    # [M, D]
        B  = self.B_proj(u)                 # [M, N]
        C  = self.C_proj(u)                 # [M, N]
        h_new, y = ssm_step(u, h, dt, B, C, self.A_log, self.D_vec)
        return y, h_new
