"""절 CM54 묶음 rtrender1: 시뮬 루프 안 실시간 3DGS 렌더 시간 (사용자 11-08 '실시간 렌더까지 포함해서 실시간').

프레임 (바깥 쪼개기 함수 = 시뮬 한 프레임) 이 끝날 때마다 같은 프로세스·같은 GPU 스트림에서
  (a) 시뮬 = 실행기의 프레임 재생 GPU 시간 s.last_ms (fem.jsonl ms 와 같은 값)
  (b) 가우시안 갱신 = 입자 -> 가우시안 위치·공분산 (+ 수박은 F 극분해 회전으로 돌린 SH 색), 영상 렌더 (exe/lb_render.py · exe/wm_render.py) 와 같은 정의
  (c) 3DGS 래스터화 = 같은 래스터라이저·카메라·해상도, 결과 영상은 GPU 에 둠 (파일 저장·CPU 복사 없음)
를 CUDA 이벤트로 재서 <out>/rr.jsonl 에 한 줄씩. 풀이 상태는 읽기만 (결과 불변).

갱신 방식 (--rr_fast):
  0 = 영상 렌더 코드의 torch 연산을 GPU 에서 그대로 (lb_roll_to_h5 의 F·R, lb_render 의 FC0Fᵀ·좌표 변환; 수박은 wm_render 의
      유한 거름·좌표/공분산 되돌림·convert_SH. 다만 wm_render 의 CPU SVD (U Vᵀ) 대신 GPU 극분해 (배율 뉴턴 10 회, f64) -- 같은 극 인수)
  2 = 1 과 같되 수박 극분해를 f32 로, ‖ΔX‖_F < 1e-6 이면 멈춤 (최대 12 회)
  1 = 갱신을 커널 하나로 합침 (cupy; 위치·공분산 변환은 같은 선형 사상을 시작 때 수치로 뽑아 씀, 극분해 뉴턴 f64, SH 3 차 그대로;
      수박의 비유한 가우시안은 빼는 대신 카메라 위치에 두어 래스터라이저가 잘라냄 = 같은 화면)
확인 (--rr_chk 프레임들): 그 프레임 렌더를 uint8 PNG 로 <out>/rrimg/<h5 번호 4 자리>.png (시간 잰 뒤, 시간 밖),
  fast 1 이면 같은 프레임에서 fast 0 갱신도 해서 가우시안 값·화면 차이를, 수박이면 GPU 극분해 대 CPU SVD 차이도 <out>/rr_chk.jsonl 에.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

W = "/home/dkta/work"
MODEL = {"lego": "lego_whitebg-trained", "ficus": "ficus_whitebg-trained", "wolf": "wolf_whitebg-trained", "bread": "bread-trained"}
MODEL.update({"chair_plasticine": "chair", "drums_foam": "drums", "materials_sand": "materials", "materials_metal": "materials", "ball_metal": "ball"})
MODEL.update({"chair_paste": "chair", "drums_honey": "drums", "bread_foam_drop": "bread-trained", "chair_paste_lego": "chair", "drums_honey_lego": "drums", "bread_foam_drop_lego": "bread-trained"})      # 절 CM189 plastfix
MODEL.update({"chair_elastic": "chair", "ship_plastic": "ship", "mic_metal": "mic", "mic_plastic": "mic", "mic_elastic": "mic", "ship_elastic": "ship", "ship_plastic_lego": "ship", "mic_metal_lego": "mic"})      # elastic_new      # 절 CM186 새 형상 (pgmodel/<물체>, 직접 학습한 3DGS·DreamPhysics ball)


class Cfg:
    kind = "pg"; shape = "lego"; traj = ""; cfg = ""; model = ""; gf = f"{W}/GaussianFluent"; fast = 0; chk = (); out = "."


C = Cfg()
ST = dict(ready=False, fi=0, log=None, clog=None, dep=0)


# ---------------------------------------------------------------- 준비 (첫 프레임 끝, 시간 밖)
def _purge(root):
    """이미 다른 곳에서 읽힌 utils·scene 모듈이 있으면 빼서 렌더러 소스 (root) 의 것을 읽게."""
    for k in list(sys.modules):
        if k.split(".")[0] in ("utils", "scene"):
            f = getattr(sys.modules[k], "__file__", None) or ""
            if not os.path.abspath(f).startswith(os.path.abspath(root)):
                print(f"[절 CM54 rtrender1] 모듈 {k} ({f}) 빼고 {root} 것으로", flush=True)
                sys.modules.pop(k)


def _setup_pg():
    cfg = json.load(open(C.cfg)); meta = json.load(open(f"{C.traj}/meta.json"))
    sys.path.append(f"{W}/i-physgaussian"); sys.path.append(f"{W}/i-physgaussian/gaussian-splatting")
    cwd = os.getcwd(); os.chdir(f"{W}/i-physgaussian")
    try:
        _purge(f"{W}/i-physgaussian")
        from scene.gaussian_model import GaussianModel
        from utils.transformation_utils import transform2origin, shift2center111
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    finally:
        os.chdir(cwd)
    import h5py
    dev = "cuda"
    mp = f"{W}/pgmodel/{MODEL[C.shape]}"
    gs = GaussianModel(3)
    it = max(int(q.split("_")[-1]) for q in os.listdir(f"{mp}/point_cloud") if q.startswith("iteration_"))
    gs.load_ply(f"{mp}/point_cloud/iteration_{it}/point_cloud.ply")
    KIDX = torch.nonzero(gs.get_opacity[:, 0] > float(cfg.get("opacity_threshold", 0.02))).squeeze(1)
    RT = torch.eye(3, device=dev)
    for dg, ax in zip(cfg.get("rotation_degree") or [], cfg.get("rotation_axis") or []):
        c_, s_ = math.cos(dg / 180.0 * 3.1415926), math.sin(dg / 180.0 * 3.1415926)
        R_ = {0: [[1, 0, 0], [0, c_, -s_], [0, s_, c_]], 1: [[c_, 0, s_], [0, 1, 0], [-s_, 0, c_]], 2: [[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]}[int(ax)]
        RT = torch.tensor(R_, device=dev, dtype=torch.float32) @ RT
    ROT = bool((RT - torch.eye(3, device=dev)).abs().max() > 0)
    TP, SO, MEAN = transform2origin(gs.get_xyz[KIDX] @ RT.T if ROT else gs.get_xyz[KIDX], float(cfg.get("scale", 1.0)))
    TP = shift2center111(TP)
    NG = KIDX.numel()
    with h5py.File(meta["h5"]) as h:
        x0 = np.array(h["x"]); x0 = x0.T if x0.shape[0] == 3 else x0
    err = float((torch.as_tensor(x0[:NG], device=dev).float() - TP).norm(dim=1).max())
    c6 = gs.get_covariance()[KIDX]
    C0 = torch.zeros(NG, 3, 3, device=dev)
    C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
    C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
    C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
    if ROT:
        C0 = RT @ C0 @ RT.T
    cam = json.load(open(f"{mp}/cameras.json"))[0]
    Rw, pos = np.array(cam["rotation"]), np.array(cam["position"])
    ctr = ((MEAN @ RT) if ROT else MEAN).cpu().numpy().astype(np.float64)
    pos = ctr + 1.6 * (pos - ctr)                                                      # lb_render 기본 --cam_back 1.6 (영상과 같음)
    W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    fx_ = 2 * math.atan(cam["width"] / (2 * cam["fx"])); fy_ = 2 * math.atan(cam["height"] / (2 * cam["fy"]))
    wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
    pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fx_, fovY=fy_).transpose(0, 1).to(dev).float()
    RAST = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
        image_height=int(cam["height"]), image_width=int(cam["width"]), tanfovx=math.tan(fx_ * 0.5), tanfovy=math.tan(fy_ * 0.5),
        bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0], sh_degree=3,
        campos=wv.inverse()[3, :3], prefiltered=False, debug=False))
    Rm = torch.as_tensor(np.array(meta["R"]), dtype=torch.float32, device=dev)         # lb_roll_to_h5: 렌더용 F = F_sim · R
    G = dict(NG=NG, H=int(cam["height"]), Wd=int(cam["width"]), C0=C0.contiguous(), RT=RT, ROT=ROT, SO=SO, MEAN=MEAN, Rm=Rm,
             SHS=gs.get_features[KIDX].contiguous(), OPA=gs.get_opacity[KIDX].contiguous(), RAST=RAST, corr_err=err)
    # 합친 커널용 선형 사상 (같은 식을 수치로): 위치 p' = A x + b, 공분산은 C0' = Rm C0 Rmᵀ 를 미리 (F Rm C0 Rmᵀ Fᵀ), 회전은 RTᵀ (.) RT
    G["C0R"] = (Rm @ C0 @ Rm.T).contiguous()
    return G


def _setup_gf():
    dev = "cuda"
    sys.path.insert(0, C.gf); sys.path.insert(1, os.path.join(C.gf, "gaussian-splatting"))
    cwd = os.getcwd(); os.chdir(C.gf)
    try:
        _purge(C.gf)
        from utils.decode_param import decode_param_json
        import types
        import utils.transformation_utils as _TU
        import utils.camera_view_utils as _CV
        import utils.render_utils as _RU
        from scene.gaussian_model import GaussianModel
        from utils.system_utils import searchForMaxIteration
        NS = {}                                                                        # wm_render 의 'from ... import *' 세 줄과 같은 이름 공간 (뒤가 앞을 덮음)
        for _m in (_TU, _CV, _RU):
            NS.update({k: v for k, v in vars(_m).items() if not k.startswith("_")})
        TU = CV = RU = types.SimpleNamespace(**NS)

        class PP:
            convert_SHs_python = False; compute_cov3D_python = True; debug = False
        material_params, bc_params, time_params, pp, camera_params = decode_param_json(C.cfg)
        d = os.path.join(C.model, "point_cloud")
        gaussians = GaussianModel(3); gaussians.load_ply(os.path.join(d, f"iteration_{searchForMaxIteration(d)}", "point_cloud.ply"))
        pipeline = PP()
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=dev)
        params = RU.load_params_from_gs(gaussians, pipeline)
        init_pos, init_cov, init_opacity, init_shs = params["pos"], params["cov3D_precomp"], params["opacity"], params["shs"]
        m = init_opacity[:, 0] > pp["opacity_threshold"]
        init_pos, init_cov, init_opacity, init_shs = init_pos[m], init_cov[m], init_opacity[m], init_shs[m]
        for k in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
            setattr(gaussians, k, getattr(gaussians, k)[m])
        R = TU.generate_rotation_matrices(torch.tensor(pp["rotation_degree"]), pp["rotation_axis"])
        rp = TU.apply_rotations(init_pos, R)
        if pp["sim_area"] is not None:
            b = pp["sim_area"]; m = torch.ones(rp.shape[0], dtype=torch.bool, device=dev)
            for i in range(3):
                m = m & (rp[:, i] > b[2 * i]) & (rp[:, i] < b[2 * i + 1])
            rp, init_cov, init_opacity, init_shs = rp[m], init_cov[m], init_opacity[m], init_shs[m]
        tp, so, mp_ = TU.transform2origin(rp)
        tp = TU.shift2center111(tp)
        init_cov = TU.apply_cov_rotations(init_cov, R) * so * so
        NG = tp.shape[0]
        C0 = torch.zeros(NG, 3, 3, device=dev); c = init_cov
        C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c[:, 0], c[:, 1], c[:, 2]
        C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c[:, 3], c[:, 4], c[:, 5]
        C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c[:, 1], c[:, 2], c[:, 4]
        vc, oc = CV.get_center_view_worldspace_and_observant_coordinate(
            torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda(),
            torch.tensor(camera_params["mpm_space_vertical_upward_axis"]).reshape((1, 3)).cuda(), R, so, mp_)
        CAMS = {}
        for t in range(0, 1001):                                                       # 카메라 궤적은 시뮬과 무관 -> 미리 (wm_render 와 같은 인자, h5 번호 t)
            if t > 0 and not camera_params["move_camera"]:
                CAMS[t] = CAMS[0]; continue
            cam = CV.get_camera_view(C.model, default_camera_index=camera_params["default_camera_index"], center_view_world_space=vc,
                                     observant_coordinates=oc, show_hint=camera_params["show_hint"], init_azimuthm=camera_params["init_azimuthm"],
                                     init_elevation=camera_params["init_elevation"], init_radius=camera_params["init_radius"],
                                     move_camera=camera_params["move_camera"], current_frame=t, delta_a=camera_params["delta_a"],
                                     delta_e=camera_params["delta_e"], delta_r=camera_params["delta_r"], width=600, height=600)
            CAMS[t] = (cam, RU.initialize_resterize(cam, gaussians, pipeline, background))
            if t > 200:
                break
    finally:
        os.chdir(cwd)
    meta = json.load(open(f"{C.traj}/meta.json"))
    Rm = torch.as_tensor(np.array(meta["R"]), dtype=torch.float32, device=dev)
    # 위치 되돌림 (affine) · 공분산 되돌림 (선형) 을 GF 함수에 단위 입력을 넣어 수치로 뽑음 (합친 커널용; f64 로 재서 f32)
    E = torch.eye(3, device=dev, dtype=torch.float32)
    f_pos = lambda X: TU.apply_inverse_rotations(TU.undotransform2origin(TU.undoshift2center111(X), so, mp_), R)
    b0 = f_pos(torch.zeros(1, 3, device=dev)); A = (f_pos(E) - b0).T                       # p' = A x + b0
    f_cov = lambda c6_: TU.apply_inverse_cov_rotations(c6_ / (so * so), R)
    L6 = f_cov(torch.eye(6, device=dev)).T                                                 # c' = L6 c
    xt = torch.randn(1000, 3, device=dev) + 1.0; ct = torch.randn(1000, 6, device=dev)
    lin_err = [float((f_pos(xt) - (xt @ A.T + b0)).abs().max()), float((f_cov(ct) - ct @ L6.T).abs().max())]
    G = dict(NG=NG, H=600, Wd=600, C0=C0.contiguous(), C0R=(Rm @ C0 @ Rm.T).contiguous(), Rm=Rm, SHS=init_shs.contiguous(), OPA=init_opacity.contiguous(),
             CAMS=CAMS, gaussians=gaussians, RU=RU, TU=TU, R=R, so=so, mp=mp_, A=A.contiguous(), b0=b0.reshape(3).contiguous(), L6=L6.contiguous(),
             lin_err=lin_err, sh_deg=int(gaussians.active_sh_degree), shs_shape=list(init_shs.shape))
    return G


# ---------------------------------------------------------------- 갱신 (b)
def _inv3(X):
    """배치 3x3 역행렬 (수반 행렬 / det), X (N,3,3)."""
    a, b, c = X[:, 0, 0], X[:, 0, 1], X[:, 0, 2]
    d, e, f = X[:, 1, 0], X[:, 1, 1], X[:, 1, 2]
    g, h, i = X[:, 2, 0], X[:, 2, 1], X[:, 2, 2]
    A_, B_, C_ = e * i - f * h, -(d * i - f * g), d * h - e * g
    det = a * A_ + b * B_ + c * C_
    adj = torch.stack([A_, -(b * i - c * h), b * f - c * e, B_, a * i - c * g, -(a * f - c * d), C_, -(a * h - b * g), a * e - b * d], 1).reshape(-1, 3, 3)
    return adj / det[:, None, None]


def polar_t(F, it=10):
    """극분해 직교 인수 (= SVD 의 U Vᵀ), 배율 뉴턴 (Higham) f64."""
    X = F.double()
    for _ in range(it):
        Xi = _inv3(X)
        g = torch.sqrt(Xi.flatten(1).norm(dim=1) / X.flatten(1).norm(dim=1))[:, None, None]
        X = 0.5 * (g * X + Xi.transpose(1, 2) / g)
    return X.float()


def upd_pg0(G, s, gi):
    x = s.x[gi]
    F = s.F.reshape(-1, 9)[gi].reshape(-1, 3, 3) @ G["Rm"]
    cov = F @ G["C0"] @ F.transpose(1, 2)
    if G["ROT"]:
        cov = G["RT"].T @ cov @ G["RT"]
    cc = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    Pm = (x - 1.0) / G["SO"] + G["MEAN"]
    if G["ROT"]:
        Pm = Pm @ G["RT"]
    return dict(P=Pm, cc=cc, shs=G["SHS"], col=None, opa=G["OPA"])


def ras_pg(G, U, t):
    P = U["P"]
    return G["RAST"](means3D=P, means2D=torch.zeros_like(P), shs=U["shs"], colors_precomp=U["col"], opacities=U["opa"], scales=None,
                     rotations=None, cov3D_precomp=U["cc"])[0]


def upd_gf0(G, s, gi, t):
    TU, RU = G["TU"], G["RU"]
    X = s.x[gi]
    Fg = s.F.reshape(-1, 9)[gi].reshape(-1, 3, 3) @ G["Rm"]
    ok = torch.isfinite(X).all(1) & torch.isfinite(Fg).all(2).all(1)
    cov = Fg @ G["C0"] @ Fg.transpose(1, 2)
    c6 = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    rot = polar_t(Fg)
    pos = TU.apply_inverse_rotations(TU.undotransform2origin(TU.undoshift2center111(X), G["so"], G["mp"]), G["R"])
    c6 = TU.apply_inverse_cov_rotations(c6 / (G["so"] * G["so"]), G["R"])
    cam = G["CAMS"][t][0]
    col = RU.convert_SH(G["SHS"][ok], cam, G["gaussians"], pos[ok], rot[ok]).float()
    return dict(P=pos[ok], cc=c6[ok], shs=None, col=col, opa=G["OPA"][ok], rot=rot, ok=ok)


def ras_gf(G, U, t):
    P = U["P"]
    return G["CAMS"][t][1](means3D=P, means2D=torch.zeros_like(P), shs=None, colors_precomp=U["col"], opacities=U["opa"], scales=None,
                           rotations=None, cov3D_precomp=U["cc"])[0]


# 합친 커널 (cupy)
_KSRC = r'''
__device__ __forceinline__ void mm3(const float* A, const float* B, float* O) {
  #pragma unroll
  for (int r = 0; r < 3; r++)
    #pragma unroll
    for (int c = 0; c < 3; c++) O[r*3+c] = A[r*3+0]*B[0*3+c] + A[r*3+1]*B[1*3+c] + A[r*3+2]*B[2*3+c];
}
extern "C" __global__ void rr_pg(const float* __restrict__ x, const float* __restrict__ F, const long long* __restrict__ gi,
    const float* __restrict__ C0R, const float* __restrict__ M, int rot, float* __restrict__ P, float* __restrict__ cc, int NG) {
  // M: RT[9], MEAN[3], 1/SO
  int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= NG) return;
  long long j = gi[n];
  float f[9], c0[9], T[9], Cv[9];
  #pragma unroll
  for (int k = 0; k < 9; k++) { f[k] = F[j*9+k]; c0[k] = C0R[(long long)n*9+k]; }
  mm3(f, c0, T);
  #pragma unroll
  for (int r = 0; r < 3; r++)
    #pragma unroll
    for (int c = 0; c < 3; c++) Cv[r*3+c] = T[r*3+0]*f[c*3+0] + T[r*3+1]*f[c*3+1] + T[r*3+2]*f[c*3+2];
  float p[3];
  #pragma unroll
  for (int d = 0; d < 3; d++) p[d] = (x[j*3+d] - 1.0f) * M[12] + M[9+d];
  if (rot) {
    float Rt[9], Q[9], R2[9]; for (int k = 0; k < 9; k++) Rt[k] = M[k];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) Q[r*3+c] = Rt[0*3+r]*Cv[0*3+c] + Rt[1*3+r]*Cv[1*3+c] + Rt[2*3+r]*Cv[2*3+c];
    mm3(Q, Rt, R2); for (int k = 0; k < 9; k++) Cv[k] = R2[k];
    float q[3]; for (int c = 0; c < 3; c++) q[c] = p[0]*Rt[0*3+c] + p[1]*Rt[1*3+c] + p[2]*Rt[2*3+c];
    p[0] = q[0]; p[1] = q[1]; p[2] = q[2];
  }
  P[n*3+0] = p[0]; P[n*3+1] = p[1]; P[n*3+2] = p[2];
  cc[n*6+0] = Cv[0]; cc[n*6+1] = Cv[1]; cc[n*6+2] = Cv[2]; cc[n*6+3] = Cv[4]; cc[n*6+4] = Cv[5]; cc[n*6+5] = Cv[8];
}
template <typename T> __device__ void polar_n(const float* f, float* Rr, int maxit, T tol) {
  T X[9]; for (int k = 0; k < 9; k++) X[k] = (T)f[k];
  for (int it = 0; it < maxit; it++) {
    T a = X[0], b = X[1], c = X[2], d = X[3], e = X[4], ff = X[5], g = X[6], h = X[7], i = X[8];
    T A_ = e*i - ff*h, B_ = -(d*i - ff*g), C_ = d*h - e*g;
    T det = a*A_ + b*B_ + c*C_;
    T adj[9] = {A_, -(b*i - c*h), b*ff - c*e, B_, a*i - c*g, -(a*ff - c*d), C_, -(a*h - b*g), a*e - b*d};
    T nx = 0, ni = 0, Xi[9];
    for (int k = 0; k < 9; k++) { Xi[k] = adj[k] / det; nx += X[k]*X[k]; ni += Xi[k]*Xi[k]; }
    T gm = sqrt(sqrt(ni) / sqrt(nx));
    T Y[9], dd = 0;
    for (int r = 0; r < 3; r++) for (int q = 0; q < 3; q++) Y[r*3+q] = (T)0.5 * (gm * X[r*3+q] + Xi[q*3+r] / gm);
    for (int k = 0; k < 9; k++) { dd += (Y[k] - X[k]) * (Y[k] - X[k]); X[k] = Y[k]; }
    if (tol > 0 && dd < tol * tol) break;
  }
  for (int k = 0; k < 9; k++) Rr[k] = (float)X[k];
}
extern "C" __global__ void rr_gf(const float* __restrict__ x, const float* __restrict__ F, const long long* __restrict__ gi,
    const float* __restrict__ C0, const float* __restrict__ Rm, const float* __restrict__ A, const float* __restrict__ b0, const float* __restrict__ L6,
    const float* __restrict__ shs, const float* __restrict__ cp, float* __restrict__ P, float* __restrict__ cc, float* __restrict__ col, int NG, int prec) {
  int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= NG) return;
  long long j = gi[n];
  float f0[9], f[9], c0[9], T[9], Cv[9];
  #pragma unroll
  for (int k = 0; k < 9; k++) { f0[k] = F[j*9+k]; c0[k] = C0[(long long)n*9+k]; }
  mm3(f0, Rm, f);                                                    // lb_roll_to_h5: F · R
  float xx = x[j*3+0], xy = x[j*3+1], xz = x[j*3+2];
  bool ok = isfinite(xx) && isfinite(xy) && isfinite(xz);
  #pragma unroll
  for (int k = 0; k < 9; k++) ok = ok && isfinite(f[k]);
  if (!ok) {                                                         // wm_render 는 빼고 그림 -> 카메라 위치에 두어 잘림 (같은 화면)
    P[n*3+0] = cp[0]; P[n*3+1] = cp[1]; P[n*3+2] = cp[2];
    for (int k = 0; k < 6; k++) cc[n*6+k] = (k == 0 || k == 3 || k == 5) ? 1e-8f : 0.f;
    col[n*3+0] = col[n*3+1] = col[n*3+2] = 0.f; return;
  }
  mm3(f, c0, T);
  #pragma unroll
  for (int r = 0; r < 3; r++)
    #pragma unroll
    for (int c = 0; c < 3; c++) Cv[r*3+c] = T[r*3+0]*f[c*3+0] + T[r*3+1]*f[c*3+1] + T[r*3+2]*f[c*3+2];
  float c6[6] = {Cv[0], Cv[1], Cv[2], Cv[4], Cv[5], Cv[8]};
  #pragma unroll
  for (int r = 0; r < 6; r++) { float s = 0.f; for (int k = 0; k < 6; k++) s += L6[r*6+k] * c6[k]; cc[n*6+r] = s; }
  float p[3];
  #pragma unroll
  for (int r = 0; r < 3; r++) p[r] = A[r*3+0]*xx + A[r*3+1]*xy + A[r*3+2]*xz + b0[r];
  P[n*3+0] = p[0]; P[n*3+1] = p[1]; P[n*3+2] = p[2];
  // 극분해 (배율 뉴턴): prec 0 = f64 10 회 고정 (polar_t 와 같은 식), 1 = f32 바뀜 ‖ΔX‖_F < 1e-6 이면 멈춤 (최대 12 회)
  float Rr[9];
  if (prec == 0) polar_n<double>(f, Rr, 10, 0.0); else polar_n<float>(f, Rr, 12, 1e-6f);
  float dx = p[0] - cp[0], dy = p[1] - cp[1], dz = p[2] - cp[2];
  float ux = Rr[0]*dx + Rr[1]*dy + Rr[2]*dz, uy = Rr[3]*dx + Rr[4]*dy + Rr[5]*dz, uz = Rr[6]*dx + Rr[7]*dy + Rr[8]*dz;
  float nr = sqrtf(ux*ux + uy*uy + uz*uz); ux /= nr; uy /= nr; uz /= nr;
  const float* s = shs + (long long)n * 48;                          // (16, 3)
  float xx2 = ux*ux, yy2 = uy*uy, zz2 = uz*uz, xyv = ux*uy, yzv = uy*uz, xzv = ux*uz;
  float bs[16];
  bs[0] = 0.28209479177387814f;
  bs[1] = -0.4886025119029199f * uy; bs[2] = 0.4886025119029199f * uz; bs[3] = -0.4886025119029199f * ux;
  bs[4] = 1.0925484305920792f * xyv; bs[5] = -1.0925484305920792f * yzv; bs[6] = 0.31539156525252005f * (2.f*zz2 - xx2 - yy2);
  bs[7] = -1.0925484305920792f * xzv; bs[8] = 0.5462742152960396f * (xx2 - yy2);
  bs[9] = -0.5900435899266435f * uy * (3.f*xx2 - yy2); bs[10] = 2.890611442640554f * xyv * uz; bs[11] = -0.4570457994644658f * uy * (4.f*zz2 - xx2 - yy2);
  bs[12] = 0.3731763325901154f * uz * (2.f*zz2 - 3.f*xx2 - 3.f*yy2); bs[13] = -0.4570457994644658f * ux * (4.f*zz2 - xx2 - yy2);
  bs[14] = 1.445305721320277f * uz * (xx2 - yy2); bs[15] = -0.5900435899266435f * ux * (xx2 - 3.f*yy2);
  for (int ch = 0; ch < 3; ch++) { float v = 0.f; for (int k = 0; k < 16; k++) v += bs[k] * s[k*3+ch]; v += 0.5f; col[n*3+ch] = v > 0.f ? v : 0.f; }
}
'''
_K = {}


def _kern(name):
    if name not in _K:
        import cupy as cp
        _K[name] = cp.RawModule(code=_KSRC, options=("--std=c++14",)).get_function(name)
    return _K[name]


def _launch(name, NG, args):
    """torch 텐서는 cupy 배열로 (복사 없이, __cuda_array_interface__), torch 현재 스트림에 띄움."""
    import cupy as cp
    aa = []
    for a in args:
        if torch.is_tensor(a):
            assert a.is_cuda and a.is_contiguous() and a.dtype in (torch.float32, torch.int64), (name, a.dtype, a.shape)
            aa.append(cp.asarray(a))
        else:
            aa.append(a)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        _kern(name)(((NG + 255) // 256,), (256,), tuple(aa))


def upd_pg1(G, s, gi):
    NG = G["NG"]
    if "bufP" not in G:
        G["bufP"] = torch.empty(NG, 3, device="cuda"); G["bufC"] = torch.empty(NG, 6, device="cuda")
        G["M"] = torch.cat([G["RT"].reshape(9), G["MEAN"].float().reshape(3), (1.0 / G["SO"]).float().reshape(1)]).contiguous()
    _launch("rr_pg", NG, (s.x, s.F, gi, G["C0R"], G["M"], np.int32(1 if G["ROT"] else 0), G["bufP"], G["bufC"], np.int32(NG)))
    return dict(P=G["bufP"], cc=G["bufC"], shs=G["SHS"], col=None, opa=G["OPA"])


def upd_gf1(G, s, gi, t):
    NG = G["NG"]
    if "bufP" not in G:
        G["bufP"] = torch.empty(NG, 3, device="cuda"); G["bufC"] = torch.empty(NG, 6, device="cuda"); G["bufK"] = torch.empty(NG, 3, device="cuda")
        G["CP"] = {}
    if t not in G["CP"]:
        G["CP"][t] = G["CAMS"][t][0].camera_center.float().reshape(3).contiguous().cuda()
    _launch("rr_gf", NG, (s.x, s.F, gi, G["C0"], G["Rm"], G["A"], G["b0"], G["L6"], G["SHS"], G["CP"][t], G["bufP"], G["bufC"], G["bufK"], np.int32(NG),
                                np.int32(1 if C.fast >= 2 else 0)))
    return dict(P=G["bufP"], cc=G["bufC"], shs=None, col=G["bufK"], opa=G["OPA"])


# ---------------------------------------------------------------- 프레임 훅
def frame(s):
    if C.kind == "pg":
        return _frame(s, lambda G, s_, gi, t: (upd_pg1 if C.fast else upd_pg0)(G, s_, gi), lambda G, U, t: ras_pg(G, U, t))
    return _frame(s, lambda G, s_, gi, t: (upd_gf1 if C.fast else upd_gf0)(G, s_, gi, t), lambda G, U, t: ras_gf(G, U, t))


def _img_u8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _frame(s, upd, ras):
    fi = ST["fi"]; ST["fi"] += 1; t = fi + 1                                            # h5 번호 (0 = 처음 상태)
    sim_ms = float(s.last_ms)
    with torch.no_grad():
        if not ST["ready"]:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            G = _setup_pg() if C.kind == "pg" else _setup_gf()
            if C.fast and C.kind == "gf":
                assert G["sh_deg"] == 3 and G["shs_shape"][1:] == [16, 3], ("합친 커널은 SH 3 차 (N,16,3) 만", G["sh_deg"], G["shs_shape"])
            ST["G"] = G; ST["ready"] = True
            ST["log"] = open(os.path.join(C.out, "rr.jsonl"), "w")
            meta = {}
            for k, v in G.items():                                                     # json 으로 쓸 수 있는 것만 (R 같은 텐서 목록은 뺌)
                try:
                    json.dumps(v); meta[k] = v
                except TypeError:
                    pass
            meta.update(kind=C.kind, shape=C.shape, fast=C.fast, setup_s=time.perf_counter() - t0, Np=int(s.x.shape[0]))
            json.dump(meta, open(os.path.join(C.out, "rr_meta.json"), "w"), ensure_ascii=False, indent=1)
            print(f"[절 CM54 rtrender1] {C.kind} {C.shape} 가우시안 {G['NG']} 해상도 {G['Wd']}x{G['H']} 갱신 방식 {C.fast} (준비 {meta['setup_s']:.1f} s)", flush=True)
        G = ST["G"]
        if ST.get("perm") is not s.perm or getattr(s, "det", False):                  # 원래 순서 앞 NG 개 = 가우시안 (비결정 판은 perm 을 처음 한 번 정함; 결정적 판은 프레임마다 다시)
            inv = torch.empty_like(s.perm); inv[s.perm] = torch.arange(s.perm.numel(), device=s.perm.device)
            ST["gi"] = inv[:G["NG"]].long().contiguous(); ST["perm"] = s.perm
        gi = ST["gi"]
        assert s.x.dtype == torch.float32 and s.F.dtype == torch.float32 and s.x.is_contiguous() and s.F.is_contiguous()
        torch.cuda.synchronize()
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        w0 = time.perf_counter()
        e0.record()
        U = upd(G, s, gi, t)
        e1.record()
        img = ras(G, U, t)
        e2.record()
        e2.synchronize()
        w1 = time.perf_counter()
        b, c = e0.elapsed_time(e1), e1.elapsed_time(e2)
        rec = dict(t=fi, a=sim_ms, b=b, c=c, tot=sim_ms + b + c, wall_bc=1e3 * (w1 - w0))
        if t in C.chk:                                                                 # 확인 (시간 밖)
            os.makedirs(os.path.join(C.out, "rrimg"), exist_ok=True)
            u8 = _img_u8(img)
            import imageio.v2 as imageio
            imageio.imwrite(os.path.join(C.out, "rrimg", f"{t:04d}.png"), u8)
            ck = dict(t=t)
            if C.fast:
                U0 = upd_pg0(G, s, gi) if C.kind == "pg" else upd_gf0(G, s, gi, t)
                img0 = ras(G, U0, t)
                if C.kind == "gf":
                    ok = U0["ok"]
                    ck.update(nok=int((~ok).sum()), dP=float((U["P"][ok] - U0["P"]).abs().max()), dC=float((U["cc"][ok] - U0["cc"]).abs().max()),
                              rC=float(((U["cc"][ok] - U0["cc"]).abs().max(1).values / U0["cc"].abs().max(1).values.clamp_min(1e-30)).max()),
                              dK=float((U["col"][ok] - U0["col"]).abs().max()))
                else:
                    ck.update(dP=float((U["P"] - U0["P"]).abs().max()), dC=float((U["cc"] - U0["cc"]).abs().max()),
                              rC=float(((U["cc"] - U0["cc"]).abs().max(1).values / U0["cc"].abs().max(1).values.clamp_min(1e-30)).max()))
                d = (img - img0).abs()
                u80 = _img_u8(img0)
                ck.update(img_max=float(d.max()), img_mean=float(d.mean()), u8_ndiff=int((u8 != u80).any(2).sum()), u8_max=int(np.abs(u8.astype(int) - u80).max()))
            if C.kind == "gf":                                                         # GPU 극분해 대 wm_render 의 CPU SVD U Vᵀ
                Fg = (s.F.reshape(-1, 9)[gi].reshape(-1, 3, 3) @ G["Rm"])
                okk = torch.isfinite(Fg).all(2).all(1)
                Uc, _, Vh = torch.linalg.svd(Fg[okk].cpu())
                rp = polar_t(Fg[okk]).cpu()
                dd = (rp - Uc @ Vh).abs().flatten(1).max(1).values
                ck.update(polar_max=float(dd.max()), polar_p999=float(torch.quantile(dd[::7].double(), 0.999)), polar_n1e3=int((dd > 1e-3).sum()),
                          detF_min=float(torch.linalg.det(Fg[okk].double()).min()))
            if ST["clog"] is None:
                ST["clog"] = open(os.path.join(C.out, "rr_chk.jsonl"), "w")
            ST["clog"].write(json.dumps(ck) + "\n"); ST["clog"].flush()
        ST["log"].write(json.dumps(rec) + "\n"); ST["log"].flush()


def install(kind, shape, traj, cfg, model, gf, fast, chk, out):
    C.kind, C.shape, C.traj, C.cfg, C.model, C.gf, C.fast, C.out = kind, shape, traj, cfg, model, gf or C.gf, int(fast), out
    C.chk = tuple(int(q) for q in chk.split(",") if q.strip()) if chk else ()
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX

    def outer(fn):
        def w(s, *a, **kw):
            ST["dep"] += 1
            try:
                r = fn(s, *a, **kw)
            finally:
                ST["dep"] -= 1
            if ST["dep"] == 0:
                frame(s)
            return r
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    print(f"[절 CM54 rtrender1] 훅 설치 {kind} {shape} 갱신 방식 {fast} 확인 프레임 {C.chk}", flush=True)
