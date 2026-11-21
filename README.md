# FusedGS (코드 이름 anchorflow)

3DGS 위에서 도는 실시간 물리 시뮬레이터. 매 프레임 현재 입자(가우시안) 상태에서 격자 기반 육면체 메시를 다시 만들고, 그 메시를 FEM 한 스텝(K=1)으로 진행한 뒤 결과를 입자로 되돌린다. MPM 의 반복 P2G/G2P·서브스텝이 없고, 메시를 매번 새로 만들어서 찢김 같은 위상 변화가 따로 처리 없이 나온다.

- 논문: `paper/cvpr2027/` (repo `wgsong0110/anchorflow-paper`, 폴더 `/home/wgsong/workspace/FusedGS/paper`)
- 코드: 이 repo (`wgsong0110/anchorflow`). 폴더 `/home/wgsong/workspace/FusedGS/master` 와 worktree `*_wt`
- 결과: R2 `r2:storage/result/anchorflow/lbours/` (FUSE `~/r2/result/anchorflow/lbours/`)
- 데이터: R2 `r2:storage/datasets/anchorflow/`, PG 계열 자산은 `lbours/_vastmirror/`

> **논문 결과용 코드는 브랜치 `paper` 에 있다.** 실험별 끝 브랜치 8개를 11-21 에 합친 브랜치다. master 는 11-02 에서 멈췄고 이 README 만 갖고 있다.
>
> ```bash
> git checkout paper
> bash repro/fetch_raw.sh     # R2 원자료(작은 파일만, 약 50 MB) → ~/.cache/fusedgs_raw
> bash repro/make_tables.sh   # 원자료 → 논문 repo 의 cvpr2027/tab/*.tex, fig/{pareto_mean,robust_bar}.pdf
> ```
>
> - 이 명령으로 논문 표·그림의 숫자가 지금 논문과 바이트 단위로 같게 나온다(11-21 확인).
> - 집계 안에는 손으로 넣은 값이 없다. 자세한 내용은 `repro/README.md` 에 있다.

## 1. 실험별 원래 브랜치 (모두 `paper` 에 합쳐짐)

| 논문 결과 | 브랜치 (worktree) | 끝 커밋 |
|---|---|---|
| 표 `stab_avg` 안정성 | `stabfinal` (`sfin_wt`) | d2a99e82 |
| 그림 `pareto_mean` 속도-정확도, 표 `abl_design`, 그림 `robust_bar` | `final_fix_lego` (`ffl_wt`, `final_pa`·`final_fix_wolfN` 포함) | e5ae450b |
| 파레토 i-PG·OmniPhysGS | `final_fix_mpmbl` (`mpmbl_wt`) | 3e34a1ef |
| 파레토·빵 찢김의 가속 베이스라인(PhysTwin·Spring-Gaus·VR-GS 소성 확장) | `final_pa_bl` (`fpabl_wt`), `final_fix_vrgs` (`vrgs_wt`) | e20a92e7, 1f29037a |
| ablation 1³ 시간 | `gq_justify` (`gqj_wt`) | 77cac21d |
| 표 `rep_l2` 표현력 | `repfinal` (`repfinal_wt`) | b2c6d5a0 |
| 새 형상(ship·mic·drums·chair) 정답 MPM | `nshape` (`nshape_wt`) | 0b99445f |

- 갈래 구조: master → k1new → nshape(0b99445f) → 위 끝 브랜치 8개.
- master 에만 있는 34 커밋(절 AM, 수박)은 끝 브랜치에 들어 있지 않다.
- `paper` 는 final_fix_lego 위에 mpmbl → vrgs → final_fix → final_pa_bl → repfinal → stabfinal → gq_justify 를 `--no-ff` 로 합쳤다. 원래 커밋 해시는 그대로다.
- 충돌은 각 브랜치가 자기 플래그로 돌 때 동작이 그대로 남도록 합집합으로 풀었다. 풀이는 merge 메시지에 적었다.
  - VR-GS 접촉은 `--vg_contact auto` 가 재질로 가른다. 탄성이면 final_fix_vrgs 판, 소성이면 final_pa_bl 판이다.
  - `pr2_grid` 의 VR-GS 반복 수는 장면별 기본값을 쓴다(wolf·bread 11, 나머지 1).
- 확인은 정적으로만 했다(실행 안 함): R2 에 기록된 명령줄 253개의 플래그가 모두 합친 코드에 있다.
- 실험별 명령 생성기는 그대로 남아 있다. 실험마다 "기본 판"이 다르기 때문이다(3절).

## 2. 환경

- **CUDA 커널:** 전부 cupy `RawModule` 런타임 컴파일이라 따로 빌드하지 않는다.
  - 핵심 코드: `lib/anchorflow/fem_an_rt.py`(시뮬레이터), `fem_mat_rt.py`·`fem_ao_rt.py`(재질), 실행기 `exe/fem_an_rt_sim.py`.
- **렌더 휠:** `diff_gaussian_rasterization`, `simple_knn`. GitHub 릴리스 `cuda-3dgs-torch251` 에 있고, sm80/90 판은 브랜치 `cuda-sm80-wheel` 에 있다.
- **vast.ai(최종 실행 대부분):**
  - 이미지 `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`.
  - 부팅 때 `pip install taichi==1.6.0 cupy-cuda12x kaolin==0.18.0 warp-lang==1.6.2` 와 휠 2개를 설치한다. 그 밖의 패키지는 버전이 고정돼 있지 않다.
  - 코드는 R2 의 `code/<접두>_<커밋>.tar` 로 받는다. 부팅 스크립트 예: `exe/vast_ffx.sh`, `exe/vast_rep4.sh`.
- **클러스터:** conda 환경 `/home/dkta/.conda/envs/af`.
- **시간 측정:** RTX A6000 단독.
  - Ours: CUDA 그래프 재생 GPU 시간(CUDA 이벤트, `fem_an_rt.py`).
  - 대부분의 베이스라인: 동기화 뒤 벽시계 시간.

## 3. 공통 입력과 기본 판

### 장면

| 장면 | 3DGS | 물성 config | 우리 입력(.pt) / 정답 MPM |
|---|---|---|---|
| 레고 | `_vastmirror/pgmodel/lego_whitebg-trained` | `_vastmirror/lbdata/config.json` (jelly E1e6 ν0.3) | `vastpkg/L000.pt` / `_vastmirror/lbdata/test_low/L000` (lb_gen 21bc4e2, 시드 7001, 서브스텝 2.0008e-5) |
| pillow2sofa | PG 공식 | `assets/pillow2sofa_config.json` | `lbours/paper_remeasure2/scenes/pillow2sofa/` (`exe/pr2_scene_fetch.sh`) |
| bread | `pgmodel/bread-trained` | `bread_plastic_config.json` (PG tear_bread 형상, plasticine 항복 200) | `lbours/fem_ag_data/bread.pt` / `_vastmirror/topo/pg_bread_plastic` |
| wolf | `wolf_whitebg-trained` | PG `wolf_config.json` (sand) | `lbours/fem_ag_data/wolf.pt` / `_vastmirror/topo/pg_wolf_sand` |
| ship·mic·drums·chair | NeRF-synthetic 을 직접 학습(3DGS `train.py` 30k), `datasets/anchorflow/nshape/model/` | `exe/nshape_cfg/<장면>_lego.json` (`exe/plastfix_lego_cfg.py`) | `datasets/anchorflow/nshape/gt_pg/<장면>` (`exe/vast_plastfix_lego.sh`) |

- PG 정답을 `.pt` 로 바꿀 때는 `exe/pg_h5_to_pt.py` 를 쓴다.
- 새 형상 정답은 원본 PhysGaussian(8339ed6)이 600 초 안에 0 프레임으로 끝나서 쓰지 못했다. 대신 i-physgaussian 사본의 명시적 러너 `gs_simulation.py` 로 만들었다(`gt_pg/<장면>/RUNNER`).

### FusedGS 기본 판(실험마다 다르다)
- 모든 실행은 진단 감싸개 사슬(`ssfix_wrap → startpt_wrap → … → ext2_wrap → fem_an_rt_sim.py`)을 앞에 붙인 긴 명령이다. 각 R2 실행 폴더에 `cmd.txt`·`commit.txt` 가 남아 있다.
- 판의 변천(문서 `doc/fem_realtime.md` 절 CM189~196):
  - CM190: `exe/fpa_cmd.py <장면> def <반복> <out>` = `--cg 4 --wrx_col 0 --mlow 1e-3 --ue_loc 51` + `ssfix_wrap --ssf_bec 1 --ssf_split 3 --ssf_bigT 0.5 --ssf_tsum 1`.
  - CM191: CM190 에 `--fpi_el 1` 추가.
  - CM192: CM191 에 `--wrx_el 1` 추가. 명령은 `FFX_ND=2 python exe/ffx_cmd.py <장면> def.bs2 <반복> <out>` 로 만든다(레고·pillow2sofa 는 `.wel`).
  - CM193/196: `--be_sub 2`·gran. K>1 에서만 영향이 있다.
- 실험별로 쓴 판:
  - 안정성 레고·pillow2sofa: `_f5` = CM190.
  - 안정성 ship·mic: `_f7` = CM191 + `--ssf_cgk 0.5`(`exe/nshape_cmd.py`).
  - 파레토·ablation·강건성·빵 찢김: CM192.

## 4. 결과별 재현

표기: **원자료** = R2 위치, **실행** = 숫자를 만든 명령, **집계** = 원자료에서 논문 tex 까지.

### 4.1 표 `stab_avg` 안정성 (`stabfinal`)
- **원자료:** `lbours/stabfinal/res/{lego,pillow2sofa,ship_plastic_lego,mic_metal_lego}/`
  - `an.json`(`exe/stab1115_an.py`): 최소 N 과 관통·떨림·Chamfer.
  - `extra.json`(`exe/stab_extra.py`): ΔV·깜빡임.
  - `vis/vis.json`(`exe/stab_vis.py`): PG N1024 렌더 대비 PSNR·LPIPS.
  - 지표 정의: `lbours/final_fix/metrics_def.md`.
- **실행:** `bash exe/stabfinal_launch.sh new <offer> <TAG> '<장면:단계>' <커밋> 4` 로 띄우고, 사다리는 `exe/stab1115_drv.py` 가 돈다.
  - 사다리 방식: N 을 2 의 거듭제곱으로 올린 뒤 사이 정수를 아래부터 본다. 같은 N 이 3 회 모두 안정이면 확정한다(mic 은 1 회).
  - PG/i-PG: `exe/stab_mpm.py --method {pg|ipg} --h5 … --config … --substeps N`, i-physgaussian 사본에서 실행한다.
  - OmniPhysGS: `exe/stab_omni.py --repo OmniPhysGS(5ab915b0)`.
  - Ours: 템플릿 `exe/stab1115_base.txt`, 큐 `lbours/stabfinal/queue/ours.txt`.
  - ship i-PG: `AF_IPG_MATRESTORE=1`(8145a086)로 재질 상태 되돌림을 고친 판이고, N7 은 1 회 판정이다. 후처리는 `exe/ship_ipg_post.sh`.
- **집계:**
  1. 원자료를 `C=~/.cache/stabfinal_col` 로 받는다(예전 `/tmp/sf_collect.sh`, repo 밖).
  2. `sfin_wt` 에서 `python3 exe/stabfinal_rank.py $C $P && SF_OVERWRITE=1 python3 exe/stabfinal_tab.py $C $C/stabhex_an.json $P` 를 돌린다. 여기서 `P=paper 폴더` 이고, 결과는 `paper/stab_final_1115.tex` 다.
  3. 논문 repo 의 `results_all/tools/rebuild.sh` 와 `paper_tables.py` 를 차례로 돌린다(5절).
- **주의:**
  - 레고 PG N43·i-PG N8 은 11-12 실행 값이다(`repro/data/stab_lego_pg_ipg_1112.json`). 그 3 회 실행의 원자료는 R2 에 없고, N_min 만 스캔 기록과 대조된다.
  - pillow2sofa PG ΔV 는 판 a 를 뺀 값이다(`stabfinal_jdevfix.py`).
  - 안정 판정의 에너지 문턱은 (E−E0)/|E0| ≤ 1 이다.

### 4.2 그림 `pareto_mean` 속도-정확도 (`final_fix_lego` 외)
- **Ours 원자료:**
  - 레고·pillow2sofa: `lbours/final_fix/runs/{L000,pillow2sofa}_def.wel_r{1,2,3}` (289ad4a).
  - bread·wolf: `runs_wolfN/{bread,wolf}_def.bs2_r{1,2,3}` (39a7abe·035f2b5).
  - 물리·시각 지표용 덤프: `final_fix/phys/ours_*`.
- **베이스라인 원자료** (`lbours/` 아래):

  | 방법 | 위치 |
  |---|---|
  | GT MPM | `final_pa/bl/T1/g/mpm_L000_n833_r*`, `paper_remeasure2/final2/all.json` |
  | i-PG·Omni | `final_fix/bl_mpm/`, `exe/pareto_mpmbl.py` |
  | VR-GS | `final_fix/vrgs_*.json`, `final_pa/blx/A/g/vrgs_*` |
  | PhysTwin | `final_pa/bl/T1/g/phystwin_*`, `final_pa/blx` |
  | Spring-Gaus | `final_fix/sg_pillow2sofa.json`, `final_pa/blx/SGB` |

- **집계:**
  1. `exe/ffx_cm192_refresh.sh` 로 원본 표 세 개 `paper/pareto_final_1115_{table,phys,vis}.tex` 를 만든다. 이 스크립트는 다음을 돌린다.
     - `ffx_nd2_cache.sh` + `fpa_table.py`: 3 회 중앙값.
     - `ffx_phys_merge.py`·`ffx_phys.py`·`ffx_vis.py`: `lb_render.py` 렌더, pyiqa 0.1.16, RAFT, alex LPIPS.
  2. 논문 repo 의 `rebuild.sh` 가 `add_mpmbl.py → recompute_avg.py → figs.py` 를 돌려 `cvpr2027/fig/pareto_mean.pdf` 를 만든다.
- **주의:**
  - i-PG·Omni 칸은 `exe/pareto_mpmbl_cells.py` 가 R2 에서 만든다. i-PG wolf 는 오차를 `N25e5_fr` 에서, 시간을 `N25e5_frt3` 에서 가져온다(`repro/data/pareto_mpmbl_sources.json`).
  - VR-GS 반복 수가 장면마다 다르다(레고·pillow2sofa 1, bread·wolf 11).
  - Spring-Gaus bread 는 RTX 6000 Ada 에서 쟀다.
  - wolf Omni 는 Corotated 판(`omni_wolf_corot`)이다.

### 4.3 표 `abl_design` (`final_fix_lego`, `gq_justify`)
- **원자료:**
  - 2³ 기본: 파레토 Ours 와 같은 실행.
  - 1³ 오차: `runs/*_q1.wel_r*`, `runs_wolfN/*_q1.bs2_r*` (`q27_wrap --q1pt 1`).
  - F 전달 교환: `*_nofpi.*`.
  - 영 에너지 모드: `final_fix/abl_eig/<장면>_<판>/eig.json` (`exe/q27_eig.py`).
- **집계:** `ffl_wt/exe/fpa_table.py` → `paper/ablation_final_1115_{tables,metrics}.tex` → 논문 repo `rebuild.sh`(`build_ablation.py` 등) → `paper_tables.py`.
- **주의:**
  - 1³ ms 17.9 는 `lbours/gq_justify/runs/*_q1_base_time_r1/gq.json` 1 회 측정이다. 같은 실행의 2³ 시간은 13.0 ms 다.
  - 1³ 영 에너지 모드 수는 하한값이다(bread 의 고윳값이 수렴하지 않음).

### 4.4 그림 `robust_bar` (`final_fix_lego`)
- **값:** `cvpr2027/tab/abl_robust.tex` 의 Err.% 열(4.3 과 같은 집계 사슬)을 `results_all/tools/robust_bar.py` 로 그린다.
- **원자료:**
  - 칸·가우스점: `final_fix/runs*` 의 셀·GQ 판, 3 회 중앙값.
  - N 1..128: 50 프레임 1 회. wolf N>1 은 `runs_wolfN/wolf_ssN.bs2.gran_r1`(29496e17)이다.
- **주의:** N 막대와 다른 두 묶음은 기준이 달라서, 같은 기본 판인데도 N=1 은 2.21, 칸×1 은 3.25 로 나온다.

### 4.5 그림 `bread_tear` (`final_fix_lego`, `final_pa_bl`)
- **렌더 PNG:** 원본 위치와 받는 명령은 `results_all/tools/bread_tear.py` 머리 주석에 있다.
  - 정답: `final_fix/phys/mpm_bread`.
  - Ours: `final_fix/phys/ours_bread_def_bs2` (`ffx_phys_drv.sh`, 39a7abe).
  - PhysTwin·VR-GS: `final_pa/blx/A/g/*_dump`.
  - Spring-Gaus: `final_pa/blx/SGB/g/*_dump`.
- **합성:** `python3 results_all/tools/bread_tear.py <PNG 폴더>` 를 돌린 뒤 `cp fig/bread_tear_strip_1116.png cvpr2027/fig/bread_tear.png` 한다.
- **주의:** Spring-Gaus 덤프는 det F=1 로 고정돼 있어, 그 행의 가우시안은 변형 없이 렌더됐다.

### 4.6 표 `rep_l2` 표현력 (`repfinal`)
- **bread·drums·chair:**
  - 실행: `bash exe/rep_final_launch.sh <offer> <태그> '<장면:판 …>' <커밋>`(b2c753b·2372fb0).
  - 원자료: `lbours/rep_final1115/<장면>/<판>/<판>_<장면>.log` 의 `RMSE` 줄.
  - 판별 명령(모두 `exe/rep_track_sg.py --opt riem`):
    - Ours: `--method hex --riem_energy logbarrier --riem_lr 0.5 --hex_dx 0.02`.
    - PhysTwin: `--method phystwin --riem_k 1 --pt_realtime --riem_lr 10`.
    - VR-GS: `--method vrgsx --riem_lr 1`.
    - Spring-Gaus: `--method springgaus --riem_lr 100`.
  - 입력: `exe/rep_prep_gf.py --h5_dir <정답 h5>` 로 만든 `rep_final1115/inputs/<장면>/`.
- **wolf 열:** `paper/repflow_results.tex` 의 값을 복사한 것이다. 네 칸이 서로 다른 실행에서 왔다.

  | 방법 | 원자료 | 코드·커밋 |
  |---|---|---|
  | Ours | `lbours/repflow_hex/main/wolf_hex.log` | `exe/rep_track_hex.py`, ff365f0 |
  | PhysTwin | `archive_20261009/r10/wolf_phystwin_k1` | `exe/rep_track2.py`, 224443c |
  | VR-GS | `lbours/repflow_vrgs/main/wolf_vrgsx.log` | 8942d80 |
  | Spring-Gaus | `lbours/repflow1115/sg/main/wolf/sg_wolf.log` | 47e462b |

- **집계:** `exe/make_rep_final_tex.py --pt best --table_out …` 로 `rep_final_1115.tex`·`results_all/rep_table.tex` 를 만든다(PhysTwin 은 k=1 한 행). 그다음 `paper_tables.py` 를 돌린다.
- **주의:** wolf 의 목표는 정답 MPM 이 아니라 가우시안 속도장(`exe/gauss_flow.py --pts`)이다.

## 5. 논문 표·그림 다시 만들기

원본 표 tex 까지 포함한 전체 과정은 `repro/make_tables.sh` 가 맡는다. 아래는 논문 repo 쪽 뒷단이다.

`paper/results_all/tools/` 의 스크립트는 자기 위치에서 논문 repo 뿌리를 찾는다. 다른 곳을 쓰려면 `PAPER=` 로 지정한다. 중간 파일은 `results_all/build/` 에 쓴다(git 에서 제외).

```bash
cd /home/wgsong/workspace/FusedGS/paper
bash results_all/tools/rebuild.sh            # 원본 표 → results_all/*_marked.tex, pareto_mean.pdf, 결과 모음 PDF
python3 results_all/tools/paper_tables.py    # → cvpr2027/tab/{stab_avg,rep_l2,abl_design,abl_robust,...}.tex
python3 results_all/tools/robust_bar.py      # → cvpr2027/fig/robust_bar.pdf
python3 results_all/tools/bread_tear.py <PNG 폴더>   # → fig/bread_tear_strip_1116.png
```

11-21 에 논문 repo 사본에서 위 명령을 다시 돌려 확인했다. `cvpr2027/tab/*.tex`·`results_all/*.tex` 는 지금 논문과 바이트 단위로 같았고, `bread_tear.png` 는 md5 가 같았다.

## 6. 처음부터 다시 돌릴 때 걸리는 것

1. 레고 PG·i-PG 의 안정성 물리 값은 원자료가 없다(4.1).
2. 실험마다 FusedGS 기본 판이 다르다(3절).
3. 반복 수와 GPU 가 칸마다 섞여 있다. 안정성 실행의 대부분은 A6000 이고, 일부는 A40·L40S·Ada 다.
4. i-physgaussian 사본과 vast pip 패키지는 버전이 고정돼 있지 않다.
5. `k1new_wt` 에 커밋 안 된 코드(CM159 `--fl_bnd`, CM188 `--aq_ctan`, 기본 꺼짐)가 있다.

과거 판(앵커 시뮬레이터·학습 stepper, 2026-07~09)에 대한 설명은 git 기록의 옛 README 에 있다.
