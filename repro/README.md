# repro: R2 원자료 → 논문 표·그림

```bash
bash repro/fetch_raw.sh      # R2 의 작은 파일만 ~/.cache/fusedgs_raw 로 (약 50 MB, FETCH_N1=1 이면 +250 MB, FETCH_PNG=1 이면 빵 그림 PNG)
bash repro/make_tables.sh    # 원본 표 tex → 논문 repo results_all/tools/rebuild.sh → paper_tables.py → robust_bar.py
# 사본에서 시험: rsync -a --exclude .git <논문 repo>/ /tmp/pc/ && PAPER=/tmp/pc bash repro/make_tables.sh
```

- 원자료 위치는 `RAW` (기본 `~/.cache/fusedgs_raw`) 아래에 `r2:storage/result/anchorflow` 와 같은 경로로 둔다.
- `LB=$RAW/lbours` 를 쓰는 생성기: `exe/ffx_cm192_refresh.sh` 와 그 하위 스크립트, `exe/pareto_mpmbl_cells.py`. `LB` 가 없으면 FUSE `~/r2/result/anchorflow/lbours` 를 쓴다.
- 2026-11-21 확인: 논문 repo 사본에서 원본 표를 모두 지우고 돌렸다. 결과는 다음과 같다.
  - `cvpr2027/tab/*.tex` 여섯 개와 `results_all/*.tex`(파레토 표 각주 한 줄 빼고)는 지금 논문과 바이트 단위로 같다.
  - `figs.py` 가 찍는 `pareto_mean` 평균점과 `robust_bar` 값도 같다. PDF 는 생성 날짜만 다르다.

## 손 값·손 편집을 대신한 것

| 예전 | 지금 |
|---|---|
| ship i-PG 칸·사다리 줄을 `stab_final_1115.tex` 에서 문자열로 고침 | R2 `stabfinal/res/ship_plastic_lego/{an,extra}.json`·`vis/vis.json`(11-17 판, i-PG N7)을 그대로 읽는다. 주석은 `exe/stabfinal_notes.tex` 를 갱신했다. |
| 레고 PG N43·i-PG N8 이 `stabfinal_tab.py`·`stabfinal_rank.py` 의 `LEGO_OLD` 에 하드코딩 | `repro/data/stab_lego_pg_ipg_1112.json` 으로 옮기고 출처를 적었다. 원자료는 없다(아래). |
| `/tmp/sf_collect.sh`, `~/.cache/stabfinal_col`, `~/.cache/sfn1/n1.json` | `fetch_raw.sh` 와 `make_tables.sh` 1 단계가 맡는다. `exe/stabfinal_n1.py <res 폴더> <출력>` 은 로컬 파일을 읽는다. |
| `add_mpmbl.py` 의 `E=`·`DV=` 사전과 고정 평균 행 | `exe/pareto_mpmbl_cells.py` 가 `bl_mpm/pareto_mpmbl.json` 과 대표 폴더의 `phys.json` 에서 `pareto_mpmbl_cells.json` 을 만든다. i-PG wolf 위치는 `repro/data/pareto_mpmbl_sources.json` 에 있다. 평균 행은 `recompute_avg.py` 가 다시 계산한다. |
| 파레토 표의 Spring-Gaus bread 행 손 삽입 | R2 `final_pa/blx/pareto_blx.json` 에 그 행이 있다. `$^\ast$` 와 각주는 행 note 의 "시간 RTX 6000 Ada 단독" 에서 만든다(`ffx_pareto_final.py`·`ffx_phys_tab.py`). |
| phys·vis 표의 i-PG·Omni 행과 Spring-Gaus pillow2sofa 행 손 채움 | `ffx_phys_tab.py` 의 작업 이름 규칙에 `ipg_*`·`omni_*`·`*_nsg` 를 넣었다. `ffx_phys_merge.py` 는 `ipg_wolf` 를 `post_N25e5_fr` 에 잇는다. |
| `build_ablation.py` 의 `Q1MS` 하드코딩 | `GQJ_RUNS/<장면>_q1_base_time_r1/gq.json` 의 `ms` 를 읽는다. |
| 표현력 표에서 PhysTwin k=1 한 행만 남긴 손 편집, `results_all/rep_table.tex` 손 작성 | `make_rep_final_tex.py --pt best --table_out …` 로 만든다. wolf 열은 `--raw` 일 때 원 로그 네 곳에서 읽고, `repflow_results.tex` 와 대조한다. |

규칙 하나를 옛 손 값과 같게 맞췄다. i-PG·Omni 의 ms 는 정수 ms 로, 오차는 소수 둘째 자리로 반올림해 넣는다. 반올림 없이 넣으면 i-PG 평균 ms 가 61762.2 에서 61762.5 로 바뀐다.

## 여전히 재현 못 하는 것

- **레고 PG N43·i-PG N8 의 물리 정합성 값.** 11-12 반복 확정 세 실행의 원자료가 R2 에 없다. 클러스터 `/home/dkta/work/stab/lego` 에만 있었다.
  - 찾아본 곳: `lbours/stab`, `lbours/stabhex`, `lbours/k1new`, `stab_a5`, `_vastmirror/stab`.
  - 정수 스캔 1 회 기록(`stab_scan_table.json`)은 남아 있다. N_min 43·8 은 이 기록과 대조한다. 물리 값(0.61·0.44·0.73 등)은 3 회 평균이라 대조하지 못한다.
- **wolf 표현력 네 칸.** 서로 다른 실행(커밋·날짜가 다름)의 로그에서 값을 읽는다. 값은 R2 로그에 있다.
- **stab_final_1115 비고 차이(논문 숫자와 무관).**
  - N-오차 표 ship i-PG 줄에 손으로 넣었던 "발산" 칸은 원자료에 없어서 빈칸이 된다.
  - 순위 표의 ship NIQE 는 i-PG N7 판을 반영해 ×가 된다(예전 표는 갱신되지 않았다).
  - `pareto_final_1115_rf_choice_table.tex` 와 `fig/pareto_*` 는 Spring-Gaus bread 를 넣어 다시 만들어졌다(26 점에서 27 점).

## lb_render 판(sim_area 자르기)

- 합친 판 `exe/lb_render.py` 는 config 에 `sim_area` 가 있으면 모든 장면을 자른다. final_fix_lego 판에는 자르기가 없다(`LB_NO_SIMAREA=1` 이면 옛 판).
- 파레토·ablation 시각 지표에 쓴 렌더는 R2 `render.log` 의 `[대응]` 줄로 확인했다. 대상은 `final_fix/phys/*`, `final_fix/bl_mpm/*`, `final_pa/blx/*/g/*_dump` 이다.
  - pillow2sofa 렌더 49 개(우리·정답 MPM·VR-GS·PhysTwin·Spring-Gaus·i-PG·Omni) 모두 "가우시안 624324, 입자 앞 624324 개와 위치 차이 0" 이다.
  - 이것은 자른 판과 같은 결과다. 시뮬 입자가 곧 sim_area 안의 가우시안이라 자르기를 해도 바뀌지 않는다.
- 자르지 않아 대응이 틀린 렌더("가우시안 624337, 위치 차이 0.879")는 `gq_justify/runs` 의 pillow2sofa 스트레스 조건 34 개(CM194 전)뿐이다. 이 렌더는 논문 표·그림에 들어가지 않는다.
- 다른 장면(레고 297877·bread 139018·wolf 53606)은 두 판에서 같다.
- 따라서 지금 판으로 다시 렌더해도 파레토 시각 지표는 바뀌지 않는다고 본다. 다시 렌더해서 확인하지는 않았다.
