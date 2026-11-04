#!/bin/bash
# 절 CM3 공통 설정 = 절 CM2 (exe/fem_cm2_scn.sh) + 결과 불변 가속 --hspec --ncomp 를 공통 묶음 BA 에 (모든 씬 같은 글자; 정의 불변).
source exe/fem_cm2_scn.sh
BA="$BA --hspec --ncomp"
