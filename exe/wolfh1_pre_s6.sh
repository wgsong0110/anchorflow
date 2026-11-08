#!/bin/bash
# 절 CM52: s4 는 /dev/shm 넘침으로 실패 -- 조정자가 /dev/shm 정리 (21 %), /tmp 는 쓰지 말 것 (11-08). 덤프는 옮기지 않고 그대로 두고 남은 자리만 기록, 실패한 vN_1 판 디렉토리만 다시 (판 스크립트가 summary 없으면 새로 함). 인자: <출력> <R2>
echo "/dev/shm $(df -h /dev/shm | tail -1)"; echo "/tmp $(df -h /tmp | tail -1) (안 씀)"
