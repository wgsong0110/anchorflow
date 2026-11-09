#!/bin/bash
export PATH=/home/dkta/bin:$PATH
mkdir -p /dev/shm/dkta
rclone cat r2:storage/result/anchorflow/lbours/speed11/code/speed11_boot.sh | bash
