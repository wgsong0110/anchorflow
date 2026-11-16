# final_fix_sg (조정자 11-16 사용자 지시): pr2_grid 의 Spring-Gaus pillow2sofa 시간을 1 회만 -- r1 이 끝나 gpu0.lock 을 놓는 순간 이 프로세스가 잠금을 잡아
# r2 시작을 막고, r1 res.json 확인 뒤 pr2_grid (python -u exe/pr2_grid.py --out) 만 끝냄. 남은 r2·r3 는 /root/sg_rep23.sh 가 맨 뒤에.
import fcntl, glob, os, signal, subprocess, sys, time
G = "/dev/shm/dkta/lbours/paper_remeasure2/stab/g"; LOG = "/root/sg_rep1_cut.log"
def say(s):
    open(LOG, "a").write(time.strftime("[%F %T] ") + s + "\n")
say("대기: r1 시작")
while not glob.glob(f"{G}/springgaus_pillow2sofa_n*_r1/card.txt"):
    if not subprocess.run(["pgrep", "-f", "python -u exe/pr2_grid.py --out"], capture_output=True).stdout.strip():
        say("pr2_grid 가 r1 전에 없어짐 -- 아무것도 안 함"); sys.exit(1)
    time.sleep(20)
r1 = glob.glob(f"{G}/springgaus_pillow2sofa_n*_r1")[0]
say(f"r1 시작 확인 {r1} -> gpu0.lock 대기")
fd = open("/home/dkta/work/gpu0.lock", "w")
fcntl.flock(fd, fcntl.LOCK_EX)
say(f"잠금 잡음; r1 res.json {os.path.exists(r1 + '/res.json')}")
if not os.path.exists(r1 + "/res.json"):
    say("r1 res.json 없음 -- 잠금 놓고 끝 (pr2_grid 그대로)"); sys.exit(1)
r2 = r1[:-1] + "2"
pids = subprocess.run(["pgrep", "-f", "python -u exe/pr2_grid.py --out"], capture_output=True, text=True).stdout.split()
time.sleep(10)                                                    # collect() 가 grid.json 쓰기를 끝낼 여유 (pr2_grid 는 잠금 대기 중)
say(f"r2 폴더 있음 {os.path.exists(r2)}; pr2_grid pid {pids} 에 SIGTERM")
for p in pids:
    os.kill(int(p), signal.SIGTERM)
time.sleep(5)
say(f"끝; 남은 pr2_grid {subprocess.run(['pgrep','-f','python -u exe/pr2_grid.py --out'], capture_output=True, text=True).stdout.split()}")
