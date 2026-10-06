# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨 (앞에 JOB 을 붙여서 보냄). 실행 중인 작업을 중단한다 (VM 은 유지)
#  작업 bash 의 하위(학습·추론 파이썬)에 SIGINT → 파이썬이 KeyboardInterrupt 로 끝나고,
#  작업 bash 가 종료 코드를 기록하고 로그를 드라이브로 복사한다.
#  20초 안에 끝나지 않으면 프로세스 그룹을 강제 종료하고 기록·복사를 직접 한다.
#  1단계 학습은 드라이브의 마지막 체크포인트가 남으므로 같은 명령으로 다시 이어갈 수 있다.
# ─────────────────────────────────────────────────────────────
import os, json, time, shutil, signal

BASE = os.environ.get("COLAB_BASE", "/content")
log, done, pidf = (f"{BASE}/logs/{JOB}.{x}" for x in ("log", "done", "pid"))
if not os.path.exists(pidf) or os.path.exists(done):
    raise SystemExit(f"[{JOB}] 실행 중인 작업이 없습니다")
pid = int(open(pidf).read())
if not os.path.exists(f"/proc/{pid}"):
    raise SystemExit(f"[{JOB}] 프로세스가 이미 없습니다")


def children(ppid):
    kids = []
    for d in filter(str.isdigit, os.listdir("/proc")):
        try:
            stat = open(f"/proc/{d}/stat").read()
        except OSError:
            continue
        if int(stat.rsplit(")", 1)[1].split()[1]) == ppid:      # "pid (comm) state ppid ..."
            kids.append(int(d))
    return kids


kids = children(pid)
for k in kids:
    os.kill(k, signal.SIGINT)
print(f"[{JOB}] 중단 신호 보냄 (pid {kids or pid})", flush=True)
for _ in range(20):
    if os.path.exists(done):
        print(f"[{JOB}] 중단됨 (종료코드 {open(done).read().strip()}, 로그는 드라이브 runs/logs/ 에 복사됨)")
        break
    time.sleep(1)
else:
    os.killpg(pid, signal.SIGKILL)                          # 작업 bash 가 프로세스 그룹 리더 (start_new_session)
    with open(done, "w") as f:
        f.write("취소(강제)")
    drive_logs = os.path.join(json.load(open(f"{BASE}/vqa_env.json"))["VQA_RUN_ROOT"], "logs")
    os.makedirs(drive_logs, exist_ok=True)
    shutil.copy(log, drive_logs)
    print(f"[{JOB}] 20초 안에 끝나지 않아 강제 종료함")
