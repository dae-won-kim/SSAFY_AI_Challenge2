# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨. GPU 사용량, GPU 를 쓰는 프로세스, 작업 목록
# ─────────────────────────────────────────────────────────────
import os, glob, shutil, subprocess

BASE = os.environ.get("COLAB_BASE", "/content")
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                          "--format=csv"], capture_output=True, text=True).stdout)
    # 어떤 프로세스가 GPU 메모리를 잡고 있는지 (OOM 이 나면 여기서 먼저 확인)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    print("GPU 를 쓰는 프로세스:" if apps else "GPU 를 쓰는 프로세스: 없음")
    for line in apps:
        pid, mem = (x.strip() for x in line.split(",", 1))
        try:
            mem = f"{float(mem) / 1024:5.1f}GB"
        except ValueError:
            pass
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = "?"
        print(f"  pid {pid:>7s}  {mem}  {cmd[:110]}")
    print()

print("작업:")
for log in sorted(glob.glob(f"{BASE}/logs/*.log"), key=os.path.getmtime):
    job = os.path.basename(log)[:-4]
    done, pidf = log[:-4] + ".done", log[:-4] + ".pid"
    if os.path.exists(done):
        st = f"완료(종료코드 {open(done).read().strip()})"
    else:
        pid = open(pidf).read().strip() if os.path.exists(pidf) else "?"
        st = f"실행 중 (pid {pid}, 학습·추론 프로세스는 그 하위)" if os.path.exists(f"/proc/{pid}") else "중단됨 (프로세스 없음)"
    print(f"  {job:<12s} {st}")
