# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨 (앞에 JOB, ARGS, ENVX, FORCE, GPUS 를 붙여서 보냄)
# 명령을 백그라운드 프로세스로 띄우고 바로 돌아온다. 로컬 터미널이 끊겨도 VM 에서는 계속 돈다.
#   로그 : /content/logs/<JOB>.log   (끝나면 드라이브 runs/logs/ 로 복사)
#   종료 : /content/logs/<JOB>.done  (종료 코드)
#
# GPU 배치 (GPU 작업 = 1·2·4단계, probe. 3·5단계는 CPU 라 검사하지 않는다)
#   GPUS 를 주면 그 GPU 에 고정한다 (예: GPUS=1 → CUDA_VISIBLE_DEVICES=1, GPUS=0,1 → 한 작업이 두 장 사용).
#   안 주면 비어 있는 GPU 를 한 장 골라 거기에 고정한다 → GPU 가 여러 장이면 작업을 나란히 돌릴 수 있다.
#   빈 GPU 가 없으면 시작하지 않는다 (한 GPU 에 학습 두 개가 겹치면 OOM 위험). 알고 겹치려면 FORCE=1.
#   모델 하나는 GPU 한 장에 올라가므로 기본은 작업당 1장이다. 두 장에 모델을 쪼개려면 GPUS=0,1 (느려질 수 있음).
# ─────────────────────────────────────────────────────────────
import os, json, shlex, shutil, subprocess

GPU_SCRIPTS = {"s1_train_choice_ce.py", "s2_predict.py", "s4_rerun_grounding.py", "probe_batch.py"}

BASE = os.environ.get("COLAB_BASE", "/content")
if not os.path.exists(f"{BASE}/vqa_env.json"):
    raise SystemExit("VM 준비가 안 되어 있습니다 (코드·데이터 미업로드) → ./colab/colab_pipeline.sh prep")
env = {**os.environ, **json.load(open(f"{BASE}/vqa_env.json")), **ENVX}
log, done, pidf = (f"{BASE}/logs/{JOB}.{x}" for x in ("log", "done", "pid"))

if os.path.exists(pidf) and not os.path.exists(done):
    pid = int(open(pidf).read())
    if os.path.exists(f"/proc/{pid}"):
        raise SystemExit(f"[{JOB}] 이미 실행 중입니다 (pid {pid}). 끝나길 기다리거나 kill 후 다시 실행하세요.")

def smi(query):
    return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()


def gpu_usage():
    """[(번호, 사용 MiB, 전체 MiB), ...] — nvidia-smi 가 답을 못 하면 빈 목록"""
    out = []
    for line in smi("gpu=index,memory.used,memory.total"):
        try:
            i, u, t = (float(x) for x in line.split(","))
            out.append((int(i), u, t))
        except ValueError:
            pass
    return out


def busy_report(usage, busy):
    """어떤 프로세스가 GPU 를 잡고 있는지 (compute-apps 는 환경에 따라 비어 나올 수 있어 안내용)"""
    lines = [f"  GPU {i}: {u / 1024:.1f}GB / {t / 1024:.1f}GB" + ("  ← 사용 중" if i in busy else "  (빈 GPU)")
             for i, u, t in usage]
    for line in smi("compute-apps=pid,used_memory"):
        gpid, mem = (x.strip() for x in line.split(",", 1))
        try:
            pcmd = open(f"/proc/{gpid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            pcmd = "?"
        lines.append(f"  pid {gpid:>7s}  {mem} MiB  {pcmd[:100]}")
    return "\n".join(lines)


gpus = [g for g in str(globals().get("GPUS", "") or "").replace(" ", "").split(",") if g]
place = ""                                               # 이 작업이 쓸 GPU (CUDA_VISIBLE_DEVICES)
uses_gpu = any(name in a for a in ARGS for name in GPU_SCRIPTS)   # bash -c "… && …" 연결 명령도 인식
if uses_gpu and shutil.which("nvidia-smi"):
    usage = gpu_usage()
    busy = {i for i, u, t in usage if u > t * 0.1}       # 전체의 10% 넘게 쓰면 사용 중
    if gpus:                                             # 직접 지정한 GPU 에 고정
        clash = [g for g in gpus if int(g) in busy]
        if clash and not FORCE:
            raise SystemExit(f"[{JOB}] 지정한 GPU {','.join(clash)} 를 이미 쓰는 중입니다:\n" + busy_report(usage, busy)
                             + "\n → 빈 GPU 번호를 주거나(GPUS=…), 끝나길 기다리세요. 알고 겹치려면 FORCE=1")
        place = ",".join(gpus)
    elif usage:
        free = [str(i) for i, u, t in usage if i not in busy]
        if not free and not FORCE:
            raise SystemExit(f"[{JOB}] 빈 GPU 가 없어 시작하지 않습니다 (한 GPU 에서 겹치면 OOM 위험):\n"
                             + busy_report(usage, busy)
                             + "\n → 끝나길 기다리거나(wait 작업) 중단한 뒤(cancel 작업) 다시 실행. "
                               "알고 겹쳐 돌리려면 앞에 FORCE=1")
        place = free[0] if free else ""                  # 작업 하나당 GPU 한 장
if place:
    env["CUDA_VISIBLE_DEVICES"] = place                  # 이 작업만 이 GPU 를 본다

if os.path.exists(done):
    os.remove(done)

drive_logs = shlex.quote(os.path.join(env["VQA_RUN_ROOT"], "logs"))
cmd = " ".join(shlex.quote(a) for a in ARGS)
# 로그를 LOG_SYNC 초마다 드라이브로 복사한다. 예전에는 작업이 끝나야 복사돼서, 도는 동안 드라이브 사본이
# 직전 실행의 것으로 남아 있었다(끊기면 로그도 통째로 사라졌다). 진행 중에도 드라이브에서 볼 수 있게 한다.
LOG_SYNC = 300
script = (f"mkdir -p {drive_logs}; cd {BASE}/vqa_pipeline && {{ "
          f"( while sleep {LOG_SYNC}; do cp {log} {drive_logs}/ 2>/dev/null; done ) & copier=$!; "
          f"{cmd} > {log} 2>&1; code=$?; "
          f"kill $copier 2>/dev/null; cp {log} {drive_logs}/; echo $code > {done}; }}")
p = subprocess.Popen(["bash", "-c", script], env=env, start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with open(pidf, "w") as f:
    f.write(str(p.pid))
print(f"[{JOB}] 시작 pid={p.pid} (GPU {place or '지정 없음'}) : {cmd}")
