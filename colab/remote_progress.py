# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨 (앞에 JOBS 를 붙여서 보냄). GPU 사용량 한 줄 + 작업별 진행률 한 줄씩
#   학습 %(문항·업데이트·남은 시간, 최근 valid) | test 예측 % | 지금 하는 일 | 제출 파일
# 작업 로그(/content/logs/<작업>.log)의 진행 막대(tqdm)와 단계 완료 줄을 읽는다.
#   JOBS = []          실행 중인 작업 전부 (없으면 가장 최근 작업)
#   JOBS = ["slot1"]   그 작업만 (sweep 작업은 지금 돌고 있는 실험 이름을 함께 표시)
# 이 파일은 호출할 때마다 로컬에서 새로 보내므로 고치면 sync 없이 바로 적용된다.
# ─────────────────────────────────────────────────────────────
import os, re, glob, shutil, subprocess

BASE = os.environ.get("COLAB_BASE", "/content")
BAR = re.compile(r"(?P<desc>[^:|]+?):\s+(?P<pct>\d+)%\|[^|]*\|\s*(?P<n>\d+)/(?P<tot>\d+)\s*"
                 r"\[(?P<el>[^<\]]*)<(?P<rem>[^,\]]*)(?P<rest>[^\]]*)\]")
UPD = re.compile(r"upd=(\d+)/(\d+)")
# 1단계가 20업데이트마다 찍는 줄 : "  e1 upd 380/1136 loss 0.2964 lr 7.8e-05 | 전체 남은 약 35분 | peak 23.7GB"
# 진행 막대는 30초마다만 기록되므로 이 줄이 더 최근일 때가 많다
UPDLINE = re.compile(r"^\s*e(\d+) upd (\d+)/(\d+) loss .*?전체 남은 약 (\d+)분")


def gpu_line():
    if not shutil.which("nvidia-smi"):
        return "GPU 없음"
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    rows = []
    for line in out.splitlines():
        name, used, total, util = (x.strip() for x in line.split(","))
        rows.append(f"{name.replace('NVIDIA ', '')} {used}/{total}MiB 사용률 {util}%")
    return " · ".join(rows) or "GPU 정보 없음"


def rem(s):
    s = s.strip()
    return "" if s in ("", "?") else f", 남은 {s}"


def summarize(job):
    log, done = f"{BASE}/logs/{job}.log", f"{BASE}/logs/{job}.done"
    if not os.path.exists(log):
        return f"{job}: 로그 없음"
    with open(log, "rb") as f:                            # 끝부분만 (긴 작업도 빠르게)
        f.seek(max(0, os.path.getsize(log) - 600_000))
        text = f.read().decode("utf-8", errors="replace")
    st = {"exp": None}

    def reset():
        st.update(train=None, train_done=False, best=None, test=None, test_done=False,
                  now=None, sub=False, final=False, rerun=None)

    reset()
    for line in text.replace("\r", "\n").splitlines():
        m = re.search(r"===== 실험 (\S+) 시작", line)
        if m:                                             # sweep : 한 작업이 실험 여러 개를 이어서 돈다
            st["exp"] = m.group(1)
            reset()
            continue
        u = UPDLINE.search(line)
        if u:
            done_u, total_u = int(u.group(2)), int(u.group(3))
            st["train"] = {"pct": 100 * done_u // max(1, total_u), "upd": (done_u, total_u), "rem": f"약 {u.group(4)}분"}
            continue
        b = BAR.search(line)
        if b:
            d, pct = b["desc"].strip(), int(b["pct"])
            info = {"pct": pct, "n": b["n"], "tot": b["tot"], "rem": b["rem"], "rest": b["rest"]}
            if d.startswith("epoch "):
                upd = UPD.search(b["rest"])
                info["upd"] = (int(upd.group(1)), int(upd.group(2))) if upd else None
                info["pct"] = 100 * info["upd"][0] // max(1, info["upd"][1]) if info["upd"] else pct
                st["train"] = info
            elif d.endswith(" test") and not d.startswith(("2패스", "추론", "프롬프트")):
                st["test"] = info
            elif d.startswith(("2패스", "1패스", "프롬프트", "추론")):
                st["rerun"] = (d, pct)
            if not d.startswith("Loading"):
                st["now"] = f"{d} {pct}%"
            continue
        if line.startswith("[평가] upd"):
            u_ev = re.search(r"\[평가\] upd (\d+)/", line)
            acc = re.search(r"valid acc ([\d.]+)", line)
            prev = re.search(r"best upd (\d+) ([\d.]+)", line)
            if "▲ best" in line and u_ev and acc:
                st["best"] = (u_ev.group(1), acc.group(1))
            elif prev:
                st["best"] = (prev.group(1), prev.group(2))
        elif line.startswith("[완료] best 업데이트") or line.startswith("[완료] valid 평가 없음"):
            st["train_done"] = True
            m = re.search(r"best 업데이트 (\d+)/\d+ .*?valid acc ([\d.]+)", line)
            if m:
                st["best"] = (m.group(1), m.group(2))
        elif re.match(r"\[test\] \d+문항", line):
            st["test_done"] = True
        elif "submission.csv" in line and "저장" in line and "final" not in line:
            st["sub"] = True
        elif "final_submission.csv" in line and "저장" in line:
            st["final"] = True

    t = st["train"]
    if st["train_done"]:
        train = "✔"
    elif t:
        train = f"{t['pct']}% (" + (f"upd {t['upd'][0]}/{t['upd'][1]}" if t.get("upd") else f"{t['n']}/{t['tot']}문항") \
            + f"{rem(t['rem'])})"
    else:
        train = "대기"
    if st["best"] and st["best"][1]:
        train += f" · best upd {st['best'][0]} valid {st['best'][1]}"
    x = st["test"]
    test = "✔" if st["test_done"] else (f"{x['pct']}% ({x['n']}/{x['tot']}{rem(x['rem'])})" if x else "대기")
    files = ("제출 submission ✔" if st["sub"] else "") + (" · final ✔" if st["final"] else "")
    if os.path.exists(done):
        code = open(done).read().strip()
        now = "끝남 ✔" if code == "0" else f"끝남 (종료코드 {code} — logs 로 확인)"
    else:
        now = st["now"] or "준비 중"
    name = f"{job}" + (f" [{st['exp']}]" if st["exp"] and st["exp"] != job else "")
    return f"{name}: 학습 {train} | test 예측 {test} | 지금 {now}" + (f" | {files.strip(' ·')}" if files else "")


jobs = [j for j in (JOBS or []) if j]
if not jobs:
    logs = sorted(glob.glob(f"{BASE}/logs/*.log"), key=os.path.getmtime)
    names = [os.path.basename(p)[:-4] for p in logs]
    running = []
    for j in names:
        pidf = f"{BASE}/logs/{j}.pid"
        try:
            pid = open(pidf).read().strip()
        except OSError:
            continue
        if not os.path.exists(f"{BASE}/logs/{j}.done") and os.path.exists(f"/proc/{pid}"):
            running.append(j)
    jobs = running or names[-1:]
print(gpu_line())
for j in jobs:
    print("  " + summarize(j))
if not jobs:
    print("  작업 없음")
