# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨 (앞에 JOB, N 을 붙여서 보냄). 로그 마지막 N 줄과 상태를 출력
# 이 파일은 호출할 때마다 로컬에서 새로 보내므로, 고치면 sync 없이 바로 적용된다.
#
# 로그를 읽기 좋게 줄이는 두 가지:
#   1) tqdm 진행 표시가 여러 줄로 쌓인 것은 같은 막대끼리 마지막 상태만 남김
#   2) 라이브러리가 매 샘플마다 찍는 반복 경고는 접어서 "(같은 경고 N줄 생략)" 으로 표시
# ─────────────────────────────────────────────────────────────
import os

NOISE = ("[transformers] Kwargs passed to `processor.__call__`",   # 템플릿이 안 쓰는 인자(enable_thinking) → 무해
         "Special tokens have been added in the vocabulary",
         "`do_sample` is set to `False`")

BASE = os.environ.get("COLAB_BASE", "/content")
log, done = f"{BASE}/logs/{JOB}.log", f"{BASE}/logs/{JOB}.done"
if os.path.exists(log):
    lines = [l for l in open(log, errors="replace").read().replace("\r", "\n").splitlines() if l.strip()]
    out, skipped = [], 0

    def flush_noise():
        global skipped
        if skipped:
            out.append(f"   … (반복 경고 {skipped}줄 생략)")
            skipped = 0

    for l in lines:
        if any(n in l for n in NOISE):
            # 경고가 진행 막대 줄 끝에 붙어 오는 경우가 있어 앞부분만 살린다
            head = l
            for n in NOISE:
                head = head.split(n)[0]
            if head.strip():
                l = head.rstrip()
            else:
                skipped += 1
                continue
        flush_noise()
        if out and l.split(":")[0] == out[-1].split(":")[0] and "%|" in l and "%|" in out[-1]:
            out[-1] = l                  # 같은 tqdm 막대는 마지막 상태만
        else:
            out.append(l)
    flush_noise()
    print("\n".join(out[-N:]))
status = f"DONE {open(done).read().strip()}" if os.path.exists(done) else ("RUNNING" if os.path.exists(log) else "NOTFOUND")
print("__STATUS__", status)
