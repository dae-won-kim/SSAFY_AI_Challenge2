# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨 (앞에 ARGS 를 붙여서 보냄). VM 의 코드 폴더에서 model_store.py 를 따로 프로세스로 실행한다
#   ARGS = ["status"]                 드라이브 보관소·VM 로컬의 모델 목록
#   ARGS = ["push", "qwen3-vl-4b"]    VM 로컬에 받아 둔 모델을 드라이브에 저장
#   ARGS = ["prefetch", "all"]        드라이브에 있는 모델을 VM 로컬로 복사
# 이 파일은 호출할 때마다 로컬에서 새로 보내므로 고치면 sync 없이 바로 적용된다 (model_store.py 는 sync 필요)
# ─────────────────────────────────────────────────────────────
import os, sys, json, subprocess

BASE = os.environ.get("COLAB_BASE", "/content")
if not os.path.exists(f"{BASE}/vqa_env.json"):
    raise SystemExit("VM 준비가 안 되어 있습니다 (코드·데이터 미업로드) → ./colab/colab_pipeline.sh prep")
env = {**os.environ, **json.load(open(f"{BASE}/vqa_env.json"))}
r = subprocess.run([sys.executable, "model_store.py", *ARGS], cwd=f"{BASE}/vqa_pipeline", env=env,
                   capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode:
    print(r.stderr.strip()[-1500:])
