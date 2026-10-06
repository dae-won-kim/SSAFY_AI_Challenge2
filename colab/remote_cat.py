# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨 (앞에 PATHS [, MAX_BYTES] 를 붙여서 보냄). 작은 파일을 base64 로 출력 → 로컬에서 복원
#   "runs 기준 경로"        예) final/final_submission.csv
#   "@결과폴더 기준 경로"    예) @*__valid_pred.csv   (VQA_RESULT_ROOT = 실험 결과 사본 폴더)
#   글롭(*) 사용 가능. 큰 파일은 출력 채널이 감당하지 못하므로 __TOOBIG__ 으로 건너뛴다 (드라이브 사본을 직접 볼 것)
# ─────────────────────────────────────────────────────────────
import os, json, glob, base64

BASE = os.environ.get("COLAB_BASE", "/content")
env = json.load(open(f"{BASE}/vqa_env.json"))
run_root = env["VQA_RUN_ROOT"]
result_root = env.get("VQA_RESULT_ROOT", f"{BASE}/vqa_pipeline/results")
limit = globals().get("MAX_BYTES", 1_500_000)

for rel in PATHS:
    root = result_root if rel.startswith("@") else run_root
    pat = rel[1:] if rel.startswith("@") else rel
    path = pat if os.path.isabs(pat) else os.path.join(root, pat)
    hits = sorted(p for p in glob.glob(path, recursive=True) if os.path.isfile(p))   # ** 로 하위 폴더까지
    if not hits:
        print("__MISSING__", path)
        continue
    for p in hits:
        size = os.path.getsize(p)
        name = os.path.relpath(p, root)
        if size > limit:
            print("__TOOBIG__", name, size)
            continue
        print("__FILE__", name, base64.b64encode(open(p, "rb").read()).decode())
