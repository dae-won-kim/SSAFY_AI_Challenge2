# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨 (앞에 FILES = {"이름": "<base64>", ...} 를 붙여서 보냄)
# 로컬 PC 에서 만든 결과 파일(앙상블 제출 파일 등)을 드라이브 results/ 와 VM 로컬 results/ 에 저장한다
# ─────────────────────────────────────────────────────────────
import os, json, base64

BASE = os.environ.get("COLAB_BASE", "/content")
env = json.load(open(f"{BASE}/vqa_env.json"))
dests = [os.path.join(env.get("VQA_DRIVE_DIR", "/content/drive/MyDrive/vqa"), "results"),
         env.get("VQA_RESULT_ROOT", f"{BASE}/results")]
for name, b64 in FILES.items():
    data = base64.b64decode(b64)
    for d in dests:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)
    print(f"__PUT__ {name} {len(data)} -> {dests[0]}")
