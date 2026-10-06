# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨 (앞에 RUNS 를 붙여서 보냄). colab_pipeline.sh fetch-best 가 PC 로 받을 best 어댑터를 준비한다.
#   best 어댑터 원본 : 드라이브 runs/<run>/adapter/ (1단계가 valid 최고 업데이트마다 교체, best.json 포함)
#   VM 로컬 사본     : /content/results/<실험이름>/adapter_best/ (1단계가 함께 저장)
# 로컬 사본이 없거나 드라이브 것과 다르면(best.json 비교) 드라이브에서 로컬로 복사한 뒤, 받을 파일 목록을 출력한다.
#   __BEST__ <run> <실험이름> <best.json base64>
#   __FILE__ <run> <원격 경로(맨 앞 / 없이, colab download 용)> <바이트> <md5>
#   __END__ <run> <파일 수>      (이 줄이 없으면 목록이 중간에 끊긴 것 → PC 쪽이 받지 않는다)
#   __NOBEST__ <run> <이유>
# PC 쪽(fetch_best.py)은 md5 로 "이미 받은 것과 같은지"와 "제대로 받았는지"를 판단한다
# ─────────────────────────────────────────────────────────────
import os, json, glob, base64, shutil, hashlib

BASE = os.environ.get("COLAB_BASE", "/content")
env = json.load(open(f"{BASE}/vqa_env.json"))
run_root, result_root = env["VQA_RUN_ROOT"], env.get("VQA_RESULT_ROOT", f"{BASE}/results")


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


runs = list(RUNS)
if runs == ["all"]:                                      # 어댑터가 있는 run 전부
    runs = sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                  for p in glob.glob(os.path.join(run_root, "*", "adapter", "adapter_model.safetensors")))
for run in runs:
    src = os.path.join(run_root, run, "adapter")
    if not os.path.isfile(os.path.join(src, "adapter_model.safetensors")):
        print("__NOBEST__", run, "어댑터_없음(1단계가_아직_best_를_저장하지_않음)")
        continue
    meta = read_json(os.path.join(run_root, run, "run.json")) or {}
    best = read_json(os.path.join(src, "best.json")) or {}
    tag = best.get("exp_tag") or meta.get("exp_tag") or run
    local = os.path.join(result_root, tag, "adapter_best")
    files = [f for f in sorted(os.listdir(src)) if os.path.isfile(os.path.join(src, f))]
    same = all(os.path.isfile(os.path.join(local, f)) and md5(os.path.join(local, f)) == md5(os.path.join(src, f))
               for f in files)
    if not same:                                         # 드라이브 → VM 로컬 (다운로드는 로컬 디스크에서 읽는 편이 빠르고 안전)
        tmp = local + "_tmp"
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp)
        if os.path.isdir(local):
            shutil.rmtree(local)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        os.replace(tmp, local)
    print("__BEST__", run, tag, base64.b64encode(json.dumps(best, ensure_ascii=False).encode()).decode())
    for f in files:
        p = os.path.join(local, f)
        print("__FILE__", run, p.lstrip("/"), os.path.getsize(p), md5(p))
    print("__END__", run, len(files), flush=True)          # 목록이 끝까지 왔다는 표시 (중간에 끊긴 목록으로 받지 않게)
