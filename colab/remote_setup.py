# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨 (colab_pipeline.sh 가 앞에 DRIVE_DIR, DATA_ZIP, TRAIN_ZIP, CODE_B64 를 붙여 colab exec 로 보냄)
#  1) 코드 zip(base64) 을 /content/vqa_pipeline 에 풀기 (매번 최신 코드로 교체)
#  2) 드라이브의 데이터 zip 을 로컬 디스크로 복사해 풀기 (이미 풀려 있으면 건너뜀)
#     - DATA_ZIP  (data.zip)       원본 : train/dev/test csv + 이미지 → /content/data       (test·제출은 항상 여기)
#     - TRAIN_ZIP (Bigdata_v2.zip) 증강 : bigdata/train.csv + train/ → /content/data_aug   (학습·valid, config.DATA)
#       TRAIN_ZIP 을 비우면 증강 데이터를 준비하지 않는다 (config.DATA['train_source']='orig' 로 쓸 때)
#  3) 경로 환경 파일 /content/vqa_env.json 작성 → 이후 모든 작업이 읽음
#     (결과 사본은 VM 로컬 /content/results 와 드라이브 <DRIVE_DIR>/results 양쪽에 저장된다)
#  4) 패키지 호환 점검
#  5) 모델 가중치 : 드라이브 보관소 <DRIVE_DIR>/models/ → /content/models (앞에 PRESETS 를 붙여 보냄, model_store.py)
# ─────────────────────────────────────────────────────────────
import os, io, sys, glob, json, base64, shutil, zipfile, subprocess

BASE = os.environ.get("COLAB_BASE", "/content")
CODE_DIR, DATA_DIR, MODEL_ROOT = f"{BASE}/vqa_pipeline", f"{BASE}/data", f"{BASE}/models"
AUG_DIR = f"{BASE}/data_aug"
RUN_ROOT = os.path.join(DRIVE_DIR, "runs")          # 산출물은 드라이브 (세션이 끊겨도 남음)
RESULT_ROOT = f"{BASE}/results"                     # 실험 결과 사본 : VM 로컬 디스크 (같은 파일이 드라이브에도 저장됨)
TRAIN_ZIP = globals().get("TRAIN_ZIP", "")


def find_data_root(base):
    for root, dirs, files in os.walk(base):
        if "train.csv" in files:
            return root
        if root[len(base):].count(os.sep) >= 2:
            dirs[:] = []
    return None


def unzip_from_drive(name, dest, what):
    """드라이브의 zip 1개를 로컬 디스크로 복사해 dest 에 풀고 train.csv 가 있는 폴더를 반환
    (이미지 수천 장을 드라이브에서 직접 읽으면 느리다). 어떤 zip 을 풀었는지 dest/.zip_source 에 남겨,
    zip 이름·크기가 바뀌면 다시 푼다"""
    src = os.path.join(DRIVE_DIR, name)
    marker = os.path.join(dest, ".zip_source")
    root = find_data_root(dest) if os.path.isdir(dest) else None
    if not os.path.isfile(src):
        if root is not None:                         # 이미 풀어 둔 것이 있으면 그대로 쓴다
            print(f"  ⚠ 드라이브에 {src} 가 없어 이미 풀어 둔 {root} 를 그대로 씁니다")
            return root
        raise SystemExit(f"드라이브에 {what} zip 이 없습니다: {src}")
    stamp = f"{name} {os.path.getsize(src)}"
    if root is not None and os.path.isfile(marker) and open(marker).read().strip() == stamp:
        return root
    if root is not None and os.path.isfile(marker):  # 다른 zip 이 풀려 있음 → 새로 푼다
        print(f"{what} zip 이 바뀌어 다시 풉니다 ({open(marker).read().strip()} → {stamp})")
        shutil.rmtree(dest)
    elif root is not None:                           # 예전 방식으로 풀린 것 (표시 없음) → 그대로 쓰고 표시만 남김
        with open(marker, "w") as f:
            f.write(stamp)
        return root
    # 임시 폴더에 다 푼 뒤에만 dest 와 바꿔 끼운다 → 풀다 끊겨도 반쯤 풀린 폴더를 완성본으로 쓰는 일이 없다
    print(f"{what} 복사 중 …", src)
    tmp_zip, tmp_dir = f"{BASE}/{os.path.basename(dest)}.zip", dest + "_tmp"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    shutil.copy(src, tmp_zip)
    with zipfile.ZipFile(tmp_zip) as z:
        z.extractall(tmp_dir)
    os.remove(tmp_zip)
    if find_data_root(tmp_dir) is None:
        raise SystemExit(f"{what} 압축을 풀었지만 train.csv 를 찾지 못했습니다. zip 구조를 확인하세요: {src}")
    with open(os.path.join(tmp_dir, ".zip_source"), "w") as f:
        f.write(stamp)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.replace(tmp_dir, dest)
    return find_data_root(dest)


if not os.path.isdir(f"{BASE}/drive/MyDrive"):
    raise SystemExit("드라이브가 마운트되지 않았습니다 → colab drivemount -s <세션>")

# 0) 실행 중인 작업이 있으면 코드를 바꾸지 않는다 : 코드 폴더를 지우고 다시 풀면 도는 작업(bash 연결 명령)의 작업 폴더가
#    사라져 다음 단계(s2~s5)가 스크립트를 못 찾고 실패한다
running = []
for pidf in glob.glob(f"{BASE}/logs/*.pid"):
    job = os.path.basename(pidf)[:-4]
    try:
        pid = int(open(pidf).read().strip())
    except (OSError, ValueError):
        continue
    if not os.path.exists(f"{BASE}/logs/{job}.done") and os.path.exists(f"/proc/{pid}"):
        running.append(job)
if running:
    raise SystemExit(f"실행 중인 작업이 있어 코드·데이터를 바꾸지 않습니다: {running} → 끝난 뒤(또는 cancel 후) 다시 sync. "
                     f"실행 중인 작업은 영향 없음")

# 1) 코드
if os.path.isdir(CODE_DIR):
    shutil.rmtree(CODE_DIR)
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall(BASE)
print("코드 →", CODE_DIR, sorted(f for f in os.listdir(CODE_DIR) if f.endswith(".py")))

# 2) 데이터
root = unzip_from_drive(DATA_ZIP, DATA_DIR, "원본 데이터")
counts = {s: len(os.listdir(os.path.join(root, s))) for s in ("train", "dev", "test") if os.path.isdir(os.path.join(root, s))}
print("원본 데이터 →", root, "| 이미지 수", counts)
aug_root = None
if TRAIN_ZIP:
    aug_root = unzip_from_drive(TRAIN_ZIP, AUG_DIR, "증강 데이터")
    n_img = len(os.listdir(os.path.join(aug_root, "train"))) if os.path.isdir(os.path.join(aug_root, "train")) else 0
    print("증강 데이터 →", aug_root, "| 이미지 수", n_img)
else:
    print("증강 데이터 → 준비 안 함 (TRAIN_ZIP 비어 있음. config.DATA['train_source']='orig' 로 쓸 것)")

# 3) 환경 파일
env = {"VQA_DATA_ROOT": root, "VQA_MODEL_ROOT": MODEL_ROOT, "VQA_RUN_ROOT": RUN_ROOT,
       "VQA_RESULT_ROOT": RESULT_ROOT, "VQA_DRIVE_DIR": DRIVE_DIR,
       "VQA_MODEL_DRIVE_ROOT": os.path.join(DRIVE_DIR, "models"), "PYTHONUNBUFFERED": "1"}
if aug_root:
    env["VQA_TRAIN_ROOT"] = aug_root
os.makedirs(RUN_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(os.path.join(DRIVE_DIR, "results"), exist_ok=True)   # 결과 사본이 쌓이는 드라이브 폴더
os.makedirs(f"{BASE}/logs", exist_ok=True)
with open(f"{BASE}/vqa_env.json", "w") as f:
    json.dump(env, f)
print("환경 →", json.dumps(env, ensure_ascii=False))

# 4) Colab 기본 torchao 가 최신 peft 의 요구 버전보다 낮으면 LoRA 를 만들 때 ImportError 가 난다.
#    peft 는 torchao 가 없으면 검사를 건너뛰고, 이 파이프라인은 torchao 를 쓰지 않으므로(4bit 는 bitsandbytes) 제거한다.
#    (torchao 를 올리면 VM 의 torch 버전과 어긋날 수 있음) 판정은 peft 자신의 검사 함수로 한다.
chk = subprocess.run([sys.executable, "-c", "from peft.import_utils import is_torchao_available as f; f()"],
                     capture_output=True, text=True)
if "incompatible version of torchao" in chk.stderr:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
    print("패키지 → peft 와 맞지 않는 torchao 제거 (이 파이프라인은 torchao 를 쓰지 않음)")

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else "GPU 없음"
print(f"GPU → {gpu} | 로컬 디스크 여유 {shutil.disk_usage(BASE).free / 1e9:.0f}GB")

# 5) 모델 가중치 : 드라이브 <DRIVE_DIR>/models/ 에 있으면 로컬 디스크로 복사 (model_store.py).
#    없으면 첫 학습·추론 때 Hugging Face 에서 받아 드라이브에 저장하고, 다음 VM 부터 여기서 복사한다.
#    PRESETS 를 비우면 config.DEFAULT_PRESET, "all" 이면 드라이브에 있는 모델 전부. 이미 로컬에 있으면 바로 넘어간다.
#    커널을 오염시키지 않게 따로 프로세스로 돌리고, 실패해도 준비는 성공으로 둔다 (학습 때 ensure_model 이 다시 찾음)
presets = [p for p in str(globals().get("PRESETS", "") or "").replace(",", " ").split() if p]
r = subprocess.run([sys.executable, "model_store.py", "prefetch", *presets], cwd=CODE_DIR,
                   env={**os.environ, **env}, capture_output=True, text=True)
print((r.stdout.strip() + ("\n" + r.stderr.strip()[-800:] if r.returncode else "")).strip() or "[모델] (출력 없음)")
