"""
모든 단계가 공유하는 설정.

각 스크립트는 이 모듈을 **가장 먼저** import 한다.
Hugging Face 캐시 경로(HF_HOME)는 transformers / huggingface_hub 가 처음 import 될 때 한 번만 읽히므로,
그 전에 여기서 환경변수를 정해야 모델·캐시가 원하는 드라이브에 저장된다.

경로는 환경변수로 바꿀 수 있다.
  VQA_DATA_ROOT   train.csv / dev.csv / test.csv 와 train/ dev/ test/ 이미지 폴더가 있는 곳 (원본 data.zip)
  VQA_TRAIN_ROOT  증강 학습 데이터(Bigdata_v2.zip 의 bigdata/ : train.csv + train/)가 있는 곳.
                  DATA["train_source"]="aug" 일 때 학습·valid 를 여기서 읽는다 (test 는 항상 VQA_DATA_ROOT)
  VQA_MODEL_ROOT  모델 가중치와 HF 캐시를 저장할 곳 (D 드라이브에 연결된 경로 권장)
  VQA_RUN_ROOT    산출물(어댑터, 예측 CSV, 제출 파일)을 저장할 곳
  VQA_RESULT_ROOT 실험 결과 사본을 모을 로컬 폴더 (기본 이 저장소의 results/)
  VQA_DRIVE_DIR   마운트된 구글 드라이브의 프로젝트 폴더 (기본 /content/drive/MyDrive/vqa).
                  있으면 결과 사본을 <여기>/results 에도 똑같이 저장한다 (마운트 안 된 PC 에서는 자동으로 건너뜀)
  VQA_MODEL_DRIVE_ROOT  모델 가중치 드라이브 보관소 (기본 <VQA_DRIVE_DIR>/models, 드라이브가 마운트됐을 때만)
  VQA_MODEL_DRIVE_CACHE "0" 이면 HF 에서 받은 모델을 드라이브에 저장하지 않음
  VQA_ONLINE      "0" 이면 오프라인 모드 (로컬·드라이브에 있는 모델만 사용)
  VQA_DISABLE_FLA "1" 이면 linear attention 빠른 커널(flash-linear-attention)을 끄고 PyTorch 참고 구현 사용 (속도 비교용)

seed·epoch·lr·LoRA 랭크·프롬프트 sampling 비율·문턱값 같은 실험 설정은 아래 "실험 설정" 절에서 한 번에 관리한다.
"""
import os


def _find_data_root():
    for d in [os.environ.get("VQA_DATA_ROOT"), ".", "..", "/workspace"]:
        if d and os.path.isfile(os.path.join(d, "train.csv")):
            return os.path.abspath(d)
    return os.path.abspath(".")


def _find_aug_root():
    """증강 데이터 bigdata/ (train.csv + train/) 위치. 없으면 None"""
    for d in [os.environ.get("VQA_TRAIN_ROOT"), os.path.join(_find_data_root(), "bigdata"), "bigdata", "../bigdata"]:
        if d and os.path.isfile(os.path.join(d, "train.csv")):
            return os.path.abspath(d)
    return None


DATA_ROOT = _find_data_root()
AUG_ROOT = _find_aug_root()
MODEL_ROOT = os.path.abspath(os.environ.get("VQA_MODEL_ROOT", "/workspace/models"))
RUN_ROOT = os.path.abspath(os.environ.get("VQA_RUN_ROOT", os.path.join(DATA_ROOT, "runs")))
ONLINE = os.environ.get("VQA_ONLINE", "1") == "1"

# 실험 결과 사본을 모으는 곳 : 로컬 폴더 + (마운트되어 있으면) 구글 드라이브
# runs/ 는 파이프라인이 주고받는 작업 폴더고, 여기에는 실험 이름이 붙은 결과 파일만 모인다
RESULT_ROOT = os.path.abspath(os.environ.get(
    "VQA_RESULT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")))
DRIVE_DIR = os.environ.get("VQA_DRIVE_DIR", "/content/drive/MyDrive/vqa")
DRIVE_RESULT_ROOT = os.path.join(DRIVE_DIR, "results")
# 모델 가중치 드라이브 보관소 (model_store.py) : <DRIVE_DIR>/models/<모델이름>/. 드라이브가 마운트된 곳(Colab)에서만 쓴다.
# 새 VM 은 Hugging Face 대신 여기서 로컬 디스크(MODEL_ROOT)로 복사한다. HF 에서 받은 모델은 여기에도 저장 (MODEL_DRIVE_CACHE)
MODEL_DRIVE_ROOT = os.environ.get("VQA_MODEL_DRIVE_ROOT") or (
    os.path.join(DRIVE_DIR, "models") if os.path.isdir(os.path.dirname(DRIVE_DIR.rstrip("/"))) else None)
MODEL_DRIVE_CACHE = os.environ.get("VQA_MODEL_DRIVE_CACHE", "1") == "1"   # "0" 이면 드라이브에 저장하지 않음 (읽기는 함)

try:
    os.makedirs(MODEL_ROOT, exist_ok=True)
except OSError:
    pass
os.environ.setdefault("HF_HOME", os.path.join(MODEL_ROOT, "hf_home"))
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(MODEL_ROOT, "modelscope_cache"))
if not ONLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
if os.environ.get("VQA_DISABLE_FLA") == "1":
    import sys
    sys.modules["fla"] = None                    # import fla 가 실패 → transformers 가 PyTorch 구현으로 대체

# ── 모델 프리셋 ──────────────────────────────────────────────
# quant : None = bf16, "4bit" = NF4 (7B 이상은 16GB GPU 에서 필수)
# coord : grounding bbox 좌표계
#         "abs"      = processor 가 리사이즈한 입력 기준 픽셀 (Qwen2.5-VL)
#         "norm1000" = 0~1000 정규화 (Qwen3 계열)
MODEL_PRESETS = {
    "qwen3-vl-4b":   dict(repo="Qwen/Qwen3-VL-4B-Instruct", quant=None,   coord="norm1000"),
    "qwen3-vl-8b":   dict(repo="Qwen/Qwen3-VL-8B-Instruct", quant=None,   coord="norm1000"),
    "qwen3.5-4b":    dict(repo="Qwen/Qwen3.5-4B",           quant=None,   coord="norm1000"),
    "qwen3.5-9b-bf16": dict(repo="Qwen/Qwen3.5-9B",         quant=None,   coord="norm1000"),
    "qwen3.5-9b":    dict(repo="Qwen/Qwen3.5-9B",           quant="4bit", coord="norm1000"),
    "qwen2.5-vl-3b": dict(repo="Qwen/Qwen2.5-VL-3B-Instruct", quant=None, coord="abs"),
    "qwen2.5-vl-7b": dict(repo="Qwen/Qwen2.5-VL-7B-Instruct", quant="4bit", coord="abs"),
}

# ═════════════════════════════════════════════════════════════
# 실험 설정 — 모든 단계의 기본값은 여기서만 바꾼다
# 각 스크립트 인자(--lr, --lora-r 등)의 기본값이 이 값이다. 인자는 한 번 시험해 볼 때만 쓰고,
# 채택한 값은 여기에 반영한다 (Colab 은 ./colab/colab_pipeline.sh sync 후 적용).
# ═════════════════════════════════════════════════════════════
SEED = 42                          # 학습 셔플·프롬프트 sampling·데이터 추출·추론 공통
DEFAULT_PRESET = "qwen3.5-9b-bf16" # 기본 모델 (2026-09-26 부터 9B 기준. 4B→8B 가 Kaggle +2%p 였던 모델 크기 효과를 이어감).
                                   # 2·4단계는 run.json 에 기록된 프리셋을 따른다
RUN_NAME = "main"                  # 산출물 폴더 runs/<이름>. LoRA 랭크 등을 비교할 땐 --run r32 처럼 나눈다
VALID_MIN_VOTES = 3                # (valid_source="dev" 일 때만) dev 5개 답 중 3표 이상 합의한 문항만 사용
USE_VALID = True                   # valid 를 떼어 평가할지. 어디서 떼는지는 DATA["valid_source"]
                                   # True  = 1단계가 eval_every 업데이트마다 valid 로 평가해 가장 높은 업데이트의 어댑터를 쓴다.
                                   #         2단계는 valid 도 예측하고, 3~최종 단계는 valid 로 문턱값·교체 방법을 고른다
                                   # False = valid 를 떼지 않고 학습 데이터 전부로 학습 (평가 없음 → 마지막 어댑터 사용).
                                   #         valid 로 학습 길이를 정한 뒤 전체 데이터로 다시 학습할 때 쓴다

# ── 학습·검증 데이터 ──
# 증강 데이터(Bigdata_v2.zip → bigdata/train.csv, 1만 문항)의 구성 (파일 이름으로 구분):
#   orig_train_NNNN.jpg           원본 train 과 동일 (6,714)
#   rot_train_NNNN_rot{L,R}30.jpg 원본 NNNN 을 ±30° 회전 (2,685 = 원본 2,427장, 질문·선지·정답은 원본과 동일)
#   auto_dev_NNNN.jpg             dev 문항 + 자동 라벨 (601, dev 5개 답과 3표 일치 524 / 1표 이하 77 → 라벨 노이즈)
# valid 는 **이미지 그룹** 단위로 뗀다 : 원본과 그 회전본, 바이트가 같은 사진(다른 id 로 중복된 것)은 한 그룹.
# valid 그룹의 회전본·중복 사진은 학습에서도 빠진다 (같은 사진을 보고 외운 답으로 valid 를 맞히는 누수 방지).
DATA = dict(
    train_source="aug",            # aug = 증강 bigdata/ (VQA_TRAIN_ROOT) | orig = data.zip 의 train.csv
    valid_source="holdout",        # holdout = 학습 데이터에서 이미지 그룹 단위로 떼어냄 (사람 라벨, 권장)
                                   # dev     = dev.csv 다수결 (라벨 신뢰도 낮음 — 2026-09-23 확인). 이때 auto_dev 는 학습에서 뺀다
    valid_frac=0.10,               # holdout 비율 (valid 후보 그룹 중). 0.10 → valid 671문항 (그룹 이름 해시 순, qtype 규칙과 무관)
    valid_kinds=["orig"],          # valid 에 넣을 종류 : 무회전 원본만 (test 와 같은 분포, 라벨 깨끗)
    train_kinds=["orig", "rot", "auto_dev"],   # 학습에 쓸 종류 (auto_dev 를 빼려면 ["orig", "rot"])
    split_seed=42,                 # valid 분할 seed. SEED 와 따로 둔다 → 실험 seed 를 바꿔도 valid 문항은 그대로
)

# 해상도
MIN_PIXELS = 256 * 28 * 28
PROC_MAX_PIXELS = 2048 * 2048      # 이미지는 직접 리사이즈하므로 processor 상한은 넉넉히
TRAIN_MAX_SIDE = 1024              # 학습 이미지 픽셀 상한 = side². 추론(1024)과 맞췄다 (2026-09-26, 이전 768).
                                   # 오답의 중심이 작은 글자 읽기라 학습도 같은 해상도로. 학습 시간·메모리는 약 1.5~2배
INFER_MAX_SIDE = 1024              # 추론 이미지 픽셀 상한 = side²

# ── LoRA (랭크 하나만 바꿔 가며 비교할 수 있게 따로 둔다) ──
# alpha=None 이면 alpha_ratio × r 로 자동 계산 → 랭크를 바꿔도 스케일(alpha/r)이 그대로다.
# 랭크를 바꾼 실험은 산출물이 섞이지 않게 --run 이름도 함께 바꾼다 (예: --run r32 --lora-r 32).
LORA = dict(
    r=16,                          # 랭크. 8 / 16 / 32 / 64 비교 대상
    alpha=None,                    # None = alpha_ratio × r
    alpha_ratio=2.0,
    dropout=0.05,
    targets="lm_all",              # LORA_TARGETS 의 키 또는 정규식 직접 지정
)
# 모듈 이름 (transformers 5.17 코드·체크포인트로 확인, 2026-09-26)
#   언어 모델  일반 어텐션 q/k/v/o_proj · MLP gate/up/down_proj
#              Qwen3.5 의 선형 어텐션(GatedDeltaNet, 9B 는 32층 중 24층) linear_attn.in_proj_qkv / in_proj_z / out_proj
#   비전 인코더 visual.blocks.N.attn.qkv / attn.proj · mlp.linear_fc1 / linear_fc2, 연결부 visual.merger.linear_fc1/2
#              (Qwen3-VL 은 visual.deepstack_merger_list.N.linear_fc1/2 도)
# 예전 attn+mlp 는 Qwen3.5 의 선형 어텐션 층을 놓쳤다 (토큰을 섞는 층의 3/4 에 LoRA 없음) → lm_all 이 기본.
# 예전 all+visual 은 비전 모듈 이름이 달라 실제로는 attn+mlp 와 같았다 → lm_all+vision 으로 바꿈
_LM = r"(?!.*(?:visual|mtp)).*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj{extra})"
_LM_LINEAR = "|in_proj_qkv|in_proj_z|out_proj"
_VISION = (r".*visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])"
           r"|.*visual\.(?:merger|deepstack_merger_list\.\d+)\.linear_fc[12]")
LORA_TARGETS = {
    "attn":          r"^(?!.*(?:visual|mtp)).*\.(?:q_proj|k_proj|v_proj|o_proj)$",
    "attn+mlp":      "^" + _LM.format(extra="") + "$",                   # 예전 기본값 (Qwen3-VL 에선 lm_all 과 같음)
    "lm_all":        "^" + _LM.format(extra=_LM_LINEAR) + "$",           # 언어 모델의 선형층 전부 (선형 어텐션 포함)
    "lm_all+vision": "^(?:" + _LM.format(extra=_LM_LINEAR) + "|" + _VISION + ")$",   # + 비전 인코더·연결부
}


def lora_spec(r=None, alpha=None, dropout=None, targets=None):
    """인자로 덮어쓴 값 + LORA 기본값 → peft LoraConfig 에 넘길 값 (r, alpha, dropout, targets 정규식)"""
    r = LORA["r"] if r is None else r
    alpha = (LORA["alpha"] if alpha is None else alpha)
    if alpha is None:
        alpha = int(round(LORA["alpha_ratio"] * r))
    targets = LORA["targets"] if targets is None else targets
    return dict(r=r, alpha=alpha, dropout=LORA["dropout"] if dropout is None else dropout,
                targets=LORA_TARGETS.get(targets, targets), targets_name=targets)


# ── 프롬프트 정책 (학습 augmentation / 추론 선택) ──
# 학습 : sample 마다 base prompt 를 균등 sampling 하고, hint_prob 확률로 qtype hint 를 덧붙인다.
#        데이터를 프롬프트 수만큼 복제하지 않는다 (같은 sample 이 epoch 마다 다른 표현으로 보인다).
# 추론 : sample 당 1회 forward. qtype 으로 프롬프트를 고르되 branch 로 나누지 않는다.
PROMPT = dict(
    base_ids=None,                 # None = model_utils.BASE_PROMPTS 전체를 균등 sampling
    hint_prob=0.30,                # qtype hint 를 붙일 확률 (나머지 70% 는 generic). valid 결과로 조정
    hint_qtypes=None,              # None = QTYPE_HINTS 에 정의된 유형 전체
    infer_mode="qtype",            # qtype(유형별 hint) | generic(hint 없음) | fixed(--prompt 로 지정)
    infer_base="p1",               # 추론에 쓸 base prompt (고정 — 추론 다양성은 쓰지 않는다)
)

# 1단계 : multi-prompt choice-CE LoRA 학습 (single adapter)
TRAIN = dict(
    epochs=1,                      # 9B 실험에서 업데이트 200 이후 정체 → 1 epoch 기준
    lr=1e-4,
    grad_accum=8,                  # 유효 배치 = 업데이트 1회가 보는 문항 수 (학습 결과를 좌우하는 값)
    micro_batch=4,                 # 한 번에 forward/backward 하는 문항 수 (≤ grad_accum). 속도만 좌우, 결과는 같다
                                   # (4B·768²·ckpt 끔 실측: 1→2.4 / 2→4.0(23.9GB) / 4→4.5(37.8GB) / 8→OOM)
                                   # 9B(19GB)·1024²·ckpt 켬 은 미실측 — 8B 는 ckpt·8 로 돌았으나 해상도가 올라 4 로 시작.
                                   # OOM 이면 그 마이크로배치만 반으로 쪼개 재시도한다
    grad_checkpointing=True,       # 9B 는 켠다 (끄면 A100 40GB 에서 1024² 학습이 어려움). 4B 는 끄는 편이 훨씬 빠름
    label_smoothing=0.0,
    eval_every=200,                # valid 평가 간격 (옵티마이저 업데이트 수). 매 평가가 best 후보이고, epoch 끝·학습 끝에도 평가한다.
                                   # 가장 높은 valid 정확도(동률이면 valid NLL 이 낮은 쪽)의 업데이트 → runs/<run>/adapter/
                                   # 4B 곡선은 500 업데이트 이후 평평 → 9B 는 평가가 느려 200 마다 (1 epoch 1,136 → 약 6회)
    eval_n=None,                   # 평가 문항 수 (None = valid 전체). 모든 평가가 같은 문항이어야 업데이트끼리 비교된다
    ckpt_every=50,                 # 이어하기용 체크포인트 간격
    eval_batch=16,                 # 평가 배치 (2단계와 같은 1장 입력). 4B 실측상 16 이상은 속도가 같다.
                                   # 학습 중 평가라 학습 메모리와 겹치므로 9B 는 16
)

# 2단계 : 단일 패스 추론 (sample 당 1 forward)
PREDICT = dict(
    splits=["valid", "test"],      # train 예측(암기·과적합 진단, 4B·A100 약 20분)은 기본에서 뺐다 (2026-09-25).
                                   # 제출·선별·최종 단계는 valid·test 만 쓴다. 필요할 때만 --splits valid test train
    batch_size=32,                 # 한 번에 forward 하는 입력 수 = 문항 수. OOM 이면 그 배치만 1개씩 재시도
                                   # A100 40GB·4B 실측(최악 입력): 8→5.5 / 16→5.7 / 32→5.7 / 80→5.8 문항/초.
                                   # 16 이상은 처리량이 포화(계산 병목)라 32 면 충분하다. 최대는 80(34.2GB)
)

# 3단계 : 재추론 대상 선별
SELECT = dict(
    conf_th=0.60,
    margin_th=0.30,
    weak_conf_th=0.90,
    weak_qtypes=None,              # None 이면 valid 에서 자동 선정 (전체보다 weak_delta 이상 낮은 유형)
    weak_delta=0.05,
    weak_min_n=20,
    max_frac=0.30,                 # 선별 비율 상한 (4단계 비용 통제)
)

# 4단계 : 선별 문항만 재추론
RERUN = dict(
    mode="grounding",              # grounding(2패스 확대) | prompt(다른 표현 프롬프트 평균) | reason(생각을 쓰고 답하기)
    pass2="adapter",               # grounding 2패스를 adapter 로 풀지 base 로 풀지
    batch_size=16,                 # 이미지 2장 입력. A100 40GB·4B 실측: 16 이상 3.8 문항/초로 포화, 32 → 27.2GB (최대 48)
                                   # 9B 는 2장·32 에서 36.9GB 라 여유를 두고 16 (처리량은 16 에서 이미 포화)
    # mode=reason : 선별 문항만 모델이 먼저 추론을 글로 쓰고(<think>…</think>) 그 뒤 a/b/c/d 확률을 읽는다.
    #   thinking 을 지원하는 모델(Qwen3.5)만. 학습한 LoRA 는 "바로 답하기"만 배워 긴 추론이 약할 수 있어 기본은 LoRA 끔
    reason_adapter=False,          # True 면 학습한 LoRA 를 켠 채로 추론 (valid 로 비교해 볼 것)
    reason_max_new_tokens=768,     # 생각 길이 상한. 넘으면 거기서 끊고 답을 읽는다 (길수록 느림)
    reason_batch=16,               # 생성 배치. 9B 선형 어텐션은 캐시가 작아 배치를 키워도 메모리 여유가 있음 (미실측)
    prompt_ids=None,               # mode=prompt 일 때 쓸 base prompt 목록 (None = BASE_PROMPTS 전체)
    prompt_hint=True,              # mode=prompt 일 때 qtype hint 를 붙인 버전도 포함
    ground_max_side=768,           # 1패스 (bbox)
    ground_batch=24,               # bbox 생성(문장 생성)은 배치 효과가 크다. 4B 실측 4→1.2 / 32→3.7 / 48→3.9 / 96→4.1 문항/초
                                   # (4B 48 에서 24.1GB). 9B 는 미실측이라 24 로 낮춤 — OOM 이면 1개씩 재시도라 매우 느려짐
    max_new_tokens=64,
    crop_pad=0.15,                 # 크롭
    crop_min_frac=0.15,
    crop_max_area=0.70,
    full_max_side=768,             # 2패스 : 전체 사진 (문맥용)
    crop_max_side=1024,            # 2패스 : 확대 사진 (읽기용)
    crop_min_side=512,             # 작은 크롭은 이 크기까지 확대
)

# 최종 : 필요한 행만 교체
FINAL = dict(
    method="auto",                 # auto | keep | replace | rule | blend
                                   # auto = 재추론 방식(grounding·reason …) × 교체 방법을 valid 로 비교해 가장 나은 조합.
                                   #        keep 보다 min_gain 이상 나아야 채택 (선별 문항이 적어 우연한 차이를 막기 위함)
    min_gain=0.003,                # valid 671문항 기준 약 2문항
    fallback="rule",               # valid 가 없을 때 auto 대신 쓸 방법 (첫 번째 재추론 방식에)
    conf_delta=0.0,
    weight=0.5,
)


def run_dir(*parts):
    """RUN_ROOT 아래 경로를 만들고 반환"""
    path = os.path.join(RUN_ROOT, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def train_root(source=None):
    """학습(·holdout valid) 데이터 폴더. aug 인데 증강 데이터가 없으면 바로 알려 준다"""
    source = source or DATA["train_source"]
    if source == "orig":
        return DATA_ROOT
    if source != "aug":
        raise ValueError(f"DATA['train_source'] 는 aug | orig : {source!r}")
    if AUG_ROOT is None:
        raise FileNotFoundError(
            "증강 데이터(bigdata/train.csv)를 찾지 못했습니다. VQA_TRAIN_ROOT 를 지정하거나 "
            "Colab 은 ./colab/colab_pipeline.sh sync 로 드라이브의 Bigdata_v2.zip 을 풀어 두세요 "
            "(원본 train 으로 돌리려면 config.DATA['train_source']='orig')")
    return AUG_ROOT


# ═════════════════════════════════════════════════════════════
# 실험 결과 저장 — 같은 파일을 로컬 results/ 와 마운트된 드라이브에 함께 남긴다
# ═════════════════════════════════════════════════════════════
def result_dirs():
    """결과를 저장할 폴더 목록. 드라이브는 마운트되어 있을 때만 (로컬 PC 에서는 자동으로 빠짐)"""
    dirs = [RESULT_ROOT]
    if os.path.isdir(os.path.dirname(DRIVE_DIR.rstrip("/"))):    # 예: /content/drive/MyDrive 가 있으면 마운트됨
        dirs.append(DRIVE_RESULT_ROOT)
    return dirs


def _fmt(v):
    """파일 이름에 쓸 수 있게 값 줄이기 (1e-4 → 1e-4, 0.3 → 0.3)"""
    if isinstance(v, float):
        s = f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e") if (v and abs(v) < 0.01) else f"{v:g}"
    else:
        s = str(v)
    return "".join(c if (c.isalnum() or c in "-.+") else "-" for c in s)


def exp_tag(run, preset, *, lora_r=None, epochs=None, lr=None, hint_prob=None, seed=None, zero_shot=False,
            data=None, extra=None):
    """결과 파일 이름 앞에 붙는 실험 이름. 중요한 하이퍼파라미터를 담아 파일만 봐도 설정을 알 수 있게 한다
    예) main_qwen3-vl-4b_r16_e1_lr1e-4_hint30_aug_s42__valid_pred.csv
    data : 학습 데이터 출처 (DATA["train_source"]). 원본(orig)은 예전 이름과 같게 표시하지 않는다"""
    parts = [_fmt(run), _fmt(preset)]
    if zero_shot:
        parts.append("zeroshot")
    else:
        if lora_r is not None:
            parts.append(f"r{_fmt(lora_r)}")
        if epochs is not None:
            parts.append(f"e{_fmt(epochs)}")
        if lr is not None:
            parts.append(f"lr{_fmt(float(lr))}")
        if hint_prob is not None:
            parts.append(f"hint{round(float(hint_prob) * 100):d}")
        if data and data != "orig":
            parts.append(_fmt(data))
    if seed is not None:
        parts.append(f"s{_fmt(seed)}")
    if extra:
        parts.append(_fmt(extra))
    return "_".join(p for p in parts if p)
