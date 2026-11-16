<div align="center">

# 🏯 흥선대원군의 AI 척화비

**AI Challenge · Team A185** · 2026.09.23 발표

이미지 + 질문 + 4지선다(a~d)로 구성된 **이미지 기반 객관식 질의응답(VQA)** 과제에 대해<br/>
데이터 정제 → 회전 증강 → LoRA 학습 → 로짓 스코어링 추론 → 저신뢰 샘플 Crop 2-pass 재추론으로 이어지는 파이프라인을 구축했습니다.

</div>

---

## 📑 목차

- [팀원 소개](#-팀원-소개)
- [개발 환경](#-개발-환경)
- [Ⅰ. 솔루션 코드 설명](#-ⅰ-솔루션-코드-설명)
- [Ⅱ. 실행 방법](#-ⅱ-실행-방법)
- [Ⅲ. 접근 전략](#-ⅲ-접근-전략)
- [Ⅳ. 실험 설계](#-ⅳ-실험-설계)
- [요약](#-요약)

---

## 👥 팀원 소개

| **김대원** | **민수영** | **강민준** | **박서영** |
| 팀장 | 팀원 | 팀원 | 팀원 |
| AI 성능 최적화, EDA | AI 구조 파이프라인 설계 | 데이터 전처리 및 증강, EDA | AI 성능 향상, 데이터 EDA |

---

## 🛠 개발 환경

| 항목 | 내용 |
| --- | --- |
| Base Model | **Qwen3.5-4B** (`Qwen/Qwen3.5-4B`, 프리셋 `qwen3.5-4b`, bf16) |
| 파인튜닝 기법 | **LoRA** (PEFT, 언어 모델 부분에만 적용) |
| 학습·추론 환경 | Google **Colab** (GPU 런타임) / 로컬 **NVIDIA GeForce RTX 5060 Ti** |
| 주요 라이브러리 | `transformers>=5.16`, `peft`, `accelerate`, `flash-linear-attention`, `huggingface_hub`, `pillow`, `pandas`, `numpy`, `tqdm` |
| 선택 라이브러리 | `bitsandbytes` (4bit 프리셋 사용 시), `modelscope` (HF 다운로드가 막힐 때) |
| 데이터 분석 보조 | Claude (해석이 어려운 문항 탐지) |

<details>
<summary><b>지원 모델 프리셋 (<code>config.MODEL_PRESETS</code>)</b></summary>

| 프리셋 | Hugging Face repo | 정밀도 | grounding 좌표계 |
| --- | --- | :---: | :---: |
| `qwen2.5-vl-3b` | Qwen/Qwen2.5-VL-3B-Instruct | bf16 | abs |
| `qwen2.5-vl-7b` | Qwen/Qwen2.5-VL-7B-Instruct | 4bit | abs |
| `qwen3-vl-8b` | Qwen/Qwen3-VL-8B-Instruct | 4bit | norm1000 |
| **`qwen3.5-4b`** | **Qwen/Qwen3.5-4B** | **bf16** | **norm1000** |
| `qwen3.5-9b` | Qwen/Qwen3.5-9B | 4bit | norm1000 |
| `qwen3.5-9b-bf16` | Qwen/Qwen3.5-9B | bf16 | norm1000 |

4bit 프리셋은 NF4 + double quant를 사용하며, **비전 인코더(`visual`)와 출력층(`lm_head`)은 양자화하지 않습니다.**

</details>

---

## 🧩 Ⅰ. 솔루션 코드 설명

> **한 줄 요약**
> VLM 하나에 **LoRA 어댑터 하나**만 학습하고, 모든 문항을 **forward 1회의 a/b/c/d 로짓 스코어링**으로 푼 뒤,
> **불확실한 문항만 골라** 비싼 재추론(Grounding 크롭 · Thinking)을 적용하고, **valid에서 이득이 확인될 때만** 답을 교체하는
> **5단계 모듈형 파이프라인**입니다.

### 핵심 설계 한눈에 보기

| # | 설계 결정 | 이유 | 구현 위치 |
| :---: | --- | --- | --- |
| 1 | **생성 대신 로짓 스코어링** | 답 파싱 실패 0건, 문항당 forward 1회, 확률을 남겨 선별·앙상블에 재활용 | `model_utils.VLM.score` |
| 2 | **학습 목표 = 추론 방식** | 추론과 같은 프롬프트 끝의 a/b/c/d 4개 로짓에만 Cross-Entropy | `VLM.choice_loss` |
| 3 | **프롬프트 다양성은 학습 증강으로** | 프롬프트 4종 균등 샘플링 + 30% qtype 힌트 + 선지 순서 셔플 → 추론은 프롬프트 1개로 고정 | `sample_prompt` / `select_prompt` |
| 4 | **비싼 연산은 불확실한 문항에만** | 정답을 보지 않는 신호(확신도·margin·약한 유형)로 최대 30%만 선별 | `s3_select_rerun.py` |
| 5 | **교체는 valid로 증명될 때만** | 재추론 방식 × 교체 방법을 valid에서 비교, keep보다 +0.3%p 이상일 때만 채택 | `s5_build_final.py` |
| 6 | **누수 없는 검증셋** | 원본·회전본·중복 사진을 **이미지 그룹**으로 묶어 그룹 단위 holdout + 지문(hash) 검증 | `data_utils.load_data` |
| 7 | **끊겨도 이어지는 학습** | 원자적 폴더 교체 + RNG까지 저장하는 체크포인트 + 데이터 지문 검사 | `s1_train_choice_ce.py` |
| 8 | **CPU / GPU 단계 분리** | 선별·최종 단계는 torch 없이 몇 초 만에 재실행 → 문턱값 튜닝 비용 최소화 | `data_utils` (CPU 전용) |
| 9 | **설정 한 곳 관리** | seed·LoRA·lr·문턱값을 `config.py` 하나에서 관리, CLI 인자는 일회성 실험용 | `config.py` |
| 10 | **원격 실험 자동화** | 로컬 터미널에서 Colab VM 할당·실행·감시·회수, VM 2대 병렬 sweep | `colab/` |

---

### 1. 전체 아키텍처

```mermaid
flowchart LR
    subgraph DATA["0. 데이터 구축 (data-preprocessing-augmentation/)"]
        RAW[원본 train 6,714] --> ROT[±30° 회전 2,685]
        DEV[dev 자동 라벨 601]
        RAW & ROT & DEV --> BIG[Bigdata_v2<br/>10,000문항]
    end

    BIG --> S1
    subgraph GPU1["GPU"]
        S1["s1 학습<br/>choice-CE LoRA"] -->|adapter/| S2["s2 1-pass 추론<br/>로짓 스코어링"]
    end
    S2 -->|"*_pred.csv<br/>p_a..p_d · conf · margin"| S3
    subgraph CPU1["CPU"]
        S3["s3 재추론 대상 선별"]
    end
    S3 -->|"*_subset.csv"| S4
    subgraph GPU2["GPU"]
        S4["s4 선별 문항 재추론<br/>grounding · reason · prompt"]
    end
    S4 -->|"*_rerun.csv"| S5
    S2 -->|submission.csv<br/>기준 제출| OUT
    subgraph CPU2["CPU"]
        S5["s5 최종 병합<br/>valid로 교체 방법 선택"]
    end
    S5 --> OUT[final_submission.csv]
```

단계끼리는 **표준 형식의 CSV 파일로만** 주고받습니다(`id, qtype, gold, p_a..p_d, pred, conf, margin`).
그래서 어느 단계든 단독으로 다시 돌릴 수 있고, 새로운 재추론 방법도 같은 형식의 `*_rerun.csv` 만 만들면 최종 단계에서 그대로 비교됩니다.

---

### 2. 디렉토리 구조

```text
SSAFY_AI_Challenge2/
├── config.py                  # 경로·모델 프리셋·실험 설정 (모든 스크립트가 가장 먼저 import)
├── data_utils.py              # [CPU] 데이터 로드·분할, qtype 분류, 확률 요약, 크롭 좌표, 제출 점검, 결과 저장
├── model_utils.py             # [GPU] 모델/LoRA 로드, 프롬프트 pool, 로짓 스코어링, grounding, thinking 추론
├── model_store.py             # 모델 가중치 드라이브 보관소 (Colab VM마다 HF에서 다시 받지 않도록)
│
├── s1_train_choice_ce.py      # 1단계: multi-prompt choice-CE LoRA 학습 (이어하기 지원)
├── s2_predict.py              # 2단계: 문항당 forward 1회 추론 → submission.csv
├── s3_select_rerun.py         # 3단계: 재추론 대상 선별 (CPU)
├── s4_rerun_grounding.py      # 4단계: 선별 문항 재추론 (grounding / reason / prompt)
├── s5_build_final.py          # 최종: valid 비교 후 필요한 행만 교체 (CPU)
│
├── ensemble.py                # 따로 학습한 어댑터들의 확률 평균/다수결 (CPU)
├── probe_batch.py             # 최악 입력 기준 GPU 배치 상한 측정 → config 배치값에 반영
├── collect_results.py         # 실험별 기록을 합쳐 비교표(summary.csv) 생성
├── run_pipeline.sh            # 1~최종 단계 순차 실행 (GPU 서버용)
│
├── colab/                     # 로컬 터미널 → Colab VM 원격 실행 도구
│   ├── colab_pipeline.sh      #   VM 할당·코드/데이터 동기화·단계 실행·감시·결과 회수
│   ├── sweep.sh / sweep.conf  #   VM 여러 대로 실험 병렬 실행
│   ├── cli_session.py         #   Colab CLI 접속 토큰 만료 문제 우회
│   └── remote_*.py            #   VM 안에서 실행되는 보조 스크립트 (작업 실행·상태·진행률·평가 조회 등)
│
└── data-preprocessing-augmentation/   # 학습 데이터 구축 파이프라인 (별도 패키지)
    ├── data_pipeline/         #   rotation · degradation · quality · merge · validation · preview
    └── scripts/               #   01_회전 증강 → 02_열화 증강 → 03_통합(Bigdata_v2)
```

---

### 3. 모듈 계층 설계

> **결론:** 의존 방향을 `config → data_utils → model_utils → 단계 스크립트` 한 방향으로 고정하고, **torch를 쓰는 코드와 쓰지 않는 코드를 파일 단위로 분리**했습니다.

```text
           config.py            ← 경로·HF 캐시 환경변수를 transformers import 전에 설정
               │
          data_utils.py         ← CPU 전용 (pandas/numpy만). torch를 import하지 않음
           │         │
   model_utils.py    │          ← GPU 전용 (torch, transformers, peft)
     │       │       │
  s1 s2 s4 probe   s3 s5 ensemble collect_results
   (GPU 단계)          (CPU 단계: GPU 없이 수 초 만에 재실행)
```

| 계층 | 책임 | 설계 포인트 |
| --- | --- | --- |
| `config.py` | 경로·프리셋·**실험 설정 전부** | `HF_HOME` 은 transformers가 처음 import될 때 한 번만 읽히므로 **반드시 가장 먼저 import**. 데이터 위치는 환경변수 → 현재/상위 폴더 순으로 자동 탐색 |
| `data_utils.py` | 데이터·평가·파일 입출력 | 3단계·최종 단계·앙상블이 이 모듈만 쓰므로 **GPU 없는 PC에서도 실행** 가능 |
| `model_utils.py` | 모델·프롬프트·스코어링 | **프롬프트 책임을 한 파일에 모음** — 학습과 추론이 같은 `build_messages()` 를 거치고 프롬프트를 고르는 *정책*만 다름 |
| `s1`~`s5` | 단계별 진입점 | 각 단계는 앞 단계의 파일(`run.json`, `*_pred.csv`)을 읽어 프리셋·실험 이름·데이터 분할을 이어받음 |

---

### 4. 단계별 상세 설명

#### 0단계 — 학습 데이터 구축 (`data-preprocessing-augmentation/`)

> **결론:** 원본 + 회전 증강 + dev 자동 라벨을 합쳐 **1만 문항의 학습셋(Bigdata_v2)** 을 만들고, 각 단계마다 무결성 검사를 통과해야만 저장합니다.

| 소스 | 파일 이름 규칙 | 문항 수 | 설명 |
| --- | --- | ---: | --- |
| 원본 | `orig_train_NNNN.jpg` | 6,714 | 대회 제공 train |
| 회전 증강 | `rot_train_NNNN_rot{L,R}30.jpg` | 2,685 | 원본 2,427장의 ±30° 회전본 (질문·선지·정답 동일, 빈 모서리는 가장자리 복제) |
| 자동 라벨 | `auto_dev_NNNN.jpg` | 601 | dev 문항 + 자동 라벨 (보기별 신뢰도 최댓값 ≥ 0.60만) |

- 패키지(`data_pipeline/`)와 실행 스크립트(`scripts/01~03`)를 분리해 로직을 재사용 가능하게 구성했습니다.
- 실제 촬영 환경을 흉내 내는 **열화 증강 9종**(기울임·렌즈·종이 휨·그림자·저조도·빛반사·블러·저해상도·노이즈 + JPEG)도 구현했으며, 원본 화질 분포를 분석해 강도를 자동 조정하고 너무 망가진 결과는 강도를 낮춰 최대 4회 재시도합니다. (통합본에는 넣지 않은 독립 산출물)
- 저장 전 검사: 이미지 경로 존재·열림 여부, id 중복, 원본 대비 증강본 수, 질문·정답 보존, 최종 개수·정답 범위.

#### 1단계 — Multi-prompt Choice-CE LoRA 학습 (`s1_train_choice_ce.py`)

> **결론:** 추론과 **똑같은 입력**을 넣고 마지막 위치의 a/b/c/d **4개 로짓에만 CE 손실**을 걸어, 학습 목표와 추론 방식을 정확히 일치시켰습니다.

```python
# model_utils.VLM.choice_loss (요약)
texts = [render(build_messages(images, q, choices, prompt)) for ...]   # 추론과 같은 채팅 템플릿
logits = model(**enc, logits_to_keep=1).logits[:, -1, :]               # 마지막 위치만 계산 (left padding)
choice_logits = logits[:, letter_ids]                                  # 어휘 전체 → a/b/c/d 4개
loss = F.cross_entropy(choice_logits, gold, label_smoothing=ls)
```

**샘플마다 입력을 다르게 (데이터를 복제하지 않는 증강)**

| 요소 | 방법 | 목적 |
| --- | --- | --- |
| 프롬프트 표현 | `BASE_PROMPTS` p1~p4 중 균등 샘플링 | 특정 문구에 과적합 방지 → 표현 변화에 강건 |
| qtype 힌트 | 30% 확률로 유형별 힌트 덧붙임 (부정형·숫자·가격·문구읽기) | 추론 때 힌트가 붙어도/없어도 동작 |
| 선지 순서 | 매번 무작위 순열 후 정답 인덱스 재계산 | 모델이 **위치가 아니라 내용**을 보도록 (위치 편향 제거) |

**학습 루프 설계**

| 항목 | 설정 | 설명 |
| --- | --- | --- |
| LoRA | r=16, alpha=2r, dropout 0.05, 대상 `lm_all` | 언어 모델의 선형층 전부(Qwen3.5의 선형 어텐션 `in_proj_qkv/in_proj_z/out_proj` 포함), 비전 인코더 제외. `lm_all+vision` 등 프리셋을 정규식으로 정의 |
| 옵티마이저 | AdamW(wd 0.01) + cosine 스케줄(warmup 3%) + grad clip 1.0 | |
| 배치 | 유효 배치 `grad_accum=8`, 마이크로배치 4 | 결과는 유효 배치가, 속도는 마이크로배치가 결정하도록 분리 |
| OOM 대응 | OOM 발생 시 **그 마이크로배치만 반으로 쪼개 재귀적으로 재시도** | 긴 학습이 OOM 한 번에 죽지 않음 |
| Best 선택 | `eval_every=200` 업데이트마다 valid 평가 → **정확도 최고(동률이면 NLL 낮은 쪽)** 어댑터를 `adapter/` 로 | epoch 단위보다 세밀하게 최적 시점 포착 |
| 이어하기 | 50 업데이트마다 체크포인트 (LoRA + optimizer + scheduler + numpy/torch/CUDA RNG + 진행 위치) | 중단 후 같은 명령으로 재실행하면 같은 순서로 이어서 학습 |

#### 2단계 — 1-pass 로짓 스코어링 추론 (`s2_predict.py`)

> **결론:** 문장을 생성하지 않고 **다음 토큰이 a/b/c/d일 확률 4개만** 계산해, 빠르고 형식 오류가 없으며 이후 단계가 쓸 **확신도 정보를 남깁니다.**

- **left padding**으로 배치의 마지막 위치를 각 문항의 "다음 토큰" 위치로 맞추고, `logits_to_keep=1` 로 마지막 위치 로짓만 계산해 메모리를 절약합니다.
- 모델 로드 시 **안전 검사** 2가지: ① a/b/c/d가 각각 토큰 1개인지 ② 채팅 템플릿 끝이 `<think>` 로 열려 있지 않은지 (Qwen3.5 thinking 모드 강제 시 오류).
- 프롬프트는 학습과 달리 **결정적으로 1개만 선택**합니다(`select_prompt`: base `p1` + 해당 qtype 힌트). 프롬프트별 branch 앙상블은 하지 않습니다.
- 출력: `p_a..p_d`, `pred`, `conf`(1위 확률), `margin`(1·2위 차이), `qtype`, `prompt` + qtype별 정확도 표 + **바로 제출 가능한 `submission.csv`**.
- `--prompt-scan`: valid에서 base prompt를 하나씩 바꿔 정확도와 **예측 일치율**을 측정하는 진단 도구 (프롬프트 민감도 확인용).

#### 3단계 — 재추론 대상 선별 (`s3_select_rerun.py`, CPU)

> **결론:** **정답을 보지 않는 신호만으로** 불확실한 문항을 고르고, valid에서는 그 규칙이 **오답을 얼마나 잡아내는지**를 수치로 보여줍니다.

| 선별 조건 (하나라도 해당) | 기본값 |
| --- | --- |
| 확신도 `conf` < 문턱값 | 0.60 |
| `margin`(1·2위 차이) < 문턱값 | 0.30 |
| **약한 qtype** 이면서 `conf` < 문턱값 | 0.90 |
| 선별 비율 상한 (초과 시 확신도 낮은 순으로 자름) | 30% |

- **약한 qtype은 valid에서 자동 선정**: 전체 정확도보다 5%p 이상 낮고 표본이 20개 이상인 유형.
- 같은 규칙을 valid와 test에 똑같이 적용하고, valid에서 `선별 문항 정확도 / 비선별 정확도 / 오답 포착률 / 4단계가 다 맞히면 얻을 수 있는 최대 이득`을 출력합니다.

#### 4단계 — 선별 문항 재추론 (`s4_rerun_grounding.py`)

> **결론:** 선별된 문항에만 세 가지 재추론 방식을 적용하고, 결과를 모두 **같은 형식의 CSV**로 남겨 최종 단계가 valid로 고르게 했습니다.

| 모드 | 동작 | 노리는 오답 |
| --- | --- | --- |
| **`grounding`** (기본) | ① LoRA를 끈 원본 모델이 "답하려면 봐야 할 영역" bbox(JSON) 생성 → ② **원본 해상도**에서 크롭 → ③ `[전체 사진] + [확대 사진]` 2장으로 재스코어링 | 작은 글자·가격표·간판 읽기 |
| **`reason`** | Qwen3.5 thinking을 켜고 생각을 먼저 생성(`<think>…</think>`)한 뒤, **바로 답하기와 같은 위치·같은 글자 토큰**의 확률을 읽음 | 부정형("없는 것"), 여러 항목 대조, 지식형 |
| `prompt` | base prompt를 바꿔 여러 번 풀고 확률 평균 | 프롬프트 표현에 민감한 문항 |

**Grounding 세부 설계**

- bbox 생성은 **LoRA를 끈 원본 모델**(`adapter_off`)로 수행 — 4지선다용 어댑터가 좌표 생성 능력을 해치지 않도록.
- 모델별 좌표계 차이를 프리셋의 `coord` 로 흡수 (`norm1000`: Qwen3 계열 0~1000 정규화 / `abs`: Qwen2.5-VL 리사이즈 입력 기준 픽셀) → `to_original()` 로 원본 좌표 복원.
- 크롭 규칙(`crop_region`): 여러 bbox는 합집합(너무 크면 첫 박스) · 여백 15% · 최소 변 15% · 가장자리는 잘라내지 않고 안쪽으로 이동 · **원본의 70% 이상이면 크롭 생략**(1장으로 재추론).
- bbox 결과는 JSON **캐시**로 저장해 같은 run의 다른 재추론 설정끼리 공유하고, 중간에 끊겨도 이어서 생성합니다.
- 빨간 박스(모델 bbox)·파란 박스(실제 크롭)를 그린 **확인용 이미지**를 저장해 좌표계 오류를 눈으로 검증합니다.

#### 최종 단계 — 필요한 행만 교체 (`s5_build_final.py`, CPU)

> **결론:** "재추론하면 좋아질 것"이라고 가정하지 않고, **valid에서 keep(교체 안 함)보다 나은 조합만 채택**합니다.

| 방법 | 선별 문항의 최종 확률 |
| --- | --- |
| `keep` | 1-pass 그대로 (기준선) |
| `replace` | 재추론 결과로 전부 교체 |
| `rule` | 재추론 확신도 ≥ 기본 확신도일 때만 교체 |
| `blend` | (1−w)·기본 + w·재추론, w ∈ {0.3, 0.5, 0.7} |
| **`auto`** (기본) | **재추론 방식(grounding·reason) × 위 방법 전부**를 valid로 비교 → keep 대비 **+0.003(약 2문항) 이상**일 때만 채택, 아니면 keep |

선별 문항 수가 적어 우연한 차이로 설정을 바꾸지 않도록 최소 이득(`min_gain`)을 둔 것이 핵심입니다.

#### 보조 도구

| 파일 | 역할 |
| --- | --- |
| `ensemble.py` | 따로 학습한 어댑터들의 `test_pred.csv` 를 **재추론 없이** 확률 가중 평균/다수결로 결합, 멤버 간 일치율 리포트 |
| `probe_batch.py` | 가장 큰 이미지 + 가장 긴 질문·프롬프트로 **최악 입력**을 만들어 학습/1장/2장/bbox 생성의 배치 상한과 처리량 측정 → config 배치값 근거 |
| `collect_results.py` | 실험별 폴더에 흩어진 `experiments_<단계>.csv` 를 합쳐 `summary.csv`(실험 비교표) 생성 |
| `model_store.py` | 모델 가중치를 드라이브에 보관하고 새 VM은 드라이브→로컬 복사. 파일별 크기 목록으로 **덜 올라간 사본을 걸러냄** |

---

### 5. 검증셋 설계 — 데이터 누수 방지

> **결론:** 원본과 그 회전본, 바이트가 같은 중복 사진을 **하나의 이미지 그룹**으로 묶고 **그룹 단위로 valid를 떼어**, "같은 사진을 보고 외운 답"으로 valid를 맞히는 누수를 막았습니다.

```text
train_0001.jpg ─┐
rot_train_0001_rotL30.jpg ─┼─▶ 그룹 "train_0001"  ──▶ valid로 뽑히면 회전본까지 학습에서 제외
rot_train_0001_rotR30.jpg ─┘
바이트가 같은 다른 id 사진 ──▶ Union-Find로 같은 그룹에 병합 (크기가 같은 파일만 md5 비교)
```

| 설계 | 내용 |
| --- | --- |
| 그룹 단위 holdout | 그룹의 10%를 valid로, 그중 **무회전 원본 문항만** valid(671문항) — test와 같은 분포, 깨끗한 사람 라벨 |
| 결정적 분할 | `md5(split_seed \| 그룹 이름)` 순으로 정렬해 앞 10% 선택 → qtype 규칙·CSV 순서·라이브러리 버전이 바뀌어도 **같은 valid** |
| 실험 seed와 분리 | `split_seed` 를 `SEED` 와 따로 두어 실험 seed를 바꿔도 valid 고정 → 증강 vs 원본 실험을 **같은 valid로 비교** |
| 지문 검증 | valid 문항 집합의 hash(`valid_hash`)를 어댑터 폴더의 `train_meta.json` 에 기록 → 2·4단계가 **분할이 달라졌으면 valid 평가를 자동으로 건너뜀** |
| dev 라벨 신뢰도 | dev 5인 다수결은 라벨 신뢰도가 낮아(EDA 결과) 기본 valid에서 제외, 옵션(`valid_source="dev"`)으로만 지원 |

---

### 6. 안정성 · 재현성 설계

> **결론:** Colab VM이 언제든 끊길 수 있는 환경을 전제로, **어느 순간에 중단돼도 완성본 하나는 남고, 다시 실행하면 같은 상태로 이어지게** 만들었습니다.

| 문제 | 해결 |
| --- | --- |
| 어댑터·체크포인트 저장 중 끊김 | `replace_dir`: 임시 폴더에 다 쓴 뒤 교체(이전 것은 `_old` 로 이동 후 삭제) → `path / _tmp / _old` 중 완성본 하나는 항상 남고, 다음 실행 때 `recover_dir` 가 완성 표시 파일로 판별해 복구 |
| 이어하기 후 다른 결과 | numpy·torch·CUDA RNG 상태와 데이터 순서·위치까지 체크포인트에 저장 → 프롬프트 샘플링·선지 셔플이 같은 순서로 재현 |
| 설정이 바뀐 체크포인트로 이어하기 | 학습 데이터·valid·평가 문항의 **지문(data_sig)** 과 LoRA 랭크를 비교해 다르면 즉시 중단 (`--fresh` 안내) |
| 실패한 재추론의 예전 파일이 쓰임 | 4단계 시작 시 예전 결과를 먼저 삭제 → 도중 실패해도 최종 단계가 낡은 파일을 읽지 않음 |
| 추론 중 OOM | `run_batched`: OOM 난 배치만 1개씩 재시도 |
| 모델 다운로드 실패 | 이어받기 + 최대 8회 지수 백오프 재시도, `modelscope` 대체 경로 |
| 제출 파일 오류 | `sample_submission.csv` 와 id 순서·개수 대조, 답이 a~d인지, id 중복, **한 선지 50% 이상 쏠림 경고** |

---

### 7. 실험 관리 · 산출물 구조

> **결론:** 실험 하나의 산출물은 `runs/<run>/` 한 폴더에, 결과 사본은 **하이퍼파라미터가 이름에 담긴 폴더**에 모아 여러 실험을 동시에 돌려도 섞이지 않게 했습니다.

```text
runs/<run>/                         ← 단계끼리 주고받는 작업 폴더
├── adapter/  (+ best.json, train_meta.json)
├── run.json  train_log.csv
├── valid_pred.csv  test_pred.csv  submission.csv
├── select/   (valid_subset.csv, test_subset.csv, qtype_report.csv)
├── rerun/<grounding_adapter | reason>/  (*_rerun.csv, crops/)
└── final/    (final_submission.csv, valid_compare.csv, final.json)

results/<run>_<preset>_r16_e1_lr1e-4_hint30_aug_s42/   ← 실험 결과 사본 (로컬 + 구글 드라이브 동시 저장)
```

- 결과 폴더 이름(`config.exp_tag`)에 **run·모델·LoRA 랭크·epoch·lr·힌트 비율·데이터·seed** 가 들어가 파일만 봐도 설정을 알 수 있습니다.
- 단계별 수치는 `experiments_<단계>.csv` 에 누적하되 **실험마다 파일을 나눠** 여러 VM이 같은 드라이브에 동시에 써도 행이 사라지지 않게 했고, 합치는 일은 로컬의 `collect_results.py` 가 합니다.

---

### 8. 실행 인프라 — 로컬 터미널에서 Colab GPU 원격 제어

> **결론:** 브라우저 노트북 대신 **Colab CLI 기반 셸 도구**를 만들어, 로컬 터미널 명령 한 줄로 VM 할당 → 실행 → 감시 → 결과 회수까지 처리하고 VM 2대로 실험을 병렬 수행했습니다.

```bash
./colab/colab_pipeline.sh up A100      # VM 할당 + 드라이브 마운트 + 패키지 + 코드·데이터·모델 준비
./colab/colab_pipeline.sh smoke        # 작은 설정으로 전 단계 점검 (runs_smoke/ 에 분리 저장)
./colab/colab_pipeline.sh pipeline     # s1 → s2 → s3 → s4(grounding, reason) → s5
./colab/colab_pipeline.sh fetch        # 결과 회수
./colab/sweep.sh plan && ./colab/sweep.sh start   # 실험 목록을 VM 2대에 나눠 병렬 실행
```

| 기능 | 구현 |
| --- | --- |
| 백그라운드 작업 | VM 안에서 프로세스로 실행 → 로컬 터미널을 꺼도 계속 진행, 로그는 5분마다 드라이브로 복사 |
| GPU 자동 배치 | GPU 작업은 빈 GPU를 골라 `CUDA_VISIBLE_DEVICES` 로 고정, 비어 있지 않으면 시작하지 않음 (학습 중복으로 인한 OOM 방지) |
| 토큰 만료 우회 | Colab CLI가 1시간 뒤 접속 토큰을 갱신하지 않아 세션을 잃는 문제를 발견 → 20분마다 할당 목록 조회로 토큰 갱신(`cli_session.py`), 잃은 세션 재등록(`adopt`) |
| 데이터 I/O | 데이터는 드라이브에 zip 1개로 두고 VM 로컬 디스크에 풀어 사용 (드라이브에서 이미지 수천 장 직접 읽기 회피) |
| 병렬 실험 | 계정 동시 할당 한도에 맞춰 `MAX_PARALLEL=2`, 실험이 더 많으면 VM이 자기 몫을 순서대로 이어서 실행 |

---

## ▶️ Ⅱ. 실행 방법

### 1. 설치

#### (A) Google Colab

```python
from google.colab import drive; drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/<코드 zip>.zip -d /content
!unzip -q /content/drive/MyDrive/<데이터 zip>.zip -d /content/data
%cd /content/<코드 폴더>
!pip install -q -r requirements.txt
```

> Colab은 PyTorch(CUDA)가 기본 설치되어 있으므로 `requirements.txt`만 설치하면 됩니다.

#### (B) 로컬 (RTX 5060 Ti)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# RTX 50 시리즈(Blackwell, sm_120)는 CUDA 12.8 이상으로 빌드된 PyTorch가 필요합니다.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> 💡 구버전 CUDA(cu121, cu124 등) 빌드의 PyTorch를 설치하면 `no kernel image is available for execution on the device` 오류가 발생할 수 있습니다.

### 2. 데이터 준비

```text
<데이터 폴더>/
├── train.csv  dev.csv  test.csv  sample_submission.csv
├── train/     # 학습 이미지
├── dev/       # dev 이미지
└── test/      # 테스트 이미지
```

- `--data-root <데이터 폴더>` 또는 환경변수 `VQA_DATA_ROOT` 로 경로를 지정합니다.
- 지정하지 않으면 **현재 폴더 → 상위 폴더 → 코드 폴더** 순으로 `train.csv` 가 있는 곳을 자동으로 찾습니다.
- `--data-root` 는 반드시 `train.csv` 가 **직접 들어 있는 폴더**여야 합니다. (zip 안에 폴더가 한 겹 더 있으면 그 폴더)

### 3. 환경변수

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `VQA_DATA_ROOT` | 자동 탐색 | 데이터 폴더 |
| `VQA_MODEL_ROOT` | `<코드 폴더>/models` | 모델 가중치 저장 위치 (HF 캐시 `HF_HOME` 도 이 아래에 생성) |
| `VQA_ONLINE` | `1` | `0` 이면 오프라인 모드 (받아 둔 모델만 사용) |
| `VQA_DISABLE_FLA` | – | `1` 이면 flash-linear-attention을 끄고 PyTorch 구현 사용 (속도 비교용) |

> 모델은 첫 실행 시 `models/` 에 자동으로 다운로드됩니다. 중간에 끊겨도 이어받기와 최대 8회 재시도(지수 백오프)를 하며, 계속 실패하면 `--download-source modelscope` 로 바꿀 수 있습니다.

### 4. 추론 실행 (`predict.py`)

처음에는 **몇 분 안에 끝나는 설정으로 끝까지 도는지 먼저 확인**합니다.

```bash
# ① 동작 점검 (valid 50문항, test 20문항) — submission.csv 에는 20문항만 담김
python predict.py --data-root <데이터 폴더> --preset qwen3.5-4b --n-valid 50 --n-test 20

# ② 본 실행: valid 정확도 측정 + test 전체 → submission.csv
python predict.py --data-root <데이터 폴더> --preset qwen3.5-4b --batch-size 8

# ③ 학습한 LoRA 어댑터를 얹어 추론
python predict.py --data-root <데이터 폴더> --preset qwen3.5-4b --adapter <어댑터 폴더> --prompt A

# ④ 선지 순서 TTA (4배 느림)
python predict.py --data-root <데이터 폴더> --preset qwen3.5-4b --tta

# ⑤ 제출 파일만 필요할 때
python predict.py --data-root <데이터 폴더> --preset qwen3.5-4b --splits test
```

> ⚠️ `config.py` 의 기본 프리셋(`DEFAULT_PRESET`)은 `qwen3.5-9b-bf16`(24GB 이상 GPU 필요)이므로, 4B 모델로 실행하려면 **`--preset qwen3.5-4b` 를 반드시 지정**해야 합니다.

#### 주요 인자

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--preset` | `qwen3.5-9b-bf16` | 모델 프리셋 |
| `--adapter` | 없음 | LoRA 어댑터 폴더 (없으면 zero-shot) |
| `--prompt` | `A` | 프롬프트 종류 (`A` / `B`) |
| `--tta` | off | 선지 순서 4가지(abcd·bcda·cdab·dabc)로 추론 후 평균 |
| `--splits` | `valid test` | 실행할 split |
| `--n-valid`, `--n-test` | 전체 | 점검용 일부 추출 (**제출용은 반드시 전체**) |
| `--batch-size` | 32 | 한 번에 forward하는 입력 수 (`--tta` 면 문항당 4개) |
| `--max-side` | 1024 | 이미지 픽셀 수 상한 = side² |
| `--out-dir` | `outputs` | 결과 저장 폴더 |
| `--download-source` | `hf` | `hf` / `modelscope` |

#### GPU별 권장 설정

| GPU | 프리셋 | 배치 |
| --- | --- | --- |
| A100 40GB | `qwen3.5-9b-bf16` | 32 (실측, 최악 입력 기준 31.1GB) |
| L4 24GB | `qwen3.5-9b-bf16` | 8부터 시도 |
| 그 이하 (RTX 5060 Ti 등) | **`qwen3.5-4b`** 또는 `qwen3.5-9b`(4bit) | 8부터 시도 |

### 5. 출력물 (`--out-dir`, 기본 `outputs/`)

| 파일 | 내용 |
| --- | --- |
| `submission.csv` | 제출 파일. `sample_submission.csv` 의 id 순서·개수·형식을 점검한 뒤 저장 |
| `test_pred.csv` | 문항별 `p_a..p_d`, `pred`, `conf`(1위 확률), `margin`(1·2위 차이), `qtype`(질문 유형) |
| `valid_pred.csv` | dev 합의 문항 예측과 `gold`(다수결 정답) |
| `run_info.json` | 실행 설정, valid 정확도, 문항당 소요 시간 |

> 제출 전 `submission.csv` 의 예측 분포가 **한 선지에 50% 이상 쏠리면 경고**가 출력됩니다.

### 6. 학습 · 2-pass 재추론 · 병합

| 단계 | 내용 | 명령어 |
| :---: | --- | --- |
| 학습 | multi-prompt choice-CE LoRA 학습 → `runs/<run>/adapter/` | `python s1_train_choice_ce.py --run main` |
| 1-pass | 전체 예측 + `conf`/`margin` 저장 → `submission.csv` | `python s2_predict.py --run main` |
| 선별 | 저신뢰 문항 선별 (CPU) | `python s3_select_rerun.py --run main` |
| 2-pass | 선별 문항 grounding → 크롭 → [전체+확대] 재추론 / thinking 재추론 | `python s4_rerun_grounding.py --run main --mode grounding --pass2 adapter`<br/>`python s4_rerun_grounding.py --run main --mode reason` |
| 병합 | valid로 교체 방법 비교 후 필요한 행만 교체 → `final/final_submission.csv` | `python s5_build_final.py --run main --rerun-tag grounding_adapter reason` |

> 전 단계를 한 번에 돌리려면 `./run_pipeline.sh` (GPU 서버) 또는 `./colab/colab_pipeline.sh pipeline` (Colab)을 사용합니다.
> 처음에는 `--n-train 200 --max-updates 20` 처럼 작은 설정으로 끝까지 도는지 먼저 확인하세요 ([PIPELINE.md](PIPELINE.md) 참고).

### 7. 자주 나는 문제

| 증상 | 해결 |
| --- | --- |
| `Found an incompatible version of torchao` (어댑터를 얹을 때) | `pip uninstall -y torchao` 후 재실행 (이 코드는 torchao를 사용하지 않음) |
| `CUDA out of memory` | `--batch-size` 를 낮추거나 더 작은 프리셋 사용. OOM이 난 배치만 1개씩 자동 재시도하며 `⚠ OOM` 이 출력됨 |
| 생각보다 느림 | `pip show flash-linear-attention` 으로 설치 확인. 없으면 Qwen3.5가 느린 PyTorch 구현으로 동작 |
| `이미지 N개를 찾지 못함` | `--data-root` 에 `train/ dev/ test/` 이미지 폴더가 있는지 확인 |
| `thinking 모드가 켜져 있어…` | 채팅 템플릿이 thinking을 강제하는 경우. 다른 프리셋 사용 |

---

## 🎯 Ⅲ. 접근 전략

전체 전략은 **"모델을 키우기보다, 데이터와 입력을 모델이 풀기 쉬운 형태로 만든다"** 는 방향으로 설계했습니다.

| # | 전략 | 해결하려는 문제 | 핵심 방법 |
| :---: | --- | --- | --- |
| 1 | **데이터 품질 우선** | 답이 모호하거나 틀린 샘플이 학습을 오염시킴 | 5인 답변 다수결 + 합의 비율 필터링 + Claude 검수 |
| 2 | **오답 기반 증강** | 회전된 이미지에서 오답이 집중됨 | 좌우 30° 회전 증강 + EXIF 회전 보정 |
| 3 | **질문 유형 분류** | 질문마다 요구하는 추론 방식이 다름 | 정규식 기반 qtype 분류 + 목적별 프롬프트 |
| 4 | **로짓 스코어링 추론** | 생성 방식은 형식 오류·파싱 실패 위험 | a/b/c/d 토큰 확률만 비교 + 선지 순서 TTA |
| 5 | **경량 파인튜닝** | 제한된 GPU(Colab, 5060 Ti) | 4B 모델 + LoRA, 4지 분류 손실 |
| 6 | **불확실한 샘플 재추론** | 작은 글자·세부 정보를 놓침 | grounding bbox로 크롭 → [전체+확대] 2-pass |

### 전략 1. 데이터 품질 우선 — 전처리 · EDA

#### 데이터셋 현황

| 구분 | 정답 형태 |
| --- | --- |
| Train data | `answer` 컬럼에 단일 정답 존재 |
| Dev data | 단일 정답 컬럼 없음 → **`answer1`~`answer5` (5명의 답변)** 만 존재 |

Dev 데이터는 정답이 하나로 정해져 있지 않기 때문에, **5명 답변의 다수결로 정답(`answer`)과 득표수(`votes`)를 계산**하고(`data_utils.majority_vote`), 답변 비율을 확인하여 답이 명확한 샘플만 선별했습니다.

#### Dev 데이터 답변 비율 분포 (총 2,683개)

| 답변 비율 | 개수 | 비중 | 학습 사용 여부 |
| --- | ---: | ---: | :---: |
| 4 : 0 | 0개 | 0.0% | – |
| 3 : 1 / 3 : 1 : 1 | **1,007개** | 37.5% | ✅ 사용 |
| 3 : 2 / 2개 이하 | 1,676개 | 62.5% | ❌ 제외 |
| **합계** | **2,683개** | 100% | |

> **→ 3:1 / 3:1:1 데이터만 사용**
> 한 선지로 답이 뚜렷하게 모이는 샘플만 남기고, 답변이 팽팽하게 갈리는 3:2 이하 샘플은 라벨 신뢰도가 낮다고 판단하여 제외했습니다.
>
> 검증(valid)에는 **5명 중 3표 이상 합의한 문항**(`VALID_MIN_VOTES = 3`)을 사용합니다.

#### 추가 검수 — 해석이 어려운 데이터 발견

Claude를 활용해 **정답이 없거나, 중복 정답이 존재하는 등 해석이 어려운 데이터**를 추가로 찾아냈습니다.

<img src="assets/eda_ambiguous_samples.jpg" alt="EDA 중 발견된 해석이 어려운 Dev 샘플" width="900"/>

| 파일 | 질문 | 선지 |
| --- | --- | --- |
| `dev_2525.jpg` | 이미지 속에 보이는 육포 제품 중 '장장 CLASSIC 육포 플러스' 제품의 가격은 얼마인가요? | A 1,200원 / B 3,000원 / C 1,700원 / D 2,500원 |
| `dev_2526.jpg` | 위 책장의 맨 윗줄에 있는 책 시리즈 이름은 무엇인가요? | A 초등 영어 리딩 파워 / B 초등 수학 문제 해결력 / C 초등 국어 독해왕 / D 초등 사회 탐구력 |
| `dev_2529.jpg` | 이 가챠머신에서 뽑을 수 없는 품목은 무엇인가요? | A 소화기 / B 엘리베이터 버튼 / C 화재 경보기 / D 화장실 비상 버튼 |
| `dev_2544.jpg` | 사진에 보이는 책 중에서 '예수 평전'이라는 제목의 책이 있는 책장은 어디인가요? | A 가운데 칸 / B 맨 아래 칸 / C 맨 위 칸 / D 책장 옆에 따로 놓인 곳 |

> 위 샘플들은 EDA 과정에서 확인한 Dev 데이터 예시입니다. (예: `dev_2525.jpg`처럼 이미지가 회전된 채 저장된 샘플도 존재)
>
> **EDA 결론 →** **답이 없는 경우, 중복 선지가 존재하는 경우** 등 실제 데이터 품질 문제가 발견되었습니다.

### 전략 2. 오답 기반 증강 — 회전 증강

| 단계 | 내용 |
| --- | --- |
| ① 샘플 테스트 | 샘플로 **10개의 data**에 대해 먼저 증강 수행 |
| ② 강도 조절 | 비율을 조정하면서 증강의 강도를 확인 |
| ③ 최종 증강 | **회전 좌우 30도씩** 진행 (`rotL30`: +30°, `rotR30`: −30°) |

<img src="assets/augmentation_rotation.jpg" alt="회전 증강 예시 (original / rotL30 / rotR30)" width="720"/>

#### ❓ Q1. 왜 좌우 회전만 했는지?

> 1차 학습 이후 **모델이 못 맞춘 이미지 데이터를 확인한 결과, 회전된 이미지가 가장 많았기 때문에**
> 오답 원인에 직접 대응하는 **회전 관련 데이터 증강만** 진행하기로 결정했습니다.

또한 이미지를 불러올 때 **EXIF 방향 정보로 회전을 보정**(`ImageOps.exif_transpose`)하여, 촬영 기기 메타데이터 때문에 이미지가 눕혀져 들어가는 경우를 먼저 제거했습니다.

### 전략 3. 질문 유형 분류 — qtype & 프롬프트

<img src="assets/prompt_flow.png" alt="질문 유형별 지시 구성 흐름" width="520"/>

```text
질문  → 규칙 기반 유형 분류 → 유형별 지시 구성 ─┐
                                               ├→ VQA 모델 → 답변
이미지 ─────────────────────────────────────────┘
```

#### 규칙 기반 질문 유형 (`data_utils.QTYPE_RULES`)

질문 텍스트를 정규식으로 검사해 **위에서부터 처음 일치하는 유형**으로 분류합니다.

| 우선순위 | 유형 | 주요 키워드 (정규식 요약) |
| :---: | --- | --- |
| 1 | 부정형 | 없는 / 아닌 / 틀린 / 않은 / 잘못된 / 다른 + 것·곳·항목 등 |
| 2 | 개수 | 몇 개·명·마리·대·권·장·잔·곳·가지, 개수 |
| 3 | 숫자·가격 | 가격, 얼마, 금액, 요금, 번호, 전화, 몇 시, 날짜, 연도, 몇 층, km, % |
| 4 | 위치·방향 | 어디, 위치, 방향, 왼쪽, 오른쪽, 위쪽, 아래쪽, 어느 곳·쪽 |
| 5 | 문구읽기 | 적힌, 쓰여, 문구, 글자, 이름, 상호, 제목, 표시된, ~라고 |
| 6 | 기타 | 위 규칙에 해당하지 않는 질문 |

qtype은 **유형별 정확도 분석(오답 분석)** 과 **재추론 대상 선별의 기준 축**으로 사용합니다.

#### 프롬프트 구성 (`model_utils.PROMPTS`)

| ID | 이미지 1장 (`single`) | 이미지 2장 (`two`: 전체 + 확대) |
| :---: | --- | --- |
| A | 이미지 속 **간판·표지판·안내문 글자를 정확히 읽는 것**을 강조 | [전체 사진]과 [확대 사진]을 함께 보고 판단 |
| B | 글자·숫자를 **한 글자씩 확인**하고, 보기끼리 비슷하면 **철자·어순·숫자를 끝까지 비교**하도록 지시 | 두 사진의 글자·숫자를 한 글자씩 확인 |

- 모든 프롬프트는 시스템 지시(`a, b, c, d 중 한 글자로만 답하라`)와 `[질문] + (a)~(d) 선지 + 답변 규칙` 형식으로 구성됩니다.
- 2장 프롬프트에는 **"확대 사진은 자동으로 잘라낸 것이라 무관할 수 있으니 전체 사진과 대조하라"** 는 주의문을 넣어, 크롭이 빗나갔을 때의 오판을 줄였습니다.
- 학습과 추론에 **같은 프롬프트 형식**을 사용하여 입력 분포를 일치시켰습니다.

<!-- TODO: 학습 시 qtype별로 다른 지시문을 붙였다면, 유형 → 지시문 매핑 표를 추가해 주세요. -->

### 전략 4. 로짓 스코어링 추론 + 선지 순서 TTA

```text
프롬프트 끝 (다음 토큰 위치)
      │
      ▼
 전체 어휘 로짓 ──▶ [a, b, c, d] 4개 토큰 로짓만 추출 ──▶ softmax ──▶ p_a, p_b, p_c, p_d
                                                                          │
                                                          pred = argmax, conf = 1위 확률,
                                                          margin = 1위 − 2위
```

- 문장을 생성하지 않고 **다음 토큰이 a/b/c/d일 확률 4개만 비교**합니다.
  → "답을 못 뽑는" 경우가 없고, 문항당 forward 1회로 빠르며, 확률을 그대로 남겨 **앙상블·재추론 선별에 활용**할 수 있습니다.
- Qwen3.5의 **thinking 모드는 끄고**(`enable_thinking=False`), 로드 시 프롬프트 끝이 `<think>` 로 열려 있으면 오류를 내도록 검사합니다.
- **선지 순서 TTA**: 선지를 4가지 순환 순서(abcd · bcda · cdab · dabc)로 바꿔 추론한 뒤, 원래 선지 기준으로 되돌려 평균합니다.
  → 모델이 특정 위치(예: 항상 a)를 선호하는 **위치 편향**을 줄이고, 4개 순서 간 일치 횟수(`tta_agree`)를 추가 신뢰도 지표로 사용합니다.

### 전략 5. 경량 파인튜닝 — Qwen3.5-4B + LoRA

| 항목 | 설정 |
| --- | --- |
| 적용 대상 | 언어 모델의 `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (**비전 인코더 제외**) |
| LoRA | r = 16, alpha = 32, dropout = 0.05, bias 없음 (코드 기본값) |
| 손실 함수 | 추론과 **동일한 프롬프트 끝에서 a/b/c/d 4개 로짓만 뽑아 4지 분류 cross-entropy** (label smoothing 옵션) |
| 메모리 절약 | gradient checkpointing, bf16 autocast |

- 생성 전체 문장이 아니라 **추론 때 실제로 보는 4개 로짓에 직접 손실을 걸기 때문에**, 학습 목표와 추론 방식이 정확히 일치합니다.
- 비전 인코더는 그대로 두고 언어 모델에만 LoRA를 얹어, 이미지 인식 능력은 유지하면서 **질문–선지 판단 능력을 조정**했습니다.

### 전략 6. 불확실한 샘플 재추론 — Crop 2-pass 파이프라인

<img src="assets/pipeline_diagram.png" alt="AI 구조 파이프라인 다이어그램" width="520"/>

```mermaid
flowchart TD
    BM[Base Model] --> TA[Trained_AI]
    FD[Filtered_Data] -->|학습| TA
    TD[Test_Data] -->|Test| INF(( 1-pass 추론 ))
    TA --> INF
    INF -->|고신뢰| SUB[Submission.csv]
    INF -->|저신뢰<br/>conf · margin| LC[Low_confidence Data]
    LC --> GR[Grounding<br/>LoRA off, 원본 모델]
    GR -->|bbox → crop_region| CR[전체 + 확대 이미지]
    CR -->|2-pass 추론| SUB
```

| 단계 | 설명 |
| :---: | --- |
| **1** | 필터링된 데이터로 학습 → `Trained_AI` |
| **2** | 테스트 전체 1-pass 추론 → `conf`(1위 확률), `margin`(1·2위 차이) 저장 |
| **3** | 신뢰도가 낮은 문항 선별 → **grounding**: "이 질문에 답하려면 어느 영역을 자세히 봐야 하는가?"를 물어 bbox(JSON) 생성 |
| **4** | bbox를 원본 좌표로 변환 후 **크롭 영역 계산** → **[전체 사진] + [확대 사진] 2장**으로 재추론 |
| **5** | 1-pass 고신뢰 결과와 2-pass 결과 **병합** → `Submission.csv` |

#### Grounding · Crop 세부 설계

- grounding은 **LoRA를 끈 원본 모델**(`adapter_off`)로 수행합니다. 4지선다용으로 학습된 어댑터가 bbox 생성 능력을 해치지 않도록 하기 위함입니다.
- grounding 입력은 768px 기준으로 축소하고, 모델이 낸 좌표는 **원본 해상도 좌표로 되돌려(`to_original`) 원본 이미지에서 잘라냅니다.** (축소본에서 자르면 확대 효과가 없음)
- 크롭 영역 계산 규칙 (`crop_region`):

| 규칙 | 값 |
| --- | --- |
| 여러 bbox | 합집합 사용, 합집합이 너무 크면 첫 번째 bbox |
| 여백(pad) | 각 변에 bbox 크기의 15% |
| 최소 변 길이 | 원본의 15% |
| 가장자리 처리 | 잘라내지 않고 안쪽으로 밀어서 크기 유지 |
| 크롭 생략 | 결과가 원본 면적의 70% 이상이면 확대 효과가 없으므로 크롭하지 않음 |

---

## 🧪 Ⅳ. 실험 설계

### 1. 실험 사이클

모든 개선은 아래 사이클을 반복하며 **"오답에서 원인을 찾고 → 그 원인만 겨냥해 수정 → 다시 검증"** 하는 방식으로 진행했습니다.

```mermaid
flowchart LR
    A[베이스라인<br/>zero-shot 추론] --> B[valid 오답 수집]
    B --> C[qtype별 정확도 ·<br/>오답 이미지 분석]
    C --> D[원인 기반 개선<br/>필터링 · 증강 · 프롬프트 · 2-pass]
    D --> E[소규모 점검<br/>--n-valid 50 --n-test 20]
    E --> F[valid 전체 재평가]
    F --> B
```

### 2. 평가 기준

| 항목 | 내용 |
| --- | --- |
| 평가 지표 | 정확도 (Accuracy, 4지선다 정답 일치율) |
| 검증 데이터 | dev 중 **5명 중 3표 이상 합의한 문항** (`VALID_MIN_VOTES = 3`), 정답 = 다수결 |
| dev가 없을 때 | train에서 **이미지 단위로 10%를 분리** (같은 이미지가 train/valid에 걸치지 않도록) |
| 세부 분석 | **qtype별 정확도·평균 확신도 표** (`qtype_table`) |
| 재현성 | `SEED = 42` 고정 (`random`, `numpy`, `torch`) |
| 기록 | 실험마다 `--out-dir` 을 다르게 주고 `run_info.json` 의 `valid_acc` 비교, 단계별 수치는 `runs/experiments_<stage>.csv` 에 누적 |

> ⚠️ **설정은 valid 결과로만 선택하고, test(리더보드) 결과를 보고 바꾸지 않는다**는 원칙으로 과적합을 방지했습니다.

### 3. 실험 항목

| # | 실험 | 비교 조건 | 확인하려는 것 |
| :---: | --- | --- | --- |
| E1 | 모델 프리셋 | zero-shot 모델 간 비교 | 제한된 GPU에서 쓸 Base Model 선택 |
| E2 | 프롬프트 | 프롬프트 `A` vs `B` | 글자 읽기 강조 지시의 효과 |
| E3 | 이미지 해상도 | `--max-side` 1024 vs 1280 | 해상도 상향의 이득 여부 |
| E4 | 선지 순서 TTA | TTA off vs on | 위치 편향 감소 효과 (속도 4배 비용) |
| E5 | 데이터 필터링 | 합의 기준별 학습 데이터 비교 (3:1 / 3:1:1만 사용) | 모호한 라벨 제거의 효과 |
| E6 | 회전 증강 | 증강 없음 vs 좌우 30° 회전 | 회전 이미지 오답 감소 여부 |
| E7 | LoRA 학습 | zero-shot vs LoRA 어댑터 | 4지 분류 손실 파인튜닝의 효과 |
| E8 | 2-pass 재추론 | 1-pass만 vs 저신뢰 문항 Crop 2-pass | 확대 재추론의 효과 |
| E9 | 재추론 임계값 | `conf` / `margin` 기준값 비교 | 2-pass 대상을 고르는 적절한 기준 |

### 4. 실험 결과

<!-- TODO: 실제 측정값으로 채워 주세요. -->

| 구성 | 필터링 | 회전 증강 | LoRA | TTA | 2-pass | valid 정확도 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-shot (Qwen3.5-4B) | | | | | | `TBD` |
| + LoRA | ✅ | | ✅ | | | `TBD` |
| + 회전 증강 | ✅ | ✅ | ✅ | | | `TBD` |
| + 선지 순서 TTA | ✅ | ✅ | ✅ | ✅ | | `TBD` |
| **+ Crop 2-pass (최종)** | ✅ | ✅ | ✅ | ✅ | ✅ | **`TBD`** |

- **E3 (해상도)**: `--max-side` 를 1024 → 1280으로 올려도 **정확도 이득이 없어** 1024를 기본값으로 유지했습니다.

### 5. 주요 하이퍼파라미터

| 항목 | 값 |
| --- | --- |
| Seed | 42 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate | `TBD` |
| Epoch | `TBD` |
| Label smoothing | `TBD` |
| 추론 이미지 픽셀 상한 (`INFER_MAX_SIDE`) | 1024² |
| 최소 픽셀 (`MIN_PIXELS`) | 256 × 28 × 28 |
| Grounding 입력 크기 | 768² |
| 크롭 pad / 최소 변 / 최대 면적 | 0.15 / 0.15 / 0.70 |
| 재추론 신뢰도 임계값 | `TBD` |
| 회전 증강 각도 | ±30° |
| 배치 크기 | `TBD` (A100 40GB·9B 기준 32) |

---

## ✅ 요약

| 영역 | 핵심 내용 |
| --- | --- |
| 데이터 정제 | Dev 5인 답변 다수결, 답변 비율 **3:1 / 3:1:1**인 **1,007개만 학습에 사용**, Claude로 정답 없음·중복 정답 데이터 검출 |
| 데이터 증강 | 오답 분석 결과 회전 이미지가 가장 많아 **좌우 30° 회전 증강** + EXIF 회전 보정 |
| 모델 · 학습 | **Qwen3.5-4B + LoRA**(언어 모델만), a/b/c/d 로짓 **4지 분류 손실** (Colab / RTX 5060 Ti) |
| 추론 | 정규식 qtype 분류, **로짓 스코어링**(생성 없음), 선지 순서 TTA |
| 후처리 | 저신뢰 문항 **grounding → 크롭 → [전체+확대] 2-pass 재추론**, 결과 병합 → `Submission.csv` |

---

<div align="center">

**감사합니다 🙇**
<br/>
*흥선대원군의 AI 척화비*

</div>
