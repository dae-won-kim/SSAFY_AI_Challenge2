# 텍스트 이미지 VQA 파이프라인 (단계별 스크립트)

이미지 + 질문 + 4지선다(a/b/c/d)에서 정답을 고르는 파이프라인입니다.
**모델 하나 + LoRA 하나를 여러 프롬프트 표현으로 학습**하고, 불확실한 문항만 비싼 추론으로 다시 봅니다.

```
[1] multi-prompt choice-CE LoRA 학습 (단 한 번)
      sample 마다 base prompt 무작위 + 30% 확률로 qtype hint
        │
[2] 단일 패스 추론 (sample 당 forward 1회, qtype 으로 프롬프트 선택)
        │                                   └─► submission.csv (기준 제출 파일)
[3] 재추론 대상 선별 (정답을 보지 않는 기준) ─┬─ 확실한 문항 → 기본 답 유지
                                              └─ 불확실한 문항
                                                     │
[4] 선별 문항만 재추론 (grounding 2패스 또는 다중 프롬프트) ◄┘
        │
[최종] 필요한 행만 교체 → final_submission.csv
```

프롬프트별 branch 를 따로 학습하거나 앙상블하지 않습니다. 프롬프트 다양성은 **학습 augmentation** 으로 넣고,
여러 프롬프트로 다시 보는 것은 4단계의 선별 문항에만 씁니다.

## 파일 구성

| 파일 | 단계 | GPU | 하는 일 | 주요 산출물 (`runs/` 아래) |
|---|---|---|---|---|
| `config.py` | 공통 | | 경로, 모델 프리셋, **실험 설정**(seed·epoch·lr·LoRA 랭크·프롬프트 비율·배치·문턱값) | |
| `data_utils.py` | 공통 | | 데이터 로드(증강 데이터 이미지 그룹 단위 train/valid 분할), qtype, 확률 요약, 크롭 좌표, 제출 점검 | |
| `model_utils.py` | 공통 | ✔ | 모델·LoRA 로드, base prompt pool·qtype hint·sampling/선택, 로짓 스코어링, grounding | |
| `s1_train_choice_ce.py` | 1 | ✔ | 4지 토큰 CE 로 LoRA 학습(한 번). 프롬프트·선지 순서 매번 무작위. `eval_every` 업데이트마다 valid 평가 → **valid 최고 업데이트의 어댑터**를 채택 | `<run>/adapter/`(+`best.json`), `adapter_e<N>/`, `run.json`, `train_log.csv` |
| `s2_predict.py` | 2 | ✔ | sample 당 1 forward 추론(valid·test, train 은 `--splits` 로 선택). `--prompt-scan` 으로 프롬프트 민감도 측정 | `<run>/{valid,test}_pred.csv`, `submission.csv` |
| `s3_select_rerun.py` | 3 | | 확신도·margin·약한 qtype 으로 선별 | `<run>/select/*_subset.csv`, `qtype_report.csv` |
| `s4_rerun_grounding.py` | 4 | ✔ | 선별 문항만 재추론 (`--mode grounding` 확대 2패스 / `--mode reason` 생각을 쓰고 답하기(Qwen3.5) / `--mode prompt`) | `<run>/rerun/<tag>/*_rerun.csv`, `crops/` |
| `s5_build_final.py` | 최종 | | 필요한 행만 교체. `--method auto`(기본)는 재추론 방식 × 교체 방법을 valid 로 비교해 keep 보다 나을 때만 채택 | `<run>/final/final_submission.csv`, `valid_compare.csv`, `final.json` |
| `model_store.py` | 공통 | | 모델 가중치 드라이브 보관소 : 드라이브 `vqa/models/` ↔ 로컬 `MODEL_ROOT` (한 번 받은 모델을 새 VM 이 드라이브에서 복사) | 드라이브 `models/<모델이름>/` |
| `probe_batch.py` | 도구 | ✔ | 배치 상한 측정 → `config.py` 의 배치에 반영 | `probe_batch.json` |
| `collect_results.py` | 도구 | | 실험 기록을 합쳐 비교표 만들기 | `results/summary.csv` |
| `colab/colab_pipeline.sh` | 실행 | | 로컬 터미널에서 Colab CLI 로 전 단계 실행·감시·결과 회수 | |
| `colab/sweep.sh` | 실행 | | VM 여러 대로 실험 여러 개를 동시에 (한 VM = GPU 1장 = 실험 1개) | |

**한 실험의 산출물은 전부 `runs/<run>/` 아래에 모입니다** (`adapter/`, `*_pred.csv`, `select/`, `rerun/`, `final/`).
그래서 여러 실험을 같은 드라이브에 동시에 돌려도 서로 덮어쓰지 않습니다.

3·최종 단계는 CPU 만 쓰므로, 문턱값을 바꿔가며 몇 초 만에 다시 돌릴 수 있습니다.

## 작업 환경 (VS Code + WSL + Colab CLI + Claude Code)

처음 설정은 **[SETUP_VSCODE_WSL.md](SETUP_VSCODE_WSL.md)**, Claude Code 가 읽는 프로젝트 안내는 **[CLAUDE.md](CLAUDE.md)** 에 있습니다.

```bash
bash scripts/setup_wsl.sh      # WSL 에서 한 번: Colab CLI · Claude Code · 스킬 설치
```

## Colab 에서 실행하기

로컬 터미널에서 Google Colab CLI 로 GPU VM 을 빌려 돌리는 방법은 **[colab/README_colab.md](colab/README_colab.md)** 에 있습니다.

```bash
./colab/colab_pipeline.sh up A100 && ./colab/colab_pipeline.sh smoke && ./colab/colab_pipeline.sh pipeline
```

아래는 GPU 가 있는 서버·Docker 에서 직접 돌리는 방법입니다.

## 실행 전 준비

```bash
pip install -r requirements.txt          # PyTorch 는 환경에 맞게 먼저 설치
export VQA_MODEL_ROOT=/workspace/models  # 모델·HF 캐시 저장 위치 (D 드라이브에 연결된 경로 권장)
export VQA_DATA_ROOT=/workspace          # 원본 data.zip : train.csv, dev.csv, test.csv, train/ dev/ test/ 가 있는 곳
export VQA_TRAIN_ROOT=/workspace/bigdata # 증강 Bigdata_v2.zip : bigdata/train.csv + bigdata/train/ (학습·valid)
```

`VQA_DATA_ROOT` 를 안 주면 현재 폴더 → 상위 폴더 → `/workspace` 순으로 `train.csv` 를 찾습니다.
`VQA_TRAIN_ROOT` 를 안 주면 `<DATA_ROOT>/bigdata` 를 봅니다. 원본 train 으로만 학습하려면 `config.DATA["train_source"] = "orig"`.
산출물은 `<DATA_ROOT>/runs/` 에 쌓입니다 (`VQA_RUN_ROOT` 로 변경 가능).

### 학습·검증 데이터 (증강 데이터에서 valid 떼기)

증강 `bigdata/train.csv`(1만 문항)는 파일 이름으로 종류가 갈립니다: `orig_train_*`(원본 6,714) ·
`rot_train_*_rot{L,R}30`(원본 2,427장의 ±30° 회전본 2,685, 질문·정답 동일) · `auto_dev_*`(dev 문항 + 자동 라벨 601).
valid 는 **이미지 그룹** 단위로 뗍니다 — 원본과 그 회전본, 바이트가 같은 사진(다른 id 로 중복된 7묶음 + dev 사진 3장)은 한 그룹.
valid 그룹은 그룹 이름의 해시 순서로 10% 를 뽑고(qtype 규칙을 고쳐도 valid 가 바뀌지 않게), 그중 **무회전 원본 문항만**
valid(671문항)로 쓰며 그 그룹의 회전본 등은 학습에서도 뺍니다. 기본값은 `config.DATA` 에 있고(`valid_frac`, `valid_kinds`,
`train_kinds`, `split_seed`), 증강·원본 어느 쪽으로 학습해도 valid 문항이 같아서 "증강 효과" 비교가 같은 valid 로 됩니다
(`run.json` 의 `data.valid_hash` = `3ec8c30c2fca`). 비교 실험은 `--train-source orig`, `--train-kinds orig rot`(auto_dev 빼기).

증강 데이터의 원본 문항 id 는 원본 data.zip 과 같게 바꿔 씁니다(`orig_train_0006.jpg` → `train_0006.jpg`) — 증강·원본 실험의
예측 CSV 를 id 로 바로 비교·앙상블할 수 있습니다.

1단계가 저장하는 모든 어댑터 폴더(`adapter/`, `adapter_e<N>/`, `adapter_best/`)에는 `train_meta.json`(무슨 데이터·분할로 학습했는지)이
함께 들어가고, 2·4단계는 이 표식을 보고 같은 분할을 씁니다. valid 에 학습 문항이 섞일 수 있으면
(`--no-valid` 학습, 이 변경 전 코드로 원본 train 전체를 학습한 예전 어댑터, 1단계 뒤 분할 설정 변경) valid 는 건너뛰고 test 만 합니다.

모델 프리셋 (`--preset`) : `qwen3-vl-4b`(기본, bf16 약 8GB) · `qwen3-vl-8b`(bf16) · `qwen3.5-4b` · `qwen3.5-9b-bf16` ·
`qwen3.5-9b`(4bit) · `qwen2.5-vl-7b`(4bit) · `qwen2.5-vl-3b`.
모델이 `VQA_MODEL_ROOT` 에 없으면 자동으로 받습니다(이어받기·재시도). 프록시 오류가 계속되면 `--download-source modelscope`.

## 실험 설정 (seed·LoRA 랭크·프롬프트 비율 등)

모든 단계의 기본값은 `config.py` 의 "실험 설정" 절 한 곳에서 관리합니다.

| 이름 | 내용 |
|---|---|
| `SEED` | 기본 42. 학습 셔플·프롬프트 sampling·데이터 추출·추론 공통 |
| `DEFAULT_PRESET` / `RUN_NAME` | 기본 모델과 산출물 폴더 이름 (2·4단계는 `run.json` 에 기록된 값을 따름) |
| `LORA` / `LORA_TARGETS` | **랭크 r**, alpha(기본 2×r), dropout, LoRA 를 붙일 모듈 묶음 |
| `PROMPT` | base prompt 목록, qtype hint 확률(기본 30%), 추론 프롬프트 정책 |
| `USE_VALID` / `DATA` | valid 사용 여부, 학습 데이터(증강/원본)·valid 분할(비율·종류·split_seed) |
| `TRAIN` | epoch, lr, grad_accum, **valid 평가 간격(`eval_every`, best 선정 단위)**·체크포인트 간격 |
| `PREDICT` / `RERUN` | 2·4단계 배치, 재추론 방식, bbox·크롭·해상도 |
| `SELECT` / `FINAL` | 3·최종 단계 문턱값과 교체 방법 |

스크립트 인자(`--lr`, `--lora-r` 등)는 이 값을 기본값으로 씁니다. 인자는 한 번 시험해 볼 때만 쓰고,
채택한 값은 `config.py` 에 반영하세요. 설정과 결과는 `runs/experiments_<단계>.csv` 에 자동으로 쌓입니다.

## 실험 결과 저장 (로컬 + 구글 드라이브)

`runs/` 는 단계끼리 주고받는 **작업 폴더**고, 실험 결과 사본은 따로 모입니다.

| 저장 위치 | 경로 | 바꾸는 법 |
|---|---|---|
| 로컬 | `results/` (Colab VM 에서는 `/content/results`) | `VQA_RESULT_ROOT` |
| 구글 드라이브 | `<드라이브>/vqa/results/` — 마운트되어 있을 때만, 없으면 자동으로 건너뜀 | `VQA_DRIVE_DIR` |

두 곳에 **같은 파일**이 저장됩니다. 드라이브 쓰기가 실패해도 로컬 사본은 남고 경고만 출력합니다.

**실험(하이퍼파라미터 조합)마다 하위 폴더 하나**에 그 실험의 파일이 다 모입니다. 폴더 이름은 `config.exp_tag()` 가 만듭니다.

```
results/
  r16_qwen3-vl-4b_r16_e1_lr1e-4_hint30_aug_s42/  ← run · preset · 랭크 · epoch · lr · hint% · 데이터(aug, 원본이면 생략) · seed
    run.json  train_log.csv  experiments_s1_train.csv
    adapter_best/                                 ← valid 최고 업데이트의 어댑터 (VM 은 학습이 저장, PC 는 fetch-best 로 받음)
    test_pred.csv  train_pred.csv  test_qtype.csv  submission.csv  predict.json  experiments_s2_predict.csv
    test_subset.csv  select_qtype_report.csv  selection.json  experiments_s3_select.csv
    rerun-grounding-adapter/                      ← 재추론 방식별로 한 단계 더
      test_rerun.csv  rerun.json  final_submission.csv  final_meta.csv  final.json
      experiments_s4_rerun.csv  experiments_s5_final.csv
  r32_qwen3-vl-4b_r32_e1_lr1e-4_hint30_s42/
  ens_r16+r32/                                    ← 앙상블 (ensemble.py)
  summary.csv  experiments_<단계>.csv              ← collect_results.py 가 하위 폴더를 훑어 만든 비교표·합본
  probe_batch.json
```

| 파일 | 내용 |
|---|---|
| `{train,valid,test}_pred.csv` | split 별 문항 단위 예측 (`p_a..p_d`, `pred`, `conf`, `margin`, `prompt`) |
| `{train,valid}_qtype.csv` | split 별 qtype 정확도 표 |
| `submission.csv` / `rerun-*/final_submission.csv` | 2단계 기준 제출 / 재추론 적용 최종 제출 |
| `run.json`, `predict.json`, `rerun-*/final.json` | 학습·추론·최종 설정과 요약 수치 |
| `train_log.csv` | valid 평가마다 한 줄 : update · epoch · loss · valid 정확도 · valid NLL · best 여부 |
| `adapter_best/` | valid 최고 업데이트의 LoRA 어댑터 + `best.json`(업데이트·점수). `runs/<run>/adapter/` 와 같은 것 |
| `{valid,test}_subset.csv`, `rerun-*/*_rerun.csv` | 선별 문항과 재추론 결과 |
| `experiments_<단계>.csv` | 그 실험의 단계별 기록 한 줄 (`runs/` 에는 `experiments_<단계>__<run>.csv`) |
| `summary.csv` (최상위) | `python collect_results.py` 가 모든 하위 폴더를 합쳐 만든 **실험 비교표** |

코드에서는 `save_result(obj, "<실험이름>__<파일>")` 처럼 `__` 로 이어 쓰면 `__` 마다 폴더가 갈립니다 (`data_utils.result_relpath`).

valid·test 예측은 2단계가 한 번에 만듭니다(`config.PREDICT["splits"]`).
train 예측(학습한 문항을 다시 풀어 **암기·과적합**을 보는 진단, 약 20분)은 제출에 쓰이지 않아 기본에서 뺐습니다.
필요하면 `--splits valid test train` — 마지막에 `train - valid` 차이를 출력합니다.

Colab 에서 로컬 PC 로 가져오려면:

```bash
./colab/colab_pipeline.sh fetch              # 요약·제출 파일 (작은 것만)
FETCH_ALL=1 ./colab/colab_pipeline.sh fetch  # 예측 CSV·선별·재추론 CSV 까지
RUN=aug_r16 ./colab/colab_pipeline.sh fetch-best   # best 어댑터 → results/<실험이름>/adapter_best/ (이미 같으면 건너뜀)
python collect_results.py                    # experiments_*__*.csv → summary.csv (비교표)
```

여러 VM 이 같은 드라이브에 동시에 쓰면 한 파일을 읽고-고쳐-쓰다 행이 사라지므로, 기록은 실험별 폴더에 남기고
합치는 일은 로컬에서 `collect_results.py` 가 합니다. `fetch` 도 하위 폴더 구조 그대로 받아옵니다.

### LoRA 랭크·적용 범위

랭크를 바꿀 때는 산출물이 섞이지 않게 `--run` 이름도 함께 바꿉니다. alpha 를 따로 주지 않으면 `2×r` 로 맞춰지므로
스케일(alpha/r)은 그대로고 랭크만 달라집니다. 지금까지 랭크 16↔32↔64 는 Kaggle 차이가 없었습니다.

적용 범위는 `config.LORA_TARGETS` (`--lora-targets`):

| 이름 | 붙는 곳 | 9B(Qwen3.5) 모듈 수 |
|---|---|---|
| `attn` | 일반 어텐션 q/k/v/o | 32 |
| `attn+mlp` | + MLP (예전 기본값 — 9B 의 선형 어텐션 24층을 놓침) | 128 |
| `lm_all` (기본) | 언어 모델 선형층 전부 (선형 어텐션 `in_proj_qkv/in_proj_z/out_proj` 포함) | 200 |
| `lm_all+vision` | + 비전 인코더 블록(`qkv/proj/linear_fc1·2`)과 연결부(`merger`) | 310 |

```bash
python s1_train_choice_ce.py --run q9b                                  # config 기본값 (9B, lm_all, 해상도 1024)
python s1_train_choice_ce.py --run q9b_vis --lora-targets lm_all+vision # 비전 인코더까지
python s1_train_choice_ce.py --run q9b_r32 --lora-r 32
```

`runs/experiments_s1_train.csv` 의 `lora_r` · `lora_targets` · `best_valid_acc` 열로 비교하세요.

## 프롬프트 엔지니어링

- **base prompt pool** : `model_utils.BASE_PROMPTS` (p1~p4). 표현만 다르고 지시하는 task 는 같아야 합니다.
- **qtype hint** : `model_utils.QTYPE_HINTS`. `data_utils.QTYPE_RULES` 의 유형 이름을 키로 씁니다.
- **출력 규칙** : `model_utils.ANSWER_RULE` 한 줄. 모든 프롬프트에서 동일해야 비교가 됩니다.

```bash
python s1_train_choice_ce.py --run gen  --hint-prob 0            # hint 없이 (기준선)
python s1_train_choice_ce.py --run hint --hint-prob 0.3          # hint 30%
python s1_train_choice_ce.py --run p12  --base-prompts p1 p2     # 일부 표현만
python s2_predict.py --run hint --prompt-scan                    # 프롬프트를 바꿨을 때 예측 일치율
```

`--prompt-scan` 은 valid 에서 base prompt 를 하나씩 바꿔 정확도와 일치율만 재는 진단입니다(제출 답에는 영향 없음).
"모든 프롬프트가 같은 답" 비율이 낮으면 아직 표현에 민감하다는 뜻입니다.

## 먼저 작게 돌려보기 (권장)

전체를 돌리기 전에 몇 분 안에 끝나는 설정으로 파이프라인이 끝까지 도는지 확인하세요.

```bash
python s1_train_choice_ce.py --run smoke --n-train 200 --max-updates 20 --eval-n 50 --eval-every 10 --ckpt-every 10 --fresh
python s2_predict.py --run smoke --n-train 100 --n-valid 100 --n-test 50
python s3_select_rerun.py --run smoke
python s4_rerun_grounding.py --run smoke --save-crops 10
python s5_build_final.py --run smoke --rerun-tag grounding_adapter
```

4단계 뒤 `runs/rerun/<tag>/crops/` 의 이미지를 꼭 열어 보세요. 빨간 박스(모델 bbox)와 파란 박스(실제 크롭)가
질문과 관련된 곳에 있어야 합니다. 박스가 일관되게 엉뚱한 곳에 있으면 `config.py` 프리셋의 `coord` 값 문제입니다.

## 전체 실행

`run_pipeline.sh` 에 순서대로 적혀 있습니다. 제출용 test 추론은 반드시 `--n-test` 없이(전체) 돌리세요.

## 실험 여러 개 돌리기 (Colab VM 여러 대)

**이 프로젝트의 기본 실험 방식입니다 — 실험은 항상 VM 2대(A100 1장씩)로 한 쌍씩 병렬로 돌립니다.**
Colab 런타임은 GPU 1장이라 병렬은 **VM 여러 대**로 합니다. 계정마다 동시 할당 한도가 있으므로
(`TooManyAssignmentsError`) VM 수는 `MAX_PARALLEL`(기본 2)로 제한하고, 실험이 더 많으면
**한 VM 이 자기 몫을 순서대로 이어서** 돌립니다. 이어지는 것도 VM 안에서 일어나므로 로컬 터미널은 꺼도 됩니다.

```bash
./colab/sweep.sh plan       # 어떤 VM 이 어떤 실험을 어떤 순서로 돌릴지
./colab/sweep.sh up         # VM 할당 + 준비 (순차)
./colab/sweep.sh start      # VM 마다 자기 몫 전부를 한 작업으로 제출, 즉시 반환
./colab/sweep.sh status     # 진행 상황
./colab/sweep.sh evals      # 실험별 업데이트 단위 valid 평가·best 현황
./colab/sweep.sh logs 40 1  # 1번 VM 로그 40줄
./colab/sweep.sh fetch      # 결과 회수 + 비교표(summary.csv)
./colab/sweep.sh fetch-best # 실험별 best 어댑터를 PC 로
./colab/sweep.sh down       # 반납
```

실험 4개 · `MAX_PARALLEL=2` 면 VM1 = 1·3번 실험, VM2 = 2·4번 실험을 차례로 돌립니다.
한 실험이 실패해도 다음 실험은 진행하고, 작업 종료 코드는 1 로 남습니다(로그에 `실험 … 실패` 표시).

실험 목록은 `colab/sweep.sh` 의 `EXPS` 또는 `colab/sweep.conf` 에 `실험이름 | s1 추가 인자` 로 적습니다.
실험 이름이 `--run` 이름이 되므로 서로 달라야 합니다.

## 단계마다 valid 로 판단할 것

| 단계 | 출력에서 볼 것 | 판단 |
|---|---|---|
| 1 | `[평가] upd N/…` 줄(= `train_log.csv`, `colab_pipeline.sh evals`), `▲ best` 표시, `hint 적용 N%` | valid 최고 업데이트(동률이면 NLL 낮은 쪽)가 `adapter/`·`adapter_best/`, epoch 끝은 `adapter_e<N>/`. best 가 학습 초반에 나오면 epoch·lr 을 줄여 볼 것. 중단되면 같은 명령으로 다시 실행 → 체크포인트에서 이어짐 |
| 2 | valid 정확도, qtype 표, `--prompt-scan` 의 일치율 | hint·랭크 설정을 바꾼 run 끼리 비교. 일치율이 낮으면 프롬프트 augmentation 을 더 넓힐 여지 |
| 3 | 선별 비율, 선별 문항 정확도, **오답 포착률** | 오답 포착률은 높고 선별 비율은 낮을수록 좋음. 약한 qtype 은 여기서 확인 |
| 4 | 선별 문항에서 고쳐짐 / 망가짐, 크롭 O·X 별 정확도 | `--pass2 adapter` 와 `base`, `--mode prompt` 를 비교해 가장 나은 하나만 채택 |
| 최종 | keep / replace / rule / blend 비교 | keep 보다 나은 방법만 채택. 차이가 작으면 keep |

문턱값은 valid 로만 정하고 test 로 정하지 마세요. valid 문항 수가 적을수록(특히 3·4단계의 선별 문항) 우연한 차이가 커집니다.

## 확장 포인트

- **qtype 규칙** : `data_utils.QTYPE_RULES`. valid 오답을 보며 계속 다듬으세요. hint 선택과 3단계 약한 유형 선정이 여기에 달려 있습니다.
- **프롬프트** : `BASE_PROMPTS` 에 표현을 추가하면 학습 sampling 과 `--prompt-scan` 에 자동으로 들어갑니다.
- **다른 재추론** : thinking 모드(부정형·지식형), 다른 계열 모델을 같은 입출력 형식(`*_rerun.csv`)으로 만들면 최종 단계에서 그대로 비교할 수 있습니다.
