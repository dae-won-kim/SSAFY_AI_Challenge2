# Colab CLI 로 파이프라인 실행하기

로컬 터미널에서 Colab GPU VM 을 빌려 파이프라인을 돌립니다. 브라우저 탭을 열어둘 필요가 없고,
명령 한 줄로 단계별 실행·진행 확인·결과 회수를 할 수 있습니다.

```
 로컬 PC (WSL Ubuntu)                         Colab VM (A100 등)
 ─────────────────────                        ───────────────────────────────────
 colab_pipeline.sh ── colab exec ───────────► /content/vqa_pipeline   코드 (매번 최신으로 교체)
   up / sync / run                             /content/data           데이터 (드라이브 zip 을 복사해 풂)
   wait / logs / fetch ◄── 로그·상태 ──────────  /content/models         모델 가중치 (로컬 디스크)
                                               /content/drive/MyDrive/vqa/runs   산출물 (드라이브 → 세션이 끊겨도 남음)
```

| 대상 | 위치 | 이유 |
|---|---|---|
| 데이터 | 드라이브에 **zip 1개**로 보관 → VM 로컬 디스크에 풀어서 사용 | 이미지 수천 장을 드라이브에서 직접 읽으면 느림 |
| 산출물 (어댑터·체크포인트·예측·제출 파일) | 드라이브 `runs/` | VM 이 사라져도 남고, 다음 세션에서 이어서 사용 |
| 모델 가중치 | 드라이브 `vqa/models/<모델이름>/` 에 보관 → VM 준비(`prep`·`sync`) 때 VM 로컬 디스크 `/content/models` 로 복사 | 처음 한 번만 Hugging Face 에서 받아 드라이브에 저장, 이후 VM 은 HF 없이 드라이브에서 복사 (`model_store.py`). 읽기는 로컬 디스크에서 |

## 1. 한 번만 하는 준비

**① WSL Ubuntu 설치 (Windows 인 경우)**
Colab CLI 는 현재 Linux·macOS 만 지원합니다. 관리자 PowerShell 에서:

```powershell
wsl --install -d Ubuntu
```

**② Ubuntu 안에서 Colab CLI 설치**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv 설치 (새 터미널을 열거나 source ~/.bashrc)
uv tool install --force google-colab-cli --overrides scripts/colab_cli_overrides.txt   # 저장소 폴더에서 (0.7.2 의존성 버그 우회)
colab version
```

**③ 코드를 WSL 홈으로 복사**

```bash
cp -r /mnt/d/<코드가 있는 폴더>/vqa_pipeline ~/
cd ~/vqa_pipeline && chmod +x colab/colab_pipeline.sh
```

**④ 데이터 zip 을 드라이브에 올리기**
`train.csv`, `dev.csv`, `test.csv`, `sample_submission.csv` 와 `train/`, `dev/`, `test/` 이미지 폴더를 zip 으로 묶어
**내 드라이브 → `vqa/data.zip`** 에 올립니다. zip 안에서 한 단계 폴더로 감싸져 있어도 자동으로 찾습니다.

```bash
cd /workspace/ai_challenge   # 데이터가 있는 곳 (zip 명령이 없어도 되도록 파이썬으로 묶음)
python3 -m zipfile -c data.zip train.csv dev.csv test.csv sample_submission.csv train dev test
```

증강 학습 데이터는 **내 드라이브 → `vqa/Bigdata_v2.zip`** (zip 안이 `bigdata/train.csv` + `bigdata/train/*.jpg`).
`sync`(·`up`·`prep`)가 두 zip 을 VM 로컬 디스크의 `/content/data`(원본, test·제출) 와 `/content/data_aug`(증강, 학습·valid) 에
각각 풀어 둡니다. 드라이브의 zip 을 새 파일로 바꾸면(이름·크기 변화) 다음 `sync` 때 다시 풉니다.
다른 이름이면 `TRAIN_ZIP=<파일명> ./colab/colab_pipeline.sh sync`, 증강 없이 원본으로만 학습하려면 `TRAIN_ZIP=` 로 비우고
`config.DATA["train_source"] = "orig"`.

**⑤ 첫 인증**
처음 `colab new` 를 실행할 때 Google 계정 인증 안내가 나옵니다. 출력되는 안내를 따라 주세요.
드라이브 마운트(`colab drivemount`)도 처음에 인증을 요구할 수 있습니다.

## 2. 실행 순서

```bash
cd ~/vqa_pipeline

./colab/colab_pipeline.sh up A100      # VM 할당 + 드라이브 + 패키지 + 코드·데이터 (GPU: T4/L4/A100/H100)
#   세션을 colab new -s vqa 로 직접 만들었다면 up 대신 : ./colab/colab_pipeline.sh prep
./colab/colab_pipeline.sh smoke        # 작은 설정으로 전체 단계 점검 → runs_smoke/ (본 실험과 분리)
#   → 드라이브 vqa/runs_smoke/rerun/A_adapter/crops/ 이미지로 bbox·크롭 위치 확인
./colab/colab_pipeline.sh pipeline     # 전체 실행 (단계마다 끝날 때까지 기다리며 진행 표시)
./colab/colab_pipeline.sh fetch        # 제출 파일·실험 기록 → ~/vqa_pipeline/results/
./colab/colab_pipeline.sh down         # VM 반납 (반납하지 않으면 컴퓨팅 단위가 계속 소모됨)
```

seed·epoch·lr·LoRA 랭크·프롬프트 비율·배치·문턱값·기본 모델은 `config.py` 의 "실험 설정" 에서 바꾸고 `sync` 로 올립니다.
잠깐 바꿔 돌릴 때는 환경변수를 씁니다 (주지 않으면 `config.py` 값).

```bash
PRESET=qwen3-vl-8b BS=16 ./colab/colab_pipeline.sh pipeline    # 모델·배치 변경
EPOCHS=2 ./colab/colab_pipeline.sh pipeline                    # 1단계 epoch 수 변경 (실험 기록 CSV 에 남음)
RUN=q9b_r32 LORA_R=32 ./colab/colab_pipeline.sh pipeline       # LoRA 랭크 비교 (산출물은 runs/q9b_r32/ 에 따로)
DRIVE_DIR=/content/drive/MyDrive/ssafy ./colab/colab_pipeline.sh up L4
```

`RUN` 은 산출물 폴더 이름입니다(기본 `main`). 설정을 바꿔 비교할 때는 `RUN` 도 함께 바꿔야 이전 결과를 덮어쓰지 않습니다.

실험 결과는 **VM 로컬 `/content/results/` 와 드라이브 `<DRIVE_DIR>/results/` 양쪽에 같은 파일**로 저장되고,
파일 이름에 모델·LoRA 랭크·epoch·lr·hint 비율·seed 가 들어갑니다
(예: `main_qwen3-vl-4b_r16_e1_lr1e-4_hint30_s42__valid_pred.csv`). VM 이 사라져도 드라이브 사본은 남습니다.

```bash
./colab/colab_pipeline.sh fetch              # 요약·제출 파일만 로컬 ./results/ 로
FETCH_ALL=1 ./colab/colab_pipeline.sh fetch  # 예측 CSV·선별·재추론 CSV 까지 (느림)
```

**GPU 별 권장 프리셋**

| GPU (메모리) | `PRESET` | `BS` |
|---|---|---|
| T4 (16GB) | `qwen3-vl-4b` (학습 인자 `--no-grad-ckpt` 없이, `--train-max-side 768`) | 8~16 |
| L4 (24GB) | `qwen3-vl-4b` · `qwen3-vl-8b` | 16~32 |
| A100 (40GB 이상) · H100 | `qwen3.5-9b-bf16` (**기본값**, 2026-09-26~) | 주지 않음 → config 값 |

`config.py` 의 배치 기본값은 A100 기준입니다. 다른 GPU 에서는 `BS` 로 줄이거나, `./colab/colab_pipeline.sh run probe python probe_batch.py` 로 그 GPU 의 상한을 재서 반영하세요.

학습 중 OOM 이 나면 `s1` 에 `--train-max-side 640` 을 주거나 더 작은 프리셋을 쓰세요. 추론 OOM 은 배치만 1개씩으로 자동 재시도합니다.

## 3. 단계별로 직접 다루기

`pipeline` 대신 단계를 하나씩 돌리며 결과를 보고 다음 설정을 정할 수도 있습니다.

```bash
./colab/colab_pipeline.sh run  s1 python s1_train_choice_ce.py --run main
./colab/colab_pipeline.sh run  r32 python s1_train_choice_ce.py --run r32 --lora-r 32   # 랭크 비교
./colab/colab_pipeline.sh logs s1 40         # 로그 마지막 40줄
./colab/colab_pipeline.sh evals main         # 업데이트별 valid 평가·best + 단계(학습→예측→재추론→제출 파일) 진행 (비용 없음)
./colab/colab_pipeline.sh models             # 모델 가중치 보관 상태 (드라이브 vqa/models/ · VM 로컬)
./colab/colab_pipeline.sh models push qwen3-vl-8b      # VM 에 받아 둔 모델을 드라이브에 저장 (보통은 자동)
PRESET=qwen3-vl-8b ./colab/colab_pipeline.sh sync      # 다른 프리셋을 드라이브에서 미리 가져오기 (PRESETS=all 이면 보관소 전부)
./colab/colab_pipeline.sh fetch-best main    # best 어댑터 → 로컬 results/<실험이름>/adapter_best/ (이미 같으면 건너뜀)
./colab/colab_pipeline.sh wait s1            # 끝날 때까지 대기
./colab/colab_pipeline.sh status             # GPU 사용량 + 작업 목록
./colab/colab_pipeline.sh progress           # 작업별 학습 % · test 예측 % · 지금 단계 · 제출 파일 (watch 도 이 내용을 주기적으로 출력)
./colab/colab_pipeline.sh stage s3 python s3_select_rerun.py --max-frac 0.2    # run + wait
./colab/colab_pipeline.sh pipeline s2        # 특정 단계부터 끝까지 (s1 s2 s3 s4g s4r s5)
./colab/colab_pipeline.sh console            # VM 쉘에 직접 접속 (nvidia-smi, ls 등)
```

GPU 작업(1·2·4단계, `probe_batch.py`)은 **빈 GPU 를 한 장 골라** 거기에 고정해서 돌립니다(`CUDA_VISIBLE_DEVICES`).
빈 GPU 가 없으면 시작하지 않습니다 (한 GPU 에 학습 두 개가 겹치면 OOM). GPU 를 직접 고르려면 `GPUS=1 …`,
한 작업에 여러 장을 주려면 `GPUS=0,1 …`, 알고 겹쳐 돌리려면 `FORCE=1 …`.

Colab 런타임은 보통 **GPU 1장**입니다(`--gpu` 는 종류만 고름). 여러 GPU 로 한 학습을 나누는 구성은 불가능하니,
병렬이 필요하면 세션을 여러 개 띄우세요: `COLAB_SESSION=vqa2 ./colab/colab_pipeline.sh up A100`.
실험 여러 개를 VM 몇 대에 나눠 돌리는 일은 `colab/sweep.sh` 가 대신합니다(동시 VM 수 = `MAX_PARALLEL`, 기본 2).

로컬에서 코드를 고쳤다면 `sync` 로 다시 올리면 됩니다(데이터는 다시 복사하지 않음).
3·최종 단계는 CPU 만 쓰므로 문턱값을 바꿔 여러 번 돌려도 빠릅니다.

## 4. 끊겼을 때

작업은 VM 안에서 **백그라운드 프로세스**로 돌기 때문에, 로컬 터미널을 닫거나 `wait` 를 중단해도 계속 진행됩니다.
다시 `logs` / `wait` 로 확인하면 됩니다.

| 상황 | 대처 |
|---|---|
| 로컬 터미널이 끊김 | 그대로 진행 중. `wait <작업>` 으로 다시 확인. 학습 중 best 는 `evals <run>` |
| VM 이 사라짐 | `up` 으로 새 VM → `pipeline <실패한 단계>`. 1단계는 드라이브의 체크포인트에서 **자동으로 이어서 학습** |
| 1단계를 처음부터 다시 | `s1` 명령에 `--fresh` |
| `Session 'vqa' appears to be lost (404/401)` | VM 은 살아 있는데 CLI 가 1시간 만료된 토큰 때문에 기록을 지운 것일 수 있음. `colab sessions` 의 `[?]` 엔드포인트로 `./colab/colab_pipeline.sh adopt <엔드포인트>` → `status`. (스크립트가 20분마다 토큰을 갱신해 새로 생기지는 않게 함) |

**VM 이 유지되는 조건**
- VM 을 유지해 주는 keep-alive 는 **로컬 PC 에서 도는 백그라운드 프로세스**입니다. 긴 학습 중에는 PC 가 절전·종료되지 않게 해 두세요.
- keep-alive 는 안전장치로 24시간 뒤 스스로 종료되므로, 그보다 긴 작업은 중간에 끊길 수 있습니다. 체크포인트 덕분에 이어서 돌릴 수 있습니다.
- 실행 중인 작업의 로그는 VM 의 `/content/logs/` 에 쓰이고, **5분마다와 작업이 끝날 때** 드라이브 `runs/logs/` 로 복사됩니다. 도중에 VM 이 사라지면 마지막 5분 치 로그는 없을 수 있습니다(학습 체크포인트·best 어댑터·`train_log.csv` 는 드라이브에 남음).

## 5. 제출

대회는 Kaggle 에 결과 파일만 올리는 온라인 방식이라, 추론 환경에 맞춰 모델을 줄일 필요가 없습니다.
`fetch` 로 받은 `results/final_submission.csv` 를 그대로 제출하면 됩니다.
