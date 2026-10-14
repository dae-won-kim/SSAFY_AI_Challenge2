# AI 챌린지 데이터 전처리 및 증강 파이프라인

이미지 기반 객관식 질의응답(VQA) 대회 데이터를 위한 **전처리·증강·통합 파이프라인**입니다.

## 📌 전체 흐름

```
대회 원본 데이터 (train.csv + train/)
        │
        ├── [1단계] 회전 증강 ────────► train_rot30_data.zip  (원본 1장당 ±30도 회전본 2장)
        │                                     │
        ├── [2단계] 열화 증강 ────────► aug_data.zip          (원본 1장당 열화본 N장, 독립 산출물)
        │                                     │
        └── [3단계] 데이터 통합 ◄─────────────┘
                 + 자동 라벨링 dev 데이터 (labeled_final.csv + dev/)
                        │
                        ▼
                 Bigdata_v2.zip  (train.csv + train/, 기본 10,000개)
```

| 단계 | 스크립트 | 입력 | 결과물 |
|---|---|---|---|
| 1 | `scripts/01_rotation_augmentation.py` | 원본 데이터 | `train_rot30_data.zip` (`train_rot30.csv`, `train_rot30/`) |
| 2 | `scripts/02_degradation_augmentation.py` | 원본 데이터 | `aug_data.zip` (`aug_train.csv`, `aug_meta.csv`, `train_aug/`) |
| 3 | `scripts/03_merge_bigdata.py` | 원본 + 1단계 결과 + 자동 라벨링 데이터 | `Bigdata_v2.zip` (`bigdata/train.csv`, `bigdata/train/`) |

> 2단계 결과물은 3단계 통합에 사용되지 않는 독립 산출물입니다. 학습 코드에서 원본 id(`source_id`)로 증강본을 골라 쓰는 용도입니다.

## 📁 폴더 구조

```
ai-challenge-data-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── data_pipeline/            # 재사용 가능한 핵심 로직 (패키지)
│   ├── __init__.py
│   ├── common.py             # 공통 컬럼 정의, 압축 해제/저장, 데이터 로드
│   ├── rotation.py           # 고정 각도 회전 증강
│   ├── degradation.py        # 9종 열화 함수 + 설정(DegradeConfig) + 증강 생성
│   ├── quality.py            # 원본 화질 분석 및 열화 강도 자동 재설정
│   ├── merge.py              # 세 데이터 소스 통합 (컬럼 통일, 샘플링, 이미지 모으기)
│   ├── validation.py         # 무결성 검사
│   └── preview.py            # 증강 결과 미리보기 이미지 저장
└── scripts/                  # 단계별 실행 스크립트 (번호 순서대로 실행)
    ├── _bootstrap.py         # 패키지 경로 등록 + 공통 실행 인자
    ├── 01_rotation_augmentation.py
    ├── 02_degradation_augmentation.py
    └── 03_merge_bigdata.py
```

## ⚙️ 설치

```bash
git clone https://github.com/<사용자명>/ai-challenge-data-pipeline.git
cd ai-challenge-data-pipeline
pip install -r requirements.txt
```

비밀번호가 걸린 압축 파일은 `7z` 명령어로 해제합니다. Colab에는 기본으로 설치되어 있으며, 로컬에서는 아래처럼 설치하실 수 있습니다.

```bash
# Ubuntu
sudo apt install p7zip-full
# macOS
brew install p7zip
```

## 🔑 압축 비밀번호 설정

```bash
export DATA_ZIP_PASSWORD="비밀번호"
```

또는 실행할 때 `--zip-password "비밀번호"` 인자로 전달하셔도 됩니다.

## 🚀 실행 방법

### 로컬에서 실행

```bash
# 1단계: 회전 증강
python scripts/01_rotation_augmentation.py --data-zip data/ssafy-16-2-ai.zip

# 2단계: 열화 증강 (원본 1장당 3장)
python scripts/02_degradation_augmentation.py --data-zip data/ssafy-16-2-ai.zip --n-aug 3

# 3단계: 데이터 통합 (총 10,000개)
python scripts/03_merge_bigdata.py \
    --data-zip data/ssafy-16-2-ai.zip \
    --rot-zip outputs/train_rot30_data.zip \
    --labeled-csv data/labeled_final.csv \
    --dev-root data
```

결과 zip과 미리보기 이미지는 `outputs/`에, 중간 작업 파일은 `work/`에 저장됩니다.
이미 압축을 풀어 두셨다면 `--data-zip` 대신 `--data-dir <폴더>`를 사용하시면 됩니다.

### Colab에서 실행

```python
# 1) Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 2) 저장소 받기 + 설치
!git clone https://github.com/<사용자명>/ai-challenge-data-pipeline.git
%cd ai-challenge-data-pipeline
!pip install -q -r requirements.txt

# 3) 비밀번호 설정 (셀에 직접 적는 대신 Colab '보안 비밀' 기능을 쓰시면 더 안전합니다)
import os
os.environ["DATA_ZIP_PASSWORD"] = "비밀번호"

# 4) 단계별 실행 — 결과를 Drive에 바로 저장합니다
DRIVE = "/content/drive/MyDrive"
!python scripts/01_rotation_augmentation.py --data-zip {DRIVE}/ssafy-16-2-ai.zip --work-dir /content/work --output-dir {DRIVE}
!python scripts/02_degradation_augmentation.py --data-zip {DRIVE}/ssafy-16-2-ai.zip --work-dir /content/work --output-dir {DRIVE}
!python scripts/03_merge_bigdata.py --data-zip {DRIVE}/ssafy-16-2-ai.zip --rot-zip {DRIVE}/train_rot30_data.zip \
    --search-root {DRIVE} --work-dir /content/work --output-dir {DRIVE}
```

> `--work-dir`을 `/content/...`(로컬 디스크)로 두는 이유는 Drive에서 직접 이미지를 읽고 쓰면 매우 느리기 때문입니다.
> 3단계의 `--search-root`를 지정하면 `labeled_final.csv`와 `dev/` 폴더를 Drive 안에서 자동으로 찾습니다.

## 🔧 주요 실행 인자

**공통**

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--data-zip` / `--data-dir` | (필수, 둘 중 하나) | 원본 데이터 압축 파일 / 압축을 푼 폴더 |
| `--work-dir` | `work` | 중간 작업 폴더 |
| `--output-dir` | `outputs` | 결과 zip·미리보기 저장 폴더 |
| `--zip-password` | 환경 변수 `DATA_ZIP_PASSWORD` | 압축 비밀번호 |
| `--seed` | `42` | 랜덤 시드 |
| `--no-zip` | - | zip으로 묶지 않고 작업 폴더에만 남김 |

**단계별**

| 스크립트 | 인자 | 기본값 | 설명 |
|---|---|---|---|
| 01 | `--angle` | `30` | 회전 각도 (+각도/-각도 두 장 생성) |
| 01, 02 | `--workers` | `4` | 병렬 처리 스레드 수 |
| 01, 02 | `--no-preview` | - | 미리보기 이미지 저장 생략 |
| 02 | `--n-aug` | `3` | 원본 1장당 열화본 수 |
| 02 | `--no-tune` | - | 화질 분석 기반 강도 재설정을 생략하고 기본 강도 사용 |
| 03 | `--rot-zip` / `--rot-dir` | (필수, 둘 중 하나) | 1단계 결과 |
| 03 | `--labeled-csv`, `--dev-root` | - | 자동 라벨링 CSV, `dev/`의 부모 폴더 |
| 03 | `--search-root` | - | 위 두 경로를 자동으로 찾을 폴더 |
| 03 | `--target-total` | `10000` | 최종 데이터 개수 |
| 03 | `--n-rot` | `2685` | 회전 증강본 사용 개수 |
| 03 | `--min-confidence` | `0.60` | 자동 라벨링 데이터 최소 신뢰도 |

## 🧪 단계별 상세 설명

### 1단계 — 회전 증강
- 원본 이미지마다 반시계 30도(`rotL30`), 시계 30도(`rotR30`) 회전본 2장을 만듭니다.
- 회전 후 생기는 빈 모서리는 가장자리 픽셀을 복제해서 채웁니다.
- 결과 CSV에는 **회전본만** 들어가며, `source_id` 컬럼으로 원본 id를 확인할 수 있습니다.
- `outputs/preview_rotation.png`로 회전 방향이 의도와 맞는지 확인해 주세요.

### 2단계 — 열화 증강
실제 촬영 환경에서 생길 수 있는 화질 저하 9종 + JPEG 압축을 1~3개 랜덤 조합해 적용합니다.

| 계열 | 열화 | 설명 |
|---|---|---|
| 기하 변형 | `geo` | 기울임 + 원근 왜곡 |
| | `lens` | 렌즈 왜곡 (술통형/실패꾸러미형) |
| | `warp` | 종이 휘어짐 |
| 조명 | `shadow` | 한쪽이 어두워지는 그림자 |
| | `low_light` | 저조도 + 노이즈 |
| | `glare` | 빛반사 |
| 화질 | `blur` | 가우시안 / 모션 / 초점 흐림 |
| | `lowres` | 저해상도 |
| | `noise` | 가우시안 노이즈 |
| 저장 | `jpeg` | 마지막 저장 시 한 번만 적용 (압축 열화 누적 방지) |

- **강도 자동 재설정**: train 원본의 blur·밝기 분포를 분석해서 열화 강도와 mild/mid/severe 비율을 정합니다.
- **식별 가드**: 열화 후 너무 어둡거나, 하얗게 날아가거나, 지나치게 뭉개지면 강도를 낮춰 최대 4번까지 다시 시도합니다.
- **재현성**: 이미지 id와 번호로 시드를 고정하므로, 같은 시드로 다시 실행하면 같은 결과가 나옵니다.
- `aug_meta.csv`에 이미지별 적용 열화·강도·파라미터·화질 지표가 기록됩니다.
- 강도를 직접 조정하고 싶으시면 `data_pipeline/quality.py`의 `tune_config()` 값을 수정해 주세요.

학습 코드에서 원본 id로 증강본을 하나 골라 쓰는 예시입니다.

```python
import pandas as pd
from data_pipeline.degradation import pick_random_augmented

aug_df = pd.read_csv("aug_data/aug_train.csv", dtype=str, keep_default_na=False)
row = pick_random_augmented(aug_df, "train_0001.jpg")
print(row["path"])   # train_aug/train_0001_aug02.jpg
```

### 3단계 — 데이터 통합
| 소스 | id 접두사 | 사용량 |
|---|---|---|
| 원본 `train.csv` | `orig_` | 전체 |
| 회전 증강 `train_rot30.csv` | `rot_` | 무작위 `--n-rot`개 |
| 자동 라벨링 `labeled_final.csv` | `auto_` | 신뢰도 필터링 후, 총합이 `--target-total`이 되도록 나머지 무작위 |

- 자동 라벨링 데이터는 보기별 신뢰도(`a_confidence` ~ `d_confidence`) 중 최댓값이 `--min-confidence` 이상인 행만 사용합니다.
- 소스별 접두사를 붙여 id 충돌을 막고, 모든 이미지를 하나의 `train/` 폴더로 모읍니다.
- `labeled_final.csv`의 컬럼명이 표준 형식과 다르면 `data_pipeline/merge.py`의 `AUTO_COLUMN_MAP`을 수정해 주세요.

## ✅ 무결성 검사

각 단계는 저장 전에 아래 항목을 자동으로 검사하며, 하나라도 실패하면 즉시 중단됩니다.

- 모든 이미지 경로가 실제로 존재하는지, 샘플 이미지가 정상적으로 열리는지
- id 중복이 없는지
- (1·2단계) 원본마다 증강본 수가 정확한지, 질문·보기·정답이 원본과 같은지
- (3단계) 최종 개수가 목표와 같은지, 정답이 a~d 범위인지, 빈 보기가 없는지

## ⚠️ 주의사항

- 2단계 zip 생성 시 `train_aug/` 폴더는 복사가 아니라 이동되므로, 작업 폴더에서 다시 확인하시려면 2단계를 재실행해 주세요. (1단계의 `train_rot30/`도 동일합니다)
