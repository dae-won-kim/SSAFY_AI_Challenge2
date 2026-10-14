"""스크립트를 어느 위치에서 실행하더라도 data_pipeline 패키지를 불러올 수 있도록 경로를 등록합니다."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def add_common_args(parser):
    """세 스크립트가 공통으로 사용하는 실행 인자를 등록합니다."""
    parser.add_argument("--work-dir", default="work",
                        help="압축 해제·중간 결과를 저장할 로컬 작업 폴더입니다. (기본값: work)")
    parser.add_argument("--output-dir", default="outputs",
                        help="최종 zip과 미리보기 이미지를 저장할 폴더입니다. Colab에서는 Drive 경로를 지정하시면 됩니다. "
                             "(기본값: outputs)")
    parser.add_argument("--zip-password", default=None,
                        help="원본 데이터 압축 비밀번호입니다. 지정하지 않으면 환경 변수 DATA_ZIP_PASSWORD를 사용합니다.")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드입니다. (기본값: 42)")
    parser.add_argument("--no-zip", action="store_true", help="결과를 zip으로 묶지 않고 작업 폴더에만 남깁니다.")
    return parser
