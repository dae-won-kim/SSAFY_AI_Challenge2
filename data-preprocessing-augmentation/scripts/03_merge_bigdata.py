"""[3단계] 세 데이터 소스를 통합해서 Bigdata 학습셋 만들기 (기본 10,000개)

| 소스 | 내용 | 사용량 |
|---|---|---|
| 원본 | 대회 제공 train.csv + train/ | 전체 |
| 회전 증강 | 1단계 결과 train_rot30_data.zip | 무작위 --n-rot개 |
| 자동 라벨링 | labeled_final.csv + dev/ 이미지 (신뢰도 필터링) | 나머지(목표 - 원본 - 회전) 무작위 |

결과는 원본과 같은 형식의 train.csv + train/ 폴더이며, bigdata/ 폴더로 묶어 zip으로 저장합니다.

실행 예시:
    python scripts/03_merge_bigdata.py \\
        --data-zip ssafy-16-2-ai.zip \\
        --rot-zip outputs/train_rot30_data.zip \\
        --labeled-csv data/labeled_final.csv \\
        --dev-root data            # dev/ 폴더의 '부모' 폴더입니다
"""

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401  (data_pipeline 경로 등록)
from _bootstrap import add_common_args

from data_pipeline.common import extract_archive, find_dir, find_file, make_zip, prepare_original_data
from data_pipeline.merge import (AUTO_COLUMN_MAP, filter_by_confidence, merge_images, normalize_answers,
                                 sample_to_target, standardize)
from data_pipeline.validation import check_merged


def parse_args():
    p = argparse.ArgumentParser(description="원본·회전 증강·자동 라벨링 데이터를 하나의 학습셋으로 통합합니다.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="원본 train.csv와 train/ 폴더가 있는(또는 그 상위) 폴더입니다.")
    src.add_argument("--data-zip", help="대회 원본 데이터 압축 파일 경로입니다.")

    rot = p.add_mutually_exclusive_group(required=True)
    rot.add_argument("--rot-dir", help="train_rot30.csv가 있는(또는 그 상위) 폴더입니다.")
    rot.add_argument("--rot-zip", help="1단계 결과 train_rot30_data.zip 경로입니다.")
    p.add_argument("--rot-csv-name", default="train_rot30.csv", help="회전 증강 CSV 파일명입니다.")

    p.add_argument("--labeled-csv", help="자동 라벨링 결과 CSV(labeled_final.csv) 경로입니다.")
    p.add_argument("--dev-root", help="dev/ 이미지 폴더의 부모 폴더입니다. (CSV의 path가 'dev/xxx.jpg' 형태라고 가정합니다)")
    p.add_argument("--search-root", default=None,
                   help="--labeled-csv, --dev-root를 지정하지 않았을 때 자동으로 찾아볼 폴더입니다. "
                        "(예: /content/drive/MyDrive)")

    p.add_argument("--target-total", type=int, default=10000, help="최종 데이터 개수입니다. (기본값: 10000)")
    p.add_argument("--n-rot", type=int, default=2685, help="회전 증강본에서 뽑을 개수입니다. (기본값: 2685)")
    p.add_argument("--min-confidence", type=float, default=0.60,
                   help="자동 라벨링 데이터에서 남길 최소 신뢰도입니다. (기본값: 0.60)")
    p.add_argument("--zip-name", default="Bigdata_v2", help="결과 zip 파일 이름입니다. (기본값: Bigdata_v2)")
    return add_common_args(p).parse_args()


def resolve_sources(args, work_dir: Path):
    """세 데이터 소스의 위치를 확정합니다. 경로를 지정하지 않은 항목은 --search-root에서 찾아봅니다."""
    orig_root = prepare_original_data(args.data_dir, args.data_zip, work_dir, args.zip_password)

    if args.rot_zip:
        rot_base = work_dir / "rot30_extract"
        extract_archive(args.rot_zip, rot_base)
    else:
        rot_base = Path(args.rot_dir)
    rot_root = find_file(rot_base, args.rot_csv_name).parent
    print("회전 증강 데이터 루트:", rot_root)

    if args.labeled_csv:
        labeled_csv = Path(args.labeled_csv)
    else:
        assert args.search_root, "--labeled-csv 또는 --search-root 중 하나를 지정해 주세요."
        labeled_csv = find_file(args.search_root, "labeled_final.csv")
    print("자동 라벨링 CSV:", labeled_csv)

    if args.dev_root:
        dev_root = Path(args.dev_root)
    else:
        assert args.search_root, "--dev-root 또는 --search-root 중 하나를 지정해 주세요."
        dev_root = find_dir(args.search_root, "dev").parent   # dev/ 폴더의 '부모'를 루트로 잡습니다.
    print("dev 이미지 루트:", dev_root)
    return orig_root, rot_root, labeled_csv, dev_root


def main():
    args = parse_args()
    work_dir, output_dir = Path(args.work_dir), Path(args.output_dir)

    # ── 0. 세 데이터 소스 위치 확인 ──
    orig_root, rot_root, labeled_csv, dev_root = resolve_sources(args, work_dir)

    # ── 1. 세 소스 CSV 로드 + 컬럼 점검 ──
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)  # noqa: E731
    orig_raw = read(orig_root / "train.csv")
    rot_raw = read(rot_root / args.rot_csv_name)
    auto_raw = filter_by_confidence(read(labeled_csv), args.min_confidence)
    for name, d in [("원본 train.csv", orig_raw), (args.rot_csv_name, rot_raw),
                    (f"자동 라벨링 (신뢰도 ≥ {args.min_confidence})", auto_raw)]:
        print(f"\n=== {name} ===")
        print("행 수:", len(d), "/ 컬럼:", d.columns.tolist())

    # ── 2. 표준 컬럼으로 통일 + id 접두사 부여 (충돌 방지) ──
    orig_df = standardize(orig_raw, "orig", orig_root)
    rot_df = standardize(rot_raw, "rot", rot_root)                 # source_id 컬럼은 여기서 제외됩니다.
    auto_df = normalize_answers(standardize(auto_raw, "auto", dev_root, AUTO_COLUMN_MAP))
    for name, d in [("orig_df", orig_df), ("rot_df", rot_df), ("auto_df", auto_df)]:
        print(f"{name}: {len(d)}행, id 예시: {d['id'].iloc[0] if len(d) else '-'}")

    # ── 3. 목표 개수에 맞춰 샘플링 ──
    combined_df = sample_to_target(orig_df, rot_df, auto_df, args.target_total, args.n_rot, args.seed)

    # ── 4. 이미지 통합 (하나의 train/ 폴더로) ──
    merged_root = work_dir / "bigdata"
    merged_df = merge_images(combined_df, merged_root)
    merged_df.to_csv(merged_root / "train.csv", index=False, encoding="utf-8-sig")
    print(f"\ntrain.csv 저장을 완료했습니다: {len(merged_df)}행")

    # ── 5. 무결성 검사 ──
    check_merged(merged_df, merged_root, args.target_total)
    print("\n무결성 검사를 통과했습니다.")

    # ── 6. zip으로 묶어서 저장 ──
    if args.no_zip:
        print(f"zip 생성을 건너뛰었습니다. 결과 위치: {merged_root}")
        return
    make_zip(merged_root, work_dir / args.zip_name, output_dir)


if __name__ == "__main__":
    main()
