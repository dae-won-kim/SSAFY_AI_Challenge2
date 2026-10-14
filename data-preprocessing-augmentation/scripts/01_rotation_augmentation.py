"""[1단계] 회전 증강 → train_rot30.csv 만들기

train.csv의 모든 원본 이미지를 반시계 30도, 시계 30도로 회전한 이미지 두 장씩 만들고,
회전본만 담은 CSV(train_rot30.csv)와 이미지 폴더(train_rot30/)를 zip으로 묶어 저장합니다.
원본 행은 결과 CSV에 포함하지 않습니다.

실행 예시:
    python scripts/01_rotation_augmentation.py --data-zip ssafy-16-2-ai.zip --output-dir outputs
    python scripts/01_rotation_augmentation.py --data-dir data/original      # 이미 압축을 푼 경우
"""

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401  (data_pipeline 경로 등록)
from _bootstrap import add_common_args

from data_pipeline.common import copy_to_local, load_train_csv, make_zip, prepare_original_data, reset_dir
from data_pipeline.preview import save_rotation_preview
from data_pipeline.rotation import make_rotations
from data_pipeline.validation import check_files, check_labels_preserved, check_per_source


def parse_args():
    p = argparse.ArgumentParser(description="원본 이미지를 ±각도로 회전한 증강 데이터셋을 만듭니다.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="train.csv와 train/ 폴더가 있는(또는 그 상위) 폴더입니다.")
    src.add_argument("--data-zip", help="대회 원본 데이터 압축 파일 경로입니다.")
    p.add_argument("--angle", type=int, default=30, help="회전 각도입니다. +각도/-각도 두 장을 만듭니다. (기본값: 30)")
    p.add_argument("--workers", type=int, default=4, help="병렬 처리 스레드 수입니다. (기본값: 4)")
    p.add_argument("--no-preview", action="store_true", help="미리보기 이미지를 저장하지 않습니다.")
    return add_common_args(p).parse_args()


def main():
    args = parse_args()
    work_dir, output_dir = Path(args.work_dir), Path(args.output_dir)

    # 양수 = 반시계(왼쪽), 음수 = 시계(오른쪽)입니다.
    angles = [args.angle, -args.angle]
    tags = [f"rotL{args.angle}", f"rotR{args.angle}"]
    name = f"train_rot{args.angle}"

    # ── 0. 데이터 준비 (Drive는 느리므로 로컬로 복사해서 작업합니다) ──
    orig_root = prepare_original_data(args.data_dir, args.data_zip, work_dir, args.zip_password)
    img_root = copy_to_local(orig_root, work_dir / "data")

    # ── 1. 데이터 로드 + 점검 ──
    df = load_train_csv(img_root)
    print("\n증강 대상(train 전체):", len(df))

    # ── 2. 미리보기 (회전 방향·각도 확인) ──
    if not args.no_preview:
        save_rotation_preview(df, img_root, angles, tags, output_dir / "preview_rotation.png", seed=args.seed)

    # ── 3. 회전 증강 생성 ──
    reset_dir(img_root / name)
    rot_df, skipped = make_rotations(df, img_root, name, angles, tags, workers=args.workers)
    rot_df.to_csv(img_root / f"{name}.csv", index=False, encoding="utf-8-sig")
    n_source = len(df) - len(skipped)
    print(f"\n{name}.csv 저장을 완료했습니다: {len(rot_df)}행 (원본 {n_source}장 x 회전본 {len(angles)}장)")

    # ── 4. 무결성 검사 ──
    check_files(rot_df, img_root)
    check_per_source(rot_df, len(angles))
    check_labels_preserved(rot_df, df)
    print("무결성 검사를 통과했습니다.")

    # ── 5. zip으로 묶어서 저장 ──
    # 원본 이미지(train/)는 원본 데이터셋에 이미 있으므로 zip에 포함하지 않습니다.
    if args.no_zip:
        print(f"zip 생성을 건너뛰었습니다. 결과 위치: {img_root}")
        return
    stage_dir = reset_dir(work_dir / "rot_package" / f"{name}_data")
    shutil.copy2(img_root / f"{name}.csv", stage_dir / f"{name}.csv")
    shutil.move(str(img_root / name), str(stage_dir / name))   # 통째로 이동해서 중복 복사를 피했습니다.
    make_zip(stage_dir, work_dir / f"{name}_data", output_dir)


if __name__ == "__main__":
    main()
