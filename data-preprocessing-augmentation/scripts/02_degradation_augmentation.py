"""[2단계] 열화 이미지 증강 → aug_train.csv 만들기

train 데이터의 모든 원본 이미지에 실제 촬영 환경을 흉내 낸 열화를 걸어서,
증강본만 담은 aug_train.csv, 열화 정보 aug_meta.csv, 이미지 폴더 train_aug/를 zip으로 묶어 저장합니다.
열화 강도는 train 원본 이미지의 화질 분석 결과를 바탕으로 자동 재설정됩니다.

실행 예시:
    python scripts/02_degradation_augmentation.py --data-zip ssafy-16-2-ai.zip --n-aug 3
    python scripts/02_degradation_augmentation.py --data-dir data/original --no-tune   # 기본 강도 그대로 사용
"""

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401  (data_pipeline 경로 등록)
from _bootstrap import add_common_args

from data_pipeline.common import copy_to_local, load_train_csv, make_zip, prepare_original_data, reset_dir
from data_pipeline.degradation import DegradeConfig, make_aug, pick_random_augmented
from data_pipeline.preview import save_degradation_preview
from data_pipeline.quality import print_quality_report, quality_from_paths, tune_config
from data_pipeline.validation import check_files, check_labels_preserved, check_per_source

AUG_SUBDIR = "train_aug"


def parse_args():
    p = argparse.ArgumentParser(description="원본 이미지에 열화를 적용한 증강 데이터셋을 만듭니다.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="train.csv와 train/ 폴더가 있는(또는 그 상위) 폴더입니다.")
    src.add_argument("--data-zip", help="대회 원본 데이터 압축 파일 경로입니다.")
    p.add_argument("--n-aug", type=int, default=3, help="원본 1장당 만들 증강본 수입니다. (기본값: 3)")
    p.add_argument("--workers", type=int, default=4, help="병렬 처리 스레드 수입니다. (기본값: 4)")
    p.add_argument("--no-tune", action="store_true",
                   help="화질 분석 기반 강도 재설정을 건너뛰고 기본 강도를 그대로 사용합니다.")
    p.add_argument("--no-preview", action="store_true", help="미리보기 이미지를 저장하지 않습니다.")
    return add_common_args(p).parse_args()


def main():
    args = parse_args()
    work_dir, output_dir = Path(args.work_dir), Path(args.output_dir)

    # ── 0. 데이터 준비 ──
    orig_root = prepare_original_data(args.data_dir, args.data_zip, work_dir, args.zip_password)
    img_root = copy_to_local(orig_root, work_dir / "data")

    # ── 1. 데이터 로드 + 점검 ──
    # valid는 학습 단계에서 source_id 기준으로 나눌 예정이라, 여기서는 원본 전체를 증강 대상으로 잡습니다.
    train_df = load_train_csv(img_root)
    print("\n증강 대상(train 전체):", len(train_df))

    # ── 2. 원본 화질 분석 → 3. 강도 재설정 ──
    cfg = DegradeConfig()
    if not args.no_tune:
        quality_df = quality_from_paths(img_root, train_df["path"])
        print_quality_report(quality_df)
        print()
        cfg = tune_config(quality_df, cfg)

    # ── 4. 미리보기 (재설정된 강도 확인) ──
    if not args.no_preview:
        save_degradation_preview(train_df, img_root, cfg, output_dir / "preview_degradation.png", seed=args.seed)

    # ── 5. 증강 생성 (원본 제외) ──
    aug_dir = reset_dir(img_root / AUG_SUBDIR)
    aug_train_df, aug_meta = make_aug(train_df, img_root, AUG_SUBDIR, args.n_aug, args.seed, cfg,
                                      include_original=False, split="train", workers=args.workers)
    aug_train_df.to_csv(img_root / "aug_train.csv", index=False, encoding="utf-8-sig")
    aug_meta.to_csv(img_root / "aug_meta.csv", index=False, encoding="utf-8-sig")
    print(f"\naug_train.csv 저장을 완료했습니다: {len(aug_train_df)}행 (원본 {len(train_df)}장 x 증강본 {args.n_aug}장)")

    # ── 6. 열화 분포 확인 ──
    # 특정 열화나 severe로 너무 치우쳤다면 data_pipeline/quality.py의 tune_config 값을 조정해 주세요.
    aug_only = aug_meta[~aug_meta["is_original"]]
    print("\n열화 강도 분포입니다:")
    print(aug_only["synth_severity"].value_counts())
    print("\n열화 태그 상위 조합입니다:")
    print(aug_only["synth_tags"].value_counts().head(8))

    # ── 7. 무결성 검사 ──
    print()
    check_files(aug_train_df, img_root)
    check_per_source(aug_train_df, args.n_aug)
    check_labels_preserved(aug_train_df, train_df)
    print("무결성 검사를 통과했습니다.")

    # ── 8. 사용 예시: 원본 id로 증강본 하나를 무작위로 가져오기 ──
    print("\n원본 id로 증강본을 무작위로 가져오는 예시입니다:")
    for sid in aug_train_df["source_id"].drop_duplicates().sample(min(3, len(train_df)), random_state=0):
        print(f"  {sid} -> {pick_random_augmented(aug_train_df, sid)['path']}")

    # ── 9. zip으로 묶어서 저장 ──
    if args.no_zip:
        print(f"zip 생성을 건너뛰었습니다. 결과 위치: {img_root}")
        return
    stage_dir = reset_dir(work_dir / "aug_package" / "aug_data")
    shutil.copy2(img_root / "aug_train.csv", stage_dir / "aug_train.csv")
    shutil.copy2(img_root / "aug_meta.csv", stage_dir / "aug_meta.csv")
    shutil.move(str(aug_dir), str(stage_dir / AUG_SUBDIR))   # 통째로 이동해서 중복 복사를 피했습니다.
    make_zip(stage_dir, work_dir / "aug_data", output_dir)


if __name__ == "__main__":
    main()
