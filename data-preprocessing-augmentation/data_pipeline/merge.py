"""세 데이터 소스를 하나의 학습셋(train.csv + train/ 이미지 폴더)으로 통합하는 모듈입니다.

| 소스 | 내용 | id 접두사 |
|---|---|---|
| 원본 | 대회 제공 train.csv + train/ | orig_ |
| 회전 증강 | train_rot30.csv + train_rot30/ | rot_ |
| 자동 라벨링 | labeled_final.csv + dev/ 이미지 (신뢰도 기준 필터링) | auto_ |
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .common import COLS

CONF_COLS = ["a_confidence", "b_confidence", "c_confidence", "d_confidence"]

# labeled_final.csv의 컬럼명이 표준 컬럼(COLS)과 다를 경우 이 딕셔너리만 수정해 주세요.
# 형식: {표준 컬럼명: 실제 CSV 컬럼명}
AUTO_COLUMN_MAP = {c: c for c in COLS}


def filter_by_confidence(dev_raw: pd.DataFrame, min_conf: float) -> pd.DataFrame:
    """보기별 신뢰도(a~d_confidence) 중 최댓값이 min_conf 이상인 행만 남깁니다."""
    d = dev_raw.copy()
    for col in CONF_COLS:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d[d[CONF_COLS].max(axis=1) >= min_conf].reset_index(drop=True)


def standardize(df: pd.DataFrame, prefix: str, src_root: str | Path,
                column_map: dict | None = None) -> pd.DataFrame:
    """표준 컬럼(COLS)만 남기고, id 충돌을 막기 위해 소스별 접두사를 붙입니다.

    src_root는 이미지 통합 단계에서 path를 어느 폴더 기준으로 찾을지 표시하는 용도입니다.
    """
    if column_map:
        missing = [c for c in column_map.values() if c not in df.columns]
        assert not missing, (f"CSV에 없는 컬럼이 있습니다: {missing} — AUTO_COLUMN_MAP을 실제 컬럼명으로 "
                             f"고쳐 주세요. 현재 컬럼: {df.columns.tolist()}")
        df = df.rename(columns={v: k for k, v in column_map.items()})
    out = df[COLS].copy()
    out["id"] = f"{prefix}_" + out["id"]
    out["src_root"] = str(src_root)
    return out


def normalize_answers(df: pd.DataFrame) -> pd.DataFrame:
    """answer 값을 소문자 a~d로 통일하고, 범위를 벗어난 값이 없는지 확인합니다."""
    df = df.copy()
    df["answer"] = df["answer"].str.strip().str.lower()
    assert df["answer"].isin(list("abcd")).all(), "answer 값에 a~d가 아닌 값이 있습니다."
    return df


def sample_to_target(orig_df: pd.DataFrame, rot_df: pd.DataFrame, auto_df: pd.DataFrame,
                     target_total: int, n_rot: int, seed: int) -> pd.DataFrame:
    """원본은 전체, 회전 증강본은 n_rot개, 자동 라벨링 데이터는 총합이 target_total이 되도록 나머지를 뽑습니다."""
    rot_sample = rot_df.sample(n=min(n_rot, len(rot_df)), random_state=seed).reset_index(drop=True)
    if len(rot_sample) < n_rot:
        print(f"⚠ 회전 증강 데이터 행 수({len(rot_df)})가 {n_rot}개보다 적어서 전체를 사용했습니다.")

    n_auto_needed = target_total - len(orig_df) - len(rot_sample)
    print(f"원본 {len(orig_df)}개 + 회전 {len(rot_sample)}개 -> 자동 라벨링에서 필요한 개수: {n_auto_needed}")
    assert n_auto_needed > 0, (
        f"원본+회전만으로 이미 {len(orig_df) + len(rot_sample)}개라 목표({target_total})를 넘었거나 딱 맞습니다. "
        f"--n-rot을 줄이거나 --target-total을 조정해 주세요."
    )

    if n_auto_needed > len(auto_df):
        print(f"⚠ 자동 라벨링 데이터 행 수({len(auto_df)})가 필요한 개수({n_auto_needed})보다 적어서 전체를 사용합니다.")
        auto_sample = auto_df.copy()
    else:
        auto_sample = auto_df.sample(n=n_auto_needed, random_state=seed).reset_index(drop=True)

    combined = pd.concat([orig_df, rot_sample, auto_sample], ignore_index=True)
    print(f"\n최종 합계: 원본 {len(orig_df)} + 회전 {len(rot_sample)} + 자동 {len(auto_sample)} = {len(combined)}행")
    return combined


def merge_images(combined_df: pd.DataFrame, merged_root: str | Path, img_subdir: str = "train") -> pd.DataFrame:
    """각 소스의 이미지를 새 id를 파일명으로 해서 merged_root/img_subdir 하나로 모읍니다.

    path 컬럼도 새 폴더 기준으로 다시 쓰고, 이미지를 찾지 못한 행은 제외합니다.
    결과 DataFrame에서는 src_root 컬럼을 제거합니다.
    """
    merged_root = Path(merged_root)
    img_dir = merged_root / img_subdir
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    new_paths, missing_rows = [], []
    for row in combined_df.to_dict("records"):
        src = Path(row["src_root"]) / row["path"]
        if not src.is_file():
            missing_rows.append(row["id"])
            new_paths.append(None)
            continue
        ext = src.suffix
        fname = row["id"] if row["id"].lower().endswith(ext.lower()) else f"{row['id']}{ext}"
        dst = img_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)
        new_paths.append(f"{img_subdir}/{fname}")

    print("이미지를 찾지 못한 행:", len(missing_rows), missing_rows[:5])
    out = combined_df.copy()
    out["path"] = new_paths
    out = out.dropna(subset=["path"]).reset_index(drop=True)
    return out[COLS]
