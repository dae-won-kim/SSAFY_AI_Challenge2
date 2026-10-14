"""고정 각도 회전 증강 모듈입니다.

열화 증강의 op_geo(회전 + 원근 왜곡)에서 원근 왜곡과 랜덤성을 제거하고,
정확히 지정한 각도(기본 ±30도)로만 회전하도록 단순화한 버전입니다.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import COLS

# 기본 회전 설정입니다.
# OpenCV 기준으로 양수 각도는 반시계 방향(왼쪽), 음수 각도는 시계 방향(오른쪽)입니다.
ROT_ANGLES = [30, -30]
ROT_TAGS = ["rotL30", "rotR30"]   # 각 각도에 대응하는 파일명 태그입니다.


def op_geo_rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """이미지를 중심 기준으로 angle_deg만큼 고정 회전합니다. (원근 왜곡 없음, 랜덤성 없음)

    회전 후 생기는 빈 모서리는 가장자리 픽셀을 복제(BORDER_REPLICATE)해서 채웁니다.
    눈으로 봤을 때 방향이 반대라고 느껴지시면 각도 부호만 뒤집어 주세요.
    """
    h, w = img.shape[:2]
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, R, (w, h), borderMode=cv2.BORDER_REPLICATE)


def source_id(i: str) -> str:
    """회전본 id에서 원본 id를 복원합니다. 예: 'train_0001_rotL30.jpg' -> 'train_0001.jpg'"""
    return re.sub(r"_rot[LR]\d+(?=\.[^.]+$)", "", i)


def rotated_id(base_id: str, ext: str, tag: str) -> str:
    """원본 id에 회전 태그를 붙인 새 id를 만듭니다. 예: 'train_0001.jpg' + 'rotL30' -> 'train_0001_rotL30.jpg'"""
    stem = base_id[: -len(ext)] if base_id.lower().endswith(ext.lower()) else base_id
    return f"{stem}_{tag}{ext}"


def make_rotations(src_df: pd.DataFrame, img_root: str | Path, out_subdir: str,
                   angles=ROT_ANGLES, tags=ROT_TAGS, workers: int = 4) -> tuple[pd.DataFrame, list[str]]:
    """src_df의 모든 원본 이미지에 대해 각도별 회전본을 만들어 img_root/out_subdir에 저장합니다.

    반환값은 (회전본만 담은 DataFrame, 이미지 로드에 실패해 건너뛴 원본 id 목록)입니다.
    결과 DataFrame에는 COLS와 함께 원본 id를 가리키는 source_id 컬럼이 포함됩니다.
    """
    img_root = Path(img_root)
    out_dir = img_root / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    def work(row):
        img = cv2.imread(str(img_root / row["path"]))
        if img is None:
            return None
        ext = Path(row["path"]).suffix
        new_rows = []
        for angle, tag in zip(angles, tags):
            rotated = op_geo_rotate(img, angle)
            new_id = rotated_id(row["id"], ext, tag)
            ok, buf = cv2.imencode(ext, rotated)
            if not ok:
                continue
            (out_dir / new_id).write_bytes(buf.tobytes())
            # 질문·보기·정답은 원본 그대로 두고, id와 이미지 경로만 바꿉니다.
            new_rows.append({**row, "id": new_id, "path": f"{out_subdir}/{new_id}"})
        return new_rows

    records = src_df.to_dict("records")
    with ThreadPoolExecutor(workers) as ex:
        raw = list(ex.map(work, records))

    skipped = [rec["id"] for rec, r in zip(records, raw) if r is None]
    if skipped:
        print(f"⚠ 이미지 로드 실패로 건너뛴 원본 {len(skipped)}장 (앞 5개): {skipped[:5]}")

    rot_rows = [r for rows in raw if rows is not None for r in rows]
    rot_df = pd.DataFrame(rot_rows, columns=COLS)
    rot_df["source_id"] = rot_df["id"].map(source_id)   # 나중에 원본 id로 회전본을 찾을 때 사용합니다.
    return rot_df, skipped
