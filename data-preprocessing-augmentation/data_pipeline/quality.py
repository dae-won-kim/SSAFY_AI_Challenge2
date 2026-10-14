"""원본 이미지 화질 분석과, 그 결과를 기반으로 열화 강도를 재설정하는 모듈입니다.

test.csv에는 라벨이 없지만 train과 test가 같은 방식으로 수집된 사진이므로,
train 원본 자체의 화질 편차(blur·밝기·대비)를 기준으로 열화 강도를 정했습니다.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .degradation import DegradeConfig


def measure_quality(paths) -> pd.DataFrame:
    """이미지 목록의 blur(선명도), 밝기, 대비, 크기를 측정해서 DataFrame으로 반환합니다."""
    rows = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows.append({
            "blur": cv2.Laplacian(g, cv2.CV_64F).var(),   # 낮을수록 흐립니다
            "bright": g.mean(),                            # 평균 밝기
            "contrast": g.std(),                           # 대비
            "w": g.shape[1], "h": g.shape[0],
        })
    return pd.DataFrame(rows)


def print_quality_report(quality_df: pd.DataFrame) -> None:
    """화질 분포 요약과 분위수(5% / 50% / 95%)를 출력합니다."""
    cols = ["blur", "bright", "contrast"]
    print("=== train 원본 이미지 품질 분포입니다 ===")
    print(quality_df[cols].describe().round(2))
    print("\n=== 분위수 (5% / 50% / 95%) ===")
    print(quality_df[cols].quantile([0.05, 0.5, 0.95]).round(2))


def tune_config(quality_df: pd.DataFrame, base: DegradeConfig | None = None) -> DegradeConfig:
    """화질 분석 결과를 바탕으로 열화 강도를 재설정한 새 DegradeConfig를 반환합니다."""
    cfg = base or DegradeConfig()

    # ── LEVEL_P, MAX_SEVERITY: 원본 안에 이미 존재하는 화질 편차를 기준으로 정했습니다 ──
    # blur 하위 5% 값이 중앙값보다 많이 낮으면(비율이 작으면) 원본 안에도 이미 다양한 화질이
    # 섞여 있다는 뜻이므로 인위적인 열화를 더 약하게 줘도 충분하다고 판단했습니다.
    # 반대로 원본이 대체로 균일하게 선명하면, 다양한 촬영 조건에 대비해 mild~mid 비중을 조금 높였습니다.
    blur_spread_ratio = quality_df["blur"].quantile(0.05) / quality_df["blur"].quantile(0.50)
    print("원본 blur 하위5%/중앙값 비율:", round(blur_spread_ratio, 3))
    if blur_spread_ratio < 0.3:
        level_p, max_severity = [0.6, 0.3, 0.1], 0.5      # 편차가 이미 큰 경우 -> mild 위주, severe 최소화
    else:
        level_p, max_severity = [0.5, 0.35, 0.15], 0.6    # 대체로 균일한 경우 -> mild~mid 위주로 살짝 늘림
    print("재설정된 LEVEL_P:", level_p, "/ MAX_SEVERITY:", max_severity)

    # ── SHADOW_DARK, LOWLIGHT_GAIN: 원본 중 가장 어두운 축(하위 5%)보다 지나치게 어둡게
    #    만들지 않도록 하한을 잡았습니다. 실제로 존재하지 않을 만큼 어두운 이미지는
    #    실전과 무관한 노이즈가 될 수 있다고 판단했습니다.
    bright_q05 = quality_df["bright"].quantile(0.05)
    print("원본 밝기 하위 5%:", round(bright_q05, 1))
    gain_floor = float(np.clip(bright_q05 / 255 * 1.3, 0.5, 0.75))
    shadow_dark = (0.95, round(gain_floor + 0.05, 2))     # 기존 (0.9, 0.5)
    lowlight_gain = (0.9, round(gain_floor, 2))           # 기존 (0.8, 0.45)
    print("재설정된 SHADOW_DARK:", shadow_dark, "/ LOWLIGHT_GAIN:", lowlight_gain)

    # ── 나머지 열화(회전·렌즈·물결 왜곡, 반사광, 블러, 저해상도, 노이즈, JPEG 압축)는
    #    픽셀 통계만으로 직접 측정하기 어려워서 "실제로 촬영해서 올릴 법한 수준"을
    #    기준으로 기본값보다 보수적으로 낮췄습니다.
    tuned = replace(
        cfg,
        level_p=level_p,
        max_severity=max_severity,
        shadow_dark=shadow_dark,
        lowlight_gain=lowlight_gain,
        geo_rot=(5, 25),               # 기존 (-30, 30) — 너무 크게 기울면 비현실적입니다
        geo_persp=(0.005, 0.02),       # 기존 (0.005, 0.03)
        lens_k=(0.05, 0.15),           # 기존 (0.1, 0.2)
        warp_amp=(0.005, 0.015),       # 기존 (0.010, 0.020)
        glare_size=(0.15, 0.3),        # 기존 (0.2, 0.4)
        glare_strength=(0.35, 0.65),   # 기존 (0.45, 0.85) — 0.85면 화면 대부분이 흰색으로 덮입니다
        blur_gauss_sigma=(0.5, 2.0),   # 기존 (1.0, 3.0)
        blur_motion_len=(2, 10),       # 기존 (3, 15)
        blur_defocus_r=(1, 4),         # 기존 (2, 6)
        lowres_scale=(0.5, 0.3),       # 기존 (0.4, 0.2)
        noise_sigma=(2, 15),           # 기존 (3, 25)
        jpeg_q=(85, 45),               # 기존 (80, 35)
    )
    print("나머지 열화 강도도 보수적인 값으로 재설정했습니다.")
    return tuned


def quality_from_paths(img_root: str | Path, rel_paths) -> pd.DataFrame:
    """img_root 기준 상대 경로 목록으로 화질을 측정하는 편의 함수입니다."""
    img_root = Path(img_root)
    return measure_quality([img_root / p for p in rel_paths])
