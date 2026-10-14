"""증강 결과를 눈으로 확인할 수 있도록 미리보기 이미지를 저장하는 모듈입니다.

스크립트 환경(화면이 없는 서버, Colab의 !python 실행 등)에서도 동작하도록
창을 띄우지 않고 PNG 파일로 저장합니다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")   # 화면 없이 파일로만 저장하는 백엔드입니다.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .degradation import DegradeConfig, degrade, sev_label
from .rotation import op_geo_rotate


def _show(ax, img, title):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_rotation_preview(df: pd.DataFrame, img_root: str | Path, angles, tags, out_path: str | Path,
                          n_samples: int = 4, seed: int = 42) -> Path:
    """원본과 각도별 회전본을 나란히 그린 미리보기를 저장합니다. 회전 방향이 의도와 맞는지 확인해 주세요."""
    samples = df.sample(min(n_samples, len(df)), random_state=seed).to_dict("records")
    fig, axes = plt.subplots(len(samples), 1 + len(angles), figsize=(4 * (1 + len(angles)), 4 * len(samples)),
                             squeeze=False)
    for i, row in enumerate(samples):
        img = cv2.imread(str(Path(img_root) / row["path"]))
        _show(axes[i, 0], img, "original")
        for j, (angle, tag) in enumerate(zip(angles, tags), start=1):
            _show(axes[i, j], op_geo_rotate(img, angle), f"{tag} ({angle} deg)")
    return _save(fig, out_path)


def save_degradation_preview(df: pd.DataFrame, img_root: str | Path, cfg: DegradeConfig, out_path: str | Path,
                             n_samples: int = 4, n_variants: int = 3, seed: int = 42) -> Path:
    """원본과 열화본 여러 장을 나란히 그린 미리보기를 저장합니다. 글자가 안 보일 정도면 강도를 낮춰 주세요."""
    samples = df.sample(min(n_samples, len(df)), random_state=seed).to_dict("records")
    fig, axes = plt.subplots(len(samples), 1 + n_variants, figsize=(4 * (1 + n_variants), 4 * len(samples)),
                             squeeze=False)
    for i, row in enumerate(samples):
        img = cv2.imread(str(Path(img_root) / row["path"]))
        _show(axes[i, 0], img, "original")
        for j in range(1, n_variants + 1):
            rng = np.random.default_rng([seed, i, j])
            buf, tags, s, q, params, m = degrade(img, rng, cfg, cfg.max_severity)
            _show(axes[i, j], cv2.imdecode(buf, cv2.IMREAD_COLOR), f"{'+'.join(tags)} / {sev_label(s)}")
    return _save(fig, out_path)


def _save(fig, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    print(f"미리보기를 저장했습니다: {out_path}")
    return out_path
