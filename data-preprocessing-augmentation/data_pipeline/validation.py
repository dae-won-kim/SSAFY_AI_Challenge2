"""결과물 무결성 검사 함수 모음입니다.

검사에 실패하면 AssertionError가 발생하므로, 잘못된 데이터가 다음 단계로 넘어가지 않습니다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd


def check_files(d: pd.DataFrame, root: str | Path, n_decode: int = 200) -> None:
    """모든 path가 실제로 존재하는지, 일부 샘플 이미지가 정상적으로 열리는지 확인합니다."""
    root = Path(root)
    ex = d["path"].map(lambda p: (root / p).exists())
    print(f"{len(d)}행 / 파일 없음 {(~ex).sum()} / id 중복 {d['id'].duplicated().sum()}")
    assert ex.all(), "존재하지 않는 이미지 경로가 있습니다."
    assert d["id"].is_unique, "id 중복이 있습니다."

    sample = d["path"].sample(min(n_decode, len(d)), random_state=0)
    bad = [p for p in sample if cv2.imread(str(root / p)) is None]
    print("열리지 않는 이미지:", len(bad))
    assert not bad, f"열리지 않는 이미지가 있습니다: {bad[:5]}"


def check_per_source(d: pd.DataFrame, expected: int) -> None:
    """원본(source_id)마다 증강본이 정확히 expected장씩 만들어졌는지 확인합니다."""
    per_src = d.groupby("source_id").size()
    assert (per_src == expected).all(), "원본당 증강본 수가 예상과 다릅니다."
    print(f"원본 {len(per_src)}장 x 증강본 {expected}장 = {len(d)}행이 맞습니다.")


def check_labels_preserved(d: pd.DataFrame, orig_df: pd.DataFrame) -> None:
    """증강본의 질문·보기·정답이 원본과 완전히 같은지 확인합니다."""
    orig = orig_df.set_index("id")
    for c in ["question", "a", "b", "c", "d", "answer"]:
        assert (d[c].values == orig.loc[d["source_id"], c].values).all(), f"{c} 값이 원본과 다릅니다."
    print("질문·보기·정답이 원본과 동일하게 보존되었습니다.")


def check_merged(d: pd.DataFrame, root: str | Path, target_total: int) -> None:
    """통합 데이터셋의 개수·경로·정답·보기를 검사하고 소스별 구성을 출력합니다."""
    print(f"전체 {len(d)}행 (목표 {target_total}개)")
    assert len(d) == target_total, f"목표 개수와 다릅니다: {len(d)} != {target_total}"
    check_files(d, root, n_decode=300)
    assert d["answer"].isin(list("abcd")).all(), "answer 값에 a~d가 아닌 값이 있습니다."
    assert (d[list("abcd")].apply(lambda c: c.str.strip() != "")).all().all(), "보기(a~d)가 빈 값인 행이 있습니다."

    print("\n소스별 구성:")
    print(d["id"].str.extract(r"^(orig|rot|auto)_")[0].value_counts())
