"""여러 단계에서 공통으로 사용하는 상수와 입출력 유틸리티입니다."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

# 원본 train.csv와 동일한 표준 컬럼입니다. 모든 결과 CSV는 이 형식을 따릅니다.
COLS = ["id", "path", "question", "a", "b", "c", "d", "answer"]

# 데이터 압축 파일 비밀번호를 읽어올 환경 변수 이름입니다.
# 비밀번호를 코드에 직접 적으면 깃허브에 그대로 공개되므로, 환경 변수나 실행 인자로 전달해 주세요.
PASSWORD_ENV = "DATA_ZIP_PASSWORD"


# ── 압축 해제 ──────────────────────────────────────────────────────────
def extract_archive(zip_path: str | Path, out_dir: str | Path, password: str | None = None) -> None:
    """압축 파일을 out_dir에 해제합니다.

    7z 명령어가 설치되어 있으면(Colab에는 기본 설치되어 있습니다) 7z를 우선 사용하고,
    없으면 파이썬 내장 zipfile로 해제합니다.
    (zipfile은 AES 암호화 zip을 풀지 못하므로, 그런 경우에는 7z를 설치해 주세요.)
    """
    zip_path, out_dir = Path(zip_path), Path(out_dir)
    if not zip_path.is_file():
        raise FileNotFoundError(f"압축 파일을 찾을 수 없습니다: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("7z"):
        cmd = ["7z", "x", "-y", f"-o{out_dir}", str(zip_path)]
        if password:
            cmd.insert(3, f"-p{password}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    else:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir, pwd=password.encode() if password else None)
    print(f"압축 해제를 완료했습니다: {zip_path.name} -> {out_dir}")


def find_file(root: str | Path, filename: str, exclude: tuple[str, ...] = ()) -> Path:
    """root 아래에서 filename을 재귀적으로 찾아 첫 번째 결과를 반환합니다.

    압축 파일 안의 폴더 구조가 환경마다 다를 수 있어서(예: ./train.csv 또는 ./train_data/train.csv)
    위치를 고정하지 않고 직접 찾도록 했습니다. exclude에 포함된 문자열이 경로에 있으면 제외합니다.
    """
    hits = sorted(glob.glob(f"{root}/**/{filename}", recursive=True))
    hits = [h for h in hits if not any(word in h for word in exclude)]
    if not hits:
        raise FileNotFoundError(f"{root} 아래에서 {filename}을(를) 찾지 못했습니다. 폴더 구조를 확인해 주세요.")
    return Path(hits[0])


def find_dir(root: str | Path, dirname: str) -> Path:
    """root 아래에서 이름이 dirname인 폴더를 재귀적으로 찾아 첫 번째 결과를 반환합니다."""
    hits = sorted(Path(p) for p in glob.glob(f"{root}/**/{dirname}", recursive=True) if Path(p).is_dir())
    if not hits:
        raise FileNotFoundError(f"{root} 아래에서 {dirname}/ 폴더를 찾지 못했습니다. 경로를 직접 지정해 주세요.")
    return hits[0]


def prepare_original_data(data_dir: str | None, data_zip: str | None, work_dir: str | Path,
                          password: str | None = None) -> Path:
    """원본 데이터(train.csv + train/)가 있는 루트 폴더를 준비해서 반환합니다.

    - data_dir가 주어지면 그 안에서 train.csv를 찾아 그대로 사용합니다.
    - data_zip이 주어지면 work_dir에 압축을 해제한 뒤 train.csv 위치를 찾습니다.
    """
    if data_dir:
        root = Path(data_dir)
    elif data_zip:
        root = Path(work_dir) / "original"
        extract_archive(data_zip, root, password or os.environ.get(PASSWORD_ENV))
    else:
        raise ValueError("--data-dir 또는 --data-zip 중 하나는 반드시 지정해 주세요.")

    csv_path = find_file(root, "train.csv", exclude=("rot30", "aug"))
    print(f"원본 데이터 루트: {csv_path.parent}")
    return csv_path.parent


def copy_to_local(src_root: str | Path, local_root: str | Path, csv_name: str = "train.csv",
                  img_subdir: str = "train") -> Path:
    """CSV와 이미지 폴더를 로컬 작업 폴더로 복사합니다.

    Google Drive에서 직접 읽고 쓰면 매우 느리기 때문에, 로컬 디스크로 복사한 뒤 작업합니다.
    이미 같은 위치라면 복사하지 않습니다.
    """
    src_root, local_root = Path(src_root).resolve(), Path(local_root).resolve()
    if src_root == local_root:
        return local_root
    local_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_root / csv_name, local_root / csv_name)
    if not (local_root / img_subdir).exists():
        shutil.copytree(src_root / img_subdir, local_root / img_subdir)
    print(f"로컬 작업 폴더로 복사를 완료했습니다: {local_root}")
    return local_root


# ── 데이터 로드 ────────────────────────────────────────────────────────
def load_train_csv(img_root: str | Path, csv_name: str = "train.csv") -> pd.DataFrame:
    """train.csv를 읽고 기본 점검을 수행한 뒤, 이미지가 실제로 존재하는 행만 반환합니다."""
    img_root = Path(img_root)
    df = pd.read_csv(img_root / csv_name, dtype=str, keep_default_na=False)

    missing_cols = [c for c in COLS if c not in df.columns]
    assert not missing_cols, f"없는 컬럼이 있습니다: {missing_cols}"
    df = df[COLS].copy()
    assert df["id"].is_unique, "id 중복이 있습니다."

    exists = df["path"].map(lambda p: (img_root / p).exists())
    print(f"전체 {len(df)}행 / 이미지 없음 {(~exists).sum()}행")
    df = df[exists].reset_index(drop=True)

    print("\n보기(a~d)가 채워진 개수 분포:")
    print((df[list("abcd")].apply(lambda c: c.str.strip() != "")).sum(axis=1).value_counts())
    print("\nanswer 상위 값:")
    print(df["answer"].value_counts().head())
    return df


# ── 결과 저장 ──────────────────────────────────────────────────────────
def reset_dir(path: str | Path) -> Path:
    """폴더가 이미 있으면 비우고 새로 만듭니다. (이전 실행 결과가 섞이지 않도록 하기 위함입니다.)"""
    path = Path(path)
    if path.exists():
        n_old = sum(1 for _ in path.iterdir())
        shutil.rmtree(path)
        print(f"기존 {path.name}/ 폴더를 삭제했습니다: 파일 {n_old}개")
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_zip(src_dir: str | Path, zip_base: str | Path, output_dir: str | Path | None = None) -> Path:
    """src_dir 폴더를 zip으로 묶고, output_dir가 주어지면 그곳(예: Google Drive)에 복사합니다.

    zip을 풀면 src_dir 폴더 이름 그대로 한 단계 폴더가 생기도록 묶습니다.
    """
    src_dir = Path(src_dir)
    zip_path = Path(shutil.make_archive(str(zip_base), "zip",
                                        root_dir=str(src_dir.parent), base_dir=src_dir.name))
    print(f"압축을 완료했습니다: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dst = output_dir / zip_path.name
        if dst.resolve() != zip_path.resolve():
            shutil.copy2(zip_path, dst)
            print(f"결과 폴더에 저장을 완료했습니다: {dst}")
        zip_path = dst
    print(f'사용하실 때는 unzip -q "{zip_path}" -d <원하는 폴더> 로 풀면 {src_dir.name}/ 폴더가 생깁니다.')
    return zip_path
