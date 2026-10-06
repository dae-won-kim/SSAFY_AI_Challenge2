"""
실험 기록 모아 보기 (CPU, 로컬에서 실행)

결과 폴더는 실험(하이퍼파라미터 조합)마다 하위 폴더 하나다 :
  results/<실험이름>/experiments_<단계>.csv                (그 실험의 단계별 기록)
  results/<실험이름>/rerun-<방식>/experiments_<단계>.csv   (재추론·최종 단계)
여러 VM 이 같은 드라이브에 동시에 써도 서로 다른 파일이라 행이 사라지지 않는다. 이 스크립트가 그것들을 합친다.

  results/*/**/experiments_<단계>.csv  →  results/experiments_<단계>.csv   (단계별 전체 기록, 한 장)
                                       →  results/summary.csv              (run 한 줄씩, 단계별 핵심 수치)
(예전 평면 배치 experiments_<단계>__<run>.csv 도 같이 읽는다)

예)
  python collect_results.py                  # results/ (config.RESULT_ROOT)
  python collect_results.py --dir /content/drive/MyDrive/vqa/results
"""
import config  # noqa: F401

import os
import re
import glob
import argparse

import pandas as pd

# summary.csv 에 뽑을 열 : {단계: [(원본 열, 표시 이름), ...]}
PICK = {
    "s1_train":     [("preset", "preset"), ("train_source", "data"), ("lora_r", "lora_r"), ("lora_targets", "lora대상"),
                     ("hint_prob", "hint"),
                     ("epochs", "ep"), ("lr", "lr"), ("best_update", "best_upd"), ("best_valid_acc", "s1_valid"),
                     ("valid_hash", "valid_id"), ("minutes", "분")],
    "s2_predict":   [("acc_valid", "valid"), ("acc_train", "train"), ("conf_mean_valid", "conf"),
                     ("sec_per_q", "초/문항")],
    "s2_prompt_scan": [("all_agree", "프롬프트일치")],
    "s3_select":    [("test_selected_frac", "test선별률"), ("valid_selected_frac", "valid선별률"),
                     ("valid_error_capture", "오답포착")],
    "s4_rerun":     [("mode", "재추론"), ("acc_base_subset", "선별전"), ("acc_rerun_subset", "선별후")],
    "s5_final":     [("rerun_tag", "채택재추론"), ("method", "교체"), ("valid_acc_keep", "keep_valid"),
                     ("valid_acc", "최종valid"), ("n_changed", "교체수")],
}
_STAGE_RE = re.compile(r"^experiments_(.+?)(?:__.*)?\.csv$")


def parse_args():
    p = argparse.ArgumentParser(description="실험 기록 합치기")
    p.add_argument("--dir", default=config.RESULT_ROOT, help=f"결과 폴더 (기본 {config.RESULT_ROOT})")
    p.add_argument("--no-write", action="store_true", help="합친 파일을 저장하지 않고 화면에만 출력")
    return p.parse_args()


def record_files(d):
    """실험별 기록 파일 전부 (하위 폴더 + 예전 평면 배치). 이 스크립트가 만든 최상위 합본은 제외"""
    new = glob.glob(os.path.join(d, "*", "**", "experiments_*.csv"), recursive=True)
    old = glob.glob(os.path.join(d, "experiments_*__*.csv"))
    return sorted(set(new + old))


def merge_stage(files):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["_src"] = os.path.relpath(f)
            dfs.append(df)
        except Exception as e:                                  # noqa: BLE001
            print(f"  ⚠ 건너뜀 {f} ({type(e).__name__})")
    if not dfs:
        return None
    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values("time").reset_index(drop=True) if "time" in out else out


def main():
    args = parse_args()
    d = args.dir
    if not os.path.isdir(d):
        raise SystemExit(f"결과 폴더가 없습니다: {d}")

    by_stage = {}
    for f in record_files(d):
        m = _STAGE_RE.match(os.path.basename(f))
        if m:
            by_stage.setdefault(m.group(1), []).append(f)
    if not by_stage:
        raise SystemExit(f"{d} 아래에 experiments_<단계>.csv 가 없습니다 (fetch 를 먼저 하세요)")

    summary = None
    for stage in sorted(by_stage):
        df = merge_stage(by_stage[stage])
        if df is None:
            continue
        if not args.no_write:
            df.drop(columns=["_src"]).to_csv(os.path.join(d, f"experiments_{stage}.csv"), index=False,
                                             encoding="utf-8-sig")
        print(f"[{stage}] {len(df)}행 / 파일 {len(by_stage[stage])}개 / run {df['run'].nunique() if 'run' in df else '?'}개")

        cols = [(src, dst) for src, dst in PICK.get(stage, []) if src in df.columns]
        if not cols or "run" not in df.columns:
            continue
        part = df.groupby("run", as_index=False).last()[["run"] + [c for c, _ in cols]]   # 같은 run 은 마지막 것
        part.columns = ["run"] + [dst for _, dst in cols]
        summary = part if summary is None else summary.merge(part, on="run", how="outer")

    if summary is not None:
        summary = summary.sort_values("run").reset_index(drop=True)
        if not args.no_write:
            summary.to_csv(os.path.join(d, "summary.csv"), index=False, encoding="utf-8-sig")
        print("\n══ 실험 비교 ══")
        print(summary.to_string(index=False))
        if not args.no_write:
            print(f"\n▶ {os.path.join(d, 'summary.csv')}")


if __name__ == "__main__":
    main()
