"""
여러 실험의 예측 확률을 평균해 제출 파일을 만든다 (CPU, GPU 불필요)

각 실험의 test_pred.csv 에는 문항별 p_a..p_d 가 저장돼 있으므로, 다시 추론하지 않고 합칠 수 있다.
같은 모델·어댑터에서 프롬프트만 바꾼 branch 앙상블(CLAUDE.md 에서 제외한 것)이 아니라,
**따로 학습한 어댑터들**의 확률 평균이다. 검증셋 없이도 비교적 안전하게 점수를 올리는 방법이지만,
효과는 Kaggle 점수로만 확인할 수 있다 (또는 train holdout valid 가 생긴 뒤).

  python ensemble.py r16 r32                       # runs/<run>/test_pred.csv 들
  python ensemble.py results/*__test_pred.csv      # 파일 경로 직접 (fetch 로 받은 사본)
  python ensemble.py r16 r32 b8 --weights 1 1 2    # 가중 평균
  python ensemble.py r16 r32 --method vote         # 다수결 (동률이면 평균 확률)

산출물 : runs/ensemble/<이름>/test_pred.csv, submission.csv  +  results/<이름>__submission.csv 등
"""
import config  # noqa: F401

import os
import re
import glob
import argparse
import itertools

import numpy as np
import pandas as pd

from data_utils import PROB_COLS, LETTERS, load_pred, summarize, write_submission, save_result, save_json, append_log


def parse_args():
    p = argparse.ArgumentParser(description="실험 여러 개의 확률 평균 → 제출 파일")
    p.add_argument("members", nargs="+", help="run 이름(runs/<run>/test_pred.csv) 또는 *_pred.csv 경로")
    p.add_argument("--split", default="test", choices=["test", "valid", "train"], help="run 이름으로 줄 때 읽을 split")
    p.add_argument("--method", default="mean", choices=["mean", "vote"])
    p.add_argument("--weights", type=float, nargs="*", default=None, help="member 순서대로 가중치 (기본 균등)")
    p.add_argument("--name", default=None, help="산출물 이름 (기본 ens_<member1>+<member2>+…)")
    return p.parse_args()


def resolve(member, split):
    """run 이름 → runs/<run>/<split>_pred.csv (없으면 results/<run>_*/<split>_pred.csv), 경로면 그대로. (경로, 짧은 이름)"""
    if os.path.isfile(member):
        base = os.path.basename(member)
        if "__" in base:                                    # 예전 평면 배치 r16_…__test_pred.csv
            short = base.split("_")[0]
        else:                                               # results/<실험이름>/test_pred.csv 또는 runs/<run>/test_pred.csv
            parent = os.path.basename(os.path.dirname(os.path.abspath(member)))
            short = parent.split("_")[0] if parent not in ("", "results", "runs") else os.path.splitext(base)[0]
        return member, short
    path = os.path.join(config.RUN_ROOT, member, f"{split}_pred.csv")
    tag = config._fmt(member)                               # 실험 이름에서는 run 이름의 '_' 가 '-' 로 바뀐다
    cands = sorted(set(glob.glob(os.path.join(config.RESULT_ROOT, f"{tag}_*", f"{split}_pred.csv"))
                       + glob.glob(os.path.join(config.RESULT_ROOT, f"{tag}_*__{split}_pred.csv"))))
    if not os.path.isfile(path) and cands:                 # fetch 로 받은 사본에서라도
        if len(cands) > 1:                                 # 같은 run 이름으로 설정을 바꿔 여러 번 돌린 경우 : 추측하지 않는다
            raise SystemExit(f"{member}: results/ 에 후보가 {len(cands)}개입니다. 쓸 파일 경로를 직접 주세요:\n  "
                             + "\n  ".join(cands))
        path = cands[0]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{member}: {path} 도, results/{member}_*/{split}_pred.csv 도 없습니다")
    return path, member


def main():
    args = parse_args()
    paths, names = zip(*(resolve(m, args.split) for m in args.members))
    if len(set(names)) != len(names):
        names = tuple(f"{n}{i}" for i, n in enumerate(names))
    w = np.asarray(args.weights if args.weights else [1.0] * len(paths), dtype=float)
    assert len(w) == len(paths), "--weights 개수가 member 수와 다릅니다"
    w = w / w.sum()
    name = args.name or "ens_" + "+".join(names) + ("" if args.method == "mean" else f"_{args.method}")

    dfs = []
    for pth, n in zip(paths, names):
        d = load_pred(pth).set_index("id")
        print(f"[{n}] {len(d)}문항  {pth}")
        dfs.append(d)
    ids = dfs[0].index
    for d, n in zip(dfs[1:], names[1:]):
        assert set(d.index) == set(ids), f"[{n}] 문항 id 집합이 다릅니다 (같은 split·전체 문항이어야 함)"
    P = np.stack([d.loc[ids, PROB_COLS].to_numpy(dtype=float) for d in dfs])   # (M, N, 4)
    preds = np.stack([d.loc[ids, "pred"].to_numpy() for d in dfs])            # (M, N)

    # ── 합치기 ──
    mean = np.tensordot(w, P, axes=1)                                          # (N, 4) 가중 평균
    if args.method == "mean":
        E = mean
    else:                                                                      # 다수결, 동률은 평균 확률
        votes = np.zeros_like(mean)
        for m in range(len(dfs)):
            votes[np.arange(len(ids)), [LETTERS.index(x) for x in preds[m]]] += w[m]
        E = votes + 1e-3 * mean
    pred, conf, margin = summarize(E)

    # ── 보고 : 멤버끼리 얼마나 다르고, 앙상블이 각 멤버를 얼마나 바꿨나 ──
    print("\n멤버 간 답 일치율:")
    for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
        print(f"  {a} vs {b}: {(preds[i] == preds[j]).mean():.4f} (다른 문항 {(preds[i] != preds[j]).sum()})")
    print("앙상블 vs 각 멤버:")
    for m, n in enumerate(names):
        print(f"  {n}: 답이 바뀐 문항 {(pred != preds[m]).sum()}")
    all_agree = (preds == preds[0]).all(axis=0)
    print(f"모든 멤버가 같은 답 {all_agree.mean():.4f} | 분쟁 문항 {(~all_agree).sum()}개에서 앙상블 확신도 평균 "
          f"{conf[~all_agree].mean():.3f} (전체 {conf.mean():.3f})")

    base = dfs[0].loc[ids]
    out = pd.DataFrame({"id": ids, "qtype": base["qtype"].values, "gold": base["gold"].values})
    for k, c in enumerate(PROB_COLS):
        out[c] = mean[:, k]
    out["pred"], out["conf"], out["margin"] = pred, conf, margin
    out["n_agree"] = (preds == pred).sum(axis=0)
    out_dir = config.run_dir("ensemble", name)
    out.to_csv(os.path.join(out_dir, f"{args.split}_pred.csv"), index=False)
    save_result(out, f"{name}__{args.split}_pred.csv", quiet=True)

    gold_ok = out["gold"].astype(str).isin(LETTERS)
    if gold_ok.any():                                                          # valid/train 이면 정확도도
        for m, n in enumerate(names):
            print(f"  정확도 {n}: {(preds[m][gold_ok.values] == out.loc[gold_ok, 'gold'].values).mean():.4f}")
        print(f"  정확도 앙상블: {(pred[gold_ok.values] == out.loc[gold_ok, 'gold'].values).mean():.4f}")

    if args.split == "test":
        sub_path = os.path.join(out_dir, "submission.csv")
        write_submission(out["id"], out["pred"], config.DATA_ROOT, sub_path)
        save_result(sub_path, f"{name}__submission.csv")
    info = {"name": name, "method": args.method, "members": dict(zip(names, paths)), "weights": w.tolist(),
            "n": int(len(ids)), "all_agree_frac": round(float(all_agree.mean()), 4),
            "changed_vs_member": {n: int((pred != preds[m]).sum()) for m, n in enumerate(names)}}
    save_json(info, os.path.join(out_dir, "ensemble.json"))
    save_result(info, f"{name}__ensemble.json", quiet=True)
    append_log(config.RUN_ROOT, {"stage": "ensemble", "run": name, "method": args.method,
                                 "members": "+".join(names), "all_agree_frac": info["all_agree_frac"]})


if __name__ == "__main__":
    main()
