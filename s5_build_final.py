"""
최종 단계 : 필요한 행만 교체해 제출 파일 생성 (CPU)

2단계 예측(기본 답)에서, 3단계가 선별하고 4단계가 다시 푼 문항만 바꾼다.
재추론 방식(--rerun-tag, 여러 개 가능 : grounding_adapter reason …)과 교체 방법을 valid 에서 비교해 고르고
test 에 그대로 적용한다.

  keep    : 교체하지 않음 (기준선 = 2단계 submission.csv 와 같음)
  replace : 선별 문항은 모두 재추론 답으로
  rule    : 재추론 확신도 >= 기본 확신도 + --conf-delta 일 때만 교체
  blend   : 선별 문항은 (1-w)·기본 확률 + w·재추론 확률 (auto 는 w = 0.3 / 0.5 / 0.7 을 비교)
  auto    : (기본) 재추론 방식 × 위 방법 전부를 valid 로 비교해 가장 나은 조합. 단 keep 보다 --min-gain 이상
            나아야 채택하고 아니면 keep (선별 문항이 적어 우연히 나온 작은 차이로 바꾸지 않게).
            valid 가 없으면 첫 번째 재추론 방식에 config.FINAL['fallback'] 방법

산출물 : runs/<run>/final/final_submission.csv , final_meta.csv , final.json , valid_compare.csv
         results/<실험이름>/final/…  (사본)

예)
  python s5_build_final.py --run q9b --rerun-tag grounding_adapter reason     # 둘 중 valid 로 나은 쪽 (auto)
  python s5_build_final.py --run q9b --rerun-tag reason --method replace      # 방법을 직접 지정
"""
import config  # noqa: F401

import os
import argparse

import numpy as np
import pandas as pd

from data_utils import (PROB_COLS, load_pred, probs, summarize, write_submission, append_log,
                        load_json, save_json, save_result)


def parse_args():
    F = config.FINAL                                   # 기본값은 config.py 에서 관리
    p = argparse.ArgumentParser(description="최종: 필요한 행만 교체")
    p.add_argument("--rerun-tag", nargs="+", default=["grounding_adapter"],
                   help="runs/<run>/rerun/<tag> — 여러 개 주면 auto 가 valid 로 고른다")
    p.add_argument("--run", default=config.RUN_NAME, help="기본 답을 가져올 run (runs/<run>/*_pred.csv)")
    p.add_argument("--method", default=F["method"], choices=["auto", "keep", "replace", "rule", "blend"])
    p.add_argument("--min-gain", type=float, default=F["min_gain"], help="auto 가 교체를 채택할 최소 valid 향상")
    p.add_argument("--conf-delta", type=float, default=F["conf_delta"])
    p.add_argument("--weight", type=float, default=F["weight"])
    return p.parse_args()


def apply(base, rerun, method, conf_delta, weight):
    """base 전체 + rerun(부분) → 최종 확률, 교체 여부"""
    P = probs(base).copy()
    idx = base.reset_index().set_index("id").loc[rerun["id"], "index"].values   # rerun 행의 base 위치
    Pb, Pr = P[idx], probs(rerun)
    if method == "keep":
        new, changed = Pb, np.zeros(len(idx), bool)
    elif method == "replace":
        new, changed = Pr, np.ones(len(idx), bool)
    elif method == "rule":
        take = rerun["conf"].values >= base["conf"].values[idx] + conf_delta
        new, changed = np.where(take[:, None], Pr, Pb), take
    else:
        new, changed = (1 - weight) * Pb + weight * Pr, np.ones(len(idx), bool)
    P[idx] = new
    flag = np.zeros(len(base), bool)
    flag[idx] = changed
    return P, flag


def main():
    args = parse_args()
    run_dir = os.path.join(config.RUN_ROOT, args.run)
    out_dir = config.run_dir(args.run, "final")
    exp = load_json(os.path.join(run_dir, "run.json"), {}).get("exp_tag") or config.exp_tag(args.run, config.DEFAULT_PRESET)
    rdir = lambda t: os.path.join(run_dir, "rerun", t)  # noqa: E731
    tags = [t for t in args.rerun_tag if os.path.isdir(rdir(t))]
    for t in args.rerun_tag:
        if t not in tags:
            print(f"  ⚠ 재추론 결과가 없어 건너뜀: {rdir(t)}")

    # ── valid 에서 재추론 방식 × 교체 방법 비교 ──
    vp = os.path.join(run_dir, "valid_pred.csv")
    has_valid = config.USE_VALID and os.path.isfile(vp)
    rows, keep_acc = [], None
    if has_valid:
        base_v = load_pred(vp)
        gold = base_v["gold"].values
        keep_acc = float((base_v["pred"].values == gold).mean())
        rows.append({"rerun_tag": "-", "method": "keep", "param": "", "valid_acc": keep_acc, "n_rerun": 0})
        for t in tags:
            vr = os.path.join(rdir(t), "valid_rerun.csv")
            if not os.path.isfile(vr):
                print(f"  ⚠ {t}: valid 재추론이 없어 비교에서 뺌")
                continue
            rerun_v = load_pred(vr)
            for m, v in [("replace", ""), ("rule", args.conf_delta), ("blend", 0.3), ("blend", 0.5), ("blend", 0.7)]:
                P, _ = apply(base_v, rerun_v, m, conf_delta=v if m == "rule" else 0.0,
                             weight=v if m == "blend" else 0.5)
                rows.append({"rerun_tag": t, "method": m, "param": v,
                             "valid_acc": float((summarize(P)[0] == gold).mean()), "n_rerun": len(rerun_v)})
        table = pd.DataFrame(rows)
        table["gain"] = (table["valid_acc"] - keep_acc).round(4)
        table["valid_acc"] = table["valid_acc"].round(4)
        print(f"[valid] 전체 {len(base_v)}문항 | 교체 방법 비교 (gain = keep 대비)")
        print(table.to_string(index=False))
    else:
        table = pd.DataFrame(rows)
        print("[알림] valid 가 없어 비교를 건너뜁니다 (config.USE_VALID 또는 valid_pred.csv)")

    # ── 채택할 조합 ──
    if args.method == "auto":
        cands = [r for r in rows if r["method"] != "keep"]
        if has_valid and cands:
            best = max(cands, key=lambda r: r["valid_acc"])
            if best["valid_acc"] >= keep_acc + args.min_gain:
                tag, method, param = best["rerun_tag"], best["method"], best["param"]
            else:
                tag, method, param = None, "keep", ""
                print(f"  → 최고 {best['rerun_tag']}/{best['method']} {best['valid_acc']:.4f} 가 keep {keep_acc:.4f} 보다 "
                      f"{args.min_gain} 이상 낫지 않아 교체하지 않습니다")
        elif tags:                                    # valid 없음 → 설정값 그대로
            tag, method = tags[0], config.FINAL["fallback"]
            param = args.conf_delta if method == "rule" else (args.weight if method == "blend" else "")
        else:
            tag, method, param = None, "keep", ""
    else:
        tag = tags[0] if (tags and args.method != "keep") else None
        method = args.method if (tag or args.method == "keep") else "keep"
        param = args.conf_delta if method == "rule" else (args.weight if method == "blend" else "")
    chosen_valid = None
    if has_valid:
        hit = [r for r in rows if r["rerun_tag"] == (tag or "-") and r["method"] == method
               and (method in ("keep", "replace") or r["param"] == param)]
        chosen_valid = hit[0]["valid_acc"] if hit else None
    print(f"  ▶ 채택 : {tag or '재추론 없음'} / {method}" + (f" ({param})" if param != "" else "")
          + (f" → valid {chosen_valid:.4f} (keep {keep_acc:.4f})" if chosen_valid is not None else ""))

    # ── test 에 적용 ──
    base_t = load_pred(os.path.join(run_dir, "test_pred.csv"))
    Pt, flag = probs(base_t), np.zeros(len(base_t), bool)
    if method != "keep" and tag:
        tr = os.path.join(rdir(tag), "test_rerun.csv")
        if os.path.isfile(tr):
            Pt, flag = apply(base_t, load_pred(tr), method, conf_delta=param if method == "rule" else 0.0,
                             weight=param if method == "blend" else 0.5)
        else:
            print(f"  ⚠ {tag} 의 test 재추론이 없어 기본 답 그대로 사용")
    pred, conf, margin = summarize(Pt)
    print(f"[test] {len(base_t)}문항 중 {flag.sum()}문항 교체, 답이 바뀐 문항 {(pred != base_t['pred'].values).sum()}")

    meta = base_t[["id", "qtype"]].copy()
    meta["base_pred"], meta["final_pred"], meta["changed"] = base_t["pred"].values, pred, flag
    meta["final_conf"], meta["final_margin"] = conf, margin
    for i, c in enumerate(PROB_COLS):
        meta[c] = Pt[:, i]
    meta.to_csv(os.path.join(out_dir, "final_meta.csv"), index=False)
    sub_path = os.path.join(out_dir, "final_submission.csv")
    write_submission(base_t["id"], pred, config.DATA_ROOT, sub_path)
    save_result(sub_path, f"{exp}__final__final_submission.csv")
    save_result(meta, f"{exp}__final__final_meta.csv", quiet=True)
    if len(table):
        table.to_csv(os.path.join(out_dir, "valid_compare.csv"), index=False, encoding="utf-8-sig")
        save_result(table, f"{exp}__final__valid_compare.csv", quiet=True)

    info = {"run": args.run, "exp_tag": exp, "rerun_tags": args.rerun_tag, "method_arg": args.method,
            "chosen_tag": tag, "chosen_method": method, "chosen_param": param, "min_gain": args.min_gain,
            "valid_keep": keep_acc, "valid_final": chosen_valid, "valid_compare": rows, "n_changed": int(flag.sum())}
    save_json(info, os.path.join(out_dir, "final.json"))
    save_result(info, f"{exp}__final__final.json", quiet=True)
    append_log(config.RUN_ROOT, {"stage": "s5_final", "run": args.run, "exp_tag": f"{exp}__final",
                                 "rerun_tag": tag or "-", "method": method, "param": param,
                                 "valid_acc": None if chosen_valid is None else round(chosen_valid, 4),
                                 "valid_acc_keep": None if keep_acc is None else round(keep_acc, 4),
                                 "n_changed": int(flag.sum())})


if __name__ == "__main__":
    main()
