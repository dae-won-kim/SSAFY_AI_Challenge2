"""
3단계 : 재추론 대상 선별 (CPU)

정답을 보지 않는 신호만으로 "불확실한 문항"을 고른다. 같은 규칙을 valid 와 test 에 똑같이 적용하고,
valid 에서는 그 규칙이 오답을 얼마나 잡아내는지(오답 포착률)를 보여준다.
전 문항에 재추론을 돌리지 않기 위한 단계다 (비용이 드는 방법은 선별 문항에만).

선별 조건 (하나라도 해당하면 선별):
  1) 확신도 < --conf-th
  2) margin(1·2위 차이) < --margin-th
  3) 약한 qtype 이면서 확신도 < --weak-conf-th   (약한 qtype 은 valid 에서 자동 선정 또는 --weak-qtypes)
선별 비율이 --max-frac 를 넘으면 확신도가 낮은 순으로 잘라낸다.

산출물 : runs/<run>/select/valid_subset.csv , test_subset.csv , selection.json , qtype_report.csv

예)
  python s3_select_rerun.py
  python s3_select_rerun.py --run r32 --weak-qtypes 숫자·가격 부정형 --max-frac 0.25
"""
import config  # noqa: F401

import os
import argparse

import numpy as np
import pandas as pd

from data_utils import QTYPES, load_pred, accuracy, qtype_table, append_log, load_json, save_json, save_result


def parse_args():
    S = config.SELECT                                  # 기본값은 config.py 에서 관리
    p = argparse.ArgumentParser(description="3단계: 재추론 대상 선별")
    p.add_argument("--run", default=config.RUN_NAME, help="2단계 예측을 가져올 run (runs/<run>/*_pred.csv)")
    p.add_argument("--conf-th", type=float, default=S["conf_th"])
    p.add_argument("--margin-th", type=float, default=S["margin_th"])
    p.add_argument("--weak-conf-th", type=float, default=S["weak_conf_th"])
    p.add_argument("--weak-qtypes", nargs="*", default=S["weak_qtypes"], help=f"직접 지정. 후보: {QTYPES}")
    p.add_argument("--weak-delta", type=float, default=S["weak_delta"], help="전체 정확도보다 이만큼 낮으면 약한 qtype")
    p.add_argument("--weak-min-n", type=int, default=S["weak_min_n"])
    p.add_argument("--max-frac", type=float, default=S["max_frac"], help="선별 비율 상한 (4단계 비용 통제)")
    return p.parse_args()


def select(df, rule):
    reasons = []
    for _, r in df.iterrows():
        why = []
        if r["conf"] < rule["conf_th"]:
            why.append("low_conf")
        if r["margin"] < rule["margin_th"]:
            why.append("low_margin")
        if r["qtype"] in rule["weak_qtypes"] and r["conf"] < rule["weak_conf_th"]:
            why.append("weak_qtype")
        reasons.append("+".join(why))
    reasons = pd.Series(reasons, index=df.index)
    mask = reasons != ""
    cap = int(np.floor(rule["max_frac"] * len(df)))
    if mask.sum() > cap:                                  # 확신도가 낮은 순으로 상한까지
        keep = df.loc[mask].sort_values("conf").index[:cap]
        mask = df.index.isin(keep)
    return pd.Series(mask, index=df.index), reasons


def main():
    args = parse_args()
    run_dir = os.path.join(config.RUN_ROOT, args.run)
    out_dir = config.run_dir(args.run, "select")
    meta = load_json(os.path.join(run_dir, "run.json"), {})
    tag = meta.get("exp_tag") or config.exp_tag(args.run, meta.get("preset", config.DEFAULT_PRESET))
    paths = {s: os.path.join(run_dir, f"{s}_pred.csv") for s in ("valid", "test")}
    if not os.path.isfile(paths["test"]):
        raise FileNotFoundError(f"{paths['test']} 가 없습니다 (2단계를 먼저: python s2_predict.py --run {args.run})")
    test = load_pred(paths["test"])
    valid = load_pred(paths["valid"]) if (config.USE_VALID and os.path.isfile(paths["valid"])) else None

    if valid is None:                                   # config.USE_VALID=False : 정답이 없어 진단은 못 한다
        print("[알림] valid 를 쓰지 않습니다(config.USE_VALID=False 또는 파일 없음) → 오답 포착률·약한 qtype 자동 선정 건너뜀 "
              "(문턱값은 config.SELECT 값을 그대로 적용)")
        stale = os.path.join(out_dir, "valid_subset.csv")
        if os.path.isfile(stale):                       # 예전 실행의 valid 선별을 4단계가 읽지 않게
            os.remove(stale)
            print(f"  (예전 {stale} 삭제)")
        overall, table = float("nan"), qtype_table(test)
        weak = list(args.weak_qtypes or [])
    else:
        overall = accuracy(valid)
        table = qtype_table(valid)
        if args.weak_qtypes is None:
            weak = table[(table["qtype"] != "전체") & (table["n"] >= args.weak_min_n)
                         & (table["acc_pred"] < overall - args.weak_delta)]["qtype"].tolist()
        else:
            weak = list(args.weak_qtypes)
    rule = {"run": args.run, "conf_th": args.conf_th, "margin_th": args.margin_th,
            "weak_conf_th": args.weak_conf_th, "weak_qtypes": weak, "max_frac": args.max_frac}

    if valid is not None:
        print(f"[valid] 전체 정확도 {overall:.4f} | 약한 qtype: {weak or '없음'}")
    table["weak"] = table["qtype"].isin(weak)

    record = {"stage": "s3_select", "exp_tag": tag, **{k: str(v) for k, v in rule.items()}}
    for name, df in [("valid", valid), ("test", test)]:
        if df is None:
            continue
        mask, reasons = select(df, rule)
        sub = df.loc[mask].copy()
        sub["reason"] = reasons[mask].values
        sub.to_csv(os.path.join(out_dir, f"{name}_subset.csv"), index=False)
        save_result(sub, f"{tag}__{name}_subset.csv", quiet=True)
        with open(os.path.join(out_dir, f"{name}_subset_ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sub["id"].astype(str)))
        print(f"[{name}] 선별 {mask.sum()}/{len(df)} ({mask.mean():.1%})")
        record[f"{name}_selected"] = int(mask.sum())
        record[f"{name}_selected_frac"] = round(float(mask.mean()), 3)
        reason_counts = pd.Series([x for r in reasons[mask] for x in r.split("+")]).value_counts()
        print("  선별 사유:", reason_counts.to_dict())

        if name == "valid":
            wrong = valid["pred"] != valid["gold"]
            captured = (wrong & mask).sum() / max(1, wrong.sum())
            acc_in, acc_out = accuracy(valid[mask]), accuracy(valid[~mask])
            print(f"  선별 문항 정확도 {acc_in:.4f} | 비선별 {acc_out:.4f} | 전체 오답 중 선별에 포함 {captured:.1%}")
            print(f"  → 4단계가 선별 문항을 전부 맞히면 최대 +{(wrong & mask).sum() / len(valid):.1%}p 까지 가능")
            record.update(valid_acc_selected=round(acc_in, 4), valid_error_capture=round(float(captured), 3))
        if name == ("valid" if valid is not None else "test"):     # qtype 표의 선별 수 (valid 가 있으면 valid 기준)
            sel_by_q = df.loc[mask, "qtype"].value_counts()
            table["selected"] = table["qtype"].map(sel_by_q).fillna(0).astype(int)
            table.loc[table["qtype"] == "전체", "selected"] = int(mask.sum())
            print(table.to_string(index=False))

    append_log(config.RUN_ROOT, record)                     # valid 가 없어도 test 선별 수는 기록한다
    print(f"[선별 결과] {out_dir}")
    table.to_csv(os.path.join(out_dir, "qtype_report.csv"), index=False, encoding="utf-8-sig")
    save_json(rule, os.path.join(out_dir, "selection.json"))
    save_result(table, f"{tag}__select_qtype_report.csv")
    save_result(rule, f"{tag}__selection.json", quiet=True)


if __name__ == "__main__":
    main()
