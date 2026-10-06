"""
2단계 : 단일 패스 추론 (sample 당 forward 1회)

1단계에서 여러 프롬프트 표현으로 학습한 어댑터 하나를 쓰고, 추론에서는 문항마다 프롬프트를 하나만 고른다.
  --prompt-mode qtype    (기본) base prompt + 그 문항 qtype 의 hint
  --prompt-mode generic  base prompt 만
  --prompt-mode fixed    --prompt 로 준 것 그대로 (예: p3, p3+부정형)
프롬프트별 branch 를 만들어 앙상블하지 않는다. 여러 프롬프트로 다시 보는 것은 4단계에서 선별 문항에만 한다.

예측할 split 은 config.PREDICT["splits"] 이고, config.USE_VALID 가 False 면 valid 는 자동으로 빠진다.
--splits 로 직접 지정하면 그 값이 우선이다. train·valid 는 1단계와 같은 분할(config.DATA, data_utils.load_data)이고,
1단계 run.json 의 valid 지문(valid_hash)과 다르면 경고한다. train 예측(암기·과적합 진단)은 기본에서 뺐다 — 필요하면
--splits valid test train.

--prompt-scan 을 주면 valid 에서 base prompt 를 하나씩 바꿔 가며 정확도와 예측 일치율만 측정한다
(프롬프트 민감도 진단용. 제출 답에는 영향을 주지 않는다).

산출물
  runs/<run>/{valid,test,train}_pred.csv , submission.csv     다음 단계가 읽는 작업 파일
  results/<실험이름>/{split}_pred.csv 등                       실험 결과 사본 (로컬 + 마운트된 드라이브)
  예측 CSV 컬럼 : id, qtype, gold, p_a..p_d, pred, conf, margin, prompt

예)
  python s2_predict.py                                        # runs/main 의 어댑터로 valid·test
  python s2_predict.py --run r32                              # 랭크 32 로 학습한 어댑터
  python s2_predict.py --run zs --adapter none                # zero-shot 기준선
  python s2_predict.py --splits valid --prompt-scan           # 프롬프트 민감도만 확인
"""
import config  # noqa: F401

import os
import time
import argparse

import numpy as np
import pandas as pd

from data_utils import (load_datasets, trained_data, describe_split, make_pred_df, accuracy, qtype_table, append_log,
                        load_json, save_json, save_result, write_submission)
from model_utils import (BASE_PROMPTS, VLM, Prompt, select_prompt, seed_everything, load_image, resize_to,
                         predict_rows)


def parse_args():
    p = argparse.ArgumentParser(description="2단계: 단일 패스 추론")
    p.add_argument("--run", default=config.RUN_NAME, help="산출물 폴더 (runs/<run>). 기본 어댑터도 여기서 찾는다")
    p.add_argument("--adapter", default=None,
                   help="어댑터 경로 (상대 경로면 runs/ 기준, 예: main/adapter_e2). 기본 runs/<run>/adapter, "
                        "'none' 이면 zero-shot")
    p.add_argument("--prompt-mode", default=None, choices=["qtype", "generic", "fixed"],
                   help="기본값은 run.json 의 값, 없으면 config.PROMPT['infer_mode']")
    p.add_argument("--prompt", default=None, help="--prompt-mode fixed 일 때 쓸 프롬프트 (예: p1, p1+부정형)")
    p.add_argument("--base-prompt", default=None, choices=list(BASE_PROMPTS),
                   help=f"qtype·generic 모드에서 쓸 base prompt (기본 {config.PROMPT['infer_base']})")
    p.add_argument("--preset", default=None, choices=list(config.MODEL_PRESETS), help="기본값은 run.json 의 값")
    default_splits = [x for x in config.PREDICT["splits"] if x != "valid" or config.USE_VALID]
    p.add_argument("--splits", nargs="+", default=default_splits, choices=["train", "valid", "test"],
                   help=f"기본 {default_splits} (config.PREDICT['splits'], USE_VALID=False 면 valid 제외)")
    p.add_argument("--prompt-scan", action="store_true",
                   help="valid 에서 base prompt 별 정확도·일치율 측정 (프롬프트 수만큼 느림)")
    p.add_argument("--batch-size", type=int, default=config.PREDICT["batch_size"],
                   help="한 번에 forward 하는 문항 수. OOM 이면 그 배치만 1개씩 재시도")
    p.add_argument("--infer-max-side", type=int, default=config.INFER_MAX_SIDE)
    p.add_argument("--n-train", type=int, default=None, help="None 이면 train 전체 (train split 을 넣었을 때만 씀)")
    p.add_argument("--n-valid", type=int, default=None, help="None 이면 합의 문항 전체")
    p.add_argument("--n-test", type=int, default=None, help="None 이면 test 전체 (제출용은 반드시 None)")
    p.add_argument("--tag", default=None, help="결과 파일 이름 (기본 run.json 의 실험 이름)")
    p.add_argument("--download-source", default="hf", choices=["hf", "modelscope"])
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(config.SEED)
    out_dir = config.run_dir(args.run)
    meta = load_json(os.path.join(out_dir, "run.json"), {})
    adapter = None if args.adapter == "none" else (args.adapter or os.path.join(out_dir, "adapter"))
    if adapter and not os.path.isdir(adapter) and os.path.isdir(os.path.join(config.RUN_ROOT, adapter)):
        adapter = os.path.join(config.RUN_ROOT, adapter)      # runs/ 기준 상대 경로 (예: main/adapter_e2)
    # 다른 run 의 어댑터를 쓰면 그 run 의 학습 설정(프리셋·실험 이름)을 따른다
    src_meta = load_json(os.path.join(os.path.dirname(os.path.abspath(adapter)), "run.json"), {}) if adapter else {}
    if adapter and not os.path.isdir(adapter):
        raise FileNotFoundError(f"어댑터가 없습니다: {adapter} (1단계를 먼저 돌리거나 --adapter none)")
    preset = args.preset or meta.get("preset") or src_meta.get("preset", config.DEFAULT_PRESET)
    mode = args.prompt_mode or meta.get("infer_mode") or config.PROMPT["infer_mode"]
    base = args.base_prompt or config.PROMPT["infer_base"]
    # 결과 파일 이름 : 1단계가 만든 실험 이름을 그대로 이어 쓴다 (없으면 여기서 만든다)
    tag = (args.tag or meta.get("exp_tag") or src_meta.get("exp_tag")
           or config.exp_tag(args.run, preset, seed=config.SEED, zero_shot=adapter is None))
    pick = lambda r: select_prompt(r["qtype"], mode, base=base, fixed=args.prompt)  # noqa: E731

    # train·valid 는 어댑터를 학습할 때와 같은 분할이어야 한다 (어댑터 폴더의 train_meta.json 을 따른다).
    # valid 에 학습 문항이 섞일 수 있으면(--no-valid 학습, 예전 코드의 어댑터, 분할이 바뀜) valid 는 예측하지 않는다
    trained, note = trained_data(adapter, src_meta or meta)
    train_df, valid_df, test_df, data_info = load_datasets(
        n_train=args.n_train if "train" in args.splits else 0,
        n_valid=args.n_valid if "valid" in args.splits else 0,
        n_test=args.n_test if "test" in args.splits else 0, use_valid=trained.get("use_valid"),
        train_source=trained.get("train_source"), train_kinds=trained.get("train_kinds"))
    print(f"[데이터] {describe_split(data_info)}")
    skip = None
    if "valid" not in args.splits:
        if note:
            print(f"  ⚠ {note}")
    elif note:
        skip = note
    elif not len(valid_df):
        skip = "valid 가 없습니다 (어댑터를 valid 없이 학습 = valid 문항까지 학습에 씀)"
    elif trained.get("valid_hash") and trained["valid_hash"] != data_info["valid_hash"]:
        skip = (f"어댑터를 학습한 valid(id {trained['valid_hash']})와 지금 분할(id {data_info['valid_hash']})이 다릅니다 "
                f"(config.DATA 변경?) → 지금 valid 에는 학습한 문항이 섞여 있을 수 있음")
    if skip:
        print(f"  ⚠ {skip} → valid 예측은 건너뜁니다")
        args.splits = [s for s in args.splits if s != "valid"]
        stale = os.path.join(out_dir, "valid_pred.csv")
        if os.path.isfile(stale):                           # 예전 실행의 valid 예측을 3단계가 읽지 않게
            os.remove(stale)
            print(f"  (예전 {stale} 삭제)")
    dfs = {"train": train_df, "valid": valid_df, "test": test_df}
    vlm = VLM(preset, adapter_dir=adapter, download_source=args.download_source)
    image_fn = lambda r: [resize_to(load_image(r["img_path"]), args.infer_max_side)]  # noqa: E731
    print(f"[설정] 실험 {tag} | 어댑터 {adapter or 'none(zero-shot)'} | 프롬프트 {mode} (base {base})")

    summary = {"exp_tag": tag, "run": args.run, "preset": preset, "adapter": adapter or "none",
               "prompt_mode": mode, "base_prompt": base, "fixed_prompt": args.prompt,
               "batch_size": args.batch_size, "infer_max_side": args.infer_max_side, "seed": config.SEED,
               "lora": meta.get("lora"), "best_epoch": meta.get("best_epoch"), "best_update": meta.get("best_update"),
               "data": {k: data_info[k] for k in ("train_source", "valid_source", "valid_hash", "n_valid_full")},
               "splits": {}}
    accs = {}
    for split in args.splits:
        df = dfs[split]
        t0 = time.time()
        P, prompts = predict_rows(vlm, [r for _, r in df.iterrows()], image_fn, pick,
                                  batch_size=args.batch_size, desc=f"{args.run} {split}")
        out = make_pred_df(df, P, prompt=prompts)
        out.to_csv(os.path.join(out_dir, f"{split}_pred.csv"), index=False)      # 다음 단계가 읽는 작업 파일
        save_result(out, f"{tag}__{split}_pred.csv")                             # 실험 결과 사본
        sec = (time.time() - t0) / len(df)
        acc = accuracy(out)                                   # gold 가 없는 test 는 nan
        summary["splits"][split] = {"n": len(df), "acc": None if np.isnan(acc) else round(acc, 4),
                                    "conf_mean": round(float(out["conf"].mean()), 3), "sec_per_q": round(sec, 3)}
        print(f"[{split}] {len(df)}문항, {sec:.2f}초/문항" + ("" if np.isnan(acc) else f" | 정확도 {acc:.4f}"))

        if not np.isnan(acc):
            accs[split] = acc
            table = qtype_table(out)
            print(table.to_string(index=False))
            save_result(table, f"{tag}__{split}_qtype.csv", quiet=True)
        if split == "test":
            # 4단계까지 가기 전에도 바로 낼 수 있는 기준 제출 파일
            sub_path = os.path.join(out_dir, "submission.csv")
            write_submission(out["id"], out["pred"], config.DATA_ROOT, sub_path)
            save_result(sub_path, f"{tag}__submission.csv")

    # ── 프롬프트 민감도 진단 (valid 전용, 제출 답에는 영향 없음) ──
    if args.prompt_scan and "valid" in args.splits and len(valid_df):
        ref, rows_out, preds = None, [], {}
        for pid in BASE_PROMPTS:
            t0 = time.time()
            P, _ = predict_rows(vlm, [r for _, r in valid_df.iterrows()], image_fn, Prompt(pid, None),
                                batch_size=args.batch_size, desc=f"scan {pid}")
            d = make_pred_df(valid_df, P)
            preds[pid] = d["pred"].values
            ref = ref if ref is not None else preds[pid]
            rows_out.append({"prompt": pid, "valid_acc": round(accuracy(d), 4),
                             "conf_mean": round(float(d["conf"].mean()), 3),
                             "agree_with_first": round(float((preds[pid] == ref).mean()), 4),
                             "sec_per_q": round((time.time() - t0) / len(valid_df), 2)})
        allp = np.stack(list(preds.values()))
        same_all = float((allp == allp[0]).all(axis=0).mean())
        scan = pd.DataFrame(rows_out)
        scan.to_csv(os.path.join(out_dir, "prompt_scan.csv"), index=False, encoding="utf-8-sig")
        save_result(scan, f"{tag}__prompt_scan.csv")
        print("\n[프롬프트 민감도] (generic, hint 없음)")
        print(scan.to_string(index=False))
        print(f"  모든 프롬프트가 같은 답: {same_all:.1%} (낮을수록 표현에 민감 → hint_prob·base prompt 조정 대상)")
        summary["prompt_scan"] = {"all_agree": round(same_all, 4), "by_prompt": rows_out}
        append_log(config.RUN_ROOT, {"stage": "s2_prompt_scan", "run": args.run, "exp_tag": tag, "preset": preset,
                                     "n_valid": len(valid_df), "all_agree": round(same_all, 4),
                                     **{f"acc_{r['prompt']}": r["valid_acc"] for r in rows_out}})

    save_json(summary, os.path.join(out_dir, "predict.json"))
    save_result(summary, f"{tag}__predict.json", quiet=True)
    append_log(config.RUN_ROOT, {"stage": "s2_predict", "run": args.run, "exp_tag": tag, "preset": preset,
                                 "adapter": adapter or "none", "best_update": meta.get("best_update"),
                                 "train_source": data_info["train_source"], "valid_hash": data_info["valid_hash"],
                                 "prompt_mode": mode, "base_prompt": base,
                                 **{f"n_{s}": v["n"] for s, v in summary["splits"].items()},
                                 **{f"acc_{s}": v["acc"] for s, v in summary["splits"].items() if v["acc"] is not None},
                                 "conf_mean_valid": summary["splits"].get("valid", {}).get("conf_mean"),
                                 "sec_per_q": summary["splits"].get("valid", {}).get("sec_per_q")})
    if "train" in accs and "valid" in accs:
        print(f"\n[요약] train {accs['train']:.4f} / valid {accs['valid']:.4f} "
              f"(차이 {accs['train'] - accs['valid']:+.4f} — 크면 train 에 과적합)")


if __name__ == "__main__":
    main()
