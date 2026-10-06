"""
4단계 : 3단계가 선별한 문항만 다시 푼다 (전 문항에는 적용하지 않는다)

--mode grounding (기본)
  [1패스] 이미지(축소) + 질문 + 선지 → LoRA 를 끈 base 모델이 "봐야 할 영역" bbox 생성 (캐시)
  [크롭]  bbox + 여백을 원본 해상도에서 잘라냄 → 작은 글자가 크게 보임
  [2패스] [전체 사진] + [확대 사진] + 질문 + 선지 → a/b/c/d 확률
  크롭할 가치가 없는 문항(bbox 없음, 크롭이 원본의 대부분)은 전체 사진 1장으로 다시 푼다.
  2패스를 LoRA 로 풀지(--pass2 adapter) base 로 풀지(--pass2 base)는 valid 로 비교해서 정한다.

--mode prompt
  같은 이미지 1장을 base prompt 를 바꿔 가며 여러 번 풀고 확률을 평균한다 (--prompt-ids, --prompt-hint).
  학습 때 프롬프트 표현을 섞었으므로 남은 민감도가 있는 문항에서만 효과가 있는지 valid 로 확인한다.

--mode reason  (thinking 을 지원하는 Qwen3.5 계열만)
  모델이 먼저 생각을 글로 쓴 뒤(<think>…</think>) 답한다. 부정형("없는 것")·가격 대조·지식형처럼
  여러 단계를 거쳐야 하는 문항을 바로 답하기보다 잘 풀기를 기대하는 방법 (model_utils.VLM.reason).
  기본은 학습한 LoRA 를 끈 원래 모델로 생각한다 (--reason-adapter 로 켬). 생성이라 느려서 선별 문항에만.
  최종 단계(s5 --method auto)가 grounding 과 이 결과 중 valid 로 나은 쪽(또는 교체 안 함)을 고른다.

산출물 : runs/<run>/rerun/<tag>/valid_rerun.csv , test_rerun.csv , rerun.json (, grounding 이면 crops/)
         tag 기본 : grounding_adapter · grounding_base · prompt · reason (· reason_adapter)
         runs/<run>/rerun/ground_cache_<모델>_<해상도>.json (1패스 캐시, 같은 run 의 tag 끼리 공유)

예)
  python s4_rerun_grounding.py                                  # runs/main + grounding + adapter
  python s4_rerun_grounding.py --pass2 base
  python s4_rerun_grounding.py --mode prompt
  python s4_rerun_grounding.py --mode reason                    # 생각하기 재추론 (LoRA 끔)
"""
import config  # noqa: F401

import os
import time
import argparse

import numpy as np
from PIL import Image, ImageDraw

from data_utils import (load_datasets, trained_data, load_pred, make_pred_df, crop_region, append_log,
                        load_json, save_json, save_result, progress)
from model_utils import (BASE_PROMPTS, VLM, Prompt, prompt_str, select_prompt, seed_everything,
                         load_image, resize_to, predict_rows, run_batched)


def parse_args():
    R = config.RERUN                                   # 기본값은 config.py 에서 관리
    p = argparse.ArgumentParser(description="4단계: 선별 문항 재추론")
    p.add_argument("--run", default=config.RUN_NAME, help="어댑터·프리셋·프롬프트 정책을 가져올 run")
    p.add_argument("--mode", default=R["mode"], choices=["grounding", "prompt", "reason"])
    p.add_argument("--pass2", default=R["pass2"], choices=["adapter", "base"], help="grounding 2패스 모델")
    p.add_argument("--tag", default=None, help="산출물 폴더 이름. 기본 <mode>[_<pass2>] (runs/<run>/rerun/<tag>)")
    p.add_argument("--select-dir", default=None, help="기본 runs/<run>/select")
    p.add_argument("--splits", nargs="+", default=["valid", "test"], choices=["valid", "test"])
    p.add_argument("--batch-size", type=int, default=R["batch_size"])
    # mode=prompt
    p.add_argument("--prompt-ids", nargs="*", default=R["prompt_ids"], choices=list(BASE_PROMPTS),
                   help="평균할 base prompt 목록 (기본 전체)")
    p.add_argument("--prompt-hint", action=argparse.BooleanOptionalAction, default=R["prompt_hint"],
                   help="qtype hint 를 붙인 버전도 포함")
    # mode=reason
    p.add_argument("--reason-adapter", action=argparse.BooleanOptionalAction, default=R["reason_adapter"],
                   help="학습한 LoRA 를 켠 채로 생각 (기본 끔 = 원래 모델)")
    p.add_argument("--reason-max-new-tokens", type=int, default=R["reason_max_new_tokens"], help="생각 길이 상한")
    p.add_argument("--reason-batch", type=int, default=R["reason_batch"])
    # mode=grounding : 1패스
    p.add_argument("--ground-max-side", type=int, default=R["ground_max_side"])
    p.add_argument("--ground-batch", type=int, default=R["ground_batch"])
    p.add_argument("--max-new-tokens", type=int, default=R["max_new_tokens"])
    # 크롭
    p.add_argument("--crop-pad", type=float, default=R["crop_pad"])
    p.add_argument("--crop-min-frac", type=float, default=R["crop_min_frac"])
    p.add_argument("--crop-max-area", type=float, default=R["crop_max_area"])
    # 2패스 해상도
    p.add_argument("--full-max-side", type=int, default=R["full_max_side"], help="전체 사진 (문맥용)")
    p.add_argument("--crop-max-side", type=int, default=R["crop_max_side"], help="확대 사진 (읽기용)")
    p.add_argument("--crop-min-side", type=int, default=R["crop_min_side"], help="작은 크롭은 이 크기까지 확대")
    p.add_argument("--single-max-side", type=int, default=config.INFER_MAX_SIDE, help="크롭 생략 시 1장 해상도")
    p.add_argument("--save-crops", type=int, default=20, help="크롭 확인용 이미지 저장 개수 (valid)")
    p.add_argument("--download-source", default="hf", choices=["hf", "modelscope"])
    return p.parse_args()


def save_preview(row, boxes, cb, path):
    img = load_image(row["img_path"])
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle(b, outline=(255, 0, 0), width=max(2, img.width // 300))
    if cb:
        draw.rectangle(cb, outline=(0, 90, 255), width=max(2, img.width // 300))
    left = resize_to(img, 700)
    right = resize_to(load_image(row["img_path"]).crop(cb), 700, 400) if cb else None
    W = left.width + (right.width + 10 if right else 0)
    H = max(left.height, right.height if right else 0)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(left, (0, 0))
    if right:
        canvas.paste(right, (left.width + 10, 0))
    canvas.save(path, quality=85)


def main():
    args = parse_args()
    seed_everything(config.SEED)
    run_dir = os.path.join(config.RUN_ROOT, args.run)
    meta = load_json(os.path.join(run_dir, "run.json"), {})
    preset = meta.get("preset", config.DEFAULT_PRESET)
    mode_infer = meta.get("infer_mode") or config.PROMPT["infer_mode"]
    variant = {"grounding": f"_{args.pass2}", "reason": "_adapter" if args.reason_adapter else ""}.get(args.mode, "")
    tag = args.tag or (args.mode + variant)
    # 결과 파일 이름 : 학습 실험 이름 + 재추론 태그 (기본 태그면 예전 이름 rerun-<mode>[-<variant>] 와 같다.
    # --tag 를 주면 그 이름으로 — 예전에는 모드만 봐서 --tag reason2k 가 rerun-reason 사본을 덮어썼다)
    exp = (meta.get("exp_tag") or config.exp_tag(args.run, preset)) + "__rerun-" + tag.replace("_", "-")
    out_dir = config.run_dir(args.run, "rerun", tag)
    # 예전 실행의 결과를 먼저 지운다 : 이번 실행이 도중에 실패하면 최종 단계가 예전 파일을 쓰는 일이 없게
    for f in [f"{s}_rerun.csv" for s in args.splits] + ["rerun.json"]:
        if os.path.isfile(os.path.join(out_dir, f)):
            os.remove(os.path.join(out_dir, f))
    select_dir = args.select_dir or os.path.join(run_dir, "select")

    # ── 선별 문항 불러오기 (id → 이미지·질문·선지) ──
    # valid 는 1·2단계와 같은 분할에서 찾는다 (1단계 run.json 의 data). 학습 문항이 섞일 수 있으면 valid 는 하지 않는다
    # 2단계가 쓴 어댑터(runs/<run>/adapter/)의 표식을 본다. 어댑터가 없으면 zero-shot run (2단계 --adapter none)
    run_adapter = os.path.join(run_dir, "adapter")
    trained, note = trained_data(run_adapter if os.path.isdir(run_adapter) else None, meta)
    _, valid_all, test_all, data_info = load_datasets(n_train=0, use_valid=trained.get("use_valid"),
                                                      train_source=trained.get("train_source"),
                                                      train_kinds=trained.get("train_kinds"))
    valid_ok = (config.USE_VALID and len(valid_all) > 0 and not note
                and trained.get("valid_hash") in (None, data_info["valid_hash"]))
    if "valid" in args.splits and not valid_ok:
        print("  ⚠ valid 는 건너뜁니다 : " + (note or ("1단계 valid 와 지금 분할이 다름" if len(valid_all) else "valid 없음")))
        stale = os.path.join(out_dir, "valid_rerun.csv")
        if os.path.isfile(stale):                       # 예전 실행의 valid 재추론을 최종 단계가 읽지 않게
            os.remove(stale)
    subsets, rows_by_split = {}, {}
    for split in args.splits:
        sub_path = os.path.join(select_dir, f"{split}_subset.csv")
        if split == "valid" and not valid_ok:
            continue                                      # valid 를 쓰지 않는 모드
        if not os.path.isfile(sub_path):
            print(f"[{split}] 선별 파일이 없어 건너뜁니다 ({sub_path})")
            continue
        sub = load_pred(sub_path)
        full = (valid_all if split == "valid" else test_all).copy()
        full["id"] = full["id"].astype(str)
        full = full.set_index("id")
        missing = set(sub["id"]) - set(full.index)
        assert not missing, f"[{split}] 선별 id {len(missing)}개가 원본 데이터에 없음"
        rows = full.loc[sub["id"]].reset_index()
        if split == "valid":
            rows["answer"] = sub["gold"].values           # 2·3단계가 쓴 정답과 동일
        subsets[split], rows_by_split[split] = sub, rows
        print(f"[{split}] 재추론 대상 {len(rows)}문항")

    if not rows_by_split:
        raise SystemExit("재추론할 split 이 없습니다 (3단계를 먼저 돌리세요)")

    adapter = os.path.join(run_dir, "adapter")
    use_adapter = not ((args.mode == "grounding" and args.pass2 == "base")
                       or (args.mode == "reason" and not args.reason_adapter))    # 원래 모델만 쓰면 LoRA 를 올리지 않음
    if use_adapter and not os.path.isdir(adapter):
        raise FileNotFoundError(f"어댑터가 없습니다: {adapter} (--pass2 base 로 돌리거나 1단계 먼저)")
    vlm = VLM(preset, adapter_dir=adapter if use_adapter else None, download_source=args.download_source)
    if args.mode == "reason" and not vlm.can_think:
        raise SystemExit(f"--mode reason 은 thinking 을 지원하는 모델(Qwen3.5 계열)만 됩니다. 이 run 의 프리셋: {preset}")

    cache, crop_info = {}, {}
    key = lambda r: f"{r['img_path']}||{r['question']}"  # noqa: E731

    if args.mode == "grounding":
        # ── 1패스 : bbox (LoRA 끈 base 모델, 캐시) ──
        cache_path = os.path.join(
            config.run_dir(args.run, "rerun"),
            f"ground_cache_{config.MODEL_PRESETS[preset]['repo'].split('/')[-1]}_{args.ground_max_side}.json")
        cache = load_json(cache_path, {})
        todo = {}
        for rows in rows_by_split.values():
            for _, r in rows.iterrows():
                if key(r) not in cache:
                    todo[key(r)] = r
        print(f"[1패스] 대상 {len(todo)} (캐시 재사용 {sum(len(v) for v in rows_by_split.values()) - len(todo)})")
        t0 = time.time()
        keys, todo_rows = list(todo.keys()), list(todo.values())
        CH = args.ground_batch * 8                      # 이만큼마다 캐시 저장 (끊겨도 이어서)
        bar = progress(total=len(todo_rows), desc="1패스 bbox", unit="문항") if todo_rows else None
        for s in range(0, len(todo_rows), CH):
            res = run_batched(todo_rows[s:s + CH],
                              lambda ch: vlm.ground(ch, args.ground_max_side, args.max_new_tokens),
                              args.ground_batch, bar=bar)
            for k, (raw, boxes) in zip(keys[s:s + CH], res):
                cache[k] = {"raw": raw, "boxes": boxes}
            save_json(cache, cache_path)
        if todo:
            bar.close()
            print(f"[1패스] {(time.time() - t0) / len(todo):.2f}초/문항 → {cache_path}")

    def crop_of(r, W, H):
        k = key(r)
        if k not in crop_info:
            crop_info[k] = crop_region(cache.get(k, {}).get("boxes", []), W, H,
                                       args.crop_pad, args.crop_min_frac, args.crop_max_area)
        return crop_info[k]

    def image_fn_ground(r):
        img = load_image(r["img_path"])
        cb = crop_of(r, *img.size)
        if cb is None:
            return [resize_to(img, args.single_max_side)]
        return [resize_to(img, args.full_max_side), resize_to(img.crop(cb), args.crop_max_side, args.crop_min_side)]

    image_fn_single = lambda r: [resize_to(load_image(r["img_path"]), args.single_max_side)]  # noqa: E731
    # mode=prompt 에서 평균할 프롬프트 목록 (문항의 qtype 에 따라 hint 버전이 붙는다)
    prompt_ids = args.prompt_ids or list(BASE_PROMPTS)

    def variants(r):
        vs = [Prompt(pid, None) for pid in prompt_ids]
        if args.prompt_hint:
            hinted = select_prompt(r["qtype"], "qtype", base=prompt_ids[0])
            if hinted.hint:
                vs.append(hinted)
        return vs

    for split, rows in rows_by_split.items():
        if len(rows) == 0:
            continue
        t0 = time.time()
        row_list = [r for _, r in rows.iterrows()]
        if args.mode == "grounding":
            pick = lambda r: select_prompt(r["qtype"], mode_infer)  # noqa: E731
            P, _ = predict_rows(vlm, row_list, image_fn_ground, pick,
                                batch_size=args.batch_size, desc=f"2패스 {split}")
            has_crop = [crop_info.get(key(r)) is not None for r in row_list]
            extra = {"has_crop": has_crop,
                     "n_boxes": [len(cache.get(key(r), {}).get("boxes", [])) for r in row_list],
                     "ground_raw": [cache.get(key(r), {}).get("raw", "")[:200] for r in row_list]}
            note = f"크롭 적용 {np.mean(has_crop):.0%}"
        elif args.mode == "reason":
            # 2단계와 같은 프롬프트(qtype hint 포함)·같은 해상도 1장으로, 생각을 먼저 쓰고 답한다
            pick = lambda r: select_prompt(r["qtype"], mode_infer)  # noqa: E731

            def fn(chunk):
                items = [(image_fn_single(r), r["question"], [str(r[c]) for c in "abcd"], pick(r)) for r in chunk]
                P, thoughts, closed = vlm.reason(items, args.reason_max_new_tokens, use_adapter=args.reason_adapter)
                return list(zip(P, thoughts, closed))

            res = run_batched(row_list, fn, args.reason_batch, desc=f"추론 {split}")
            P = np.stack([r[0] for r in res])
            thoughts, closed = [r[1] for r in res], [bool(r[2]) for r in res]
            extra = {"think_closed": closed, "think_chars": [len(t) for t in thoughts],
                     "think": [" ".join(t.split())[:400] for t in thoughts]}    # 앞부분만 (전체는 길다)
            note = (f"생각을 스스로 맺음 {np.mean(closed):.0%} (나머지는 {args.reason_max_new_tokens}토큰에서 끊음) | "
                    f"평균 {np.mean([len(t) for t in thoughts]):.0f}자")
        else:
            # 문항마다 변형 수가 다를 수 있다 (hint 가 있는 qtype 만 +1) → 그 문항이 실제로 쓴 개수로 나눈다
            vars_by_row = [variants(r) for r in row_list]
            vars_by_id = {str(r["id"]): vs for r, vs in zip(row_list, vars_by_row)}
            n_var = max(len(v) for v in vars_by_row)
            acc_P = np.zeros((len(row_list), 4))
            counts = np.zeros(len(row_list))
            votes = np.full((n_var, len(row_list)), -1)
            for k in range(n_var):
                idx = [i for i, v in enumerate(vars_by_row) if k < len(v)]
                Pk, _ = predict_rows(vlm, [row_list[i] for i in idx], image_fn_single,
                                     lambda r, k=k: vars_by_id[str(r["id"])][k],
                                     batch_size=args.batch_size, desc=f"프롬프트 {k + 1}/{n_var} {split}")
                acc_P[idx] += Pk
                counts[idx] += 1
                votes[k, idx] = Pk.argmax(1)
            P = acc_P / counts[:, None]
            agree = (votes == P.argmax(1)).sum(0)
            extra = {"n_prompts": counts.astype(int), "prompt_agree": agree,
                     "prompts": [",".join(prompt_str(v) for v in vs) for vs in vars_by_row]}
            note = f"문항당 프롬프트 {int(counts.min())}~{int(counts.max())}개 평균, 모두 일치 {np.mean(agree == counts):.0%}"

        out = make_pred_df(rows, P, **extra)
        out.to_csv(os.path.join(out_dir, f"{split}_rerun.csv"), index=False)
        save_result(out, f"{exp}__{split}_rerun.csv")
        print(f"[{split}] 재추론 {len(rows)}문항 {(time.time() - t0) / len(rows):.2f}초/문항 | {note}")

        if split == "valid":
            base = subsets["valid"].set_index("id").loc[out["id"]]
            gold = out["gold"].values
            b_ok, r_ok = base["pred"].values == gold, out["pred"].values == gold
            print(f"  선별 문항 정확도  기본 {b_ok.mean():.4f} → 재추론 {r_ok.mean():.4f} "
                  f"(고쳐짐 {int((~b_ok & r_ok).sum())} / 망가짐 {int((b_ok & ~r_ok).sum())})")
            if args.mode == "grounding":
                hc = np.array(extra["has_crop"])
                if hc.any() and (~hc).any():
                    print(f"  크롭 O {hc.sum()}문항: {b_ok[hc].mean():.3f} → {r_ok[hc].mean():.3f} | "
                          f"크롭 X {(~hc).sum()}문항: {b_ok[~hc].mean():.3f} → {r_ok[~hc].mean():.3f}")
            if args.mode == "reason":
                cl = np.array(extra["think_closed"])
                if cl.any() and (~cl).any():
                    print(f"  생각을 맺음 {cl.sum()}문항: {b_ok[cl].mean():.3f} → {r_ok[cl].mean():.3f} | "
                          f"끊김 {(~cl).sum()}문항: {b_ok[~cl].mean():.3f} → {r_ok[~cl].mean():.3f}")
            append_log(config.RUN_ROOT, {"stage": "s4_rerun", "tag": tag, "exp_tag": exp, "run": args.run,
                                         "mode": args.mode,
                                         "pass2": args.pass2 if args.mode == "grounding" else "",
                                         "n_valid_subset": len(rows),
                                         "acc_base_subset": round(float(b_ok.mean()), 4),
                                         "acc_rerun_subset": round(float(r_ok.mean()), 4),
                                         "fixed": int((~b_ok & r_ok).sum()), "broken": int((b_ok & ~r_ok).sum())})
            if args.mode == "grounding":
                # 크롭 확인용 이미지 (빨강 = 모델 bbox, 파랑 = 실제 크롭)
                prev_dir = config.run_dir(args.run, "rerun", tag, "crops")
                for i, r in enumerate(row_list[:args.save_crops]):
                    save_preview(r, cache.get(key(r), {}).get("boxes", []), crop_info.get(key(r)),
                                 os.path.join(prev_dir, f"{i:02d}_{r['id']}".replace("/", "_") + ".jpg"))
                print(f"  크롭 확인용 이미지 → {prev_dir}  (좌표가 엉뚱하면 config 의 coord 설정 확인)")

    info = {"run": args.run, "exp_tag": exp, "mode": args.mode, "pass2": args.pass2, "preset": preset, **vars(args)}
    save_json(info, os.path.join(out_dir, "rerun.json"))
    save_result(info, f"{exp}__rerun.json", quiet=True)


if __name__ == "__main__":
    main()
