"""
1단계 : multi-prompt choice-CE LoRA 학습 (모델 하나 · 어댑터 하나, 중단돼도 이어서 학습)

추론과 똑같은 프롬프트를 넣고, 마지막 위치의 a/b/c/d 네 로짓만 뽑아 4지 분류 cross-entropy 로 학습한다.
  - sample 마다 base prompt 를 균등 sampling (model_utils.BASE_PROMPTS) → 표현 변화에 강해진다
  - --hint-prob 확률로 그 sample 의 qtype hint 를 덧붙인다 (나머지는 generic)
  - 데이터를 프롬프트 수만큼 복제하지 않는다. 같은 sample 이 epoch 마다 다른 표현으로 보일 뿐이다
  - 선지 순서도 매번 섞어서 모델이 '위치'가 아니라 '내용'을 보게 한다
프롬프트 sampling 과 선지 셔플은 체크포인트에 저장되는 rng(CPU·CUDA)를 쓰므로 이어하기해도 같은 순서로 진행된다
(GPU 연산 자체의 비결정성 때문에 수치는 완전히 같지 않을 수 있다). 체크포인트의 학습 데이터·valid 지문이 지금과 다르면
(예전 코드로 만든 체크포인트 포함) 이어하지 않고 멈춘다 → --fresh.

학습 데이터는 config.DATA (기본 : 증강 bigdata/ = 원본 + ±30° 회전본 + auto_dev, --train-source / --train-kinds 로 바꿈).
valid 는 같은 데이터에서 이미지 그룹 단위로 떼어낸 원본 문항이다 (그 사진의 회전본·중복 사진은 학습에서도 빠진다
→ data_utils.load_data). 원본(orig)으로 학습해도 valid 문항은 같다 → 증강 효과를 같은 valid 로 비교할 수 있다.

업데이트 기준 best : --eval-every 업데이트마다(그리고 epoch 끝·학습 끝에) 같은 valid 문항으로 평가하고,
가장 높은 valid 정확도(동률이면 valid NLL 이 낮은 쪽)의 어댑터를 adapter/ 로 둔다 (adapter/best.json 에 업데이트·점수).
평가는 추론과 같은 정책(qtype 기반 단일 프롬프트)으로 한다. 2·4단계는 이 adapter/ 를 그대로 쓴다.
PC 로 받기 : ./colab/colab_pipeline.sh fetch-best <run>  → results/<실험이름>/adapter_best/

LoRA 랭크 비교는 --run 이름을 함께 바꿔서 산출물이 섞이지 않게 한다.
  python s1_train_choice_ce.py --run r16
  python s1_train_choice_ce.py --run r32 --lora-r 32
  python s1_train_choice_ce.py --run gen --hint-prob 0          # generic prompt 만 (hint 효과 비교)
  python s1_train_choice_ce.py --run orig --train-source orig   # 증강 없이 원본 train 만 (같은 valid 로 증강 효과 비교)
  (seed·epoch·lr·LoRA·프롬프트 비율·평가 간격·데이터 기본값은 config.py 의 SEED / TRAIN / LORA / PROMPT / DATA)

산출물 : runs/<run>/adapter/ (best 업데이트 + best.json), adapter_e1/ … (epoch 끝), ckpt/ (이어하기용),
         run.json, train_log.csv (평가할 때마다 갱신), valid_ids.txt
"""
import config  # noqa: F401  (반드시 가장 먼저)

import os
import math
import time
import shutil
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from data_utils import (LETTERS, TRAIN_META, load_datasets, describe_split, take, id_hash, train_meta, append_log,
                        save_json, save_result, load_json, accuracy, mean_nll, make_pred_df, progress, result_relpath)
from model_utils import (BASE_PROMPTS, VLM, prompt_str, sample_prompt, select_prompt, seed_everything,
                         load_image, resize_to, predict_rows)


def parse_args():
    T, P = config.TRAIN, config.PROMPT                 # 기본값은 config.py 에서 관리
    p = argparse.ArgumentParser(description="1단계: multi-prompt choice-CE LoRA 학습")
    p.add_argument("--run", default=config.RUN_NAME, help="산출물 폴더 이름 (runs/<run>)")
    p.add_argument("--preset", default=config.DEFAULT_PRESET, choices=list(config.MODEL_PRESETS))
    p.add_argument("--epochs", type=int, default=T["epochs"])
    p.add_argument("--max-updates", type=int, default=None, help="옵티마이저 업데이트 수 상한 (빠른 점검용)")
    p.add_argument("--lr", type=float, default=T["lr"])
    p.add_argument("--grad-accum", type=int, default=T["grad_accum"],
                   help="유효 배치 = 업데이트 1회가 보는 문항 수 (학습 결과를 좌우)")
    p.add_argument("--micro-batch", type=int, default=T["micro_batch"],
                   help="한 번에 forward/backward 하는 문항 수 (≤ grad_accum). 속도만 좌우하고 결과는 같다")
    p.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=T["grad_checkpointing"],
                   help="gradient checkpointing (끄면 빠르고 메모리를 더 씀)")
    # LoRA (랭크별 비교용)
    p.add_argument("--lora-r", type=int, default=None, help=f"기본 {config.LORA['r']}")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help=f"기본 None = {config.LORA['alpha_ratio']}×r (랭크를 바꿔도 스케일 유지)")
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--lora-targets", default=None,
                   help=f"LoRA 를 붙일 모듈: {list(config.LORA_TARGETS)} 또는 정규식 (기본 {config.LORA['targets']})")
    # 프롬프트 augmentation
    p.add_argument("--base-prompts", nargs="*", default=P["base_ids"], choices=list(BASE_PROMPTS),
                   help="학습에 쓸 base prompt 목록 (기본 전체 균등 sampling)")
    p.add_argument("--hint-prob", type=float, default=P["hint_prob"], help="qtype hint 를 붙일 확률 (0 이면 generic 만)")
    p.add_argument("--eval-prompt-mode", default=P["infer_mode"], choices=["qtype", "generic", "fixed"],
                   help="평가 프롬프트 정책 (2단계 추론과 같게 둘 것)")
    p.add_argument("--eval-prompt", default=None, help="--eval-prompt-mode fixed 일 때 쓸 프롬프트 (예: p1, p1+부정형)")
    p.add_argument("--label-smoothing", type=float, default=T["label_smoothing"])
    p.add_argument("--no-shuffle-choices", action="store_true")
    p.add_argument("--train-source", default=config.DATA["train_source"], choices=["aug", "orig"],
                   help="학습 데이터 : aug = 증강 bigdata/, orig = 원본 train.csv (valid 문항은 어느 쪽이든 같다)")
    p.add_argument("--train-kinds", nargs="+", default=config.DATA["train_kinds"], choices=["orig", "rot", "auto_dev", "other"],
                   help="학습에 쓸 종류 (예: --train-kinds orig rot = auto_dev 빼기)")
    p.add_argument("--n-train", type=int, default=None, help="None 이면 train 전체")
    p.add_argument("--eval-every", type=int, default=T["eval_every"],
                   help="이 업데이트마다 valid 평가 → 가장 높은 업데이트의 어댑터를 adapter/ 로 (epoch 끝에도 평가)")
    p.add_argument("--eval-n", type=int, default=T["eval_n"],
                   help="평가 문항 수 (기본 valid 전체). 모든 평가가 같은 문항을 쓴다")
    p.add_argument("--valid", action=argparse.BooleanOptionalAction, default=config.USE_VALID,
                   help="valid 를 떼어 평가 (--no-valid 면 valid 없이 학습 데이터 전부로 학습하고 마지막 어댑터를 쓴다)")
    p.add_argument("--eval-at-start", action="store_true", help="학습 전 zero-shot 정확도도 기록 (업데이트 0 도 best 후보)")
    p.add_argument("--ckpt-every", type=int, default=T["ckpt_every"], help="이 업데이트마다 이어하기용 체크포인트 저장")
    p.add_argument("--fresh", action="store_true", help="체크포인트를 무시하고 처음부터")
    p.add_argument("--train-max-side", type=int, default=config.TRAIN_MAX_SIDE)
    p.add_argument("--infer-max-side", type=int, default=config.INFER_MAX_SIDE)
    p.add_argument("--eval-batch", type=int, default=T["eval_batch"])
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--download-source", default="hf", choices=["hf", "modelscope"])
    return p.parse_args()


def replace_dir(path, fill):
    """fill(임시 폴더) 로 다 쓴 뒤 path 와 바꿔 끼운다. 이전 것은 _old 로 옮겼다가 지우므로 어느 순간에 끊겨도
    path · path_tmp · path_old 중 완성본이 하나는 남는다 (다음 실행 때 recover_dir 가 되살림)"""
    tmp, old = path + "_tmp", path + "_old"
    for d in (tmp, old):
        if os.path.isdir(d):
            shutil.rmtree(d)
    fill(tmp)
    if os.path.isdir(path):
        os.replace(path, old)
    os.replace(tmp, path)
    shutil.rmtree(old, ignore_errors=True)


def recover_dir(path, done_file):
    """path 에 완성 표시 파일이 없으면 path_tmp(새것) → path_old(이전 것) 순으로 완성본을 찾아 되살린다"""
    if os.path.isfile(os.path.join(path, done_file)):
        return
    for cand in (path + "_tmp", path + "_old"):
        if os.path.isfile(os.path.join(cand, done_file)):
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.replace(cand, path)
            print(f"[복구] 저장 도중 끊긴 {os.path.basename(path)}/ 를 {os.path.basename(cand)}/ 에서 되살림 → {path}")
            return


def main():
    args = parse_args()
    seed_everything(args.seed)
    out_dir = config.run_dir(args.run)
    adapter_dir = os.path.join(out_dir, "adapter")
    ckpt_dir = os.path.join(out_dir, "ckpt")
    # 폴더 교체(replace_dir) 도중 끊겼으면 완성된 쪽(_tmp 새것 → _old 이전 것)을 되살린다. 완성 표시 = 마지막에 쓰는 파일
    recover_dir(ckpt_dir, "state.pt")
    recover_dir(adapter_dir, "best.json")
    if args.fresh:
        for d in (ckpt_dir, ckpt_dir + "_tmp", ckpt_dir + "_old"):
            if os.path.isdir(d):
                shutil.rmtree(d)
    lora = config.lora_spec(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_targets)

    train_df, valid_df, _, data_info = load_datasets(n_train=args.n_train, n_test=0, use_valid=args.valid,
                                                     train_source=args.train_source, train_kinds=args.train_kinds)
    data_info["use_valid"] = args.valid
    eval_df = take(valid_df, args.eval_n, config.SEED)          # 모든 평가가 같은 문항 → 업데이트끼리 비교 가능
    # 이어하기 검증용 지문 : 학습 데이터·valid·평가 문항이 체크포인트와 같아야 best 점수를 이어서 비교할 수 있다
    data_sig = {"train_source": data_info["train_source"], "train_kinds": data_info["train_kinds"],
                "valid_source": data_info["valid_source"], "train_hash": data_info["train_hash"],
                "valid_hash": data_info["valid_hash"], "eval_hash": id_hash(eval_df) if len(eval_df) else None}
    # 결과 파일 이름에 붙일 실험 이름 (2~최종 단계도 run.json 에서 이 값을 읽어 같은 이름을 쓴다)
    kinds_note = "" if sorted(args.train_kinds) == sorted(config.DATA["train_kinds"]) else "-" + "+".join(args.train_kinds)
    tag = config.exp_tag(args.run, args.preset, lora_r=lora["r"], epochs=args.epochs, lr=args.lr,
                         hint_prob=args.hint_prob, seed=args.seed, data=data_info["train_source"] + kinds_note)
    base_ids = args.base_prompts or list(BASE_PROMPTS)
    print(f"[데이터] {describe_split(data_info)}")
    if len(eval_df):
        with open(os.path.join(out_dir, "valid_ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(valid_df["id"].astype(str)))
        print(f"[평가] {args.eval_every}업데이트마다 + epoch 끝 · valid {len(eval_df)}문항 "
              f"→ 가장 높은 업데이트의 어댑터를 {adapter_dir}")
    else:
        print("[평가] 없음 (--no-valid 또는 config.USE_VALID=False) → 마지막 어댑터를 사용")
    print(f"[프롬프트] base {base_ids} 균등 sampling | qtype hint {args.hint_prob:.0%} "
          f"| 평가 정책 {args.eval_prompt_mode}")
    print(f"[실험] {tag} | 결과 사본 → " + " , ".join(config.result_dirs()))

    vlm = VLM(args.preset, train_lora=True, lora=lora, grad_ckpt=args.grad_ckpt,
              download_source=args.download_source)
    model = vlm.model

    eval_prompt = lambda r: select_prompt(r["qtype"], args.eval_prompt_mode, fixed=args.eval_prompt)  # noqa: E731

    def evaluate(label, df):                           # label 은 진행 막대 표시용 (실험 이름 tag 와 다름)
        """valid 평가 → {"acc", "nll", "n"} (평가 문항이 없으면 None)"""
        if len(df) == 0:
            return None
        model.eval()                                   # 평가 중에는 dropout 끔 (호출한 쪽에서 다시 train())
        rows = [r for _, r in df.iterrows()]
        P, _ = predict_rows(vlm, rows, lambda r: [resize_to(load_image(r["img_path"]), args.infer_max_side)],
                            eval_prompt, batch_size=args.eval_batch, desc=f"평가 {label}", leave=False)
        res = {"acc": accuracy(make_pred_df(df, P)), "nll": mean_nll(P, df["answer"]), "n": len(df)}
        torch.cuda.empty_cache()
        return res

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_updates = args.epochs * math.ceil(len(train_df) / args.grad_accum)
    if args.max_updates:
        total_updates = min(total_updates, args.max_updates)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(total_updates * 0.03)), total_updates)
    rng = np.random.default_rng(args.seed)             # 데이터 순서 · 선지 셔플 · 프롬프트 sampling 공용

    # ── 이어하기 상태 ──
    # best_* 는 업데이트 단위 평가 중 최고 (정확도, 동률이면 NLL). last_eval = 마지막으로 평가한 업데이트
    # epoch_accs = {epoch: epoch 끝 정확도}
    # run_id : 이 학습(처음 시작 ~ 이어하기)의 고유 id. adapter/best.json 과 체크포인트가 같은 학습의 것인지 가린다
    st = {"epoch": 1, "pos": 0, "order": None, "update": 0, "best_acc": -1.0, "best_nll": None, "best_update": None,
          "best_epoch": None, "last_eval": None, "last_acc": None, "epoch_accs": {}, "log": [], "elapsed": 0.0,
          "prompt_use": {}, "run_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"}
    t0 = time.time()

    def save_ckpt(epoch, pos, order):
        def fill(d):
            model.save_pretrained(d)
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,  # LoRA dropout
                        "epoch": epoch, "pos": pos, "order": order, "update": st["update"],
                        "best_acc": st["best_acc"], "best_nll": st["best_nll"], "best_update": st["best_update"],
                        "best_epoch": st["best_epoch"], "last_eval": st["last_eval"], "last_acc": st["last_acc"],
                        "epoch_accs": st["epoch_accs"], "log": st["log"], "prompt_use": st["prompt_use"],
                        "run_id": st["run_id"], "elapsed": st["elapsed"] + (time.time() - t0), "lora": lora,
                        "n_train": len(train_df), "total_updates": total_updates,
                        "data_sig": data_sig}, os.path.join(d, "state.pt"))
        replace_dir(ckpt_dir, fill)

    def best_info():
        """adapter/best.json — 이 어댑터가 몇 번째 업데이트의 것이고 valid 점수가 얼마인지"""
        return {"run": args.run, "exp_tag": tag, "run_id": st["run_id"], "update": st["best_update"], "epoch": st["best_epoch"],
                "total_updates": total_updates,
                "valid_acc": None if math.isnan(st["best_acc"]) else st["best_acc"],   # 평가 없음 → null
                "valid_nll": None if st["best_nll"] is None else st["best_nll"], "eval_n": len(eval_df),
                "valid_hash": data_info["valid_hash"], "eval_hash": data_sig["eval_hash"],
                "train_source": data_info["train_source"], "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save_adapter(d, best=True):
        """어댑터 + train_meta.json(학습 데이터 표식, 2·4단계가 봄) (+ best.json : 완성 표시라서 마지막에 쓴다)"""
        model.save_pretrained(d)
        save_json(train_meta(data_info, st["run_id"]), os.path.join(d, TRAIN_META))
        if best:
            save_json(best_info(), os.path.join(d, "best.json"))

    # 채택 어댑터 저장 : runs/<run>/adapter/ (드라이브, 2·4단계가 읽음) + 로컬 결과 폴더 사본
    # (Colab 은 VM 디스크 /content/results/<실험이름>/adapter_best/ → fetch-best 가 PC 의 results/ 로 받아 간다)
    local_best = os.path.join(config.RESULT_ROOT, result_relpath(f"{tag}__adapter_best"))

    def save_best():
        replace_dir(adapter_dir, save_adapter)
        try:
            os.makedirs(os.path.dirname(local_best), exist_ok=True)
            replace_dir(local_best, save_adapter)
        except OSError as e:                            # 로컬 사본 실패로 학습을 멈추지 않는다 (드라이브 사본은 있음)
            tqdm.write(f"  ⚠ 로컬 사본 저장 실패: {local_best} ({e})")

    def write_meta(**extra):
        """runs/<run>/run.json — 2·4단계가 프리셋·프롬프트 정책·실험 이름을 여기서 읽는다"""
        meta = {"run": args.run, "exp_tag": tag, "preset": args.preset, "lora": lora,
                "base_prompts": base_ids, "hint_prob": args.hint_prob,
                "infer_mode": args.eval_prompt_mode, "infer_prompt": args.eval_prompt,
                "best_valid_acc": st["best_acc"], "best_valid_nll": st["best_nll"], "best_update": st["best_update"],
                "best_epoch": st["best_epoch"], "update": st["update"], "total_updates": total_updates,
                "eval_every": args.eval_every, "eval_n": len(eval_df), "epoch_accs": st["epoch_accs"],
                "data": data_info, "prompt_use": st["prompt_use"], "args": vars(args), **extra}
        save_json(meta, os.path.join(out_dir, "run.json"))
        save_result(meta, f"{tag}__run.json", quiet=True)          # 결과 폴더(로컬 + 드라이브) 사본

    def save_log():
        log = pd.DataFrame(st["log"])
        log.to_csv(os.path.join(out_dir, "train_log.csv"), index=False)     # 도는 중에도 colab_pipeline.sh evals 로 확인
        save_result(log, f"{tag}__train_log.csv", quiet=True)

    def eval_and_track(epoch, label, loss=None):
        """valid 평가 1회 → 기록하고, 지금까지 최고면 adapter/ 를 이 업데이트의 어댑터로 바꾼다. 반환 정확도"""
        r = evaluate(label, eval_df)
        if r is None:
            return None
        u = st["update"]
        better = (st["best_update"] is None or math.isnan(st["best_acc"]) or r["acc"] > st["best_acc"]
                  or (r["acc"] == st["best_acc"] and r["nll"] < (math.inf if st["best_nll"] is None else st["best_nll"])))
        st["last_eval"], st["last_acc"] = u, r["acc"]
        st["log"].append({"update": u, "epoch": epoch, "loss": loss, "valid_acc": round(r["acc"], 4),
                          "valid_nll": round(r["nll"], 4), "eval_n": r["n"],
                          "best": better or u == st["best_update"]})     # 이어하기로 받아들인 best 를 다시 평가한 경우도 표시
        if better:
            st.update(best_acc=r["acc"], best_nll=r["nll"], best_update=u, best_epoch=epoch)
            save_best()
        tqdm.write(f"[평가] upd {u}/{total_updates} (epoch {epoch}) valid acc {r['acc']:.4f} nll {r['nll']:.4f} "
                   f"({r['n']}문항) " + ("▲ best → adapter/" if better
                                        else f"| best upd {st['best_update']} {st['best_acc']:.4f}"))
        save_log()
        write_meta()
        model.train()
        return r["acc"]

    state_path = os.path.join(ckpt_dir, "state.pt")
    if os.path.isfile(state_path):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved.get("n_train") != len(train_df) or saved.get("total_updates") != total_updates:
            raise RuntimeError("체크포인트와 현재 설정(n_train/epochs/grad_accum)이 다릅니다. 같은 설정으로 돌리거나 --fresh")
        if saved.get("lora", {}).get("r") not in (None, lora["r"]):
            raise RuntimeError(f"체크포인트의 LoRA 랭크는 {saved['lora']['r']} 입니다. --lora-r 를 맞추거나 "
                               f"다른 --run 이름으로 새로 시작하세요")
        if saved.get("data_sig") != data_sig:            # 예전 코드의 체크포인트(지문 없음)도 여기서 걸린다
            diff = [k for k in data_sig if (saved.get("data_sig") or {}).get(k) != data_sig[k]]
            raise RuntimeError(f"체크포인트({ckpt_dir})를 만들 때와 학습 데이터·valid 가 다릅니다 (다른 항목: {diff}). "
                               f"예전 실행의 체크포인트이거나 config.DATA·--train-*·--n-train·--eval-n 을 바꾼 것 → "
                               f"처음부터 하려면 --fresh, 이어서 하려면 설정을 되돌리세요")
        set_peft_model_state_dict(model, load_file(os.path.join(ckpt_dir, "adapter_model.safetensors")))
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        if saved.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        st.update({k: saved[k] for k in st if k in saved})
        # 체크포인트 뒤에 best 가 더 저장됐다가 끊긴 경우(best 저장 → 체크포인트 저장 사이) adapter/ 쪽이 더 새롭다
        disk = load_json(os.path.join(adapter_dir, "best.json"), {}) or {}
        if (disk.get("run_id") == st["run_id"] and disk.get("eval_hash") == data_sig["eval_hash"]
                and disk.get("valid_acc") is not None and disk.get("update") is not None
                and (st["best_update"] is None or (disk["valid_acc"], -(disk.get("valid_nll") or 0))
                     > (st["best_acc"], -(st["best_nll"] or 0)))):
            st.update(best_acc=disk["valid_acc"], best_nll=disk.get("valid_nll"), best_update=disk["update"],
                      best_epoch=disk.get("epoch"))
            print(f"[이어하기] adapter/ 에 체크포인트 이후의 best(upd {disk['update']}, acc {disk['valid_acc']:.4f})가 있어 "
                  f"그 점수를 기준으로 이어갑니다")
        print(f"[이어하기] epoch {st['epoch']} / {st['pos']}번째 샘플 / 업데이트 {st['update']}/{total_updates} 부터 "
              + (f"(best upd {st['best_update']} {st['best_acc']:.4f})" if st["best_update"] is not None
                 else "(아직 평가 전)"))
    elif args.eval_at_start and len(eval_df):
        acc0 = eval_and_track(0, "zero-shot")          # 업데이트 0 (LoRA 가 아직 0) 도 best 후보
        print(f"[zero-shot] valid acc {acc0:.4f} ({len(eval_df)}문항)")

    if args.micro_batch > args.grad_accum:
        raise SystemExit(f"--micro-batch({args.micro_batch}) 는 --grad-accum({args.grad_accum}) 보다 클 수 없습니다")
    print(f"[학습] {args.epochs} epoch, 업데이트 {total_updates}회 (유효 배치 {args.grad_accum}, "
          f"마이크로배치 {args.micro_batch}), 체크포인트 {args.ckpt_every}업데이트마다")
    upd_at_start = st["update"]
    running = 0.0
    used = Counter(st["prompt_use"])
    torch.cuda.reset_peak_memory_stats()
    model.train()
    done = st["update"] >= total_updates
    for epoch in range(st["epoch"], args.epochs + 1):
        # 마지막 업데이트 직후(epoch 끝 평가·저장 전)에 끊겼다면 학습은 건너뛰고 epoch 끝 처리만 한다
        resuming = epoch == st["epoch"] and st["order"] is not None
        if done and not resuming:
            break
        order = np.asarray(st["order"]) if resuming else rng.permutation(len(train_df))
        start = st["pos"] if resuming else 0
        # epoch 진행 막대 (문항 단위, 이어하기면 멈춘 위치부터). 남은 시간은 이 epoch 기준
        bar = progress(total=len(order), initial=start, desc=f"epoch {epoch}/{args.epochs}", unit="문항")

        def backward_batch(batch):
            """마이크로배치 하나를 forward/backward. OOM 이면 반으로 쪼개 재시도. 반환: loss 합(문항 수 × 평균)"""
            try:
                loss = vlm.choice_loss(batch, args.label_smoothing)
                (loss * len(batch) / args.grad_accum).backward()   # 유효 배치(grad_accum) 기준 평균이 되도록
                return loss.item() * len(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch) == 1:
                    raise
                bar.write(f"  ⚠ OOM → 마이크로배치 {len(batch)}개를 반으로 나눠 재시도 "
                          f"(자주 보이면 config.TRAIN['micro_batch'] 를 줄일 것)")
                half = len(batch) // 2
                return backward_batch(batch[:half]) + backward_batch(batch[half:])

        i, since = start, 0                                 # i = 이번 epoch 에서 처리한 문항 수, since = 이번 업데이트 누적
        while i < len(order) and not done:
            mb = min(args.micro_batch, args.grad_accum - since, len(order) - i)
            batch = []
            for j in order[i:i + mb]:
                r = train_df.iloc[j]
                img = resize_to(load_image(r["img_path"]), args.train_max_side)
                choices = [str(r[c]) for c in LETTERS]
                gold = LETTERS.index(r["answer"])
                if not args.no_shuffle_choices:
                    perm = rng.permutation(4)                   # perm[새 위치] = 원래 인덱스
                    choices = [choices[k] for k in perm]
                    gold = int(np.where(perm == gold)[0][0])
                prompt = sample_prompt(rng, r["qtype"], hint_prob=args.hint_prob, base_ids=base_ids)
                used[prompt_str(prompt)] += 1
                batch.append(([img], r["question"], choices, gold, prompt))

            running += backward_batch(batch)
            i += len(batch); since += len(batch)
            bar.update(len(batch))

            if since >= args.grad_accum or i == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                st["update"] += 1
                avg, running, since = running / max(1, since), 0.0, 0
                u = st["update"]
                bar.set_postfix(upd=f"{u}/{total_updates}", loss=f"{avg:.4f}", refresh=False)
                if u % 20 == 0:                         # 로그에 남는 기록 (전체 남은 시간)
                    el = time.time() - t0
                    eta = el / max(1, u - upd_at_start) * (total_updates - u)
                    bar.write(f"  e{epoch} upd {u}/{total_updates} loss {avg:.4f} lr {scheduler.get_last_lr()[0]:.1e} "
                              f"| 전체 남은 약 {eta / 60:.0f}분 | peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
                if len(eval_df) and u % args.eval_every == 0:
                    eval_and_track(epoch, f"upd{u}", loss=round(avg, 4))     # best 후보 (업데이트 기준)
                if u % args.ckpt_every == 0:
                    st["prompt_use"] = dict(used)
                    save_ckpt(epoch, i, order)
                if u >= total_updates:
                    done = True
        bar.close()
        st["prompt_use"] = dict(used)
        if len(eval_df):
            # epoch 끝(= 학습 끝 포함)도 best 후보. 방금 그 업데이트에서 평가했다면 다시 하지 않는다
            acc = st["last_acc"] if st["last_eval"] == st["update"] else eval_and_track(epoch, f"epoch{epoch}")
            st["epoch_accs"][epoch] = round(acc, 4)
            print(f"[epoch {epoch} 끝] upd {st['update']} valid acc {acc:.4f} | epoch 별 {st['epoch_accs']} "
                  f"| best upd {st['best_update']} ({st['best_acc']:.4f})", flush=True)
            write_meta()
        else:                                           # 평가 없음 → 마지막 어댑터가 곧 채택본
            st["best_epoch"], st["best_update"], st["best_acc"] = epoch, st["update"], float("nan")
            save_best()
            print(f"[epoch {epoch}] 평가 없이 어댑터 저장 → {adapter_dir} (+ {local_best})", flush=True)
            write_meta()
        save_adapter(os.path.join(out_dir, f"adapter_e{epoch}"), best=False)   # epoch 별 어댑터 (다른 epoch 로도 추론 가능)
        model.train()
        st["order"] = None
        save_ckpt(epoch + 1, 0, None)                   # 다음 epoch 처음부터 이어가도록

    pd.DataFrame(st["log"]).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)
    minutes = (st["elapsed"] + time.time() - t0) / 60
    hint_frac = sum(v for k, v in used.items() if "+" in k) / max(1, sum(used.values()))
    write_meta()
    save_result(pd.DataFrame(st["log"]), f"{tag}__train_log.csv")
    append_log(config.RUN_ROOT, {"stage": "s1_train", "run": args.run, "exp_tag": tag,
                                 "preset": args.preset, "seed": args.seed,
                                 "epochs": args.epochs, "lr": args.lr, "grad_accum": args.grad_accum,
                                 "micro_batch": args.micro_batch, "grad_ckpt": args.grad_ckpt,
                                 "lora_r": lora["r"], "lora_alpha": lora["alpha"], "lora_targets": lora["targets_name"],
                                 "base_prompts": ",".join(base_ids), "hint_prob": args.hint_prob,
                                 "hint_used_frac": round(hint_frac, 3),
                                 "train_source": data_info["train_source"], "valid_source": data_info["valid_source"],
                                 "n_train": len(train_df), "train_kinds": data_info["kinds_train"],
                                 "n_valid": len(eval_df), "valid_hash": data_info["valid_hash"],
                                 "eval_every": args.eval_every, "total_updates": total_updates,
                                 "best_update": st["best_update"], "best_epoch": st["best_epoch"],
                                 "best_valid_acc": round(st["best_acc"], 4),
                                 "best_valid_nll": None if st["best_nll"] is None else round(st["best_nll"], 4),
                                 "valid_acc_by_epoch": st["epoch_accs"], "minutes": round(minutes, 1)})
    nll_txt = "-" if st["best_nll"] is None else f"{st['best_nll']:.4f}"
    acc_txt = (f"best 업데이트 {st['best_update']}/{total_updates} (epoch {st['best_epoch']}) "
               f"valid acc {st['best_acc']:.4f} nll {nll_txt} ({len(eval_df)}문항) | epoch 끝 {st['epoch_accs']}"
               if len(eval_df) else f"valid 평가 없음 → 마지막 업데이트 {st['best_update']}")
    print(f"[완료] {acc_txt} | hint 적용 {hint_frac:.0%} | 어댑터 {adapter_dir} (로컬 사본 {local_best})")
    print(f"  PC 로 받기: ./colab/colab_pipeline.sh fetch-best {args.run}   → results/{tag}/adapter_best/")


if __name__ == "__main__":
    main()
