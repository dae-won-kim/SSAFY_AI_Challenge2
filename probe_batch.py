"""
배치 크기 상한 측정 (GPU). 결과를 config.py 의 PREDICT / RERUN / TRAIN 배치에 반영하기 위한 도구

실제 코드 경로(VLM.score, VLM.ground, 1단계 학습 스텝)를 그대로 쓰고, 입력은 최악 조건으로 만든다.
  이미지 : test 에서 픽셀 수가 가장 큰 이미지들 (리사이즈 후 시각 토큰이 가장 많음)
  텍스트 : 질문+선지가 가장 긴 문항 + 가장 긴 base prompt + qtype hint (프롬프트 중 최악)
배치를 키워 가며 peak 메모리와 처리량을 재고, reserved 메모리가 GPU 전체의 --limit 이하인 최대 배치를 고른다.
(실제 추론에서 OOM 이 나면 run_batched 가 그 배치를 1개씩 다시 돌려 매우 느려지므로 여유를 둔다)

측정 항목 → config 키
  train  : 학습 forward+backward (1단계)                 → TRAIN["micro_batch"]
  single : 이미지 1장 스코어링 (2단계, 1단계 평가)      → PREDICT["batch_size"], TRAIN["eval_batch"]
  two    : [전체 + 확대] 2장 스코어링 (4단계 grounding)  → RERUN["batch_size"]
  ground : bbox 생성 (4단계 1패스)                      → RERUN["ground_batch"]
배치 = 한 번에 forward 하는 문항 수 (추론은 sample 당 1 forward).
GPU 를 다른 작업이 쓰는 중이면(사용량이 전체의 10% 초과) 시작하지 않는다. 학습이 끝난 뒤, 다른 작업이 없을 때 실행할 것.

산출물 : runs/probe_batch.json (fetch 로 로컬 results/ 에 받아짐)

예)
  ./colab/colab_pipeline.sh run probe python probe_batch.py              # 상한 찾기 (8 → 256)
  ./colab/colab_pipeline.sh run probe python probe_batch.py --sizes 32   # 32 만 확인 (ground 는 건너뜀)
  ./colab/colab_pipeline.sh logs probe 60
"""
import config  # noqa: F401  (반드시 가장 먼저)

import os
import sys
import time
import argparse
import subprocess

import numpy as np
import torch
from PIL import Image

from data_utils import LETTERS, load_split, save_json
from model_utils import BASE_PROMPTS, QTYPE_HINTS, VLM, Prompt, load_image, prompt_text, resize_to

TRAIN_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 48]
SCORE_SIZES = [8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256]
GROUND_SIZES = [4, 8, 16, 24, 32, 48, 64, 96, 128]


def parse_args():
    p = argparse.ArgumentParser(description="배치 크기 상한 측정")
    p.add_argument("--preset", default=config.DEFAULT_PRESET, choices=list(config.MODEL_PRESETS))
    p.add_argument("--limit", type=float, default=0.90, help="reserved 메모리 상한 (GPU 전체 대비 비율)")
    p.add_argument("--n-images", type=int, default=16, help="최악 조건 후보 이미지 수")
    p.add_argument("--repeat", type=int, default=2,
                   help="크기마다 워밍업 1회 뒤 이만큼 더 재서 중앙값 (Triton 컴파일·자동 튜닝 시간 제외)")
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="이 배치만 확인 (예: --sizes 32). single·two 에 적용하고 ground 는 건너뜀. 없으면 상한을 찾는다")
    p.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=config.TRAIN["grad_checkpointing"],
                   help="학습 측정에 gradient checkpointing 사용 (끄면 빠르고 메모리를 더 씀)")
    p.add_argument("--download-source", default="hf", choices=["hf", "modelscope"])
    return p.parse_args()


def worst_rows(n):
    """픽셀 수가 가장 큰 이미지 n 장에, 가장 긴 질문·선지를 붙인 행 목록"""
    df = load_split(config.DATA_ROOT, "test")
    px = df["img_path"].map(lambda p: int(np.prod(Image.open(p).size)))      # 헤더만 읽음
    text_len = df["question"].astype(str).str.len() + sum(df[c].astype(str).str.len() for c in LETTERS)
    longest = df.loc[text_len.idxmax()]
    rows = []
    for _, r in df.loc[px.sort_values(ascending=False).index[:n]].iterrows():
        r = r.copy()
        for c in ["question"] + LETTERS:
            r[c] = longest[c]
        rows.append(r)
    print(f"[입력] 최대 이미지 {px.max() / 1e6:.1f}MP ~ {px.sort_values().iloc[-n] / 1e6:.1f}MP {n}장, "
          f"질문+선지 {text_len.max()}자")
    return rows


def gpu_memory_used():
    """다른 프로세스를 포함한 GPU 0 의 (사용, 전체) 바이트. 이 프로세스가 CUDA 를 쓰기 전에 호출할 것
    nvidia-smi 기준 : WSL 에서는 torch.cuda.mem_get_info 가 다른 프로세스의 사용량을 보지 못한다"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.splitlines()[0]
        used, total = (float(x) * 2 ** 20 for x in out.split(","))          # MiB → 바이트
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        free, total = torch.cuda.mem_get_info()
        used = total - free
    return used, total


def measure(name, run, sizes, limit_bytes, repeat):
    """sizes 를 차례로 시도해 한도 안에서 성공한 최대 크기를 반환
    크기마다 첫 회는 워밍업(그 크기의 컴파일·자동 튜닝 포함), 이후 repeat 회의 중앙값을 처리량으로 쓴다"""
    run(max(1, sizes[0] // 2))                           # 워밍업 (커널 초기화 시간 제외)
    best = None
    for b in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        oom = False
        try:
            secs = []
            for _ in range(1 + repeat):
                t0 = time.time()
                run(b)
                torch.cuda.synchronize()
                secs.append(time.time() - t0)
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            print(f"  [{name}] {b:>4d} → OOM", flush=True)
            break
        alloc, resv = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        over = resv > limit_bytes
        sec = float(np.median(secs[1:])) if repeat else secs[0]
        print(f"  [{name}] {b:>4d} → allocated {alloc / 1e9:5.1f}GB  reserved {resv / 1e9:5.1f}GB  "
              f"| {b / sec:6.1f} 입력/초 (첫 회 {b / secs[0]:.1f})" + ("  (한도 초과)" if over else ""), flush=True)
        if over:
            break
        best = b
    torch.cuda.empty_cache()
    return best


def main():
    args = parse_args()
    # 다른 작업(학습 등)이 GPU 를 쓰는 중이면 측정값이 틀리고, 그 작업까지 OOM 으로 죽을 수 있으므로 시작하지 않는다
    used, total = gpu_memory_used()
    if used > total * 0.1:
        raise SystemExit(f"GPU 를 다른 프로세스가 쓰고 있습니다 (사용 중 {used / 1e9:.1f}GB / {total / 1e9:.1f}GB). "
                         f"./colab/colab_pipeline.sh status 로 확인하고, 그 작업이 끝난 뒤 다시 실행하세요.")
    props = torch.cuda.get_device_properties(0)
    limit = props.total_memory * args.limit
    print(f"[GPU] {props.name} {props.total_memory / 1e9:.1f}GB | 한도(reserved) {limit / 1e9:.1f}GB | 프리셋 {args.preset}")

    # 가장 긴 프롬프트 (base + hint) 로 잰다
    worst_base = max(BASE_PROMPTS, key=lambda k: len(BASE_PROMPTS[k]["two"]))
    worst_hint = max(QTYPE_HINTS, key=lambda k: len(QTYPE_HINTS[k])) if QTYPE_HINTS else None
    prompt = Prompt(worst_base, worst_hint)

    rows = worst_rows(args.n_images)
    origs = [load_image(r["img_path"]) for r in rows]
    R = config.RERUN
    single = [[resize_to(im, config.INFER_MAX_SIDE)] for im in origs]
    # 4단계 최악 조건 : 확대 사진이 crop_max_side 까지 꽉 차는 경우
    two = [[resize_to(im, R["full_max_side"]), resize_to(im, R["crop_max_side"], R["crop_min_side"])] for im in origs]
    r0 = rows[0]
    choices = [str(r0[c]) for c in LETTERS]

    # 1단계와 같은 조건 : LoRA 학습 모델 + 옵티마이저 상태가 올라간 상태에서 평가
    lora = config.lora_spec()
    vlm = VLM(args.preset, train_lora=True, lora=lora, grad_ckpt=args.grad_ckpt,
              download_source=args.download_source)
    fla_on = sys.modules.get("fla") is not None               # transformers 가 모델을 읽을 때 fla 를 가져왔는지
    print(f"[모델] 가중치 로드 후 {torch.cuda.memory_allocated() / 1e9:.1f}GB | linear attention : "
          + ("flash-linear-attention" if fla_on else "PyTorch 참고 구현"))
    print(f"[프롬프트] 최악 조건: {prompt_text(prompt, two=True)[:80]}…")

    # ── 학습 : 마이크로배치를 키워 가며 (1단계와 같은 경로: choice_loss → backward → step) ──
    trainable = [p for p in vlm.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=config.TRAIN["lr"])
    img_train = resize_to(origs[0], config.TRAIN_MAX_SIDE)
    vlm.model.train()

    def train_step(b):
        batch = [([img_train], r0["question"], choices, 0, prompt) for _ in range(b)]
        vlm.choice_loss(batch).backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    print(f"[학습] 마이크로배치 상한 ({config.TRAIN_MAX_SIDE}², gradient checkpointing "
          f"{'켬' if args.grad_ckpt else '끔'}) — 처리량은 문항/초")
    best_train = measure("train", train_step, sorted(args.sizes) if args.sizes else TRAIN_SIZES,
                         limit, args.repeat)
    vlm.model.eval()
    torch.cuda.empty_cache()

    def items(imgs_list, b):
        return [(imgs_list[i % len(imgs_list)], r0["question"], choices, prompt) for i in range(b)]

    score_sizes = sorted(args.sizes) if args.sizes else SCORE_SIZES
    with torch.no_grad():
        best_single = measure("single", lambda b: vlm.score(items(single, b)), score_sizes, limit, args.repeat)
        best_two = measure("two", lambda b: vlm.score(items(two, b)), score_sizes, limit, args.repeat)
        best_ground = None if args.sizes else measure(
            "ground", lambda b: vlm.ground([rows[i % len(rows)] for i in range(b)],
                                          R["ground_max_side"], R["max_new_tokens"]), GROUND_SIZES, limit, args.repeat)

    if args.sizes:                                        # 확인 모드 : 가능 여부를 한 줄로
        for name, best, key in (("single", best_single, "PREDICT·TRAIN.eval"), ("two", best_two, "RERUN")):
            ok = [b for b in score_sizes if best is not None and b <= best]
            print(f"[확인] {name:<6s} ({key} 배치) : " + (f"{ok} 가능" if ok else "불가")
                  + (f", {[b for b in score_sizes if b not in ok]} 불가" if ok and len(ok) < len(score_sizes) else ""))

    result = {"gpu": props.name, "total_gb": round(props.total_memory / 1e9, 1), "preset": args.preset,
              "lora_r": lora["r"], "lora_targets": lora["targets_name"], "grad_checkpointing": args.grad_ckpt,
              "limit": args.limit, "checked_sizes": args.sizes,
              "linear_attention": "fla" if fla_on else "torch",
              "TRAIN.micro_batch": best_train,
              "PREDICT.batch_size": best_single, "TRAIN.eval_batch": best_single,
              "RERUN.batch_size": best_two, "RERUN.ground_batch": best_ground}
    save_json(result, os.path.join(config.RUN_ROOT, "probe_batch.json"))
    print("__RESULT__", result)


if __name__ == "__main__":
    main()
