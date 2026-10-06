"""
GPU 공통 유틸: 모델 다운로드·로드(LoRA), 프롬프트 pool·sampling·선택, a/b/c/d 로짓 스코어링, grounding(bbox) 생성.
이 모듈을 import 하기 전에 config 가 먼저 import 되어 있어야 한다 (HF 캐시 경로).

프롬프트 책임은 전부 이 파일에 모은다.
  BASE_PROMPTS   의미가 같은 generic prompt pool (표현만 다름)
  QTYPE_HINTS    qtype 별 optional hint
  sample_prompt  학습용 stochastic sampling (rng 로 재현 가능)
  select_prompt  추론용 deterministic 선택 (qtype 기반)
학습과 추론은 같은 build_messages() / VLM.render() 를 거치고, 프롬프트를 만드는 정책만 다르다.
"""
import config  # noqa: F401  (반드시 transformers 보다 먼저)

import os
import time
import random
import contextlib
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import transformers
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model

import model_store
from data_utils import LETTERS, parse_boxes, to_original, progress

Image.MAX_IMAGE_PIXELS = None
DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True     # A100 TensorFloat-32 (bf16 autocast 밖 연산도 빠르게)
torch.backends.cudnn.allow_tf32 = True


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════
# 이미지
# ═════════════════════════════════════════════════════════════
def load_image(path):
    """원본 해상도로 로드 (EXIF 회전 보정). 크롭은 반드시 이 원본에서 자른다"""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def resize_to(img, max_side, min_side=None):
    """픽셀 수가 max_side² 를 넘으면 축소, min_side² 보다 작으면 확대 (비율 유지)"""
    w, h = img.size
    px, max_px = w * h, max_side * max_side
    min_px = min_side * min_side if min_side else None
    if px > max_px:
        s = (max_px / px) ** 0.5
    elif min_px and px < min_px:
        s = (min_px / px) ** 0.5
    else:
        return img
    return img.resize((max(32, round(w * s)), max(32, round(h * s))), Image.LANCZOS)


# ═════════════════════════════════════════════════════════════
# 프롬프트
#   single : 이미지 1장 (기본 추론·학습)
#   two    : [전체 사진] + [확대 사진] (4단계 grounding 재추론)
# base prompt 들은 표현만 다르고 지시하는 task 는 같아야 한다. 출력 규칙(ANSWER_RULE)은 모든 프롬프트에서 동일.
# ═════════════════════════════════════════════════════════════
SYSTEM_INSTRUCT = ("You are a helpful visual question answering assistant. "
                   "Answer using exactly one letter among a, b, c, or d. No explanation.")
# 출력 형식은 항상 같다 (= Answer only: a, b, c, or d). 표현을 바꾸면 기존 실험과 비교가 끊기므로 여기만 고친다
ANSWER_RULE = "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
_TWO_NOTE = ("확대 사진은 자동으로 잘라낸 것이라 정답과 무관한 부분일 수도 있으니, "
             "전체 사진과 대조해서 판단하세요.")

BASE_PROMPTS = {
    "p1": {"single": "이미지와 질문을 보고 가장 적절한 답을 고르세요.",
           "two": "[전체 사진]과 질문 관련 부분을 확대한 [확대 사진]을 보고 가장 적절한 답을 고르세요. " + _TWO_NOTE},
    "p2": {"single": "각 선택지를 이미지 속 정보와 하나씩 비교한 뒤 답을 고르세요.",
           "two": "각 선택지를 [전체 사진]과 [확대 사진] 속 정보와 하나씩 비교한 뒤 답을 고르세요. " + _TWO_NOTE},
    "p3": {"single": "이미지에 실제로 보이는 정보를 일반 상식보다 우선해서 답을 고르세요.",
           "two": "[전체 사진]과 [확대 사진]에 실제로 보이는 정보를 일반 상식보다 우선해서 답을 고르세요. " + _TWO_NOTE},
    "p4": {"single": "질문의 핵심 조건을 확인한 뒤, 관련된 영역과 선택지를 대조해 답을 고르세요.",
           "two": "질문의 핵심 조건을 확인한 뒤, [전체 사진]과 [확대 사진]의 관련 영역과 선택지를 대조해 답을 고르세요. "
                  + _TWO_NOTE},
}

# qtype 별 hint — 실제로 관찰된 오답 유형에만 둔다 (valid 에서 반복 확인될 때만 추가)
# 키는 data_utils.QTYPE_RULES 의 이름과 같아야 한다
QTYPE_HINTS = {
    "부정형":    "이 질문은 부정 조건입니다. '없는 것/아닌 것'을 묻는지 먼저 확인하고, 네 선택지를 모두 이미지와 대조하세요.",
    "숫자·가격": "질문이 가리키는 항목과 직접 연결된 숫자만 읽으세요. 옆 항목의 가격·숫자와 섞이지 않게 확인하세요.",
    "문구읽기":  "이미지에 실제로 적힌 글자와 어순을 그대로 따르세요. 더 자연스러운 표현으로 고쳐 읽지 마세요.",
}

Prompt = namedtuple("Prompt", "base hint")      # base: BASE_PROMPTS 키, hint: QTYPE_HINTS 키 또는 None


def as_prompt(x):
    """"p2" / "p2+부정형" / Prompt / None → Prompt (CSV·JSON·CLI 에 문자열로 저장하고 되읽기 위해)"""
    if isinstance(x, Prompt):
        return x
    if x is None:
        return Prompt(config.PROMPT["infer_base"], None)
    base, _, hint = str(x).partition("+")
    return Prompt(base, hint or None)


def prompt_str(p):
    p = as_prompt(p)
    return p.base + (f"+{p.hint}" if p.hint else "")


def prompt_text(p, two=False):
    """지시문 = base prompt (+ qtype hint)"""
    p = as_prompt(p)
    head = BASE_PROMPTS[p.base]["two" if two else "single"]
    return head + (" " + QTYPE_HINTS[p.hint] if p.hint else "")


def sample_prompt(rng, qtype=None, hint_prob=None, base_ids=None, hint_qtypes=None):
    """학습용 : base prompt 를 균등 sampling 하고, hint_prob 확률로 그 sample 의 qtype hint 를 덧붙인다.
    rng 는 학습 루프가 쓰는 numpy Generator (체크포인트에 저장되므로 이어하기해도 재현된다)"""
    ids = base_ids or config.PROMPT["base_ids"] or list(BASE_PROMPTS)
    hint_prob = config.PROMPT["hint_prob"] if hint_prob is None else hint_prob
    allowed = hint_qtypes if hint_qtypes is not None else config.PROMPT["hint_qtypes"]
    base = ids[int(rng.integers(len(ids)))]
    hint = qtype if (qtype in QTYPE_HINTS and (allowed is None or qtype in allowed)
                     and rng.random() < hint_prob) else None
    return Prompt(base, hint)


def select_prompt(qtype=None, mode=None, base=None, fixed=None, hint_qtypes=None):
    """추론용 : 같은 입력에는 항상 같은 프롬프트 (branch 로 나누지 않는다)
      qtype   : base + 그 유형의 hint (없으면 generic 과 같음)
      generic : base 만
      fixed   : --prompt 로 준 값 그대로 (예: p3, p3+부정형)"""
    mode = mode or config.PROMPT["infer_mode"]
    if mode == "fixed":
        return as_prompt(fixed)
    base = base or config.PROMPT["infer_base"]
    allowed = hint_qtypes if hint_qtypes is not None else config.PROMPT["hint_qtypes"]
    if mode == "qtype" and qtype in QTYPE_HINTS and (allowed is None or qtype in allowed):
        return Prompt(base, qtype)
    return Prompt(base, None)


def build_messages(images, question, choices, prompt=None):
    """학습·추론 공통 메시지. prompt 는 Prompt 또는 "p2"/"p2+부정형" 문자열"""
    if len(images) == 2:
        content = [{"type": "text", "text": "[전체 사진]"}, {"type": "image", "image": images[0]},
                   {"type": "text", "text": "[확대 사진]"}, {"type": "image", "image": images[1]}]
    else:
        content = [{"type": "image", "image": images[0]}]
    head = prompt_text(prompt, two=len(images) == 2)
    a, b, c, d = choices
    content.append({"type": "text", "text":
                    f"{head}\n\n[질문]\n{question}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n{ANSWER_RULE}"})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
            {"role": "user", "content": content}]


GROUND_PROMPT = (
    "Question: {q}\n"
    "Options: (a) {a} (b) {b} (c) {c} (d) {d}\n\n"
    "To answer this question, which part of the image must be read closely? "
    "Locate that region (for example the text, sign, label, price tag, menu, or screen) "
    "and output its bbox coordinates in JSON format."
)


# ═════════════════════════════════════════════════════════════
# 모델 준비
# ═════════════════════════════════════════════════════════════
def ensure_model(preset_name, source="hf", tries=8, workers=2):
    """MODEL_ROOT/<모델이름> 의 경로를 반환. 로컬 → 드라이브 보관소 → Hugging Face 순서로 찾는다 (model_store).
    HF 에서 받았으면 드라이브에도 저장해 다음 VM 은 드라이브에서 복사한다 (config.MODEL_DRIVE_CACHE)"""
    repo = config.MODEL_PRESETS[preset_name]["repo"]
    local = model_store.local_dir(preset_name)
    if model_store.is_complete(local) or model_store.fetch_from_drive(preset_name):
        return local
    if not config.ONLINE:
        raise FileNotFoundError(f"오프라인인데 {local} 에도, 드라이브 보관소에도 모델이 없습니다.")
    os.makedirs(local, exist_ok=True)
    for i in range(1, tries + 1):
        try:
            if source == "modelscope":
                try:
                    from modelscope import snapshot_download as ms_snapshot
                except ImportError as e:
                    raise SystemExit("modelscope 가 없습니다: pip install modelscope") from e
                ms_snapshot(repo, local_dir=local)
            else:
                from huggingface_hub import snapshot_download
                snapshot_download(repo, local_dir=local, max_workers=workers)
            if model_store.is_complete(local):
                if config.MODEL_DRIVE_CACHE and config.MODEL_DRIVE_ROOT:
                    model_store.save_to_drive(preset_name)          # 실패해도 경고만 (로컬 사본으로 계속)
                return local
        except Exception as e:  # noqa: BLE001
            wait = min(120, 10 * 2 ** (i - 1))
            print(f"[다운로드 {i}/{tries}] {type(e).__name__}: {str(e)[:150]} → {wait}초 후 재시도")
            time.sleep(wait)
    raise RuntimeError(f"{repo} 다운로드 실패 (--download-source modelscope 로 바꿔 보세요)")


class VLM:
    """processor + model + 로짓 스코어링.
    adapter_dir 를 주면 학습된 LoRA 를 얹고, train_lora=True 면 lora(=config.lora_spec(...)) 로 새 LoRA 를 만든다"""

    def __init__(self, preset_name, adapter_dir=None, train_lora=False, lora=None, grad_ckpt=None,
                 download_source="hf"):
        self.preset_name = preset_name
        self.preset = config.MODEL_PRESETS[preset_name]
        model_path = ensure_model(preset_name, source=download_source)
        print(f"[모델] {preset_name} | {model_path} | {self.preset['quant'] or 'bf16'}")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"      # 배치에서 마지막 위치 = 다음 토큰
        self.tok = self.processor.tokenizer
        # thinking 토글은 그런 템플릿에만 넘긴다. 템플릿이 안 쓰는 인자를 주면 transformers 5 가
        # apply_chat_template 호출마다 경고를 찍어 로그가 경고로 뒤덮인다 (Qwen3-VL 은 토글이 없음)
        tmpl = self.processor.chat_template
        tmpl_text = " ".join(tmpl.values()) if isinstance(tmpl, dict) else str(tmpl or "")
        self._render_kw = {"enable_thinking": False} if "enable_thinking" in tmpl_text else {}
        self._set_pixels(config.PROC_MAX_PIXELS, config.MIN_PIXELS)
        self.patch_size = getattr(self.processor.image_processor, "patch_size", 14)

        kw = dict(device_map="auto", attn_implementation="sdpa")
        kw["dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"] = torch.bfloat16
        if self.preset["quant"] == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["visual", "lm_head"])  # 비전 인코더·출력층은 양자화하지 않음
        from transformers import AutoModelForImageTextToText
        base = AutoModelForImageTextToText.from_pretrained(model_path, **kw)
        base.config.use_cache = False

        if train_lora:
            spec = lora or config.lora_spec()
            # gradient checkpointing : 켜면 메모리를 아끼고 20~40% 느리다. A100 40GB·4B 는 꺼도 되므로 기본 끔
            if config.TRAIN["grad_checkpointing"] if grad_ckpt is None else grad_ckpt:
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                base.enable_input_require_grads()
            self.model = get_peft_model(base, LoraConfig(
                r=spec["r"], lora_alpha=spec["alpha"], lora_dropout=spec["dropout"],
                bias="none", task_type="CAUSAL_LM", target_modules=spec["targets"]))
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if n_train == 0:
                raise RuntimeError(f"LoRA 대상 모듈이 하나도 잡히지 않았습니다 (targets={spec['targets_name']}). "
                                   f"이 모델의 모듈 이름을 확인하고 config.LORA_TARGETS 를 고치세요")
            ckpt_on = config.TRAIN["grad_checkpointing"] if grad_ckpt is None else grad_ckpt
            print(f"[LoRA] r={spec['r']} alpha={spec['alpha']} dropout={spec['dropout']} "
                  f"대상={spec['targets_name']} | 학습 파라미터 {n_train / 1e6:.1f}M "
                  f"| gradient checkpointing {'켬' if ckpt_on else '끔'}")
        elif adapter_dir:
            self.model = PeftModel.from_pretrained(base, adapter_dir)
            print(f"[모델] LoRA 로드: {adapter_dir}")
        else:
            self.model = base
        self.model.eval()

        self.letter_ids = []
        for l in LETTERS:
            ids = self.tok.encode(l, add_special_tokens=False)
            assert len(ids) == 1, f"'{l}' 이 토큰 1개가 아님: {ids}"
            self.letter_ids.append(ids[0])
        tail = self.render(build_messages([Image.new("RGB", (64, 64))], "q", ["1", "2", "3", "4"]))[-60:]
        if "<think>" in tail and "</think>" not in tail:
            raise RuntimeError(f"thinking 모드가 켜져 있어 다음 토큰이 정답 글자가 아님: {tail!r}")

    # ── 기본 동작 ──
    def _set_pixels(self, max_px, min_px):
        ip = self.processor.image_processor
        for k, v in (("min_pixels", min_px), ("max_pixels", max_px)):
            if hasattr(ip, k):
                setattr(ip, k, v)
        if isinstance(getattr(ip, "size", None), dict):
            size = dict(ip.size)
            if "shortest_edge" in size: size["shortest_edge"] = min_px
            if "longest_edge" in size: size["longest_edge"] = max_px
            ip.size = size

    @property
    def can_think(self):
        """템플릿에 thinking 토글이 있는 모델인지 (Qwen3.5 계열. Qwen3-VL-Instruct 는 없음)"""
        return bool(self._render_kw)

    def render(self, msgs, thinking=False):
        """학습·추론 공통 생성 프롬프트. 평소엔 thinking 을 끄고(답이 바로 첫 토큰), thinking=True 면 켠다
        (Qwen3.5 템플릿 : 끄면 '<think>\\n\\n</think>\\n\\n' 뒤 답, 켜면 '<think>\\n' 뒤에 생각을 이어 쓴다)"""
        kw = {"enable_thinking": True} if (thinking and self.can_think) else self._render_kw
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def _last_logits(self, enc):
        try:
            out = self.model(**enc, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = self.model(**enc, use_cache=False)
        return out.logits[:, -1, :]

    def adapter_off(self):
        """LoRA 를 잠시 끄는 컨텍스트 (grounding 은 base 모델 능력으로)"""
        return self.model.disable_adapter() if isinstance(self.model, PeftModel) else contextlib.nullcontext()

    # ── 학습 : 4지 토큰 cross-entropy (마이크로배치) ──
    def choice_loss(self, batch, label_smoothing=0.0):
        """batch: [(images, question, choices, gold_idx, prompt), ...] → 배치 평균 CE
        추론과 똑같은 프롬프트 끝에서 a/b/c/d 네 로짓만 뽑아 4지 분류로 학습한다.
        패딩이 왼쪽이라 배치로 묶어도 마지막 위치가 각 문항의 마지막 토큰이다"""
        texts, flat, golds = [], [], []
        for images, q, ch, gold, p in batch:
            texts.append(self.render(build_messages(images, q, ch, p)))
            flat.extend(images)
            golds.append(gold)
        enc = self.processor(text=texts, images=flat, padding=True, return_tensors="pt").to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self._last_logits(enc)
        choice_logits = logits[:, self.letter_ids].float()
        target = torch.tensor(golds, device=choice_logits.device)
        return F.cross_entropy(choice_logits, target, label_smoothing=label_smoothing)

    # ── 추론 : 여러 문항 한 번에 스코어링 ──
    @torch.no_grad()
    def score(self, items):
        """items: [(images, question, choices, prompt), ...] → (N,4) 확률"""
        texts, flat = [], []
        for images, q, ch, p in items:
            texts.append(self.render(build_messages(images, q, ch, p)))
            flat.extend(images)
        enc = self.processor(text=texts, images=flat, padding=True, return_tensors="pt").to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self._last_logits(enc)
        return torch.softmax(logits[:, self.letter_ids].float(), dim=-1).cpu().numpy()

    # ── 추론 모드 재추론 : 생각을 먼저 쓰고 그 뒤 a/b/c/d 확률 (4단계 --mode reason) ──
    @torch.no_grad()
    def reason(self, items, max_new_tokens=768, use_adapter=False):
        """items: [(images, question, choices, prompt), ...] → (N,4) 확률, 생각 글, 생각을 스스로 맺었는지(</think>)
        1) thinking 을 켠 프롬프트(…<think>\\n)에서 생각을 생성 (그리디, max_new_tokens 에서 끊음)
        2) 프롬프트 + 생각 + '\\n</think>\\n\\n' 다음 토큰의 a/b/c/d 확률 — 바로 답하기(thinking 끔)와 같은 위치·같은 글자
           토큰이라 결과를 같은 형식(p_a..p_d)으로 비교할 수 있다
        use_adapter=False 면 학습한 LoRA 를 끄고 원래 모델로 생각한다 (LoRA 는 '바로 답하기'만 배움)"""
        if not self.can_think:
            raise RuntimeError(f"{self.preset_name} 은 thinking 을 지원하지 않아 추론 모드를 쓸 수 없습니다 "
                               f"(Qwen3.5 계열 프리셋에서만)")
        texts, flat = [], []
        for images, q, ch, p in items:
            texts.append(self.render(build_messages(images, q, ch, p), thinking=True))
            flat.extend(images)
        ctx = contextlib.nullcontext if use_adapter else self.adapter_off     # 호출할 때마다 새 컨텍스트
        enc = self.processor(text=texts, images=flat, padding=True, return_tensors="pt").to(DEVICE)
        with ctx(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        gen = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)
        stops = [s for s in (self.tok.eos_token, "<|im_end|>", self.tok.pad_token) if s]
        thoughts, closed, full = [], [], []
        for t, g in zip(texts, gen):
            end = g.find("</think>")
            body = g[:end] if end >= 0 else g
            for s in stops:                                   # 생각을 맺지 못하고 끝난 경우 끝 표시·패딩 제거
                if s in body:
                    body = body[:body.find(s)]
            thoughts.append(body.strip())
            closed.append(end >= 0)
            full.append(t + body.rstrip() + "\n</think>\n\n")
        enc2 = self.processor(text=full, images=flat, padding=True, return_tensors="pt").to(DEVICE)
        with ctx(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self._last_logits(enc2)
        P = torch.softmax(logits[:, self.letter_ids].float(), dim=-1).cpu().numpy()
        return P, thoughts, closed

    # ── grounding : 질문에 답하려면 봐야 할 영역의 bbox ──
    @torch.no_grad()
    def ground(self, rows, max_side=768, max_new_tokens=64):
        """rows: pd.Series 목록 → [(raw_text, [원본 좌표 bbox, ...]), ...]"""
        origs = [load_image(r["img_path"]) for r in rows]
        smalls = [resize_to(im, max_side) for im in origs]
        texts = []
        for r, sm in zip(rows, smalls):
            prompt = GROUND_PROMPT.format(q=r["question"], a=r["a"], b=r["b"], c=r["c"], d=r["d"])
            msgs = [{"role": "user", "content": [{"type": "image", "image": sm}, {"type": "text", "text": prompt}]}]
            texts.append(self.render(msgs))
        enc = self.processor(text=texts, images=smalls, padding=True, return_tensors="pt").to(DEVICE)
        with self.adapter_off(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        raws = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        results = []
        for im, raw, (_, gh, gw) in zip(origs, raws, enc["image_grid_thw"].tolist()):
            resized = (gw * self.patch_size, gh * self.patch_size)
            boxes = [to_original(b, resized, im.size, self.preset["coord"]) for b in parse_boxes(raw)]
            results.append((raw, [b for b in boxes if b is not None]))
        return results


# ═════════════════════════════════════════════════════════════
# 배치 실행
# ═════════════════════════════════════════════════════════════
def run_batched(items, fn, batch_size, desc=None, bar=None, leave=True):
    """items 를 batch_size 씩 fn 에 넣어 결과를 이어붙임. OOM 나면 그 배치만 1개씩 재시도
    진행 표시는 문항 단위. bar 를 주면 그 막대를 이어서 쓰고(닫지 않음), 없으면 desc 로 새로 만든다"""
    own = bar is None
    if own:
        bar = progress(total=len(items), desc=desc, unit="문항", leave=leave)
    out = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        try:
            out.extend(fn(chunk))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bar.write(f"  ⚠ OOM → 이 배치 {len(chunk)}문항을 1개씩 재시도 (자주 보이면 config 의 배치를 줄일 것)")
            for it in chunk:
                out.extend(fn([it]))
        bar.update(len(chunk))
    if own:
        bar.close()
    return out


def predict_rows(vlm, rows, image_fn, prompt_fn, batch_size=32, desc="predict", leave=True):
    """rows(pd.Series 목록)를 sample 당 1회 forward 로 예측.
    image_fn(row)   → 이미지 목록 (1장 또는 [전체, 확대] 2장)
    prompt_fn       → Prompt/문자열(모든 행 공통) 또는 row 를 받아 Prompt 를 돌려주는 함수
    leave=False 면 끝난 뒤 진행 막대를 지움 (학습 중 평가처럼 다른 막대 안에서 돌 때)
    반환: probs (N,4), 사용한 프롬프트 문자열 목록 (N,)"""
    pick = prompt_fn if callable(prompt_fn) else (lambda r: prompt_fn)

    def fn(chunk):
        items, used = [], []
        for r in chunk:
            p = as_prompt(pick(r))
            used.append(prompt_str(p))
            items.append((image_fn(r), r["question"], [str(r[c]) for c in LETTERS], p))
        P = vlm.score(items)
        return list(zip(P, used))

    res = run_batched(list(rows), fn, batch_size, desc, leave=leave)
    return np.stack([r[0] for r in res]), [r[1] for r in res]
