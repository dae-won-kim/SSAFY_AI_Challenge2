"""열화(degradation) 이미지 증강 모듈입니다.

실제 촬영 환경에서 생길 수 있는 화질 저하를 흉내 내는 9종의 열화 함수와,
이를 1~3개 랜덤 조합해 적용하는 degrade(), 데이터셋 전체에 적용하는 make_aug()로 구성되어 있습니다.

- 기하 변형 계열 : geo(기울임·원근), lens(렌즈 왜곡), warp(종이 휘어짐)
- 조명 계열      : shadow(그림자), low_light(저조도), glare(빛반사)
- 화질 계열      : blur(블러), lowres(저해상도), noise(노이즈)
- JPEG 압축      : 함수가 아니라 마지막 저장 단계에서 딱 한 번만 적용합니다. (압축 열화 누적 방지)

모든 강도 관련 값은 DegradeConfig 하나에 모여 있습니다.
노트북에서는 전역 변수를 덮어써서 강도를 바꿨는데, 스크립트에서는 설정 객체를 넘기는 방식으로 바꿨습니다.
"""

from __future__ import annotations

import json
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import COLS

# 적용 순서입니다: 기하 -> 조명 -> 화질 -> (마지막에) JPEG 저장
# "jpeg"는 OPS에 없고 degrade()에서 저장 품질로 처리합니다.
ORDER = ["geo", "lens", "warp", "shadow", "blur", "lowres", "low_light", "glare", "noise", "jpeg"]

# 강도 구간입니다. 이 범위 안에서 기본 강도(base)를 뽑습니다. (mild / mid / severe)
LEVELS = [(0.05, 0.33), (0.33, 0.66), (0.66, 1.0)]

# 리사이즈 보간 방식 이름 -> OpenCV 상수 (op_lowres에서 사용합니다)
INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR,
          "cubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST}


# ══════════════════════════════════════════════════════════════════════
#  설정
#  (a, b) 튜플은 a = 강도 0일 때 값, b = 강도 1일 때 값입니다.
#  SHADOW_DARK, LOWLIGHT_GAIN, LOWRES_SCALE, JPEG_Q는 "값이 작을수록 강한" 항목입니다.
# ══════════════════════════════════════════════════════════════════════
@dataclass
class DegradeConfig:
    """열화 강도와 조합 방식을 모아 둔 설정입니다. 기본값은 노트북 4단계의 초기값과 같습니다."""

    # ── 강도 구간 선택 확률 / 기본 강도 상한 ──
    level_p: list = field(default_factory=lambda: [0.3, 0.4, 0.3])   # mild / mid / severe 확률
    max_severity: float = 0.85                                        # 기본 강도(base) 상한

    # ── 열화별 세부 범위 ──
    geo_rot: tuple = (-30, 30)              # [op_geo] 최대 회전각(도)
    geo_persp: tuple = (0.005, 0.03)        # [op_geo] 원근 왜곡 (꼭짓점 이동 비율)
    lens_k: tuple = (0.1, 0.2)              # [op_lens] 렌즈 왜곡 계수 크기 (부호는 랜덤)
    warp_amp: tuple = (0.010, 0.020)        # [op_warp] 물결 최대 변위 (긴 변 대비 비율)
    shadow_dark: tuple = (0.9, 0.5)         # [op_shadow] 가장 어두운 쪽의 밝기 배율
    lowlight_gamma: tuple = (1.3, 3.0)      # [op_low_light] 어두운 부분을 뭉개는 정도
    lowlight_gain: tuple = (0.8, 0.45)      # [op_low_light] 전체 밝기 배율
    lowlight_noise: tuple = (0.005, 0.03)   # [op_low_light] 노이즈 표준편차 (0~1 스케일)
    glare_size: tuple = (0.2, 0.4)          # [op_glare] 반사 크기 (가로/세로 대비 비율)
    glare_strength: tuple = (0.45, 0.85)    # [op_glare] 반사 세기 (흰색으로 덮는 비율)
    blur_gauss_sigma: tuple = (1.0, 3.0)    # [op_blur] 가우시안 블러 세기
    blur_motion_len: tuple = (3, 15)        # [op_blur] 모션 블러 길이(px)
    blur_defocus_r: tuple = (2, 6)          # [op_blur] 초점 흐림 반경(px)
    lowres_scale: tuple = (0.4, 0.2)        # [op_lowres] 축소 비율 (최소 16px 보장)
    noise_sigma: tuple = (3, 25)            # [op_noise] 노이즈 표준편차 (0~255 스케일)
    jpeg_q: tuple = (80, 35)                # [jpeg] JPEG 저장 품질

    # ── 열화 조합 / 강도 보정 ──
    op_weight: dict = field(default_factory=lambda: {   # 열화가 뽑히는 상대 확률 (기하 3종은 0.5씩)
        "geo": 1, "lens": 0.5, "warp": 0.5, "shadow": 1, "blur": 1,
        "lowres": 1, "low_light": 1, "glare": 1, "noise": 1, "jpeg": 1})
    op_count_p: list = field(default_factory=lambda: [0.5, 0.35, 0.15])  # 열화 1개/2개/3개 확률
    combo_decay: float = 0.15       # 열화가 1개 늘 때마다 강도를 깎는 비율 (2개면 x0.85, 3개면 x0.70)
    op_jitter: float = 0.05         # 열화마다 강도를 흔드는 정도 (표준편차)
    s_min: float = 0.02             # 열화 하나에 적용되는 최소 강도
    save_q: int = 95                # JPEG 열화가 뽑히지 않았을 때의 저장 품질
    op_max: dict = field(default_factory=lambda: {t: 1.0 for t in ORDER})   # 열화별 강도 상한 (1.0 = 상한 없음)

    # ── 재시도 ──
    max_retry: int = 4              # 식별 가드에 걸렸을 때 최대 시도 횟수
    retry_decay: float = 0.8        # 재시도마다 강도에 곱하는 비율 (x1, x0.8, x0.64, x0.51)

    # ── 식별 가드 기준 ──
    guard_min_bright: float = 25    # 결과 평균 밝기 하한 (원본이 더 어두우면 원본의 절반까지 허용)
    guard_sat_gain: float = 0.25    # 하얗게 날아간 픽셀 비율이 원본보다 늘어도 되는 상한
    guard_blur_ratio: float = 0.05  # 블러 지표가 원본 대비 최소 이 비율은 남아야 함

    @property
    def op_p(self) -> np.ndarray:
        """op_weight를 합이 1이 되도록 정규화한 확률 배열입니다. (ORDER 순서)"""
        p = np.array([self.op_weight[t] for t in ORDER], dtype=float)
        return p / p.sum()


# ── 공통 유틸 ──────────────────────────────────────────────────────────
def _lerp(a, b, s):
    """강도 s에 따라 a(s=0) ~ b(s=1)를 선형으로 잇습니다."""
    return a + (b - a) * s


def jpeg_quality(s: float, cfg: DegradeConfig) -> int:
    """강도 s를 JPEG 저장 품질로 변환합니다."""
    return int(_lerp(cfg.jpeg_q[0], cfg.jpeg_q[1], s))


def metrics(img: np.ndarray) -> dict:
    """열화 정도를 재는 지표입니다. (식별 가드와 meta 기록에 사용합니다)"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"m_blur": float(cv2.Laplacian(g, cv2.CV_64F).var()),   # 작을수록 흐립니다
            "m_bright": float(g.mean()),                           # 평균 밝기
            "m_sat": float((g >= 250).mean())}                     # 거의 흰색인 픽셀 비율


def source_id(i: str) -> str:
    """증강본 id에서 원본 id를 복원합니다. 예: 'train_0001_aug01.jpg' -> 'train_0001.jpg'"""
    return re.sub(r"_aug\d+(?=\.[^.]+$)", "", i)


def sev_label(s: float) -> str:
    """강도 값을 mild / mid / severe 라벨로 변환합니다."""
    return "mild" if s < 0.33 else ("mid" if s < 0.66 else "severe")


# ══════════════════════════════════════════════════════════════════════
#  열화 함수
#  모든 op_* 함수는 (img, s, rng, cfg) -> (열화된 img, 사용한 파라미터 dict) 형태입니다.
#    img : BGR uint8 이미지 / s : 강도 0~1 / rng : np.random.Generator / cfg : DegradeConfig
# ══════════════════════════════════════════════════════════════════════

# ── 1. 기하 변형 계열 (geo, lens, warp) ───────────────────────────────
#    글자 위치·모양이 바뀌므로 강도가 커지면 가장자리 글자가 잘릴 수 있습니다.

def op_geo(img, s, rng, cfg):
    """기울임(회전) + 원근 왜곡입니다. (비스듬히 찍은 사진)"""
    h, w = img.shape[:2]
    ang = float(rng.uniform(-1, 1) * _lerp(*cfg.geo_rot, s))   # 최대 회전각(도)
    j = _lerp(*cfg.geo_persp, s)                               # 원근 왜곡(꼭짓점 이동 비율)

    # 원근: 네 꼭짓점을 j 비율만큼 랜덤하게 이동합니다.
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-j, j, (4, 2)).astype(np.float32) * np.float32([w, h])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    # 회전: 이미지 중심 기준입니다.
    R = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    out = cv2.warpAffine(out, R, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, {"rot": round(ang, 2), "persp": round(float(j), 4)}


def op_lens(img, s, rng, cfg):
    """렌즈 왜곡입니다. 술통형(+) 또는 실패꾸러미형(-)을 랜덤으로 선택합니다."""
    h, w = img.shape[:2]
    sign = float(rng.choice([-1.0, 1.0]))          # +: 술통(barrel) / -: 실패꾸러미(pincushion)
    k = sign * _lerp(*cfg.lens_k, s)

    # 왜곡 중심을 이미지 중심에서 ±5% 정도 흔듭니다.
    cx, cy = w / 2 + rng.uniform(-0.05, 0.05) * w, h / 2 + rng.uniform(-0.05, 0.05) * h
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xn, yn = (xs - cx) / (w / 2), (ys - cy) / (h / 2)    # 중심 기준 정규화 좌표
    # 변(가장자리) 중앙은 고정하도록 정규화해서, 가장자리 글자가 프레임 밖으로 잘리지 않게 했습니다.
    f = (1 + k * (xn ** 2 + yn ** 2)) / (1 + k)
    out = cv2.remap(img, (xn * f * (w / 2) + cx).astype(np.float32),
                    (yn * f * (h / 2) + cy).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, {"k": round(float(k), 3), "type": "barrel" if k > 0 else "pincushion"}


def op_warp(img, s, rng, cfg):
    """종이가 휘거나 구겨진 듯한 부드러운 물결 왜곡입니다."""
    h, w = img.shape[:2]
    amp = _lerp(*cfg.warp_amp, s) * max(h, w)     # 최대 변위 = 긴 변 x 비율 (px)

    grid = 4                                       # 난수 격자 크기입니다. 키우면 더 잘게 구겨집니다.
    dx = cv2.resize(rng.uniform(-1, 1, (grid, grid)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(rng.uniform(-1, 1, (grid, grid)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = cv2.remap(img, xs + dx * amp, ys + dy * amp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, {"amp_px": round(float(amp), 1)}


# ── 2. 조명 계열 (shadow, low_light, glare) ───────────────────────────

def op_shadow(img, s, rng, cfg):
    """한쪽이 어두워지는 부드러운 그림자(그라데이션)입니다."""
    h, w = img.shape[:2]
    th = float(rng.uniform(0, 2 * np.pi))          # 그림자가 생기는 방향
    dark = _lerp(*cfg.shadow_dark, s)              # 가장 어두운 쪽의 밝기 배율

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / w - 0.5) * np.cos(th) + (yy / h - 0.5) * np.sin(th)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)   # 0~1로 정규화
    t = t * t * (3 - 2 * t)                          # smoothstep으로 부드럽게
    mult = 1 - (1 - dark) * t                        # 1(밝음) -> dark(어두움)
    out = np.clip(img.astype(np.float32) * mult[..., None], 0, 255).astype(np.uint8)
    return out, {"dark": round(float(dark), 2), "angle": round(th, 2)}


def op_low_light(img, s, rng, cfg):
    """저조도입니다. 어둡게 만든 뒤 노이즈를 더합니다. (실제 카메라와 같은 순서)"""
    gamma = _lerp(*cfg.lowlight_gamma, s)
    gain = _lerp(*cfg.lowlight_gain, s)
    sig = _lerp(*cfg.lowlight_noise, s)

    f = (img.astype(np.float32) / 255.0) ** gamma * gain          # 0~1로 바꿔 어둡게
    f = f + rng.normal(0, sig, f.shape).astype(np.float32)        # 어둡게 한 "뒤에" 노이즈
    out = np.clip(f * 255, 0, 255).astype(np.uint8)
    return out, {"gamma": round(float(gamma), 2), "gain": round(float(gain), 2)}


def op_glare(img, s, rng, cfg):
    """빛반사입니다. 타원형 가우시안 하이라이트를 랜덤 위치에 흰색으로 블렌딩합니다."""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h   # 반사 중심 (랜덤 위치)

    sx = _lerp(*cfg.glare_size, s) * w * rng.uniform(0.7, 1.3)      # 반사 가로 크기
    sy = _lerp(*cfg.glare_size, s) * h * rng.uniform(0.7, 1.3)      # 반사 세로 크기
    strength = _lerp(*cfg.glare_strength, s)                         # 반사 세기

    th = float(rng.uniform(0, np.pi))              # 타원 방향
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xx - cx, yy - cy
    xr = x * np.cos(th) + y * np.sin(th)
    yr = -x * np.sin(th) + y * np.cos(th)
    mask = np.exp(-(xr ** 2 / (2 * sx ** 2) + yr ** 2 / (2 * sy ** 2)))   # 중심 1 -> 바깥 0
    f = img.astype(np.float32)
    out = np.clip(f + (255 - f) * (mask[..., None] * strength), 0, 255).astype(np.uint8)
    return out, {"strength": round(float(strength), 2), "cx": round(float(cx / w), 2), "cy": round(float(cy / h), 2)}


# ── 3. 화질 계열 (blur, lowres, noise) ────────────────────────────────

def op_blur(img, s, rng, cfg):
    """블러입니다. 가우시안 / 모션 / 초점 흐림(defocus) 중 하나를 랜덤으로 선택합니다."""
    h, w = img.shape[:2]
    sc = max(h, w) / 1024                          # 이미지 크기 보정 (1024px 기준)
    kind = str(rng.choice(["gaussian", "motion", "defocus"]))

    if kind == "gaussian":
        sig = _lerp(*cfg.blur_gauss_sigma, s) * sc
        out = cv2.GaussianBlur(img, (0, 0), sig)
        p = {"kind": kind, "sigma": round(float(sig), 2)}

    elif kind == "motion":
        k = max(3, int(_lerp(*cfg.blur_motion_len, s) * sc) | 1)   # 모션 블러 길이(px, 홀수)
        ang = float(rng.uniform(0, 180))                          # 흔들린 방향
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1                                        # 가로선 커널
        M = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), ang, 1)
        ker = cv2.warpAffine(ker, M, (k, k))                      # 커널을 ang만큼 회전
        ker /= ker.sum() + 1e-8
        out = cv2.filter2D(img, -1, ker)
        p = {"kind": kind, "len": k, "angle": round(ang, 1)}

    else:  # defocus
        r = max(1.0, _lerp(*cfg.blur_defocus_r, s) * sc)          # 초점 흐림 반경(px)
        R = int(np.ceil(r))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        ker = ((xx ** 2 + yy ** 2) <= r * r).astype(np.float32)   # 원판 커널
        ker /= ker.sum()
        out = cv2.filter2D(img, -1, ker)
        p = {"kind": kind, "radius": round(float(r), 2)}

    return out, p


def op_lowres(img, s, rng, cfg):
    """저해상도입니다. 작게 줄였다가 원래 크기로 키워서 뭉개진 이미지를 만듭니다. (썸네일, 원거리 촬영 등)"""
    h, w = img.shape[:2]
    scale = _lerp(*cfg.lowres_scale, s)

    nw, nh = max(16, int(w * scale)), max(16, int(h * scale))                 # 최소 16px 보장
    down = str(rng.choice(["area", "linear", "cubic"]))                       # 줄이는 방식 (랜덤)
    up = str(rng.choice(["linear", "cubic", "nearest"], p=[0.4, 0.4, 0.2]))   # 키우는 방식 (nearest는 네모 픽셀 느낌)
    small = cv2.resize(img, (nw, nh), interpolation=INTERP[down])
    out = cv2.resize(small, (w, h), interpolation=INTERP[up])                 # 모델 입력 해상도 유지
    return out, {"scale": round(float(scale), 2), "down": down, "up": up}


def op_noise(img, s, rng, cfg):
    """가우시안 노이즈입니다."""
    sig = _lerp(*cfg.noise_sigma, s)
    n = rng.normal(0, sig, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return out, {"sigma": round(float(sig), 1)}


# 함수 이름 -> 함수 (degrade()가 이름으로 꺼내 씁니다)
OPS = {"geo": op_geo, "lens": op_lens, "warp": op_warp,
       "shadow": op_shadow, "blur": op_blur, "lowres": op_lowres, "low_light": op_low_light,
       "glare": op_glare, "noise": op_noise}


# ══════════════════════════════════════════════════════════════════════
#  열화 적용 (degrade) + 증강 데이터 생성 (make_aug)
# ══════════════════════════════════════════════════════════════════════

def is_legible(m: dict, m0: dict, cfg: DegradeConfig) -> bool:
    """결과(m)가 원본(m0) 대비 글자를 읽을 수 있는 상태인지 검사합니다. (식별 가드)"""
    return (m["m_bright"] >= min(cfg.guard_min_bright, 0.5 * m0["m_bright"])   # 너무 어두워지지 않았는지
            and m["m_sat"] <= m0["m_sat"] + cfg.guard_sat_gain                 # 반사로 하얗게 날아가지 않았는지
            and m["m_blur"] >= cfg.guard_blur_ratio * m0["m_blur"])            # 너무 뭉개지지 않았는지


def degrade(img: np.ndarray, rng: np.random.Generator, cfg: DegradeConfig, max_s: float = 1.0):
    """이미지 1장에 열화 1~3개를 랜덤 조합해서 적용합니다.

    글자를 읽기 어려울 정도로 망가지면(식별 가드 실패) 강도를 낮춰서 다시 시도합니다.
    반환값: (jpeg 인코딩 버퍼, 적용한 열화 태그, 실제 강도, jpeg 품질, 파라미터 dict, 결과 지표)
    """
    n = int(rng.choice([1, 2, 3], p=cfg.op_count_p))                                       # 열화 개수
    tags = sorted(rng.choice(ORDER, n, replace=False, p=cfg.op_p).tolist(), key=ORDER.index)   # 어떤 열화
    lo, hi = LEVELS[int(rng.choice(3, p=cfg.level_p))]                                     # mild/mid/severe 중 하나
    base = min(float(rng.uniform(lo, hi)), max_s)                                          # 기본 강도 (max_s로 상한)
    m0 = metrics(img)

    for attempt in range(cfg.max_retry):           # 글자가 뭉개졌으면 강도를 낮춰 재시도합니다.
        s_eff = base * (1 - cfg.combo_decay * (n - 1)) * (cfg.retry_decay ** attempt)
        out, params = img, {}                      # 매 시도마다 원본에서 다시 시작합니다. (열화 누적 방지)
        for t in tags:
            if t == "jpeg":
                continue                           # JPEG는 아래 저장 단계에서 처리합니다.
            s_t = float(np.clip(s_eff + rng.normal(0, cfg.op_jitter), cfg.s_min, max(cfg.s_min, cfg.op_max[t])))
            out, params[t] = OPS[t](out, s_t, rng, cfg)
        q = cfg.save_q
        if "jpeg" in tags:
            q = jpeg_quality(float(np.clip(s_eff, 0, cfg.op_max["jpeg"])), cfg)
            params["jpeg"] = {"quality": q}
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])   # 저장될 그대로 인코딩합니다.
        m = metrics(cv2.imdecode(buf, cv2.IMREAD_COLOR))
        if is_legible(m, m0, cfg):
            break
    return buf, tags, s_eff, q, params, m


def make_aug(src_df: pd.DataFrame, img_root: str | Path, out_subdir: str, n_aug: int, seed: int,
             cfg: DegradeConfig, include_original: bool = True, max_s: float | None = None,
             split: str = "", workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """src_df의 각 이미지마다 열화본을 n_aug장씩 만들어 img_root/out_subdir에 저장합니다.

    - max_s : 기본 강도 상한입니다. None이면 cfg.max_severity를 사용합니다.
    - 이미지·번호별로 고정 seed를 쓰기 때문에, 같은 seed로 다시 실행하면 같은 결과가 나옵니다.
    반환값: (COLS + source_id 컬럼 DataFrame, 열화 정보 meta DataFrame)
    """
    img_root = Path(img_root)
    if max_s is None:
        max_s = cfg.max_severity
    (img_root / out_subdir).mkdir(parents=True, exist_ok=True)

    def work(row):
        img = cv2.imread(str(img_root / row["path"]))
        if img is None:
            return None
        stem = Path(row["id"]).stem
        m0 = metrics(img)
        rows = []
        metas = [{"id": row["id"], "source_id": row["id"], "is_original": True, "split": split,
                  "synth_tags": "", "synth_severity": "", "synth_params": "", "seed": "",
                  "jpeg_quality": "", **m0}]
        for k in range(1, n_aug + 1):
            rng = np.random.default_rng([seed, zlib.crc32(row["id"].encode()), k])   # 이미지·번호별 고정 seed
            buf, tags, s, q, params, m = degrade(img, rng, cfg, max_s)
            new_id = f"{stem}_aug{k:02d}.jpg"
            rel = f"{out_subdir}/{new_id}"
            (img_root / rel).write_bytes(buf.tobytes())
            rows.append({**row, "id": new_id, "path": rel})   # 질문·보기·정답은 원본 그대로 유지합니다.
            metas.append({"id": new_id, "source_id": row["id"], "is_original": False, "split": split,
                          "synth_tags": "|".join(tags), "synth_severity": sev_label(s),
                          "synth_params": json.dumps(params, ensure_ascii=False),
                          "seed": f"{seed}-{k}", "jpeg_quality": q, **m})
        return rows, metas

    records = src_df.to_dict("records")
    with ThreadPoolExecutor(workers) as ex:
        raw = list(ex.map(work, records))

    # 이미지를 못 읽은 원본은 조용히 빠지지 않도록 보고하고, 원본 행도 함께 제외합니다.
    skipped = [rec["id"] for rec, r in zip(records, raw) if r is None]
    if skipped:
        print(f"⚠ 이미지 로드 실패로 건너뛴 원본 {len(skipped)}장 (앞 5개): {skipped[:5]}")
    results = [r for r in raw if r is not None]

    aug_rows = [r for rows, _ in results for r in rows]
    metas = [m for _, ms in results for m in ms]
    out_df = pd.DataFrame(aug_rows, columns=COLS)
    if include_original:
        keep = src_df[~src_df["id"].isin(skipped)]
        out_df = pd.concat([keep[COLS], out_df], ignore_index=True)

    out_df["source_id"] = out_df["id"].map(source_id)   # 학습 단계에서 원본 id로 증강본을 찾을 때 사용합니다.
    return out_df, pd.DataFrame(metas)


def pick_random_augmented(aug_df: pd.DataFrame, target_source_id: str, random_state=None) -> pd.Series:
    """원본 id(source_id)에 해당하는 증강본 중 하나를 무작위로 골라 반환합니다.

    학습 코드에서 다음과 같이 사용하실 수 있습니다.

        aug_df = pd.read_csv(".../aug_train.csv", dtype=str, keep_default_na=False)
        row = pick_random_augmented(aug_df, "train_0001.jpg")
        row["path"]  # -> "train_aug/train_0001_aug02.jpg"
    """
    candidates = aug_df[aug_df["source_id"] == target_source_id]
    if candidates.empty:
        raise KeyError(f"source_id가 {target_source_id}인 증강본이 없습니다.")
    return candidates.sample(1, random_state=random_state).iloc[0]
