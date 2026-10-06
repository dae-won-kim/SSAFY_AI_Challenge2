"""
CPU 전용 공통 유틸 (torch 불필요).
데이터 로드, 질문 유형(qtype), 확률 요약, 예측 CSV 입출력, 크롭 좌표 계산, 제출 파일 점검, 진행 표시,
실험 결과 저장(로컬 results/ + 마운트된 드라이브).
3단계(선별)와 최종 단계는 이 모듈만 쓰므로 GPU 없이도 돌아간다.
"""
import config  # noqa: F401  (경로·결과 폴더 설정. transformers 를 끌어오지 않으므로 CPU 단계에서도 가볍다)

import os
import re
import sys
import json
import shutil
import hashlib
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

LETTERS = ["a", "b", "c", "d"]


def progress(iterable=None, **kw):
    """tqdm 진행 표시. 터미널이 아니면(Colab 로그 파일) 30초 간격으로만 갱신해 로그가 커지지 않게 한다.
    로그에 쌓인 갱신 줄은 colab_pipeline.sh logs / wait 가 마지막 상태 한 줄로 줄여 보여준다."""
    if not sys.stderr.isatty():
        kw.setdefault("mininterval", 30)
        kw.setdefault("maxinterval", 60)
        kw.setdefault("position", 0)                # 겹친 막대(학습 중 평가)가 커서 이동 코드를 로그에 남기지 않게
    return tqdm(iterable, **kw)
PROB_COLS = [f"p_{l}" for l in LETTERS]
SPLIT_DIRS = {"train": "train", "valid": "dev", "dev": "dev", "test": "test"}


# ═════════════════════════════════════════════════════════════
# 데이터 로드
# ═════════════════════════════════════════════════════════════
def take(df, n, seed):
    """n 개 무작위 추출 (원래 순서 유지). n=None 이면 전체"""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)


def resolve_images(df, split, data_root):
    """csv 의 path 를 실제 이미지 절대경로(img_path)로 변환"""
    img_dir = os.path.join(data_root, SPLIT_DIRS[split])

    def resolve(p):
        p = str(p).replace("\\", "/")
        cands = [p] if os.path.isabs(p) else []
        cands += [os.path.join(img_dir, p), os.path.join(data_root, p), os.path.join(img_dir, os.path.basename(p))]
        return next((os.path.abspath(c) for c in cands if os.path.isfile(c)), None)

    df = df.copy()
    df["img_path"] = df["path"].map(resolve)
    miss = df["img_path"].isna()
    if miss.any():
        raise FileNotFoundError(
            f"[{split}] 이미지 {miss.sum()}/{len(df)}개를 찾지 못함. 예: {df.loc[miss, 'path'].head(3).tolist()}")
    return df


def majority_vote(row):
    """answer1~5 다수결. (정답, 득표수)"""
    votes = [str(row[c]).strip().lower() for c in [f"answer{k}" for k in range(1, 6)]
             if c in row.index and pd.notna(row[c]) and str(row[c]).strip().lower() in LETTERS]
    if not votes:
        return pd.Series({"answer": None, "votes": 0})
    letter, cnt = Counter(votes).most_common(1)[0]
    return pd.Series({"answer": letter, "votes": cnt})


def load_split(data_root, split):
    """split 전체를 읽어 img_path·qtype(·dev 는 다수결 answer, votes) 을 붙여 반환"""
    name = SPLIT_DIRS[split]
    df = pd.read_csv(os.path.join(data_root, f"{name}.csv"))
    if name == "dev" and "answer" not in df.columns:
        df[["answer", "votes"]] = df.apply(majority_vote, axis=1)
    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(str).str.strip().str.lower()
    df = resolve_images(df, split, data_root)
    df["qtype"] = df["question"].map(qtype)
    return df


# ── 학습 데이터 종류·이미지 그룹 (증강 데이터를 train / valid 로 나눌 때 쓰는 단위) ──
# 증강 bigdata/train.csv 의 이미지 이름 규칙 (원본 data.zip 의 train_NNNN.jpg 도 같은 규칙으로 읽힌다)
_KIND_RE = re.compile(r"^(?:(?P<pre>orig|rot|auto)_)?(?P<split>train|dev)_(?P<num>\d+)(?:_(?P<rot>rot[A-Za-z]*\d+))?$")
KINDS = ["orig", "rot", "auto_dev", "other"]


def image_kind(path):
    """이미지 파일 이름 → (종류, 원본 사진 키). 회전본은 원본과 같은 키라서 같은 그룹이 된다
      orig_train_0001.jpg / train_0001.jpg → ("orig", "train_0001")
      rot_train_0001_rotL30.jpg            → ("rot", "train_0001")
      auto_dev_0001.jpg                    → ("auto_dev", "dev_0001")
      그 밖의 이름                          → ("other", 파일 이름) — 자기 혼자 한 그룹"""
    stem = os.path.splitext(os.path.basename(str(path).replace("\\", "/")))[0]
    m = _KIND_RE.match(stem)
    if not m:
        return "other", stem
    key = f"{m['split']}_{m['num']}"
    if m["pre"] == "rot" or m["rot"]:
        return "rot", key
    if m["pre"] == "auto" or m["split"] == "dev":
        return "auto_dev", key
    return "orig", key


def _same_bytes(paths):
    """바이트가 똑같은 이미지 파일 묶음 목록 (다른 id 로 중복된 사진). 크기가 같은 파일만 해시하므로 빠르다"""
    by_size = {}
    for p in dict.fromkeys(paths):
        by_size.setdefault(os.path.getsize(p), []).append(p)
    out = []
    for same in by_size.values():
        if len(same) < 2:
            continue
        by_hash = {}
        for p in same:
            h = hashlib.md5()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            by_hash.setdefault(h.hexdigest(), []).append(p)
        out += [g for g in by_hash.values() if len(g) > 1]
    return out


def image_groups(keys, img_paths):
    """문항별 원본 사진 키 + 이미지 경로 → 문항별 그룹 이름.
    같은 키(원본·회전본) 와 바이트가 같은 사진은 한 그룹으로 합친다. 반환 (그룹 목록, 합쳐진 중복 묶음 수)
    그룹 이름은 train_ 키를 dev_ 키보다 우선한다 → 증강(aug)·원본(orig) 어느 쪽으로 학습해도 원본 문항의 그룹 이름과
    valid 분할이 같다 (auto_dev 사진이 원본 사진과 같아 합쳐져도 이름이 바뀌지 않게)"""
    parent = {}
    rank = lambda k: (not k.startswith("train_"), k)     # noqa: E731

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    key_of = dict(zip(img_paths, keys))
    dups = _same_bytes(img_paths)
    for grp in dups:
        roots = sorted({find(key_of[p]) for p in grp}, key=rank)
        for r in roots[1:]:
            parent[r] = roots[0]
    return [find(k) for k in keys], len(dups)


def load_train_table(root):
    """학습용 train.csv 전체 + img_path · qtype · kind(종류) · group(이미지 그룹)"""
    df = pd.read_csv(os.path.join(root, "train.csv"))
    df["id"] = df["id"].astype(str)
    df["answer"] = df["answer"].astype(str).str.strip().str.lower()
    df = resolve_images(df, "train", root)
    df["qtype"] = df["question"].map(qtype)
    kinds = [image_kind(p) for p in df["path"]]
    df["kind"] = [k for k, _ in kinds]
    # 원본 문항 id 를 원본 data.zip 과 같게 (orig_train_0006.jpg → train_0006.jpg) : 증강·원본으로 학습한 실험의
    # valid·train 예측 CSV 를 id 로 바로 비교·앙상블할 수 있게
    df.loc[df["kind"] == "orig", "id"] = df.loc[df["kind"] == "orig", "id"].str.replace(r"^orig_", "", regex=True)
    if not df["id"].is_unique:
        raise ValueError(f"{root}/train.csv : id 가 겹칩니다 (orig_ 접두어를 떼면서 생긴 중복)")
    df["group"], n_dup = image_groups([g for _, g in kinds], df["img_path"].tolist())
    df.attrs["n_dup"] = n_dup
    return df


def id_hash(df):
    """valid 문항 집합의 짧은 지문 — 단계·실험마다 같은 valid 를 쓰는지 확인용.
    id 대신 (원본 사진 키, 질문)으로 만든다 → 증강(orig_train_0001)·원본(train_0001) 어느 쪽 id 든 같은 문항이면 같은 값"""
    keys = [f"{image_kind(p)[1]}|{q}" for p, q in zip(df["path"], df["question"])]
    return hashlib.md5("\n".join(sorted(keys)).encode()).hexdigest()[:12]


def holdout_groups(df, frac, seed, valid_kinds):
    """valid 로 뗄 이미지 그룹 집합. valid 후보(valid_kinds 문항이 있는 그룹)를 md5(split_seed|그룹 이름) 순으로 세워 앞 frac 만큼.
    그룹 이름만으로 정해지므로 qtype 규칙을 고치거나 csv 순서·라이브러리 버전이 바뀌어도 같은 valid 가 나온다
    (qtype 층화 추출은 규칙을 한 줄만 고쳐도 valid 절반이 바뀌어 학습 문항이 valid 로 섞여 들어가서 쓰지 않는다)"""
    groups = sorted(set(df.loc[df["kind"].isin(valid_kinds), "group"]))
    groups.sort(key=lambda g: hashlib.md5(f"{seed}|{g}".encode()).hexdigest())
    return set(groups[:int(round(len(groups) * frac))])


def load_data(data_root, n_train=None, n_valid=None, n_test=None, *, valid_min_votes, seed, train_root=None,
              data_cfg, use_valid=True):
    """train / valid / test DataFrame + 분할 정보(info)
      train_root : 학습 csv 가 있는 폴더 (증강 bigdata/ 또는 원본 data_root)
      data_cfg   : config.DATA (train_source · valid_source · valid_frac · valid_kinds · train_kinds · split_seed)
      use_valid  : False 면 valid 를 떼지 않고 학습 데이터 전부를 train 으로 (valid 는 빈 표)
    valid 분할은 n_train·n_valid 와 무관하게 전체 데이터로 정한 뒤 그 안에서 n 개를 뽑는다 → 단계마다 같은 valid.
    valid_min_votes·seed 등은 호출하는 쪽에서 config 값을 넘긴다 (기본값을 두 곳에 두지 않도록 → load_datasets)"""
    train_root = train_root or data_root
    table = load_train_table(train_root)
    train_kinds, valid_kinds = list(data_cfg["train_kinds"]), list(data_cfg["valid_kinds"])
    source = data_cfg["valid_source"] if use_valid else "none"
    info = {"train_source": data_cfg["train_source"], "train_root": train_root, "valid_source": source,
            "n_table": len(table), "n_groups": int(table["group"].nunique()), "dup_merged": table.attrs["n_dup"]}
    in_hold = pd.Series(False, index=table.index)   # valid 로 뗀 이미지 그룹의 문항
    if source == "holdout":
        in_hold = table["group"].isin(holdout_groups(table, data_cfg["valid_frac"], data_cfg["split_seed"],
                                                     valid_kinds))
        valid = table[in_hold & table["kind"].isin(valid_kinds)].sort_values("id")
        info.update(valid_frac=data_cfg["valid_frac"], split_seed=data_cfg["split_seed"], valid_kinds=valid_kinds,
                    n_hold_groups=int(valid["group"].nunique()),
                    n_dropped=int((in_hold & ~table["kind"].isin(valid_kinds)).sum()))   # valid 사진의 회전본 등
    elif source == "dev":
        dev = load_split(data_root, "dev")
        valid = dev[dev["answer"].isin(LETTERS) & (dev["votes"] >= valid_min_votes)]
        if "auto_dev" in train_kinds:               # auto_dev = dev 문항 → valid 와 겹치므로 학습에서 뺀다
            print("  ⚠ valid_source='dev' 라서 auto_dev(dev 문항 자동 라벨)는 학습에서 뺍니다")
            train_kinds = [k for k in train_kinds if k != "auto_dev"]
        # dev valid 사진과 바이트가 같은 학습 사진(과 그 회전본)이 속한 그룹도 학습에서 뺀다
        dev_paths = set(valid["img_path"])
        same = {p for grp in _same_bytes(table["img_path"].tolist() + sorted(dev_paths))
                if dev_paths.intersection(grp) for p in grp if p not in dev_paths}
        in_hold = table["group"].isin(set(table.loc[table["img_path"].isin(same), "group"]))
        info.update(valid_min_votes=valid_min_votes, n_dropped=int(in_hold.sum()))
    elif source == "none":
        valid = table.head(0)
    else:
        raise ValueError(f"DATA['valid_source'] 는 holdout | dev : {source!r}")
    train = table[~in_hold & table["kind"].isin(train_kinds)]
    info.update(train_kinds=train_kinds, n_excluded_kind=int((~in_hold & ~table["kind"].isin(train_kinds)).sum()),
                n_valid_full=len(valid), valid_hash=id_hash(valid) if len(valid) else None)

    train = take(train, n_train, seed)
    valid = take(valid, n_valid, seed)
    test = take(load_split(data_root, "test"), n_test, seed)
    info.update(n_train=len(train), n_valid=len(valid), kinds_train=train["kind"].value_counts().to_dict(),
                kinds_valid=valid["kind"].value_counts().to_dict() if "kind" in valid else {},
                train_hash=id_hash(train) if len(train) else None)   # 이어하기 때 같은 학습 데이터인지 확인용
    return train, valid, test, info


def load_datasets(n_train=None, n_valid=None, n_test=None, use_valid=None, train_source=None, train_kinds=None):
    """config 설정대로 load_data (1·2·4단계 공통). 인자를 주면 그 값이 우선 (1단계 --train-source 등,
    2·4단계는 1단계 run.json 에 기록된 값을 넘겨 같은 분할을 쓴다). use_valid 기본값 = config.USE_VALID"""
    cfg = dict(config.DATA)
    if train_source:
        cfg["train_source"] = train_source
    if train_kinds:
        cfg["train_kinds"] = list(train_kinds)
    return load_data(config.DATA_ROOT, n_train, n_valid, n_test, valid_min_votes=config.VALID_MIN_VOTES,
                     seed=config.SEED, train_root=config.train_root(cfg["train_source"]), data_cfg=cfg,
                     use_valid=config.USE_VALID if use_valid is None else use_valid)


TRAIN_META = "train_meta.json"      # 1단계가 저장하는 모든 어댑터 폴더에 함께 두는 표식 (무슨 데이터로 학습했는지)


def train_meta(data_info, run_id):
    """어댑터 폴더에 둘 학습 데이터 표식. 2·4단계가 run.json 이 아니라 어댑터 자체를 보고 같은 분할을 고른다
    (--fresh 뒤에 남은 예전 adapter_e2/ 처럼 run.json 과 어긋난 어댑터도 제대로 가려진다)"""
    return {"run_id": run_id, **{k: data_info.get(k) for k in
                                 ("use_valid", "train_source", "train_kinds", "valid_source", "valid_hash")}}


def trained_data(adapter, meta=None):
    """2·4단계 : 어댑터를 학습한 데이터 설정 → (load_datasets 인자 + valid_hash, 경고 문구)
      표식(train_meta.json) 있음 → 그 값
      표식 없음 + run.json 있음 → 이 변경 전 코드의 어댑터 = 원본 train 전체 학습 → valid 금지
      둘 다 없음 → 모름 (경고)"""
    if adapter is None:                             # zero-shot : 학습 데이터가 없으니 config 그대로
        return {}, None
    mark = load_json(os.path.join(adapter, TRAIN_META))
    if mark:
        return mark, None
    if meta:
        return ({"use_valid": False, "train_source": "orig"},
                "이 어댑터는 예전 코드로 원본 train 전체를 학습했습니다(어댑터에 train_meta.json 없음) → holdout valid 도 "
                "학습한 문항이라 valid 는 건너뜁니다")
    return {}, "어댑터에 train_meta.json 도, run.json 도 없어 어떤 데이터로 학습했는지 모릅니다 → valid 는 건너뜁니다"


def describe_split(info):
    """분할 정보 한 줄 요약 (로그용)"""
    kinds = lambda d: " / ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda x: KINDS.index(x[0])  # noqa: E731
                                                             if x[0] in KINDS else 9))
    s = f"학습 데이터 {info['train_source']}({info['n_table']}문항, 이미지 그룹 {info['n_groups']}"
    s += f", 바이트 중복 {info['dup_merged']}묶음 병합)" if info.get("dup_merged") else ")"
    s += f" → train {info['n_train']} ({kinds(info['kinds_train'])})"
    if info["valid_source"] == "holdout":
        s += (f" | valid {info['n_valid']} (그룹 {info['valid_frac']:.0%} holdout, {'+'.join(info['valid_kinds'])} 만, "
              f"split_seed {info['split_seed']}, id {info['valid_hash']}) | valid 그룹이라 학습에서 뺀 문항 {info['n_dropped']}")
    elif info["valid_source"] == "dev":
        s += (f" | valid {info['n_valid']} (dev {info['valid_min_votes']}표 이상 다수결, id {info['valid_hash']}) "
              f"| dev 사진과 같은 사진이라 학습에서 뺀 문항 {info['n_dropped']}")
    else:
        s += " | valid 없음 (전체 학습)"
    if info.get("n_excluded_kind"):
        s += f" | 종류 제외 {info['n_excluded_kind']} (train_kinds={info['train_kinds']})"
    return s


# ═════════════════════════════════════════════════════════════
# 질문 유형 (qtype) — 오답 분석과 재추론 선별의 축. dev 오답을 보며 계속 다듬을 것
# ═════════════════════════════════════════════════════════════
# 위에서부터 먼저 걸리는 규칙 하나가 유형이 된다 (겹치면 앞 규칙이 이김).
# 주의점 (실데이터로 확인) :
#   · 한글은 \w 라서 \b위치 는 "샌드위치"·"닌텐도 스위치" 안의 '위치'에는 걸리지 않고 "위치한/위치는" 에만 걸린다
#   · 활용형은 어간으로 잡는다 : 적힌/적혀 → 적[힌혀], 표시된/표시되어 → 표시[된되]
QTYPE_RULES = [
    ("부정형",    re.compile(r"(없는|아닌|틀린|않은|잘못된|다른)\s*(것|말|곳|사람|메뉴|내용|설명|항목)")),
    ("개수",      re.compile(r"(몇\s*(개|명|마리|대|권|장|잔|곳|가지|병|봉|팩|칸|회)|개수|수량)")),
    ("숫자·가격", re.compile(r"(가격|얼마|금액|요금|숫자|번호|전화|연락처|날짜|연도|시간|km|kcal|칼로리|용량|함량|무게|중량|퍼센트|%"
                            r"|몇\s*(시|층|번|미터|m|분|초|월|일|년|주년|배|도|살|세|kg|g|ml|mm|cm))")),
    ("위치·방향", re.compile(r"(어디|\b위치|방향|왼쪽|오른쪽|위쪽|아래쪽|쪽으로|어느\s*(곳|쪽))")),
    ("문구읽기",  re.compile(r"(적[힌혀]|쓰[여인]|써\s*있|문구|글자|이름|상호|제목|표시[된되]|표기|새겨|주소|라고)")),
]
QTYPES = [name for name, _ in QTYPE_RULES] + ["기타"]


def qtype(question):
    q = str(question)
    for name, pat in QTYPE_RULES:
        if pat.search(q):
            return name
    return "기타"


# ═════════════════════════════════════════════════════════════
# 확률 → 예측 요약, 예측 CSV
# ═════════════════════════════════════════════════════════════
def summarize(P):
    """(N,4) 확률 → 예측 글자, 확신도(top1), margin(top1 - top2)"""
    P = np.asarray(P, dtype=float)
    srt = np.sort(P, axis=1)
    return np.array(LETTERS)[P.argmax(1)], srt[:, -1], srt[:, -1] - srt[:, -2]


def make_pred_df(rows, P, **extra):
    """예측 CSV 의 표준 형식: id, qtype, gold, p_a..p_d, pred, conf, margin, (추가 컬럼)"""
    pred, conf, margin = summarize(P)
    out = pd.DataFrame({
        "id": rows["id"].values,
        "qtype": rows["qtype"].values if "qtype" in rows else [qtype(q) for q in rows["question"]],
        "gold": rows["answer"].values if "answer" in rows else "",
    })
    for i, c in enumerate(PROB_COLS):
        out[c] = np.asarray(P)[:, i]
    out["pred"], out["conf"], out["margin"] = pred, conf, margin
    for k, v in extra.items():
        out[k] = v
    return out


def load_pred(path):
    df = pd.read_csv(path, keep_default_na=False)
    df["id"] = df["id"].astype(str)
    return df


def probs(df):
    return df[PROB_COLS].to_numpy(dtype=float)


def accuracy(df, pred_col="pred"):
    m = df["gold"].astype(str).isin(LETTERS)
    return float((df.loc[m, pred_col] == df.loc[m, "gold"]).mean()) if m.any() else float("nan")


def mean_nll(P, answers):
    """정답 선지 확률의 평균 음의 로그 (valid 손실). 정확도가 같은 체크포인트끼리 가를 때 쓴다"""
    P, ans = np.asarray(P, dtype=float), [str(a) for a in answers]
    rows = [i for i, a in enumerate(ans) if a in LETTERS]
    if not rows:
        return float("nan")
    p = P[rows, [LETTERS.index(ans[i]) for i in rows]]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())


def qtype_table(df, pred_cols=("pred",)):
    """qtype 별 n, 정확도(여러 예측 컬럼), 평균 확신도"""
    rows = []
    for qt in QTYPES + ["전체"]:
        sub = df if qt == "전체" else df[df["qtype"] == qt]
        if len(sub) == 0:
            continue
        rec = {"qtype": qt, "n": len(sub), "conf_mean": round(sub["conf"].mean(), 3)}
        for c in pred_cols:
            rec[f"acc_{c}"] = round(accuracy(sub, c), 3)
        rows.append(rec)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════
# grounding bbox → 크롭 영역 (원본 좌표계)
# ═════════════════════════════════════════════════════════════
_BOX_RE = re.compile(r'bbox_2d"?\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]')
_ANY4_RE = re.compile(r'\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]')


def parse_boxes(text):
    found = _BOX_RE.findall(text) or _ANY4_RE.findall(text)
    return [[float(v) for v in m] for m in found]


def to_original(box, resized_wh, orig_wh, coord_mode):
    """모델 좌표 → 원본 이미지 좌표. 범위 밖은 자르고, 퇴화한 박스는 None"""
    rw, rh = resized_wh
    W, H = orig_wh
    sx, sy = (W / 1000, H / 1000) if coord_mode == "norm1000" else (W / rw, H / rh)
    x1, x2 = sorted((box[0] * sx, box[2] * sx))
    y1, y2 = sorted((box[1] * sy, box[3] * sy))
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def crop_region(boxes, W, H, pad=0.15, min_frac=0.15, max_area=0.70):
    """bbox 목록 → 최종 크롭 (x1,y1,x2,y2). 크롭할 가치가 없으면 None
    - 박스 여러 개면 합집합, 합집합이 너무 크면 첫 박스
    - 각 변에 pad 비율 여백, 최소 변 길이 = 원본의 min_frac
    - 가장자리에 걸리면 잘라내지 않고 안쪽으로 밀어 크기 유지
    - 결과가 원본의 max_area 이상이면 확대 효과가 없으므로 None"""
    if not boxes:
        return None
    u = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    box = u if (u[2] - u[0]) * (u[3] - u[1]) <= max_area * W * H else boxes[0]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
    if x2 - x1 < W * min_frac:
        c = (x1 + x2) / 2; x1, x2 = c - W * min_frac / 2, c + W * min_frac / 2
    if y2 - y1 < H * min_frac:
        c = (y1 + y2) / 2; y1, y2 = c - H * min_frac / 2, c + H * min_frac / 2
    if x1 < 0: x2, x1 = x2 - x1, 0
    if y1 < 0: y2, y1 = y2 - y1, 0
    if x2 > W: x1, x2 = x1 - (x2 - W), W
    if y2 > H: y1, y2 = y1 - (y2 - H), H
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(np.ceil(x2))), min(H, int(np.ceil(y2)))
    if (x2 - x1) * (y2 - y1) > max_area * W * H:
        return None
    return (x1, y1, x2, y2)


# ═════════════════════════════════════════════════════════════
# 제출 파일, 기록
# ═════════════════════════════════════════════════════════════
def write_submission(ids, answers, data_root, out_path):
    """sample_submission.csv 가 있으면 그 id 순서·컬럼에 맞춰 저장하고 점검"""
    sub = pd.DataFrame({"id": pd.Series(ids).astype(str).values, "answer": list(answers)})
    sample_path = os.path.join(data_root, "sample_submission.csv")
    if os.path.isfile(sample_path):
        sample = pd.read_csv(sample_path)
        id_col, ans_col = sample.columns[0], sample.columns[1]
        sample[id_col] = sample[id_col].astype(str)
        missing = set(sample[id_col]) - set(sub["id"])
        if missing:
            print(f"  ⚠ sample_submission 의 id {len(missing)}개가 예측에 없음 (test 일부만 예측했다면 정상)")
        merged = sample[[id_col]].merge(sub.rename(columns={"id": id_col, "answer": ans_col}), on=id_col, how="inner")
        if not missing:
            assert len(merged) == len(sample) and (merged[id_col].values == sample[id_col].values).all(), \
                "sample_submission 과 id 순서·개수가 다름"
        sub = merged
    ans = sub.iloc[:, 1]
    assert ans.isin(LETTERS).all() and ans.notna().all(), "답이 a/b/c/d 가 아닌 행이 있음"
    assert sub.iloc[:, 0].is_unique, "id 중복"
    dist = ans.value_counts(normalize=True).reindex(LETTERS, fill_value=0).round(3).to_dict()
    print(f"  예측 분포 {dist}" + ("  ⚠ 한 선지에 50% 이상 쏠림" if max(dist.values()) > 0.5 else ""))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"  저장 {out_path} ({len(sub)}행)")
    return sub


def _append_csv(path, rec):
    new = pd.DataFrame([rec])
    if os.path.exists(path):
        try:
            new = pd.concat([pd.read_csv(path), new], ignore_index=True)   # 컬럼이 늘어도 안전하게 합침
        except Exception as e:                              # noqa: BLE001  (드라이브 사본이 깨져도 본 기록은 남긴다)
            print(f"  ⚠ 기존 기록을 읽지 못해 새로 씀: {path} ({type(e).__name__})")
    new.to_csv(path, index=False, encoding="utf-8-sig")


def result_relpath(filename):
    """결과 파일 이름 → 결과 폴더 안의 상대 경로. `__` 가 폴더 경계다.
      "r16_qwen3-vl-4b_r16_e1_lr1e-4_hint30_s42__test_pred.csv"
          → r16_qwen3-vl-4b_r16_e1_lr1e-4_hint30_s42/test_pred.csv
      "<실험이름>__rerun-grounding-adapter__final_submission.csv"
          → <실험이름>/rerun-grounding-adapter/final_submission.csv
    즉 하이퍼파라미터 조합(실험 이름)마다 하위 폴더 하나에 그 실험의 파일이 다 모인다"""
    parts = [p for p in str(filename).split("__") if p]
    return os.path.join(*parts) if parts else str(filename)


def append_log(run_root, record):
    """단계별 주요 수치를 experiments_<stage>.csv 에 누적 (단계마다 컬럼이 달라 파일을 나눔).
      runs/                : experiments_<stage>__<run>.csv        (작업 폴더, 예전과 같음)
      results/<실험이름>/  : experiments_<stage>.csv                (실험별 하위 폴더, 드라이브에도 동일)
    실험별로 파일을 나누는 이유 : 여러 VM 이 같은 드라이브에 동시에 쓰면 한 파일을 읽고-고쳐-쓰다 행이 사라진다.
    비교용 한 장짜리 표는 나중에 `python collect_results.py` 가 하위 폴더를 훑어 합친다"""
    stage = record.get("stage", "misc")
    key = str(record.get("exp_tag") or record.get("run") or "misc")     # 결과 하위 폴더 (rerun 은 한 단계 더)
    run = config._fmt(record.get("run") or key)
    rec = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **{k: str(v) if isinstance(v, (list, dict)) else v
                                                                    for k, v in record.items()}}
    targets = [os.path.join(run_root, f"experiments_{stage}__{run}.csv")]
    targets += [os.path.join(d, result_relpath(f"{key}__experiments_{stage}.csv")) for d in config.result_dirs()]
    for path in targets:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _append_csv(path, rec)
        except OSError as e:
            print(f"  ⚠ 실험 기록 저장 실패: {path} ({e})")


def save_result(obj, filename, *, index=False, quiet=False):
    """실험 결과 사본을 결과 폴더 모두(로컬 results/ + 마운트된 드라이브)에 저장.
    obj : DataFrame | dict/list(json) | str(텍스트) | 원본 파일 경로
    filename 은 "<실험이름>__<파일>" 꼴. `__` 마다 하위 폴더로 나뉘어 results/<실험이름>/<파일> 에 놓인다 (result_relpath)"""
    saved = []
    for d in config.result_dirs():
        path = os.path.join(d, result_relpath(filename))
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(path, index=index, encoding="utf-8-sig")
            elif isinstance(obj, (dict, list)):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
            elif isinstance(obj, str) and os.path.isfile(obj):
                shutil.copyfile(obj, path)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(str(obj))
            saved.append(path)
        except OSError as e:                                # 드라이브가 잠깐 끊겨도 로컬 사본은 남긴다
            print(f"  ⚠ 결과 저장 실패: {path} ({e})")
    if saved and not quiet:
        print(f"  결과 → {saved[0]}" + (f" (+ 드라이브 {len(saved) - 1}곳)" if len(saved) > 1 else ""))
    return saved


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path, default=None):
    """깨진 파일(드라이브에 쓰다 끊긴 캐시 등)은 기본값으로 넘어간다 — 읽기 때문에 작업이 죽지 않게"""
    if not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        print(f"  ⚠ 읽지 못해 무시함: {path} ({type(e).__name__}: {str(e)[:80]})")
        return default
