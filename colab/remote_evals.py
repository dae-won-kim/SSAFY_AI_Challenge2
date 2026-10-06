# ─────────────────────────────────────────────────────────────
# Colab VM 에서 실행됨 (앞에 RUNS, N 을 붙여서 보냄). 1단계의 업데이트별 valid 평가 현황을 보여준다.
# 1단계는 평가할 때마다 runs/<run>/train_log.csv · run.json 을 갱신하므로 학습 도중에도 볼 수 있다.
#   RUNS = ["main"]  (["all"] 이면 run.json 이 있는 run 전부, 최근 것부터)
# 이 파일은 호출할 때마다 로컬에서 새로 보내므로 고치면 sync 없이 바로 적용된다.
# ─────────────────────────────────────────────────────────────
import os, csv, glob, json, time

BASE = os.environ.get("COLAB_BASE", "/content")
run_root = json.load(open(f"{BASE}/vqa_env.json"))["VQA_RUN_ROOT"]
n_show = int(globals().get("N", 15))


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def num(v, fmt):
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "-"


def stages(d, meta):
    """학습 → 예측 → 선별 → 재추론 → 제출 파일까지 어디까지 됐는지 (full·sweep 작업은 이 순서로 한 번에 돈다).
    1단계가 마지막으로 쓴 run.json 보다 오래된 파일은 같은 run 이름의 예전 실행 것이라 '예전' 으로 표시"""
    t_run = os.path.getmtime(os.path.join(d, "run.json")) - 2

    def mark(rel):
        p = os.path.join(d, rel)
        if not os.path.isfile(p):
            return "…"
        return "✔" if os.path.getmtime(p) >= t_run else "예전"

    def when(rel):
        p = os.path.join(d, rel)
        if mark(rel) != "✔":
            return "아직" if mark(rel) == "…" else "예전 실행의 파일"
        m = (time.time() - os.path.getmtime(p)) / 60
        return f"있음, {m:.0f}분 전" if m < 120 else f"있음, {m / 60:.1f}시간 전"

    trained = meta.get("update") is not None and meta.get("update") == meta.get("total_updates")
    extra = f" train {mark('train_pred.csv')}" if mark("train_pred.csv") == "✔" else ""   # --splits … train 로 돌렸을 때만
    print(f"   단계: 1 학습 {'✔' if trained else '…'} → 2 예측 valid {mark('valid_pred.csv')} test {mark('test_pred.csv')}"
          f"{extra} → 3 선별 {mark('select/test_subset.csv')} → 4 재추론 "
          f"grounding {mark('rerun/grounding_adapter/test_rerun.csv')} reason {mark('rerun/reason/test_rerun.csv')} "
          f"→ 최종 {mark('final/final_submission.csv')}   (… = 아직)")
    print(f"   제출 파일: submission.csv (2단계, {when('submission.csv')}) | "
          f"final/final_submission.csv (재추론 반영, {when('final/final_submission.csv')})  → PC 로: fetch")


runs = list(RUNS)
if runs == ["all"]:
    metas = glob.glob(os.path.join(run_root, "*", "run.json"))
    runs = [os.path.basename(os.path.dirname(p)) for p in sorted(metas, key=os.path.getmtime, reverse=True)]
for run in runs:
    d = os.path.join(run_root, run)
    meta = read_json(os.path.join(d, "run.json"))
    if not meta:
        print(f"── {run} : run.json 이 없습니다 (1단계가 아직 첫 평가 전이거나 run 이름이 다름)\n")
        continue
    rows = []
    if os.path.isfile(os.path.join(d, "train_log.csv")):
        with open(os.path.join(d, "train_log.csv"), encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("valid_acc") not in (None, "")]
    best = read_json(os.path.join(d, "adapter", "best.json"))
    data = meta.get("data") or {}
    print(f"── {run}  [{meta.get('exp_tag')}]")
    print(f"   데이터 {data.get('train_source', '?')} train {data.get('n_train', '?')} / valid {meta.get('eval_n', '?')}문항 "
          f"({data.get('valid_source', '?')}, id {data.get('valid_hash')}) | 평가 {meta.get('eval_every', '?')}업데이트마다")
    print(f"   진행 upd {meta.get('update')}/{meta.get('total_updates', '?')} (마지막 평가 시점) | 평가 {len(rows)}회 "
          f"| best upd {meta.get('best_update')} (epoch {meta.get('best_epoch')}) "
          f"acc {num(meta.get('best_valid_acc'), '.4f')} nll {num(meta.get('best_valid_nll'), '.4f')}"
          + (f" | adapter/ 저장 {best.get('saved_at')}" if best else ""))
    if rows:
        print(f"   {'upd':>6} {'epoch':>5} {'loss':>8} {'valid_acc':>9} {'nll':>7}")
        if len(rows) > n_show:
            print(f"   … 앞 {len(rows) - n_show}회 생략 (N 으로 조절)")
        for r in rows[-n_show:]:
            mark = "  ▲ best" if str(r.get("best")).lower() == "true" else ""
            print(f"   {r.get('update', ''):>6} {r.get('epoch', ''):>5} {num(r.get('loss'), '.4f'):>8} "
                  f"{num(r.get('valid_acc'), '.4f'):>9} {num(r.get('valid_nll'), '.4f'):>7}{mark}")
    stages(d, meta)
    print()
