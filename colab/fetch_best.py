# ─────────────────────────────────────────────────────────────
# 로컬(PC)에서 실행됨 : colab_pipeline.sh fetch-best 가 remote_best.py 의 출력을 넘겨 준다 (표준 입력).
# 각 run 의 best 어댑터를 colab download 로 받아 results/<실험이름>/adapter_best/ 에 둔다.
# 이미 받은 파일과 md5 가 모두 같으면 건너뛰고(어댑터 파일은 수백 MB 라 매번 받지 않게), 받은 뒤에도 md5 로 확인한다.
# 임시 폴더에 다 받은 다음에만 adapter_best/ 와 바꿔 끼우므로, 받다 끊겨도 예전 것은 온전하다.
#   python3 colab/fetch_best.py <results 폴더> <세션 이름>  < remote_best 출력
# ─────────────────────────────────────────────────────────────
import os, sys, json, base64, shutil, hashlib, subprocess

out_root, session = sys.argv[1], sys.argv[2]
runs, order = {}, []
for line in sys.stdin:
    p = line.split()
    if len(p) == 4 and p[0] == "__BEST__":
        runs[p[1]] = {"tag": p[2], "best": json.loads(base64.b64decode(p[3]).decode()), "files": []}
        order.append(p[1])
    elif len(p) == 5 and p[0] == "__FILE__" and p[1] in runs:
        runs[p[1]]["files"].append((p[2], int(p[3]), p[4]))
    elif len(p) == 3 and p[0] == "__END__" and p[1] in runs:
        runs[p[1]]["end"] = int(p[2])
    elif p and p[0] == "__NOBEST__":
        print(f"  [{p[1]}] 건너뜀 : {' '.join(p[2:]).replace('_', ' ')}")
    elif line.strip() and not line.startswith("__"):
        print("  " + line.rstrip())                       # 원격 오류 메시지 등은 그대로 보여준다


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def num(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


failed = 0
for run in order:
    r = runs[run]
    if ".." in r["tag"] or "/" in r["tag"]:
        print(f"  [{run}] 이상한 실험 이름이라 건너뜀: {r['tag']}")
        failed += 1
        continue
    names = {os.path.basename(rp) for rp, _, _ in r["files"]}
    if r.get("end") != len(r["files"]) or not {"adapter_model.safetensors", "adapter_config.json"} <= names:
        print(f"  [{run}] ✘ 원격 파일 목록이 중간에 끊겼거나 어댑터 파일이 없습니다 (받은 {sorted(names)}) → 건너뜀. "
              f"잠시 뒤 다시 fetch-best (학습이 어댑터를 저장하는 중이었을 수 있음)")
        failed += 1
        continue
    dst = os.path.join(out_root, r["tag"], "adapter_best")
    best = r["best"]
    desc = (f"upd {best.get('update')}/{best.get('total_updates')} valid acc {num(best.get('valid_acc'))}"
            if best else "best.json 없음(예전 방식 어댑터)")
    if r["files"] and all(os.path.isfile(os.path.join(dst, os.path.basename(rp)))
                          and md5(os.path.join(dst, os.path.basename(rp))) == h for rp, _, h in r["files"]):
        print(f"  [{run}] 이미 최신 → {dst} ({desc})")
        continue
    tmp = dst + "_tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    ok = True
    for rp, size, h in r["files"]:
        local = os.path.join(tmp, os.path.basename(rp))
        print(f"  [{run}] 받는 중 {os.path.basename(rp)} ({size / 1e6:.1f}MB) …", flush=True)
        res = subprocess.run(["colab", "download", "-s", session, rp, local], capture_output=True, text=True)
        if res.returncode != 0 or not os.path.isfile(local) or md5(local) != h:
            print(f"  [{run}] ✘ 받기 실패 {rp}: {(res.stdout + res.stderr).strip()[-300:] or 'md5 불일치'}")
            ok = False
            break
    if not ok:
        shutil.rmtree(tmp, ignore_errors=True)
        failed += 1
        continue
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    print(f"  [{run}] ✔ {dst} ({desc})")
sys.exit(1 if failed or not order else 0)
