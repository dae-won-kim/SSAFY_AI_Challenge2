"""
모델 가중치(Hugging Face 체크포인트) 보관소 : 구글 드라이브 <DRIVE_DIR>/models/<모델이름>/ ↔ VM 로컬 MODEL_ROOT/<모델이름>/

Colab VM 은 세션마다 디스크가 새것이라 모델을 매번 받아야 한다. 한 번 받은 모델을 드라이브에 두고,
새 VM 을 준비할 때(colab_pipeline.sh prep·sync → remote_setup.py) 드라이브에서 로컬 디스크로 복사해 쓴다.
  1) 로컬에 완전한 사본이 있으면 그대로
  2) 없으면 드라이브 사본을 복사 (크기 목록 .vqa_complete.json 과 맞는 완전한 사본만)
  3) 드라이브에도 없으면 model_utils.ensure_model 이 HF 에서 받고, 여기 save_to_drive 로 드라이브에 저장
     (config.MODEL_DRIVE_CACHE). 다음 VM 부터는 2) 로 끝난다
모델은 로컬 디스크에서 읽는다 (드라이브에서 바로 읽으면 느리다). 학습한 LoRA 어댑터·체크포인트는 원래 드라이브 runs/ 에 있다.

드라이브(FUSE)는 쓰기를 나중에 올리므로 작은 표시 파일이 큰 가중치보다 먼저 올라갈 수 있다. 그래서 표시 파일에 파일별 크기를
적어 두고, 가져올 때 실제 크기와 대조해 덜 올라간 사본은 쓰지 않는다 (→ HF 에서 다시 받아 덮어씀).

torch 를 쓰지 않으므로 VM 준비 단계에서도 가볍게 부른다:
  python model_store.py prefetch [프리셋 …|all]   드라이브 → 로컬 (기본 config.DEFAULT_PRESET, all = 드라이브에 있는 것 전부)
  python model_store.py push [프리셋 …]           로컬 → 드라이브 (이미 받아 둔 모델을 올릴 때)
  python model_store.py status                     프리셋별 로컬·드라이브 보관 상태
"""
import config  # noqa: F401  (경로 설정)

import os
import sys
import json
import glob
import time
import shutil
import socket

MARK = ".vqa_complete.json"      # 드라이브 사본이 완전하다는 표시 (파일별 크기). 가중치를 다 쓴 뒤 마지막에 쓴다


def model_name(preset):
    return config.MODEL_PRESETS[preset]["repo"].split("/")[-1]


def local_dir(preset):
    return os.path.join(config.MODEL_ROOT, model_name(preset))


def drive_dir(preset):
    return os.path.join(config.MODEL_DRIVE_ROOT, model_name(preset)) if config.MODEL_DRIVE_ROOT else None


def is_complete(d):
    """config.json 과 가중치가 다 있는지 (여러 조각이면 index 에 적힌 조각이 전부 있는지)"""
    if not d or not os.path.isfile(os.path.join(d, "config.json")):
        return False
    index = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index, encoding="utf-8") as f:
                shards = set(json.load(f)["weight_map"].values())
        except (OSError, ValueError, KeyError):
            return False
        return all(os.path.isfile(os.path.join(d, s)) for s in shards)
    return bool(glob.glob(os.path.join(d, "*.safetensors")))


def _files(d):
    """{상대 경로: 바이트}. 숨김 파일·폴더(HF 의 .cache/ 등)는 빼고"""
    out = {}
    for root, dirs, files in os.walk(d):
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        for f in sorted(files):
            if not f.startswith("."):
                p = os.path.join(root, f)
                out[os.path.relpath(p, d)] = os.path.getsize(p)
    return out


def drive_manifest(preset):
    """드라이브 사본의 {파일: 크기}. 표시가 없거나 실제 크기와 다르면(덜 올라감·중간에 끊김) None"""
    d = drive_dir(preset)
    if not d or not os.path.isfile(os.path.join(d, MARK)):
        return None
    try:
        with open(os.path.join(d, MARK), encoding="utf-8") as f:
            files = json.load(f)["files"]
    except (OSError, ValueError, KeyError):
        return None
    for rel, size in files.items():
        p = os.path.join(d, rel)
        if not os.path.isfile(p) or os.path.getsize(p) != size:
            return None
    return files


def _copy_files(src, dst, files, what):
    """files 를 src → dst 로 복사하며 진행 상황 출력. 끝나고 크기를 대조"""
    total, done, t0 = sum(files.values()), 0, time.time()
    for rel, size in files.items():
        os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
        with open(os.path.join(src, rel), "rb") as fi, open(os.path.join(dst, rel), "wb") as fo:
            shutil.copyfileobj(fi, fo, 16 << 20)
        if os.path.getsize(os.path.join(dst, rel)) != size:
            raise OSError(f"복사한 크기가 다름: {rel}")
        done += size
        if size > 200e6:
            el = max(time.time() - t0, 1e-6)
            print(f"  [{what}] {rel} {size / 1e9:.1f}GB | 누적 {done / 1e9:.1f}/{total / 1e9:.1f}GB "
                  f"{done / 1e6 / el:.0f}MB/s", flush=True)


def fetch_from_drive(preset):
    """드라이브 사본 → 로컬 디스크. 가져왔으면 True (로컬에 이미 있으면 True, 드라이브에 쓸 만한 사본이 없으면 False)"""
    dst = local_dir(preset)
    if is_complete(dst):
        return True
    files = drive_manifest(preset)
    if files is None:
        return False
    tmp = f"{dst}_tmp{os.getpid()}"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    t0 = time.time()
    print(f"[모델] 드라이브에서 복사: {drive_dir(preset)} → {dst} ({sum(files.values()) / 1e9:.1f}GB)", flush=True)
    try:
        _copy_files(drive_dir(preset), tmp, files, "드라이브→로컬")
        if not is_complete(tmp):
            raise OSError("복사본에 가중치가 모자람")
    except OSError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  ⚠ 드라이브 사본을 쓰지 못함 ({e}) → Hugging Face 에서 받습니다")
        return False
    if os.path.isdir(dst):                              # 받다 만 HF 폴더 등
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    print(f"[모델] 드라이브 사본 사용 ({time.time() - t0:.0f}초) → {dst}", flush=True)
    return True


def save_to_drive(preset, force=False):
    """로컬 사본 → 드라이브. 이미 완전한 사본이 있으면 건너뜀. 여러 VM 이 동시에 올려도 겹치지 않게 VM·프로세스별
    임시 폴더에 쓴 뒤 이름을 바꾼다 (먼저 끝난 쪽이 이기고 나머지는 지운다). 성공하면 True"""
    src, dst = local_dir(preset), drive_dir(preset)
    if dst is None:
        print("[모델] 드라이브가 마운트되지 않아 저장하지 않습니다")
        return False
    if not is_complete(src):
        print(f"[모델] 로컬에 완전한 모델이 없어 드라이브에 올리지 않습니다: {src}")
        return False
    if drive_manifest(preset) is not None and not force:
        return True
    files = _files(src)
    tmp = f"{dst}_tmp_{socket.gethostname()}_{os.getpid()}"
    t0 = time.time()
    print(f"[모델] 드라이브에 저장: {src} → {dst} ({sum(files.values()) / 1e9:.1f}GB, 다음 VM 부터는 여기서 복사)",
          flush=True)
    try:
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        _copy_files(src, tmp, files, "로컬→드라이브")
        with open(os.path.join(tmp, MARK), "w", encoding="utf-8") as f:
            json.dump({"repo": config.MODEL_PRESETS[preset]["repo"], "files": files,
                       "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname()}, f, indent=1)
        if drive_manifest(preset) is not None:           # 그 사이 다른 VM 이 먼저 올렸다
            shutil.rmtree(tmp, ignore_errors=True)
            print("[모델] 다른 VM 이 먼저 드라이브에 저장해서 그 사본을 둡니다")
            return True
        if os.path.isdir(dst):                           # 덜 올라간 예전 사본
            shutil.rmtree(dst)
        os.replace(tmp, dst)
    except OSError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  ⚠ 드라이브 저장 실패 ({e}) — 학습·추론은 로컬 사본으로 계속합니다 (용량 확인)")
        return False
    print(f"[모델] 드라이브 저장 완료 ({time.time() - t0:.0f}초). 업로드는 드라이브가 뒤에서 마저 한다 "
          f"(끝나기 전에 VM 이 사라지면 다음 VM 이 크기 대조로 알아채고 HF 에서 다시 받음)", flush=True)
    return True


def _presets(names):
    """명령줄 이름 → 프리셋 목록. 비우면 기본 프리셋, all = 드라이브에 완전한 사본이 있는 프리셋 전부"""
    if not names:
        return [config.DEFAULT_PRESET]
    if names == ["all"]:
        seen, out = set(), []
        for p in config.MODEL_PRESETS:                   # 같은 가중치를 쓰는 프리셋(bf16/4bit)은 한 번만
            if model_name(p) not in seen and drive_manifest(p) is not None:
                seen.add(model_name(p))
                out.append(p)
        return out
    bad = [n for n in names if n not in config.MODEL_PRESETS]
    if bad:
        raise SystemExit(f"알 수 없는 프리셋 {bad}. 후보: {list(config.MODEL_PRESETS)}")
    return names


def main(argv):
    cmd, names = (argv[0] if argv else "status"), argv[1:]
    if cmd == "prefetch":
        for p in _presets(names):
            ok = fetch_from_drive(p)
            print(f"[모델] {p} ({model_name(p)}) : " + (f"로컬 준비됨 {local_dir(p)}" if ok else
                  "드라이브에 사본 없음 → 첫 학습·추론 때 Hugging Face 에서 받아 드라이브에 저장합니다"))
    elif cmd == "push":
        for p in _presets(names):
            save_to_drive(p)
    elif cmd == "status":
        print(f"드라이브 보관소 : {config.MODEL_DRIVE_ROOT or '(드라이브 없음)'} | 로컬 : {config.MODEL_ROOT}")
        seen = set()
        for p in config.MODEL_PRESETS:
            if model_name(p) in seen:
                continue
            seen.add(model_name(p))
            files = drive_manifest(p)
            print(f"  {model_name(p):<28s} 드라이브 " + (f"{sum(files.values()) / 1e9:5.1f}GB" if files else "  없음 ")
                  + f" | 로컬 {'있음' if is_complete(local_dir(p)) else '없음'}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
