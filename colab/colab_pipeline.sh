#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 로컬 터미널(WSL Ubuntu / Linux / macOS)에서 Google Colab CLI 로 VQA 파이프라인을 돌리는 도구
#
#   ./colab/colab_pipeline.sh up [GPU]          VM 할당 + prep
#   ./colab/colab_pipeline.sh prep               드라이브 마운트 + 패키지 + 코드·데이터 준비 (세션을 colab new 로 직접 만든 경우)
#   ./colab/colab_pipeline.sh sync               로컬에서 고친 코드만 다시 올리기 (데이터는 처음 한 번만 복사)
#   ./colab/colab_pipeline.sh smoke              작은 설정으로 전체 단계 점검 (runs_smoke/ 에 저장)
#   ./colab/colab_pipeline.sh pipeline [단계]    전체 실행 (s1 학습 → s2 추론 → s3 선별 → s4g grounding·s4r 생각하기 재추론 → s5 최종). 단계를 주면 그 단계부터
#   ./colab/colab_pipeline.sh full               전 단계를 VM 안에서 한 작업으로 실행하고 바로 반환 (로컬을 꺼도 계속 진행)
#   ./colab/colab_pipeline.sh run  JOB 명령...   명령 하나를 백그라운드로 실행 (빈 GPU 에 자동 배치. GPUS=1 로 지정, FORCE=1 로 강제)
#   ./colab/colab_pipeline.sh wait JOB           끝날 때까지 기다리며 진행 상황 표시
#   ./colab/colab_pipeline.sh cancel JOB         실행 중인 작업 중단 (VM 은 유지, 1단계는 체크포인트에서 이어갈 수 있음)
#   ./colab/colab_pipeline.sh logs JOB [N]       로그 마지막 N 줄
#   ./colab/colab_pipeline.sh status             GPU 와 작업 목록
#   ./colab/colab_pipeline.sh evals [RUN…|all]   1단계의 업데이트별 valid 평가·best 현황 (학습 도중에도, 기본 RUN. N=표시 줄 수)
#   ./colab/colab_pipeline.sh progress [JOB…]    작업별 진행률 : 학습 % · test 예측 % · 지금 단계 · 제출 파일 (+ GPU 사용률)
#   ./colab/colab_pipeline.sh watch [JOB]        세션이 회수되지 않게 주기적으로 progress 출력 (WATCH_EVERY, 기본 600초)
#   ./colab/colab_pipeline.sh fetch              실험 결과를 로컬 ./results/ 로 가져오기 (FETCH_ALL=1 이면 예측 CSV 까지)
#   ./colab/colab_pipeline.sh fetch-best [RUN…|all]  best(valid 최고 업데이트) 어댑터를 PC 의 results/<실험이름>/adapter_best/ 로
#   ./colab/colab_pipeline.sh models [push|prefetch 프리셋…]  모델 가중치 드라이브 보관소(vqa/models/) 상태 / VM→드라이브 / 드라이브→VM
#   ./colab/colab_pipeline.sh console            VM 쉘 접속 (수동 점검용)
#   ./colab/colab_pipeline.sh down               VM 반납 (컴퓨팅 단위 절약)
#   ./colab/colab_pipeline.sh adopt ENDPOINT     로컬 기록이 지워진 VM(colab sessions 의 [?])을 세션으로 다시 등록 + keep-alive
#
# 설정은 환경변수로:
#   COLAB_SESSION (기본 vqa)   GPU (기본 A100)
#   RUN (기본 main = 산출물 폴더 runs/<RUN>. LoRA 랭크 등 설정을 비교할 때 바꾼다)
#   PRESET · BS(추론 배치) · EPOCHS · LORA_R : 비워 두면 config.py 의 기본값 (seed 등 나머지 실험 설정도 config.py)
#   DRIVE_DIR (기본 /content/drive/MyDrive/vqa)   DATA_ZIP (기본 data.zip, DRIVE_DIR 기준 — 원본, test 는 항상 여기)
#   TRAIN_ZIP (기본 Bigdata_v2.zip — 증강 학습 데이터. TRAIN_ZIP= 로 비우면 준비하지 않음 → config.DATA['train_source']='orig')
#   PRESETS (prep·sync 때 드라이브 vqa/models/ 에서 VM 으로 복사할 모델. 기본 PRESET → config 기본 프리셋, all = 보관소 전부)
#   GPUS (작업을 올릴 GPU 번호. 비우면 빈 GPU 자동 선택. 예: GPUS=1, GPUS=0,1)
#   POLL (wait 확인 간격 초, 기본 120)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

S=${COLAB_SESSION:-vqa}
GPU=${GPU:-A100}
RUN=${RUN:-main}
DRIVE_DIR=${DRIVE_DIR:-/content/drive/MyDrive/vqa}
DATA_ZIP=${DATA_ZIP:-data.zip}
TRAIN_ZIP=${TRAIN_ZIP-Bigdata_v2.zip}     # ":-" 가 아니라 "-" : TRAIN_ZIP= 로 일부러 비울 수 있게
POLL=${POLL:-120}
PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=$(dirname "$HERE")
EXTRA_ENV=${EXTRA_ENV:-"{}"}
# Colab CLI 의 파이썬 (cli_session.py 가 CLI 내부 기능을 씀)
CLI_PY=${CLI_PY:-$(dirname "$(readlink -f "$(command -v colab 2>/dev/null || echo /nonexistent)")")/python}

jsonargs() { "$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:], ensure_ascii=False))' "$@"; }

# remote <보조스크립트> "변수 = 값" ...  → 변수 정의를 앞에 붙여 VM 의 커널에서 실행
# 제한 시간 EXEC_TIMEOUT 초 (colab exec 기본 30초는 데이터 준비에 모자람)
# 주의: colab exec 는 원격 코드가 예외로 끝나도 종료 코드 0 → 성공 여부는 출력으로 판단한다
# VM 접속 토큰은 1시간 뒤 만료되는데 CLI 0.6 이 갱신하지 않아, 만료 후 요청이 404 → CLI 가 세션 기록과 keep-alive 를
# 지워 버린다 (VM 은 살아 있음). 그래서 원격 명령 전에 20분마다 토큰을 새로 받는다 (cli_session.py)
refresh_token() {
  local mark="$HOME/.config/colab-cli/.vqa_token_$S"
  [[ -n $(find "$mark" -mmin -20 2>/dev/null) ]] && return 0
  [[ -x $CLI_PY ]] && "$CLI_PY" "$HERE/cli_session.py" refresh "$S" < /dev/null && touch "$mark" || true
}

remote() {
  local helper=$1; shift
  refresh_token
  { for kv in "$@"; do printf '%s\n' "$kv"; done; cat "$HERE/$helper"; } \
    | colab exec -s "$S" --timeout "${EXEC_TIMEOUT:-120}"
}

code_b64() {
  "$PY" - "$CODE" <<'PYEOF'
import base64, io, os, sys, zipfile
code = sys.argv[1]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(code):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "runs", "results", ".git")]
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.join("vqa_pipeline", os.path.relpath(p, code)))
print(base64.b64encode(buf.getvalue()).decode())
PYEOF
}

cmd_sync() {
  local out
  # PRESETS : 드라이브 보관소에서 미리 가져올 모델 (비우면 PRESET, 그것도 없으면 config 기본 프리셋. all = 보관소 전부)
  out=$(EXEC_TIMEOUT=${EXEC_TIMEOUT:-2400} remote remote_setup.py "DRIVE_DIR = \"$DRIVE_DIR\"" "DATA_ZIP = \"$DATA_ZIP\"" \
        "TRAIN_ZIP = \"$TRAIN_ZIP\"" "PRESETS = \"${PRESETS:-${PRESET:-}}\"" "CODE_B64 = \"$(code_b64)\"" 2>&1 \
        | tee /dev/stderr) || true
  grep -q '^환경 →' <<<"$out" || { echo "✘ VM 준비 실패 (위 메시지 확인)"; return 1; }
}

# 드라이브가 실제로 붙었는지 확인 (drivemount 는 실패해도 0 을 돌려줄 때가 있다)
drive_mounted() {
  refresh_token
  echo "import os; print('MOUNT_OK' if os.path.isdir('/content/drive/MyDrive') else 'MOUNT_NO')" \
    | colab exec -s "$S" --timeout 120 2>&1 | grep -q MOUNT_OK
}

cmd_prep() {
  if drive_mounted; then
    echo "▶ 드라이브 이미 마운트됨"
  else
    echo "▶ 드라이브 마운트 (처음에는 브라우저 인증·권한 허용이 필요할 수 있음)"
    colab drivemount -s "$S" || true
    if ! drive_mounted; then
      echo "✘ 드라이브가 마운트되지 않았습니다."
      echo "   1) 아래 명령을 따로 실행하고, 화면에 나오는 인증·권한 요청을 끝까지 허용하세요:"
      echo "        colab drivemount -s $S"
      echo "   2) 그 다음 다시:  COLAB_SESSION=$S $0 prep"
      echo "   (계정을 바꾼 직후에는 Colab 에 드라이브 접근 권한을 새로 허용해야 합니다)"
      return 1
    fi
  fi
  echo "▶ 패키지 설치"
  colab install -s "$S" -r "$CODE/requirements.txt"
  echo "▶ 코드·데이터 준비"
  cmd_sync
}

# 이 이름의 세션이 서버에 살아 있는지 (colab sessions 의 "[이름] 엔드포인트" 줄)
session_alive() { colab sessions < /dev/null 2>/dev/null | grep -q "^\[$S\] "; }

cmd_up() {
  local gpu=${1:-$GPU}
  # 같은 이름의 세션이 이미 살아 있으면 VM 을 새로 받지 않는다. 새로 받으면 앞 VM 이 로컬 기록 없이([?]) 남아
  # 요금만 나가고 계정 동시 할당 한도(2대)도 차지한다 (2026-09-25 실제로 겪음). 새 VM 을 원하면 먼저 down
  if session_alive; then
    echo "▶ 세션 $S 가 이미 할당되어 있어 새로 받지 않고 준비(prep)만 합니다 (새 VM 이 필요하면 먼저 down)"
  else
    echo "▶ VM 할당 (세션 $S, GPU $gpu)"
    colab new -s "$S" --gpu "$gpu"
  fi
  cmd_prep
}

cmd_run() {
  local job=$1 out; shift
  out=$(remote remote_job.py "JOB = \"$job\"" "ARGS = $(jsonargs "$@")" "ENVX = $EXTRA_ENV" \
        "GPUS = \"${GPUS:-}\"" \
        "FORCE = $([[ -n ${FORCE:-} ]] && echo True || echo False)" 2>&1) || true
  printf '%s\n' "$out"
  grep -q "^\[$job\] 시작 pid=" <<<"$out" || { echo "✘ [$job] 시작하지 못했습니다 (위 메시지 확인)"; return 1; }
}

cmd_logs() { remote remote_poll.py "JOB = \"$1\"" "N = ${2:-30}"; }

# 1단계의 업데이트별 valid 평가 현황 (runs/<run>/train_log.csv · run.json). 인자가 없으면 RUN, all 이면 전부
cmd_evals() {
  (( $# )) || set -- "$RUN"
  remote remote_evals.py "RUNS = $(jsonargs "$@")" "N = ${N:-15}"
}

# 모델 가중치 드라이브 보관소 (model_store.py) : models → 상태, models push [프리셋…] → VM 로컬 모델을 드라이브에 저장,
# models prefetch [프리셋…|all] → 드라이브 → VM 로컬. HF 에서 받은 모델은 학습·추론이 알아서 드라이브에 저장한다
cmd_models() {
  (( $# )) || set -- status
  EXEC_TIMEOUT=${EXEC_TIMEOUT:-3600} remote remote_models.py "ARGS = $(jsonargs "$@")"
}

# best 어댑터(valid 최고 업데이트)를 PC 로 : VM 이 드라이브 runs/<run>/adapter/ 를 VM 로컬 results/<실험>/adapter_best/ 와
# 맞춘 뒤, colab download 로 로컬 ./results/<실험이름>/adapter_best/ 에 받는다. 이미 같은 best 면 건너뜀
cmd_fetch_best() {
  (( $# )) || set -- "$RUN"
  mkdir -p "$CODE/results"
  EXEC_TIMEOUT=900 remote remote_best.py "RUNS = $(jsonargs "$@")" 2>&1 | "$PY" "$HERE/fetch_best.py" "$CODE/results" "$S"
}

cmd_wait() {
  local job=$1 out st
  while true; do
    out=$(cmd_logs "$job" 3 2>&1 || true)
    st=$(printf '%s\n' "$out" | grep -o '__STATUS__ .*' | tail -1 | cut -d' ' -f2- || true)
    case "$st" in
      "DONE 0") echo "✔ [$job] 완료"; cmd_logs "$job" 15 | grep -v '__STATUS__'; return 0 ;;
      DONE*)    echo "✘ [$job] 실패 ($st)"; cmd_logs "$job" 60 | grep -v '__STATUS__'; return 1 ;;
      RUNNING)  echo "… [$job] $(date +%H:%M) $(printf '%s\n' "$out" | grep -v '__STATUS__' | tail -1 | cut -c1-150)" ;;
      NOTFOUND) echo "✘ [$job] 작업이 없습니다"; return 1 ;;
      *)        echo "? [$job] 상태 확인 실패 (VM 연결 확인: colab status -s $S)"; printf '%s\n' "$out" | tail -3 ;;
    esac
    sleep "$POLL"
  done
}

stage() { local job=$1; shift; cmd_run "$job" "$@" && cmd_wait "$job"; }

# ── 전체 파이프라인 : 단계 이름을 주면 그 단계부터 다시 시작 ──
# ${PRESET:+...} / ${BS:+...} / ${EPOCHS:+...} / ${LORA_R:+...} : 환경변수를 준 경우에만 인자로 넘김 (없으면 config.py 기본값)
cmd_pipeline() {
  local from=${1:-s1} started=0
  run_if() { local job=$1; shift; [[ $job == "$from" ]] && started=1; if (( started )); then stage "$job" "$@"; fi; }
  # 학습은 한 번만 (프롬프트 다양성은 학습 안에서 sampling). 추론은 sample 당 1 forward
  run_if s1   python s1_train_choice_ce.py --run "$RUN" ${PRESET:+--preset "$PRESET"} ${EPOCHS:+--epochs "$EPOCHS"} ${LORA_R:+--lora-r "$LORA_R"}
  if (( started )); then                   # 학습을 방금 끝냈으면 best 어댑터를 PC 로 (실패해도 파이프라인은 계속)
    cmd_evals "$RUN" || true
    cmd_fetch_best "$RUN" || echo "⚠ best 어댑터를 받지 못했습니다 → 나중에: $0 fetch-best $RUN"
  fi
  run_if s2   python s2_predict.py --run "$RUN" ${BS:+--batch-size "$BS"}
  run_if s3   python s3_select_rerun.py --run "$RUN"
  run_if s4g  python s4_rerun_grounding.py --run "$RUN" --mode grounding --pass2 adapter ${BS:+--batch-size "$BS"}
  run_if s4r  python s4_rerun_grounding.py --run "$RUN" --mode reason
  run_if s5   python s5_build_final.py --run "$RUN" --rerun-tag grounding_adapter reason
  (( started )) || { echo "알 수 없는 단계: $from"; return 1; }
  echo "▶ 4단계는 adapter/base 두 버전을 만들었습니다. valid 결과를 보고 다른 조합도 비교할 수 있습니다:"
  echo "   $0 run  s4b python s4_rerun_grounding.py --run $RUN --mode grounding --pass2 base   # grounding 2패스를 원래 모델로"
  echo "   $0 run  s4p python s4_rerun_grounding.py --run $RUN --mode prompt   # 선별 문항만 다중 프롬프트"
}

# ── 전 단계를 VM 안에서 한 작업으로 실행 ──
# pipeline 은 단계마다 로컬이 기다렸다 다음 단계를 띄우므로 로컬 터미널이 살아 있어야 한다.
# full 은 s1→s5 를 && 로 이어 붙여 VM 에 한 번만 던지므로, 로컬을 꺼도 VM 이 끝까지 진행한다.
# 실험별로 다른 학습 인자는 S1_ARGS 로 (예: S1_ARGS="--hint-prob 0")
# build_chain 은 sweep.sh 도 쓴다 (단계 목록을 한 곳에만 두기 위해 chain 명령으로 노출)
build_chain() {
  local run=$1 s1args=${2:-} c
  c="python s1_train_choice_ce.py --run $run${PRESET:+ --preset $PRESET}${EPOCHS:+ --epochs $EPOCHS}${LORA_R:+ --lora-r $LORA_R}${s1args:+ $s1args}"
  c="$c && python s2_predict.py --run $run${BS:+ --batch-size $BS}${S2_ARGS:+ $S2_ARGS}"
  c="$c && python s3_select_rerun.py --run $run"
  # 4단계 : 선별 문항만 grounding(확대 2패스) + reason(생각하기, thinking 모델만). 최종 단계가 valid 로 둘 중(또는 교체 안 함) 고름.
  # reason 은 thinking 이 없는 모델(Qwen3-VL)에서는 건너뛴다 (|| true 로 체인을 끊지 않음 — 최종 단계가 없는 결과는 뺌)
  c="$c && python s4_rerun_grounding.py --run $run --mode grounding --pass2 adapter${BS:+ --batch-size $BS}"
  c="$c && { python s4_rerun_grounding.py --run $run --mode reason || echo '(reason 재추론 건너뜀 — thinking 미지원 모델이거나 실패, 로그 확인)'; }"
  c="$c && python s5_build_final.py --run $run --rerun-tag grounding_adapter reason"
  printf '%s' "$c"
}

cmd_full() {
  cmd_run "${JOB:-$RUN}" bash -c "$(build_chain "$RUN" "${S1_ARGS:-}")"
  echo "▶ 진행 확인: $0 logs ${JOB:-$RUN} 40   /   RUN=$RUN $0 evals   /   $0 wait ${JOB:-$RUN}"
  echo "▶ 학습이 끝나면 best 어댑터를 PC 로: RUN=$RUN $0 fetch-best"
}

# ── 작은 설정으로 끝까지 도는지 점검 (결과는 runs_smoke/, 본 실험과 섞이지 않음) ──
cmd_smoke() {
  EXTRA_ENV="{\"VQA_RUN_ROOT\": \"$DRIVE_DIR/runs_smoke\"}"
  # --fresh : 점검은 항상 처음부터 (예전 smoke 체크포인트를 이어받아 학습·평가를 건너뛰는 일이 없게)
  local small=(--n-train 200 --max-updates 20 --eval-n 50 --eval-every 10 --ckpt-every 10 --fresh)
  stage t1 python s1_train_choice_ce.py --run smoke ${PRESET:+--preset "$PRESET"} ${LORA_R:+--lora-r "$LORA_R"} "${small[@]}"
  stage t2 python s2_predict.py --run smoke --n-train 100 --n-valid 100 --n-test 50 ${BS:+--batch-size "$BS"}
  stage t3 python s3_select_rerun.py --run smoke
  stage t4 python s4_rerun_grounding.py --run smoke --save-crops 10 ${BS:+--batch-size "$BS"}
  stage t4r python s4_rerun_grounding.py --run smoke --mode reason --reason-max-new-tokens 64   # 짧게 : 끝까지 도는지만
  stage t5 python s5_build_final.py --run smoke --rerun-tag grounding_adapter reason
  echo "▶ 점검 완료. 드라이브 $DRIVE_DIR/runs_smoke/smoke/rerun/grounding_adapter/crops/ 의 크롭 이미지를 꼭 확인하세요."
}

# ── 세션 유지 : 주기적으로 가벼운 원격 호출 ──
# 작업은 커널이 아니라 백그라운드 프로세스로 돌기 때문에, Colab 쪽에서 보면 커널이 "놀고" 있다.
# CLI 의 keep-alive 가 죽으면(토큰 1시간 만료 → 세션 정리) VM 이 유휴로 판단돼 회수된다.
# watch 는 WATCH_EVERY 초마다 토큰을 갱신하고 원격 명령을 한 번 보내 커널 활동을 만든다.
#   ./colab/colab_pipeline.sh watch <작업>     (로컬 터미널을 켜 둬야 함. nohup … & 로 띄워도 됨)
# 작업별 진행률 : GPU 사용량 한 줄 + 작업마다 "학습 N% | test 예측 N% | 지금 … | 제출 파일" (remote_progress.py)
# 인자가 없으면 실행 중인 작업 전부 (없으면 가장 최근 작업)
cmd_progress() { remote remote_progress.py "JOBS = $(jsonargs "$@")"; }

cmd_watch() {
  local job=${1:-} every=${WATCH_EVERY:-600} out
  echo "▶ [$S] ${every}초마다 세션 유지 + 진행 상황 (Ctrl-C 로 중단)"
  while true; do
    out=$(cmd_progress ${job:+"$job"} 2>&1 || true)
    if ! grep -q '사용률\|GPU 없음' <<<"$out"; then   # 진행 정보가 없으면 CLI 가 남긴 이유(세션 없음 등)를 보여 준다
      out="(응답 없음) $(grep -m1 -i 'not found\|lost\|error\|✘' <<<"$out" || echo "→ $0 status 로 확인")"
      session_alive || out="$out | 세션 $S 가 없습니다 (colab sessions 로 이름 확인. sweep 은 SESSIONS=\"…\")"
    else
      out=$(grep -v '^\[colab\]' <<<"$out")
    fi
    printf '%s [%s] %s\n' "$(date +%H:%M)" "$S" "$out"
    sleep "$every"
  done
}

cmd_fetch() {
  mkdir -p "$CODE/results"
  # 결과 폴더(@)는 실험마다 하위 폴더 하나 : results/<실험이름>/…, 재추론은 results/<실험이름>/rerun-<방식>/…
  # 예측 CSV 같은 큰 파일은 FETCH_ALL=1 일 때만 (드라이브에는 항상 있음). 하위 폴더 구조를 그대로 받는다
  # best 어댑터(adapter_best/, 수백 MB)는 fetch-best 로 따로 받는다 (점수·업데이트는 run.json·train_log.csv 에 있음)
  local pats="'@**/experiments_*.csv', '@**/run.json', '@**/predict.json', '@**/final.json', '@**/ensemble.json',
    '@**/train_log.csv', '@**/*_qtype.csv', '@**/prompt_scan.csv', '@**/select_qtype_report.csv', '@**/selection.json',
    '@**/submission.csv', '@**/final_submission.csv', 'probe_batch.json'"
  [[ -n ${FETCH_ALL:-} ]] && pats="$pats, '@**/*_pred.csv', '@**/*_rerun.csv', '@**/*_subset.csv', '@**/final_meta.csv'"
  remote remote_cat.py "PATHS = [$pats]" \
  | "$PY" -c '
import base64, os, sys
out = sys.argv[1]
for line in sys.stdin:
    p = line.split()
    if len(p) == 3 and p[0] == "__FILE__" and ".." not in p[1]:
        dst = os.path.join(out, p[1])                      # results/<실험이름>/<파일> 구조 유지
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(base64.b64decode(p[2])); print("  받음", p[1])
    elif p and p[0] == "__TOOBIG__":
        print(f"  건너뜀 {p[1]} ({int(p[2])/1e6:.1f}MB, 큼) → 드라이브 results/ 에서 직접 여세요")
    elif p and p[0] == "__MISSING__":
        print("  없음", p[1])
' "$CODE/results"
  echo "▶ $CODE/results/<실험이름>/   (드라이브 $DRIVE_DIR/results/ 에도 같은 구조)"
}

case "${1:-}" in
  up)       shift; cmd_up "$@" ;;
  prep)     cmd_prep ;;
  sync)     cmd_sync ;;
  run)      shift; cmd_run "$@" ;;
  stage)    shift; stage "$@" ;;
  wait)     shift; cmd_wait "$@" ;;
  cancel)   shift; remote remote_cancel.py "JOB = \"$1\"" ;;
  logs)     shift; cmd_logs "$@" ;;
  status)   remote remote_status.py ;;
  progress) shift; cmd_progress "$@" ;;
  pipeline) shift; cmd_pipeline "$@" ;;
  full)     cmd_full ;;
  chain)    build_chain "$RUN" "${S1_ARGS:-}"; echo ;;
  smoke)    cmd_smoke ;;
  watch)    shift; cmd_watch "$@" ;;
  evals)    shift; cmd_evals "$@" ;;
  fetch)    cmd_fetch ;;
  fetch-best) shift; cmd_fetch_best "$@" ;;
  models)   shift; cmd_models "$@" ;;
  console)  colab console -s "$S" ;;
  down)     colab stop -s "$S" ;;
  adopt)    shift; "$CLI_PY" "$HERE/cli_session.py" adopt "$S" "$1" && rm -f "$HOME/.config/colab-cli/.vqa_token_$S" ;;
  *) sed -n '2,35p' "$0" ;;
esac
