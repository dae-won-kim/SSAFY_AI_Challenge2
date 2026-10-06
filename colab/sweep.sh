#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 실험 여러 개를 VM 몇 대에 나눠 돌린다. 한 VM = GPU 1장 = 한 번에 실험 1개.
# 계정의 동시 할당 한도(TooManyAssignmentsError)가 있으므로 VM 수는 MAX_PARALLEL 로 제한하고,
# 실험이 더 많으면 **한 VM 이 자기 몫을 순서대로 이어서** 돌린다 (VM 안에서 이어지므로 로컬은 꺼도 된다).
#
#   ./colab/sweep.sh plan          어떤 VM 이 어떤 실험을 어떤 순서로 돌릴지 미리 보기 (chain [번호] = 그 VM 이 돌릴 명령 전체)
#   ./colab/sweep.sh up [번호]     VM 할당 + 준비 (순차, 대당 5~10분). 번호를 주면 그 VM 만
#   ./colab/sweep.sh start         VM 마다 자기 몫 전부를 한 작업으로 제출하고 즉시 반환
#   ./colab/sweep.sh status        VM 별 GPU·작업 상태와 마지막 로그 줄
#   ./colab/sweep.sh watch         VM 이 회수되지 않게 계속 지켜보기 (실행 중에는 켜 둘 것)
#   ./colab/sweep.sh logs [N] [번호]  로그 마지막 N 줄 (기본 20)
#   ./colab/sweep.sh evals [번호]  실험별 업데이트 단위 valid 평가·best 현황 (학습 도중에도)
#   ./colab/sweep.sh fetch         결과 회수 + 비교표(summary.csv)
#   ./colab/sweep.sh fetch-best [번호]  실험별 best 어댑터를 PC 의 results/<실험이름>/adapter_best/ 로
#   ./colab/sweep.sh cancel [번호]    실행 중인 작업 중단 (VM 유지)
#   ./colab/sweep.sh down [번호]      VM 반납
#
# 실험 목록은 아래 EXPS 또는 colab/sweep.conf ("실험이름 | s1 추가 인자" 한 줄에 하나).
# 실험 이름이 곧 RUN 이름(runs/<이름>/)이자 결과 파일 이름의 일부다 → 서로 달라야 한다.
#
# 이 프로젝트의 기본 실험 방식이다 : 실험은 항상 VM 2대(A100 1장씩)로 병렬 (Colab 은 VM 당 GPU 1장만 받을 수 있음)
#
# 설정: MAX_PARALLEL(기본 2 = 동시에 쓸 VM 수) SESSION_PREFIX(기본 vqa → vqa1, vqa2) GPU(기본 A100) PRESET BS EPOCHS
#       SESSIONS="vqa vqa2" (슬롯별 세션 이름을 직접 지정 — 이미 켜 둔 vqa 를 슬롯 1 로 재사용할 때)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=$(dirname "$HERE")
PIPE="$HERE/colab_pipeline.sh"
PREFIX=${SESSION_PREFIX:-vqa}
SLOTS=${MAX_PARALLEL:-2}                  # 계정 동시 할당 한도에 맞춘 VM 수
export GPU=${GPU:-A100}

# 실험 목록 : 한 번에 하나씩만 다르게 (해석 가능한 비교)
EXPS=(
  "r16    |"                      # 기준선 (config.py 기본값 : LoRA r=16, hint 30%)
  "r32    | --lora-r 32"          # 랭크만 ↑
  "hint0  | --hint-prob 0"        # qtype hint 없이
  "hint50 | --hint-prob 0.5"      # hint 비율 ↑
)
if [[ -f "$HERE/sweep.conf" ]]; then      # 파일이 있으면 그걸 씀 (# 주석, 빈 줄 무시)
  mapfile -t EXPS < <(grep -v '^\s*#' "$HERE/sweep.conf" | grep -v '^\s*$')
fi

trim() { local s=$1; s="${s#"${s%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"; }
name_of() { trim "${1%%|*}"; }            # "r32 | --lora-r 32" → r32
args_of() { [[ $1 == *"|"* ]] && trim "${1#*|}" || true; }

# 실험 i(0부터) → 슬롯 (i % SLOTS) + 1. 슬롯 1 = vqa1 = 실험 1,3,5… 을 순서대로
slot_exps() { local i=0 e; for e in "${EXPS[@]}"; do (( i % SLOTS + 1 == $1 )) && echo "$e"; i=$((i + 1)); done; }
slot_runs() { slot_exps "$1" | while read -r e; do name_of "$e"; echo; done; }   # 슬롯의 실험(run) 이름들
used_slots() { local n=${#EXPS[@]}; echo $(( n < SLOTS ? n : SLOTS )); }
# 세션 이름 : SESSIONS="vqa vqa2" 처럼 주면 그 순서대로 슬롯 1, 2 … (이미 켜 둔 VM 을 슬롯으로 재사용).
# 한 번 주면 colab/.sweep_sessions 에 기억해 두고, 이후 sweep 명령(watch·status·evals …)은 SESSIONS 없이도 같은 VM 을 본다
# (전부 down 하면 지운다). 없으면 vqa1, vqa2 … (up 은 살아 있는 세션이면 새로 받지 않고 준비만 한다)
SESS_FILE="$HERE/.sweep_sessions"
if [[ -n ${SESSIONS:-} ]]; then
  printf '%s\n' "$SESSIONS" > "$SESS_FILE"
elif [[ -f $SESS_FILE ]]; then
  SESSIONS=$(cat "$SESS_FILE")
fi
read -r -a SESSION_NAMES <<< "${SESSIONS:-}"
session_of() { local n=${SESSION_NAMES[$(( $1 - 1 ))]:-}; echo "${n:-${PREFIX}$1}"; }
job_of() { echo "slot$1"; }

# 슬롯 하나가 VM 에서 돌릴 전체 명령 : 실험들을 ; 로 이어 붙인다
# (&& 가 아니라 ; → 앞 실험이 실패해도 다음 실험은 돈다. 단계 목록 자체는 colab_pipeline.sh build_chain 한 곳에만)
slot_chain() {
  local out="fail=0" exp run args one
  while read -r exp; do
    [[ -z $exp ]] && continue
    run=$(name_of "$exp"); args=$(args_of "$exp")
    one=$(RUN="$run" S1_ARGS="$args" "$PIPE" chain)
    # 한 실험이 실패해도 다음 실험은 돌리고, 대신 작업 전체의 종료 코드를 1 로 남긴다 (wait 가 실패로 보고)
    out="$out; echo \"===== 실험 $run 시작 \$(date +%F\\ %T) =====\"; { $one; }"
    out="$out && echo \"===== 실험 $run 완료 =====\" || { echo \"===== 실험 $run 실패 (위 로그 확인) =====\"; fail=1; }"
  done < <(slot_exps "$1")
  printf '%s; exit $fail' "$out"
}

each_slot() {                             # $@ = 함수. ONLY 가 있으면 그 슬롯만
  local i
  for i in $(seq 1 "$(used_slots)"); do
    [[ -n ${ONLY:-} && ${ONLY:-} != "$i" ]] && continue
    "$@" "$i" "$(session_of "$i")" "$(job_of "$i")"
  done
}

plan() {
  echo "▶ 실험 ${#EXPS[@]}개 / VM $(used_slots)대 (MAX_PARALLEL=$SLOTS) / GPU $GPU"
  local i
  for i in $(seq 1 "$(used_slots)"); do
    printf '   %-6s %s\n' "$(session_of "$i")" "$(slot_exps "$i" | while read -r e; do printf '%s → ' "$(name_of "$e")"; done | sed 's/ → $//')"
    slot_exps "$i" | while read -r e; do
      [[ -n $(args_of "$e") ]] && printf '            %-8s %s\n' "$(name_of "$e")" "$(args_of "$e")" || true
    done
  done
  echo "   (한 VM 은 자기 실험을 순서대로 이어서 돌립니다 — 앞 실험이 실패해도 다음 실험은 진행)"
}

do_up()     { echo "── [$2] VM 할당 + 준비"; COLAB_SESSION="$2" "$PIPE" up "$GPU"; }
do_start()  { echo "── [$2] 작업 $3 제출: $(slot_exps "$1" | while read -r e; do printf '%s ' "$(name_of "$e")"; done)"
              COLAB_SESSION="$2" "$PIPE" run "$3" bash -c "$(slot_chain "$1")"; }
do_status() { echo "── [$2] $3"; COLAB_SESSION="$2" "$PIPE" status 2>&1 | sed 's/^/   /'
              COLAB_SESSION="$2" "$PIPE" logs "$3" 4 2>&1 | grep -v '__STATUS__' | tail -3 | sed 's/^/   | /' || true; }
do_logs()   { echo "── [$2] $3"; COLAB_SESSION="$2" "$PIPE" logs "$3" "${N:-20}" 2>&1 | sed 's/^/   /'; }
do_cancel() { echo "── [$2] $3 중단"; COLAB_SESSION="$2" "$PIPE" cancel "$3" 2>&1 | sed 's/^/   /' || true; }
do_down()   { echo "── [$2] 반납"; COLAB_SESSION="$2" "$PIPE" down 2>&1 | sed 's/^/   /' || true; }
do_watch()  { COLAB_SESSION="$2" WATCH_EVERY="${WATCH_EVERY:-600}" "$PIPE" watch "$3" & }
do_fetch()  { echo "── [$2] 결과 회수"; COLAB_SESSION="$2" "$PIPE" fetch 2>&1 | sed 's/^/   /' || true; }
do_evals()  { echo "── [$2]"; COLAB_SESSION="$2" N="${N:-8}" "$PIPE" evals $(slot_runs "$1") 2>&1 | sed 's/^/   /' || true; }
do_fetch_best() { echo "── [$2] best 어댑터"; COLAB_SESSION="$2" "$PIPE" fetch-best $(slot_runs "$1") 2>&1 | sed 's/^/   /' || true; }

cmd=${1:-}; shift || true
case "$cmd" in
  plan)   plan ;;
  chain)  slot_chain "${1:-1}"; echo ;;       # 슬롯이 VM 에서 실제로 돌릴 명령 (확인용)
  up)     plan; ONLY=${1:-} each_slot do_up ;;
  start)  ONLY=${1:-} each_slot do_start ;;
  status) ONLY=${1:-} each_slot do_status ;;
  logs)   N=${1:-20}; ONLY=${2:-} each_slot do_logs ;;
  cancel) ONLY=${1:-} each_slot do_cancel ;;
  down)   ONLY=${1:-} each_slot do_down
          [[ -z ${1:-} ]] && rm -f "$SESS_FILE" || true ;;     # 전부 반납하면 기억한 세션 이름도 지운다
  watch)  each_slot do_watch
          echo "▶ VM ${SLOTS}대 ($(for i in $(seq 1 "$(used_slots)"); do printf '%s ' "$(session_of "$i")"; done))를 ${WATCH_EVERY:-600}초마다 확인합니다 (Ctrl-C 로 전부 중단)"
          trap 'kill $(jobs -p) 2>/dev/null' EXIT; wait ;;
  evals)  ONLY=${1:-} each_slot do_evals ;;
  fetch)  ONLY=${1:-} each_slot do_fetch
          echo "── 비교표"; (cd "$CODE" && python3 collect_results.py) || true ;;
  fetch-best) ONLY=${1:-} each_slot do_fetch_best ;;
  *) sed -n '2,26p' "$0" ;;
esac
