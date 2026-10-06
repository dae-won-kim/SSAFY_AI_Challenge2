#!/usr/bin/env bash
# 1~4단계 + 최종 제출까지 순서대로 실행. 단계별로 따로 돌려도 된다.
set -euo pipefail
export VQA_MODEL_ROOT=${VQA_MODEL_ROOT:-/workspace/models}   # D 드라이브에 연결된 경로 권장
# seed·epoch·lr·LoRA 랭크·프롬프트 비율·문턱값·모델 기본값은 config.py 의 "실험 설정" 에서 관리.
# RUN(산출물 폴더), PRESET, LORA_R 은 환경변수로 덮어쓸 수 있다
#   예) RUN=r32 LORA_R=32 ./run_pipeline.sh
RUN=${RUN:-main}

# 1) multi-prompt choice-CE LoRA 학습 (모델 하나 · 어댑터 하나)
python s1_train_choice_ce.py --run "$RUN" ${PRESET:+--preset "$PRESET"} ${LORA_R:+--lora-r "$LORA_R"}

# 2) 단일 패스 추론 (valid + test) → runs/<RUN>/submission.csv 까지 바로 나온다
python s2_predict.py --run "$RUN"

# 3) 재추론 대상 선별 (정답을 보지 않는 기준)
python s3_select_rerun.py --run "$RUN"

# 4) 선별 문항만 재추론 (LoRA 유무 둘 다 돌려 valid 로 비교)
python s4_rerun_grounding.py --run "$RUN" --mode grounding --pass2 adapter
python s4_rerun_grounding.py --run "$RUN" --mode grounding --pass2 base
# 다중 프롬프트 재추론도 후보 : python s4_rerun_grounding.py --run "$RUN" --mode prompt

# 최종) 필요한 행만 교체 → runs/final/final_submission.csv
python s5_build_final.py --run "$RUN" --rerun-tag grounding_adapter
