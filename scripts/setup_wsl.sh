#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# WSL Ubuntu 에서 한 번 실행 : Colab CLI, Claude Code, 보조 도구 설치
#   cd ~/vqa_pipeline && bash scripts/setup_wsl.sh
# 다시 실행해도 안전하다 (이미 있는 것은 건너뛰거나 업데이트).
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "▶ 기본 패키지 (python3, unzip, curl)"
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip unzip curl
sudo apt-get install -y wslu || echo "  wslu 설치 실패 (인증 URL 을 수동으로 브라우저에 붙여넣으면 됨)"

echo "▶ uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "▶ Colab CLI"
# 0.7.2 는 jupyter-kernel-client==0.8 로 고정돼 있어 실행 시 AttributeError(JupyterSubprotocol) → 버전을 덮어써서 설치
uv tool install --force google-colab-cli --overrides scripts/colab_cli_overrides.txt || true

echo "▶ Claude Code (네이티브 설치, 자동 업데이트)"
command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash

echo "▶ 셸 설정 (~/.bashrc)"
grep -q 'HOME/.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
if command -v wslview >/dev/null; then
  grep -q 'BROWSER=wslview' ~/.bashrc || echo 'export BROWSER=wslview   # 인증 URL 을 Windows 브라우저로 열기' >> ~/.bashrc
fi

echo "▶ Colab CLI 에이전트 스킬 → .claude/skills/colab-operator"
mkdir -p .claude/skills/colab-operator
curl -fsSL https://raw.githubusercontent.com/googlecolab/google-colab-cli/main/skills/colab-operator/SKILL.md \
  -o .claude/skills/colab-operator/SKILL.md \
  && echo "  받음" || echo "  다운로드 실패 (나중에 이 스크립트를 다시 실행)"

chmod +x colab/colab_pipeline.sh scripts/*.sh

echo "▶ 확인"
colab version || echo "  colab 명령이 없음 → 새 터미널을 열고 다시 확인"
claude --version || echo "  claude 명령이 없음 → 새 터미널을 열고 다시 확인"
echo
echo "다음 단계:"
echo "  1) 새 터미널을 열거나 source ~/.bashrc"
echo "  2) claude           → Claude 계정 로그인"
echo "  3) ./colab/colab_pipeline.sh up A100   → Colab 인증 + VM 준비"
