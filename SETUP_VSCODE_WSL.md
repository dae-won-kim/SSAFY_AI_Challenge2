# VS Code + WSL + Colab CLI + Claude Code 작업 환경

```
Windows
└─ VS Code (창은 Windows 에서 뜨지만, 모든 작업은 WSL 안에서)
   └─ WSL Ubuntu  ~/vqa_pipeline
      ├─ Claude Code (VS Code 확장 패널 + 터미널 claude)  ← 코드 수정·실험 설계·결과 분석
      ├─ Colab CLI (colab)                               ← GPU VM 할당·실행
      └─ colab/colab_pipeline.sh                         ← 파이프라인 단계 실행·감시·회수
            │ colab exec
            ▼
      Colab VM (A100 등) ── 산출물 ──► Google Drive  vqa/runs/
```

## 1. WSL Ubuntu 준비 (Windows, 한 번만)

관리자 PowerShell:

```powershell
wsl --install -d Ubuntu
```

Docker Desktop 이 설치되어 있어도 `docker-desktop` 배포판은 작업용이 아니므로, **Ubuntu 배포판을 따로** 설치해 씁니다.

**프로젝트는 WSL 의 리눅스 파일 시스템(`~/`)에 둡니다.** `/mnt/c`, `/mnt/d` 아래(Windows 드라이브)에서 작업하면 파일 접근이 훨씬 느립니다.

```bash
cp -r /mnt/d/<코드가 있는 폴더>/vqa_pipeline ~/
```

## 2. VS Code 연결

1. Windows 에 VS Code 설치 → 확장 **WSL** (`ms-vscode-remote.remote-wsl`) 설치
2. WSL 터미널에서 `cd ~/vqa_pipeline && code .` → VS Code 가 WSL 에 연결된 창으로 열림 (왼쪽 아래에 `WSL: Ubuntu` 표시)
3. 추천 확장 설치 알림이 뜨면 **설치**. Claude Code 확장은 **WSL 쪽에 설치**되어야 합니다(확장 화면의 "Install in WSL: Ubuntu").

## 3. 도구 설치 (WSL 터미널, 한 번만)

```bash
cd ~/vqa_pipeline
bash scripts/setup_wsl.sh
source ~/.bashrc
```

설치되는 것: `uv`, Colab CLI(`colab`), Claude Code(`claude`, 네이티브 설치), 인증 URL 을 Windows 브라우저로 여는 `wslview`,
그리고 Colab CLI 공식 저장소의 에이전트 스킬(`.claude/skills/colab-operator`).

## 4. 로그인 (한 번만)

| 대상 | 방법 |
|---|---|
| Claude Code | 터미널에서 `claude` 실행 → 브라우저 로그인 (Pro·Max·Team·Enterprise 또는 Console 계정) |
| Claude Code 확장 패널 | VS Code 왼쪽/오른쪽의 Claude 아이콘 → 같은 계정으로 로그인 |
| Colab CLI | 처음 `./colab/colab_pipeline.sh up A100` 할 때 인증 안내가 나옴 |
| Google Drive 마운트 | `up` 도중 `colab drivemount` 가 인증을 요구할 수 있음 |

브라우저가 자동으로 안 열리면 터미널에 출력된 URL 을 복사해 Windows 브라우저에 붙여넣으면 됩니다.

## 5. 구성 파일

| 파일 | 역할 |
|---|---|
| `CLAUDE.md` | Claude Code 가 매 세션 읽는 프로젝트 안내: 구조, 실행 방법, 규칙, 지금까지의 실험 결과 |
| `.claude/settings.json` | 명령 권한. 상태·로그 확인은 자동 허용, **GPU 비용이 드는 명령(`up`, `run`, `pipeline` 등)은 매번 확인**, `colab pay`·`sudo`·`rm -rf` 는 차단 |
| `.claude/skills/vqa-challenge/` | 이 대회 유형의 일반 전략 스킬 |
| `.claude/skills/colab-operator/` | Colab CLI 사용법 스킬 (공식 저장소에서 받음) |
| `.vscode/tasks.json` | 명령 팔레트 → **Tasks: Run Task** → `Colab: up / smoke / pipeline / wait / logs / status / fetch / down` |
| `.vscode/extensions.json` | 추천 확장 (WSL, Claude Code, Python, ShellCheck) |

개인용 권한 조정은 `.claude/settings.local.json` 에 하세요(git 에 올라가지 않음).

## 6. 작업 방식

**역할 분담이 핵심입니다.** Claude Code 의 명령 실행에는 시간 제한이 있어서, 몇 시간짜리 학습을 기다리게 하면 안 됩니다.

| 누가 | 무엇을 |
|---|---|
| 사용자 (VS Code Tasks 또는 터미널) | `up`, `smoke`, `pipeline` 같은 **오래 걸리는 실행** |
| Claude Code | 코드 수정, `run` 으로 **단일 작업 띄우기**, `logs`·`status` 로 **진행 확인**, 결과 CSV **분석**, 다음 실험 제안 |

Claude Code 에 이렇게 요청하면 됩니다.

```
- smoke 결과 로그를 보고 문제가 있는지 확인해줘
- runs_smoke 의 크롭 이미지 목록을 보고 bbox 좌표계가 맞는지 판단할 방법을 알려줘
- results/ 의 실험 기록을 비교해서 s4 선별 문턱값을 어떻게 바꿀지 제안해줘
- s4 를 --max-frac 0.2 로 다시 돌려줘 (run 으로 띄우고 logs 로 확인)
- qtype 규칙에 '가격표' 유형을 추가하고 py_compile 로 점검한 뒤 sync 해줘
```

`run`·`up` 같은 명령은 실행 전에 확인 창이 뜹니다. 내용을 보고 승인하세요.

## 7. 긴 작업 중 주의

- **PC 절전을 끄세요.** Colab VM 을 유지하는 keep-alive 가 로컬(WSL)에서 돌기 때문에, PC 가 잠들면 VM 이 idle 로 정리될 수 있습니다.
- **WSL 이 계속 떠 있게** VS Code 창이나 WSL 터미널을 열어 두세요.
- 끊겨도 산출물과 학습 체크포인트는 드라이브에 남습니다. `up` → `pipeline <단계>` 로 이어서 실행합니다.
- 다 쓰면 `down`. 반납하지 않은 VM 은 컴퓨팅 단위를 계속 씁니다.
