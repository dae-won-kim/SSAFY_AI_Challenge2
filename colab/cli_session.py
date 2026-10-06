# ─────────────────────────────────────────────────────────────
# 로컬에서 실행됨 (Colab CLI 의 파이썬으로: colab_pipeline.sh 가 호출). Colab CLI 0.6 보완용
#
# 문제 : VM 접속 토큰(runtime proxy token)은 1시간(tokenExpiresInSeconds=3600) 뒤 만료되는데 CLI 가 갱신하지 않는다.
#        만료 뒤 요청은 404 → CLI 가 "세션을 잃었다"며 로컬 기록과 keep-alive 를 지운다. VM 은 서버에 살아 있는데도.
#        (오늘 VM 3대가 모두 할당 후 약 61분에 이렇게 끊김)
# 해결 : 서버 할당 목록(GET /tun/m/assignments)은 조회할 때마다 새 토큰을 준다 → 그 토큰으로 로컬 기록을 갱신.
#
#   refresh NAME                    로컬 세션의 접속 토큰을 새로 받음 (colab_pipeline.sh 가 원격 명령 전에 자동 호출)
#   adopt NAME ENDPOINT [--dry-run] 로컬 기록이 지워진 할당(colab sessions 의 [?])을 NAME 으로 다시 등록 + keep-alive 재시작
# ─────────────────────────────────────────────────────────────
import sys

from colab_cli.common import state
from colab_cli.state import SessionState


def assignment(endpoint):
    return next((a for a in state.client.list_assignments() if a.endpoint == endpoint), None)


def refresh(name):
    s = state.store.get(name)
    if not s:
        return                                           # 로컬 세션이 없으면 할 일 없음 (colab exec 가 알려 줌)
    a = assignment(s.endpoint)
    if a is None:
        print(f"[refresh] 서버에 세션 '{name}' 의 할당이 없습니다 (VM 이 반납·회수됨)", file=sys.stderr)
        return
    s.token, s.url = a.runtime_proxy_info.token, a.runtime_proxy_info.url
    state.store.add(s)


def adopt(name, endpoint, dry_run=False):
    if state.store.get(name):
        raise SystemExit(f"[adopt] 로컬 세션 '{name}' 이 이미 있습니다. 다른 이름을 쓰거나 먼저 정리하세요")
    a = assignment(endpoint)
    if a is None:
        raise SystemExit(f"[adopt] 서버에 할당 {endpoint} 이 없습니다 (colab sessions 로 확인)")
    s = SessionState(name=name, token=a.runtime_proxy_info.token, url=a.runtime_proxy_info.url, endpoint=endpoint,
                     variant=a.variant.name, accelerator=a.accelerator.value)
    if dry_run:
        print(f"[adopt] (dry-run) {endpoint} → 세션 '{name}' ({s.accelerator}, {s.variant}) 로 등록할 수 있음")
        return
    from colab_cli.commands.session import spawn_keep_alive
    state.store.add(s)                                   # keep-alive 프로세스가 세션을 찾을 수 있게 먼저 저장
    s.keep_alive_pid = spawn_keep_alive(endpoint, name, auth_provider=state.auth_provider,
                                        config_path=state.config_path)
    state.store.add(s)
    print(f"[adopt] {endpoint} → 세션 '{name}' 로 다시 등록, keep-alive 재시작 (pid {s.keep_alive_pid})")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "refresh":
        try:                                             # 갱신 실패로 파이프라인을 막지 않는다
            refresh(args[0])
        except EOFError:                                 # Google 인증 만료 → CLI 가 인증 코드를 물어봄
            print("[refresh] Google 인증이 만료됐습니다 → 터미널에서 'colab sessions' 로 다시 인증하세요", file=sys.stderr)
        except Exception as e:
            print(f"[refresh] 토큰 갱신 실패 ({type(e).__name__}: {str(e)[:120]})", file=sys.stderr)
    elif cmd == "adopt":
        adopt(args[0], args[1], dry_run="--dry-run" in args)
    else:
        raise SystemExit(f"알 수 없는 명령: {cmd}")
