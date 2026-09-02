#!/usr/bin/env python3
"""Install / start / stop the charting server Windows logon task.

Examples (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\pipeline\\charting_server_service.py install-task
  python scripts\\pipeline\\charting_server_service.py start
  python scripts\\pipeline\\charting_server_service.py task-status
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.charting_server_service import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    TASK_NAME,
    install_windows_task,
    process_status,
    query_windows_task,
    start_windows_task,
    stop_windows_task,
    uninstall_windows_task,
)


def cmd_install_task(args: argparse.Namespace) -> int:
    name = install_windows_task(ROOT, host=args.host, port=args.port)
    print("Installed logon task: %s" % name)
    print("It runs pythonw charting_server.py --service (no CMD window).")
    print("URL: http://localhost:%s  (dashboard /hot)" % args.port)
    print("Logs: logs\\charting_server\\charting_server.log")
    print()
    if not args.no_start:
        start_windows_task()
        print("Started. Open http://localhost:%s" % args.port)
        print()
    print("Useful:")
    print("  python scripts\\pipeline\\charting_server_service.py task-status")
    print("  python scripts\\pipeline\\charting_server_service.py start")
    print("  python scripts\\pipeline\\charting_server_service.py stop")
    print("  python scripts\\pipeline\\charting_server_service.py uninstall-task")
    return 0


def cmd_uninstall_task(_args: argparse.Namespace) -> int:
    try:
        stop_windows_task(ROOT)
    except RuntimeError:
        pass
    uninstall_windows_task()
    print("Removed task %s" % TASK_NAME)
    return 0


def cmd_task_status(_args: argparse.Namespace) -> int:
    st = process_status(ROOT)
    print("pid=%s running=%s" % (st["pid"] if st["pid"] is not None else "-", st["running"]))
    print("log=%s" % st["log"])
    print()
    print(query_windows_task())
    return 0


def cmd_start(_args: argparse.Namespace) -> int:
    start_windows_task()
    print("Started task %s" % TASK_NAME)
    print("URL: http://localhost:%s" % DEFAULT_PORT)
    return 0


def cmd_stop(_args: argparse.Namespace) -> int:
    stop_windows_task(ROOT)
    print("Stopped task %s" % TASK_NAME)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Charting server Windows logon task (always-on, not crontab.yaml)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("install-task", help="Register the hidden logon Windows task")
    p_ins.add_argument("--host", default=DEFAULT_HOST)
    p_ins.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_ins.add_argument("--no-start", action="store_true", help="Install without starting now")
    p_ins.set_defaults(func=cmd_install_task)

    p_un = sub.add_parser("uninstall-task", help="Stop and remove the Windows task")
    p_un.set_defaults(func=cmd_uninstall_task)

    p_st = sub.add_parser("task-status", help="Query the Windows task and pid file")
    p_st.set_defaults(func=cmd_task_status)

    p_start = sub.add_parser("start", help="Start the installed task now")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop the running server")
    p_stop.set_defaults(func=cmd_stop)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, RuntimeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
