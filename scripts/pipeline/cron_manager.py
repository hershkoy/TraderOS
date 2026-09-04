#!/usr/bin/env python3
"""CLI for the repo crontab and the hidden Windows minute-tick runner.

Examples (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\pipeline\\cron_manager.py list
  python scripts\\pipeline\\cron_manager.py add channel_touch_nightly --schedule "0 23 * * 1-5" --command "crons\\channel_touch_nightly.bat"
  python scripts\\pipeline\\cron_manager.py enable channel_touch_nightly
  python scripts\\pipeline\\cron_manager.py install-task
  python scripts\\pipeline\\cron_manager.py tick --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cron.cronexpr import parse_schedule
from utils.cron.manager import CronJob, CronManager, TASK_NAME

logger = logging.getLogger("cron_manager")


def _mgr() -> CronManager:
    return CronManager(root=ROOT)


def _setup_logging(tick: bool) -> None:
    log_dir = ROOT / "logs" / "cron"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = []
    if tick:
        day = datetime.now().strftime("%Y%m%d")
        handlers.append(logging.FileHandler(log_dir / ("tick_%s.log" % day), encoding="utf-8"))
    if sys.stdout and sys.stdout.isatty():
        handlers.append(logging.StreamHandler(sys.stdout))
    if not handlers:
        handlers.append(logging.FileHandler(log_dir / ("tick_%s.log" % datetime.now().strftime("%Y%m%d")), encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def cmd_list(args: argparse.Namespace) -> int:
    mgr = _mgr()
    jobs = mgr.jobs()
    if not jobs:
        print("No jobs in %s" % mgr.crontab_path)
        print("Add one with: python scripts\\pipeline\\cron_manager.py add NAME --schedule \"0 23 * * 1-5\" --command \"crons\\\\foo.bat\"")
        return 0
    now = datetime.now()
    print("%-8s %-28s %-22s %s" % ("ENABLED", "NAME", "NEXT", "SCHEDULE"))
    for job in jobs:
        st = mgr.job_state(job.name)
        flag = "yes" if job.enabled else "no"
        nxt = "-"
        if job.enabled:
            try:
                nxt = mgr.next_run(job, after=now).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                nxt = "(none in 8d)"
        running = " RUNNING" if st.get("running") else ""
        sched = job.schedule[0] if len(job.schedule) == 1 else ", ".join(job.schedule)
        print("%-8s %-28s %-22s %s%s" % (flag, job.name, nxt, sched, running))
        print("         %s" % job.command)
        if job.description:
            print("         %s" % job.description)
        last = st.get("last_fired") or "-"
        rc = st.get("last_exit_code")
        print("         last_fired=%s last_exit=%s" % (last, rc if rc is not None else "-"))
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    mgr = _mgr()
    job = CronJob(
        name=args.name,
        schedule=parse_schedule(args.schedule),
        command=args.command,
        enabled=not args.disabled,
        description=args.description or "",
        timezone=args.timezone,
    )
    mgr.add_job(job, overwrite=args.force)
    print("saved job %s -> %s" % (job.name, mgr.crontab_path))
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    _mgr().remove_job(args.name)
    print("removed %s" % args.name)
    return 0


def cmd_enable(args: argparse.Namespace) -> int:
    job = _mgr().set_enabled(args.name, True)
    print("enabled %s (%s)" % (job.name, job.schedule[0] if len(job.schedule) == 1 else job.schedule))
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    job = _mgr().set_enabled(args.name, False)
    print("disabled %s" % job.name)
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    _setup_logging(tick=True)
    mgr = _mgr()
    now = None
    if args.at:
        now = datetime.fromisoformat(args.at)
    started = mgr.tick(now=now, dry_run=args.dry_run, spawn=not args.inline)
    if not started:
        logger.debug("tick: no due jobs")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _setup_logging(tick=False)
    return _mgr().run_job(args.name, quiet=args.quiet)


def cmd_stop(args: argparse.Namespace) -> int:
    _setup_logging(tick=False)
    return _mgr().stop_job(args.name, timeout_sec=float(args.timeout), force=not args.no_force)


def cmd_next(args: argparse.Namespace) -> int:
    mgr = _mgr()
    if args.name:
        jobs = [mgr.job(args.name)]
    else:
        jobs = [j for j in mgr.jobs() if j.enabled]
    for job in jobs:
        try:
            nxt = mgr.next_run(job)
            print("%s  %s" % (job.name, nxt.strftime("%Y-%m-%d %H:%M %Z").strip()))
        except ValueError as exc:
            print("%s  %s" % (job.name, exc))
    return 0


def cmd_install_task(_args: argparse.Namespace) -> int:
    mgr = _mgr()
    name = mgr.install_windows_task()
    print("Installed hidden minute task: %s" % name)
    print("It runs pythonw ... cron_manager.py tick (no CMD window).")
    print("Jobs live in %s" % mgr.crontab_path)
    print()
    print("Useful:")
    print("  schtasks /Query /TN \"%s\" /V /FO LIST" % TASK_NAME)
    print("  schtasks /Run /TN \"%s\"" % TASK_NAME)
    print("  python scripts\\pipeline\\cron_manager.py uninstall-task")
    return 0


def cmd_uninstall_task(_args: argparse.Namespace) -> int:
    _mgr().uninstall_windows_task()
    print("Removed task %s" % TASK_NAME)
    return 0


def cmd_task_status(_args: argparse.Namespace) -> int:
    print(_mgr().windows_task_query())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repo cron manager (YAML crontab + 1-minute hidden Windows task)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List jobs, last run, next run")
    p_list.set_defaults(func=cmd_list)

    p_add = sub.add_parser("add", help="Add or replace a job")
    p_add.add_argument("name")
    p_add.add_argument("--schedule", required=True, help="5-field cron, e.g. '0 23 * * 1-5'")
    p_add.add_argument("--command", required=True, help="Command from repo root (bat or python ...)")
    p_add.add_argument("--description", default="")
    p_add.add_argument("--timezone", default=None, help="IANA tz or omit for crontab default (local)")
    p_add.add_argument("--disabled", action="store_true")
    p_add.add_argument("--force", action="store_true", help="Overwrite if the name exists")
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("remove", help="Remove a job")
    p_rm.add_argument("name")
    p_rm.set_defaults(func=cmd_remove)

    p_en = sub.add_parser("enable", help="Enable a job")
    p_en.add_argument("name")
    p_en.set_defaults(func=cmd_enable)

    p_dis = sub.add_parser("disable", help="Disable a job")
    p_dis.add_argument("name")
    p_dis.set_defaults(func=cmd_disable)

    p_tick = sub.add_parser("tick", help="Run due jobs for this minute (Task Scheduler calls this)")
    p_tick.add_argument("--dry-run", action="store_true")
    p_tick.add_argument("--inline", action="store_true", help="Run due jobs in this process instead of spawning")
    p_tick.add_argument("--at", default=None, help="ISO datetime to evaluate instead of now")
    p_tick.set_defaults(func=cmd_tick)

    p_run = sub.add_parser("run", help="Run one job now (ignores schedule)")
    p_run.add_argument("name")
    p_run.add_argument("--quiet", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_stop = sub.add_parser("stop", help="Stop a running job (process tree)")
    p_stop.add_argument("name")
    p_stop.add_argument("--timeout", type=float, default=60.0, help="Seconds to wait before force-kill")
    p_stop.add_argument("--no-force", action="store_true", help="Do not taskkill /F after timeout")
    p_stop.set_defaults(func=cmd_stop)

    p_next = sub.add_parser("next", help="Show next fire time")
    p_next.add_argument("name", nargs="?")
    p_next.set_defaults(func=cmd_next)

    p_ins = sub.add_parser("install-task", help="Register the hidden every-minute Windows task")
    p_ins.set_defaults(func=cmd_install_task)

    p_un = sub.add_parser("uninstall-task", help="Remove the Windows CronRunner task")
    p_un.set_defaults(func=cmd_uninstall_task)

    p_st = sub.add_parser("task-status", help="Query the Windows CronRunner task")
    p_st.set_defaults(func=cmd_task_status)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
