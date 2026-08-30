"""Load crontab.yaml, decide due jobs, and run them without a console window."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

from .cronexpr import any_cron_matches, next_match, parse_schedule, validate_cron

logger = logging.getLogger("cron_manager")

TASK_NAME = r"backTraderTest\CronRunner"
CRONTAB_HEADER = (
    "# backTraderTest crontab. The Windows CronRunner task ticks every minute\n"
    "# (pythonw, no console) and runs enabled jobs whose cron matches now.\n"
    "#\n"
    "#   python scripts\\pipeline\\cron_manager.py list\n"
    "#   python scripts\\pipeline\\cron_manager.py add NAME --schedule \"0 23 * * 1-5\" "
    "--command \"crons\\\\foo.bat\"\n"
    "#   python scripts\\pipeline\\cron_manager.py enable NAME\n"
    "#   python scripts\\pipeline\\cron_manager.py install-task\n"
    "\n"
)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class CronJob:
    name: str
    schedule: List[str]
    command: str
    enabled: bool = True
    description: str = ""
    timezone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.name,
            "enabled": self.enabled,
            "schedule": self.schedule[0] if len(self.schedule) == 1 else self.schedule,
            "command": self.command,
        }
        if self.description:
            data["description"] = self.description
        if self.timezone:
            data["timezone"] = self.timezone
        return data

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CronJob":
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("job is missing name")
        _validate_name(name)
        command = str(raw.get("command") or "").strip()
        if not command:
            raise ValueError("job %s is missing command" % name)
        schedule = parse_schedule(raw.get("schedule"))
        tz = raw.get("timezone")
        tz_s = str(tz).strip() if tz else None
        desc = str(raw.get("description") or "")
        enabled = bool(raw.get("enabled", True))
        return cls(
            name=name,
            schedule=schedule,
            command=command,
            enabled=enabled,
            description=desc,
            timezone=tz_s or None,
        )


class CronManager:
    def __init__(self, root: Optional[Path] = None, crontab_path: Optional[Path] = None):
        self.root = Path(root) if root else default_root()
        self.crontab_path = Path(crontab_path) if crontab_path else self.root / "crons" / "crontab.yaml"
        self.state_dir = self.root / "logs" / "cron"
        self.lock_dir = self.state_dir / "locks"
        self.state_path = self.state_dir / "state.json"

    def load(self) -> Dict[str, Any]:
        if not self.crontab_path.is_file():
            return {"timezone": "local", "jobs": []}
        with self.crontab_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError("crontab must be a YAML mapping")
        data.setdefault("timezone", "local")
        data.setdefault("jobs", [])
        if data["jobs"] is None:
            data["jobs"] = []
        if not isinstance(data["jobs"], list):
            raise ValueError("crontab jobs must be a list")
        return data

    def jobs(self) -> List[CronJob]:
        return [CronJob.from_dict(item) for item in self.load()["jobs"]]

    def job(self, name: str) -> CronJob:
        for item in self.jobs():
            if item.name == name:
                return item
        raise KeyError("unknown job: %s" % name)

    def save(self, timezone: str, jobs: List[CronJob]) -> None:
        names = [j.name for j in jobs]
        if len(names) != len(set(names)):
            raise ValueError("duplicate job names")
        payload = {
            "timezone": timezone,
            "jobs": [j.to_dict() for j in jobs],
        }
        self.crontab_path.parent.mkdir(parents=True, exist_ok=True)
        text = CRONTAB_HEADER + yaml.safe_dump(
            payload, sort_keys=False, default_flow_style=False, allow_unicode=True
        )
        self.crontab_path.write_text(text, encoding="utf-8")

    def add_job(self, job: CronJob, overwrite: bool = False) -> None:
        data = self.load()
        jobs = [CronJob.from_dict(item) for item in data["jobs"]]
        existing = [j for j in jobs if j.name == job.name]
        if existing and not overwrite:
            raise ValueError("job already exists: %s (use --force to replace)" % job.name)
        jobs = [j for j in jobs if j.name != job.name]
        jobs.append(job)
        self.save(str(data.get("timezone") or "local"), jobs)

    def remove_job(self, name: str) -> None:
        data = self.load()
        jobs = [CronJob.from_dict(item) for item in data["jobs"]]
        kept = [j for j in jobs if j.name != name]
        if len(kept) == len(jobs):
            raise KeyError("unknown job: %s" % name)
        self.save(str(data.get("timezone") or "local"), kept)

    def set_enabled(self, name: str, enabled: bool) -> CronJob:
        data = self.load()
        jobs = [CronJob.from_dict(item) for item in data["jobs"]]
        found = None
        for job in jobs:
            if job.name == name:
                job.enabled = enabled
                found = job
                break
        if found is None:
            raise KeyError("unknown job: %s" % name)
        self.save(str(data.get("timezone") or "local"), jobs)
        return found

    def due_jobs(self, now: Optional[datetime] = None) -> List[CronJob]:
        data = self.load()
        default_tz = str(data.get("timezone") or "local")
        out: List[CronJob] = []
        for job in self.jobs():
            if not job.enabled:
                continue
            when = _aware_now(now, job.timezone or default_tz)
            if any_cron_matches(job.schedule, when):
                out.append(job)
        return out

    def next_run(self, job: CronJob, after: Optional[datetime] = None) -> datetime:
        data = self.load()
        default_tz = str(data.get("timezone") or "local")
        when = _aware_now(after, job.timezone or default_tz)
        nxt = next_match(job.schedule, when)
        return nxt

    def tick(
        self,
        now: Optional[datetime] = None,
        dry_run: bool = False,
        spawn: bool = True,
    ) -> List[str]:
        """Start every enabled job that is due this minute.

        Returns the names that were started (or would be, on dry-run).
        """
        started: List[str] = []
        minute_key_now = now
        for job in self.due_jobs(now=now):
            data = self.load()
            default_tz = str(data.get("timezone") or "local")
            when = _aware_now(minute_key_now, job.timezone or default_tz)
            minute_key = when.strftime("%Y-%m-%dT%H:%M")
            if self._already_fired(job.name, minute_key):
                logger.info("skip %s: already fired this minute (%s)", job.name, minute_key)
                continue
            alive, pid = self.lock_status(job.name)
            if alive:
                logger.info("skip %s: still running (pid %s)", job.name, pid)
                continue
            if dry_run:
                logger.info("dry-run: would start %s at %s", job.name, minute_key)
                started.append(job.name)
                continue
            self._mark_fired(job.name, minute_key)
            if spawn:
                self._spawn_run(job.name)
            else:
                self.run_job(job.name, quiet=True)
            started.append(job.name)
            logger.info("started %s", job.name)
        return started

    def run_job(self, name: str, quiet: bool = False) -> int:
        job = self.job(name)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        alive, pid = self.lock_status(name)
        if alive:
            logger.warning("job %s already running (pid %s)", name, pid)
            return 0
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.state_dir / ("%s_%s.log" % (name, stamp))
        self._write_lock(name, os.getpid())
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root)
        env["PYTHONUNBUFFERED"] = "1"
        venv_scripts = self.root / "venv" / "Scripts"
        if venv_scripts.is_dir():
            env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")
        logger.info("running %s: %s", name, job.command)
        logger.info("log: %s", log_path)
        rc = 1
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as logf:
                logf.write("job=%s\ncommand=%s\nstarted=%s\n\n" % (
                    name, job.command, datetime.now().isoformat(timespec="seconds")
                ))
                logf.flush()
                popen_kwargs: Dict[str, Any] = {
                    "args": job.command,
                    "shell": True,
                    "cwd": str(self.root),
                    "env": env,
                    "stdout": logf,
                    "stderr": subprocess.STDOUT,
                }
                popen_kwargs.update(_hidden_popen_kwargs())
                proc = subprocess.Popen(**popen_kwargs)
                self._write_lock(name, proc.pid)
                rc = proc.wait()
                logf.write("\nfinished=%s\nexit_code=%s\n" % (
                    datetime.now().isoformat(timespec="seconds"), rc
                ))
        finally:
            self._clear_lock(name)
            self._record_result(name, rc, str(log_path))
        if not quiet:
            print("job %s exit %s (log %s)" % (name, rc, log_path))
        return rc

    def lock_status(self, name: str) -> tuple:
        path = self._lock_path(name)
        if not path.is_file():
            return False, None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pid = int(data.get("pid") or 0)
        except (OSError, ValueError, json.JSONDecodeError):
            return False, None
        if pid and _pid_alive(pid):
            return True, pid
        try:
            path.unlink()
        except OSError:
            pass
        return False, pid

    def _spawn_run(self, name: str) -> None:
        cli = self.root / "scripts" / "pipeline" / "cron_manager.py"
        python = _pythonw_path(self.root)
        args = [str(python), str(cli), "run", name, "--quiet"]
        kwargs: Dict[str, Any] = {
            "args": args,
            "cwd": str(self.root),
            "env": {**os.environ, "PYTHONPATH": str(self.root), "PYTHONUNBUFFERED": "1"},
            "close_fds": True,
        }
        kwargs.update(_hidden_popen_kwargs())
        subprocess.Popen(**kwargs)

    def _already_fired(self, name: str, minute_key: str) -> bool:
        state = self._load_state()
        last = (state.get("jobs") or {}).get(name, {}).get("last_fired")
        return last == minute_key

    def _mark_fired(self, name: str, minute_key: str) -> None:
        state = self._load_state()
        jobs = state.setdefault("jobs", {})
        rec = jobs.setdefault(name, {})
        rec["last_fired"] = minute_key
        self._save_state(state)

    def _record_result(self, name: str, rc: int, log_path: str) -> None:
        state = self._load_state()
        jobs = state.setdefault("jobs", {})
        rec = jobs.setdefault(name, {})
        rec["last_finished"] = datetime.now().isoformat(timespec="seconds")
        rec["last_exit_code"] = rc
        rec["last_log"] = log_path
        self._save_state(state)

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.is_file():
            return {"jobs": {}}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"jobs": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _lock_path(self, name: str) -> Path:
        return self.lock_dir / ("%s.lock" % name)

    def _write_lock(self, name: str, pid: int) -> None:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        payload = {"pid": pid, "started": datetime.now().isoformat(timespec="seconds")}
        self._lock_path(name).write_text(json.dumps(payload), encoding="utf-8")

    def _clear_lock(self, name: str) -> None:
        try:
            self._lock_path(name).unlink()
        except OSError:
            pass

    def job_state(self, name: str) -> Dict[str, Any]:
        rec = dict((self._load_state().get("jobs") or {}).get(name) or {})
        alive, pid = self.lock_status(name)
        rec["running"] = alive
        rec["pid"] = pid
        return rec

    def install_windows_task(self) -> str:
        """Register the hidden every-minute Task Scheduler job. Returns task name."""
        if sys.platform != "win32":
            raise RuntimeError("Windows Task Scheduler install is only supported on Windows")
        pythonw = _pythonw_path(self.root)
        cli = self.root / "scripts" / "pipeline" / "cron_manager.py"
        if not pythonw.is_file():
            raise FileNotFoundError("pythonw.exe not found at %s" % pythonw)
        xml_path = self.state_dir / "cron_runner_task.xml"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        xml_path.write_bytes(_task_xml(pythonw, cli, self.root))
        cmd = [
            "schtasks",
            "/Create",
            "/TN",
            TASK_NAME,
            "/XML",
            str(xml_path),
            "/F",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "schtasks failed (%s): %s %s" % (proc.returncode, proc.stdout, proc.stderr)
            )
        return TASK_NAME

    def uninstall_windows_task(self) -> None:
        proc = subprocess.run(
            ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "schtasks delete failed (%s): %s %s" % (proc.returncode, proc.stdout, proc.stderr)
            )

    def windows_task_query(self) -> str:
        proc = subprocess.run(
            ["schtasks", "/Query", "/TN", TASK_NAME, "/V", "/FO", "LIST"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "task %s is not installed (%s): %s" % (TASK_NAME, proc.returncode, proc.stderr)
            )
        return proc.stdout


def _validate_name(name: str) -> None:
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("job name must be alphanumeric plus _- : %r" % name)


def _aware_now(now: Optional[datetime], tz_name: str) -> datetime:
    tz = _zone(tz_name)
    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        local = datetime.now().astimezone().tzinfo
        now = now.replace(tzinfo=local)
    return now.astimezone(tz)


def _zone(tz_name: str):
    if not tz_name or tz_name.lower() == "local":
        return datetime.now().astimezone().tzinfo
    return ZoneInfo(tz_name)


def _pythonw_path(root: Path) -> Path:
    venv_w = root / "venv" / "Scripts" / "pythonw.exe"
    if venv_w.is_file():
        return venv_w
    exe = Path(sys.executable)
    if exe.name.lower() == "python.exe":
        alt = exe.with_name("pythonw.exe")
        if alt.is_file():
            return alt
    return exe


def _hidden_popen_kwargs() -> Dict[str, Any]:
    if sys.platform != "win32":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return {
        "startupinfo": startupinfo,
        "creationflags": CREATE_NO_WINDOW,
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _task_xml(pythonw: Path, cli: Path, root: Path) -> bytes:
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    xml = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>backTraderTest cron runner: every minute, hidden, no console. Runs due jobs from crons/crontab.yaml.</Description>
    <URI>\\backTraderTest\\CronRunner</URI>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>%s</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
      <Repetition>
        <Interval>PT1M</Interval>
        <Duration>P1D</Duration>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT10M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>%s</Command>
      <Arguments>"%s" tick</Arguments>
      <WorkingDirectory>%s</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
""" % (
        start,
        _xml_escape(str(pythonw)),
        _xml_escape(str(cli)),
        _xml_escape(str(root)),
    )
    return xml.encode("utf-16")


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
