"""Install the charting server as an always-on Windows Task Scheduler job.

This is not a crontab.yaml job. CronRunner ticks scheduled bats; the dashboard
must stay up, so it gets its own logon task (same pythonw / Interactive pattern).
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

TASK_NAME = r"backTraderTest\ChartingServer"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000


def current_windows_user() -> str:
    domain = os.environ.get("USERDOMAIN") or os.environ.get("COMPUTERNAME") or ""
    user = os.environ.get("USERNAME") or ""
    if not user:
        try:
            user = os.getlogin()
        except OSError:
            user = "User"
    if domain:
        return "%s\\%s" % (domain, user)
    return user


def default_log_dir(root: Path) -> Path:
    return root / "logs" / "charting_server"


def pid_path(log_dir: Path) -> Path:
    return log_dir / "charting_server.pid"


def pythonw_path(root: Path) -> Path:
    venv_w = root / "venv" / "Scripts" / "pythonw.exe"
    if venv_w.is_file():
        return venv_w
    exe = Path(sys.executable)
    if exe.name.lower() == "python.exe":
        alt = exe.with_name("pythonw.exe")
        if alt.is_file():
            return alt
    return exe


def attach_service_stdio(log_dir: Path) -> Path:
    """Send stdout/stderr to a log file so pythonw has somewhere to write."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "charting_server.log"
    stream = open(log_path, "a", encoding="utf-8", buffering=1, errors="replace")
    sys.stdout = stream
    sys.stderr = stream
    try:
        sys.stdin = open(os.devnull, "r")
    except OSError:
        pass
    _rebind_logging_streams(stream)
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("---- charting server start %s pid=%s ----" % (started, os.getpid()))
    return log_path


def _rebind_logging_streams(stream) -> None:
    """pythonw starts with stdout/stderr None; loggers captured that at import."""
    import logging

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, stream=stream, force=True)
        return
    loggers = [root]
    loggers.extend(
        obj for obj in logging.Logger.manager.loggerDict.values() if isinstance(obj, logging.Logger)
    )
    stdio = (None, sys.__stdout__, sys.__stderr__)
    for logger in loggers:
        for handler in list(logger.handlers):
            if not isinstance(handler, logging.StreamHandler):
                continue
            if isinstance(handler, logging.FileHandler):
                continue
            if getattr(handler, "stream", "missing") not in stdio:
                continue
            try:
                handler.setStream(stream)
            except Exception:
                handler.stream = stream


def write_pid(log_dir: Path, pid: Optional[int] = None) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = pid_path(log_dir)
    path.write_text(str(pid if pid is not None else os.getpid()), encoding="utf-8")
    atexit.register(_unlink_quietly, path)
    return path


def read_pid(log_dir: Path) -> Optional[int]:
    path = pid_path(log_dir)
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes

    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return False


def process_status(root: Path) -> dict:
    log_dir = default_log_dir(root)
    pid = read_pid(log_dir)
    alive = bool(pid and pid_alive(pid))
    return {
        "pid": pid,
        "running": alive,
        "log": str(log_dir / "charting_server.log"),
        "pid_file": str(pid_path(log_dir)),
    }


def build_task_xml(
    pythonw: Path,
    script: Path,
    root: Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    user_id: Optional[str] = None,
) -> bytes:
    """Task Scheduler XML: this-user logon trigger, unlimited runtime, restart on failure.

    A LogonTrigger without UserId applies to every account and requires Administrator.
    """
    user_id = user_id or current_windows_user()
    args = '"%s" --service --host %s --port %s' % (script, host, int(port))
    xml = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>backTraderTest charting server (Flask) at http://localhost:%s. pythonw, no console. Starts at logon and restarts on failure.</Description>
    <URI>\\backTraderTest\\ChartingServer</URI>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>%s</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>%s</UserId>
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
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>%s</Command>
      <Arguments>%s</Arguments>
      <WorkingDirectory>%s</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
""" % (
        int(port),
        _xml_escape(user_id),
        _xml_escape(user_id),
        _xml_escape(str(pythonw)),
        _xml_escape(args),
        _xml_escape(str(root)),
    )
    return xml.encode("utf-16")


def install_windows_task(
    root: Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> str:
    if sys.platform != "win32":
        raise RuntimeError("Windows Task Scheduler install is only supported on Windows")
    pythonw = pythonw_path(root)
    script = root / "charting_server.py"
    if not pythonw.is_file():
        raise FileNotFoundError("pythonw.exe not found at %s" % pythonw)
    if not script.is_file():
        raise FileNotFoundError("charting_server.py not found at %s" % script)
    log_dir = default_log_dir(root)
    log_dir.mkdir(parents=True, exist_ok=True)
    xml_path = log_dir / "charting_server_task.xml"
    xml_path.write_bytes(build_task_xml(pythonw, script, root, host=host, port=port))
    proc = subprocess.run(
        ["schtasks", "/Create", "/TN", TASK_NAME, "/XML", str(xml_path), "/F"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "schtasks failed (%s): %s %s" % (proc.returncode, proc.stdout, proc.stderr)
        )
    return TASK_NAME


def uninstall_windows_task() -> None:
    proc = subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "schtasks delete failed (%s): %s %s" % (proc.returncode, proc.stdout, proc.stderr)
        )


def query_windows_task() -> str:
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


def start_windows_task() -> None:
    proc = subprocess.run(
        ["schtasks", "/Run", "/TN", TASK_NAME],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "schtasks run failed (%s): %s %s" % (proc.returncode, proc.stdout, proc.stderr)
        )


def stop_windows_task(root: Path) -> None:
    subprocess.run(
        ["schtasks", "/End", "/TN", TASK_NAME],
        capture_output=True,
        text=True,
    )
    st = process_status(root)
    pid = st.get("pid")
    if pid and pid_alive(int(pid)):
        kill = subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            capture_output=True,
            text=True,
        )
        if kill.returncode != 0:
            raise RuntimeError(
                "taskkill failed (%s): %s %s" % (kill.returncode, kill.stdout, kill.stderr)
            )
    _unlink_quietly(pid_path(default_log_dir(root)))


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
