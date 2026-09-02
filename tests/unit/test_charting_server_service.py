"""Unit tests for charting-server Windows task XML and pid helpers."""
from pathlib import Path

from utils.charting_server_service import (
    TASK_NAME,
    attach_service_stdio,
    build_task_xml,
    default_log_dir,
    pid_alive,
    pid_path,
    process_status,
    pythonw_path,
    read_pid,
    write_pid,
)


def test_task_xml_is_always_on_logon_job(tmp_path: Path):
    pythonw = tmp_path / "pythonw.exe"
    script = tmp_path / "charting_server.py"
    pythonw.write_text("", encoding="utf-8")
    script.write_text("", encoding="utf-8")
    xml = build_task_xml(
        pythonw, script, tmp_path, host="127.0.0.1", port=5001, user_id=r"DESKTOP\Hezi"
    ).decode("utf-16")

    assert r"\backTraderTest\ChartingServer" in xml
    assert "<LogonTrigger>" in xml
    assert r"<UserId>DESKTOP\Hezi</UserId>" in xml
    assert "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in xml
    assert "<RestartOnFailure>" in xml
    assert "<Hidden>false</Hidden>" in xml
    assert "--service" in xml
    assert "--host 127.0.0.1" in xml
    assert "--port 5001" in xml
    assert "IgnoreNew" in xml
    assert str(pythonw) in xml
    assert str(tmp_path) in xml
    assert TASK_NAME == r"backTraderTest\ChartingServer"


def test_xml_escapes_ampersand(tmp_path: Path):
    pythonw = tmp_path / "py & thonw.exe"
    script = tmp_path / "charting_server.py"
    xml = build_task_xml(pythonw, script, tmp_path).decode("utf-16")
    assert "py &amp; thonw.exe" in xml
    assert "py & thonw.exe" not in xml


def test_pid_roundtrip_and_alive(tmp_path: Path):
    import os

    log_dir = tmp_path / "logs"
    path = write_pid(log_dir, pid=os.getpid())
    assert path == pid_path(log_dir)
    assert read_pid(log_dir) == os.getpid()
    assert pid_alive(os.getpid()) is True
    assert pid_alive(0) is False


def test_process_status_missing_pid(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "utils.charting_server_service.default_log_dir",
        lambda root: tmp_path / "missing",
    )
    st = process_status(tmp_path)
    assert st["pid"] is None
    assert st["running"] is False


def test_attach_service_stdio_writes_banner(tmp_path: Path):
    import sys

    old_out, old_err, old_in = sys.stdout, sys.stderr, sys.stdin
    try:
        log_path = attach_service_stdio(tmp_path)
        print("hello-service")
        sys.stdout.flush()
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        sys.stdin = old_in
    text = log_path.read_text(encoding="utf-8")
    assert "charting server start" in text
    assert "hello-service" in text


def test_rebind_logging_streams_replaces_none_handler(tmp_path: Path):
    import logging
    import sys

    from utils.charting_server_service import _rebind_logging_streams

    log_path = tmp_path / "x.log"
    stream = open(log_path, "w", encoding="utf-8")
    handler = logging.StreamHandler(sys.stderr)
    handler.stream = None
    logger = logging.getLogger("charting_server_service_probe")
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        _rebind_logging_streams(stream)
        logger.info("rebound-ok")
        stream.flush()
    finally:
        logger.handlers[:] = []
        stream.close()
    assert "rebound-ok" in log_path.read_text(encoding="utf-8")


def test_pythonw_prefers_venv(tmp_path: Path):
    fake = tmp_path / "venv" / "Scripts"
    fake.mkdir(parents=True)
    exe = fake / "pythonw.exe"
    exe.write_text("", encoding="utf-8")
    assert pythonw_path(tmp_path) == exe


def test_default_log_dir():
    root = Path("D:/proj")
    assert default_log_dir(root) == root / "logs" / "charting_server"
