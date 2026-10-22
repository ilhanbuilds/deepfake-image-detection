"""Actual owned Windows process trees and loopback startup; no model inference."""

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from src.desktop_job import WindowsJob
from src import desktop_runtime as runtime

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows lifecycle integration")


def test_job_close_kills_owned_grandchild_but_not_sibling():
    code = (
        "import sys,subprocess,time;sys.stdin.readline();"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "print(p.pid,flush=True);time.sleep(60)"
    )
    sibling = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    parent = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    job = WindowsJob()
    child_handle = None
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    try:
        job.assign(parent)
        parent.stdin.write(b"go\n")
        parent.stdin.flush()
        child = int(parent.stdout.readline())
        child_handle = api.OpenProcess(0x100000, False, child)  # SYNCHRONIZE only
        assert child_handle
        job.close()
        parent.wait(timeout=5)
        assert api.WaitForSingleObject(child_handle, 5000) == 0
        assert sibling.poll() is None
    finally:
        job.close()
        if child_handle:
            api.CloseHandle(child_handle)
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        parent.stdin.close()
        parent.stdout.close()
        sibling.terminate()
        sibling.wait(timeout=5)


def test_real_port_collision_recovers_and_close_releases_listener(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    choose = runtime.choose_port
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        occupied = other.getsockname()[1]
        ports = iter((occupied, choose()))
        monkeypatch.setattr(runtime, "choose_port", lambda: next(ports))
        server = runtime.DesktopServer(root)
        try:
            url = server.start(timeout=25, attempts=2)
            assert runtime.healthy(url)
            assert server.process.poll() is None
            assert other.getsockname()[1] == occupied
            assert (
                f"Port {occupied} is not available"
                in (server.session / "server-0.log").read_text()
            )
            server.stop()
            assert not runtime.healthy(url)
            assert other.fileno() != -1
        finally:
            server.stop()
