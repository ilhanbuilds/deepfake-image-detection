"""Wait for ownership assignment before entering the unchanged model worker."""

import runpy
import sys


if __name__ == "__main__":
    if sys.stdin.readline().strip() != "start":
        raise SystemExit(3)
    runpy.run_module("scripts.demo_worker", run_name="__main__")
