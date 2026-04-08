from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
NODE_MODULES = FRONTEND_DIR / "node_modules"


def run(command: list[str], cwd: Path | None = None) -> None:
    if os.name == "nt" and command[0] == "npm":
        command = ["npm.cmd", *command[1:]]
    subprocess.check_call(command, cwd=str(cwd or BASE_DIR))


def ensure_python_requirements() -> None:
    try:
        import flask  # noqa: F401
    except ModuleNotFoundError:
        try:
            import pip  # noqa: F401
        except ModuleNotFoundError:
            run([sys.executable, "-m", "ensurepip", "--upgrade"])
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def ensure_frontend_dependencies() -> None:
    if not NODE_MODULES.exists():
        run(["npm", "install"], cwd=FRONTEND_DIR)


def ensure_frontend_build() -> None:
    run(["npm", "run", "build"], cwd=FRONTEND_DIR)


def main() -> None:
    os.chdir(BASE_DIR)
    ensure_python_requirements()
    ensure_frontend_dependencies()
    ensure_frontend_build()
    os.execvp(sys.executable, [sys.executable, "server.py"])


if __name__ == "__main__":
    main()
