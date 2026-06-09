#!/usr/bin/env python3
"""Platform-independent launcher for the Stock Tools MCP server."""

import os
import sys
import shutil


def find_uv():
    """Find the uv executable in common installation locations."""
    # 1. Check if uv is in PATH
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path

    home = os.path.expanduser("~")

    # 2. Platform-specific common locations
    if sys.platform == "win32":
        candidates = [
            os.path.join(os.environ.get("APPDATA", ""), "Python", "Scripts", "uv.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "uv", "uv.exe"),
            os.path.join(home, ".local", "bin", "uv.exe"),
            os.path.join(home, ".cargo", "bin", "uv.exe"),
        ]
    elif sys.platform == "darwin":
        # macOS - check Homebrew (both Apple Silicon and Intel) and user installs
        candidates = [
            "/opt/homebrew/bin/uv",  # Homebrew on Apple Silicon
            "/usr/local/bin/uv",  # Homebrew on Intel Mac
            os.path.join(home, ".local", "bin", "uv"),  # Official installer
            os.path.join(home, ".cargo", "bin", "uv"),  # Cargo install
        ]
    else:
        # Linux and other Unix-like systems
        candidates = [
            "/usr/local/bin/uv",
            "/usr/bin/uv",
            os.path.join(home, ".local", "bin", "uv"),  # Official installer
            os.path.join(home, ".cargo", "bin", "uv"),  # Cargo install
        ]

    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def main():
    """Launch the Stock Tools MCP server using uv."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    uv_executable = find_uv()
    if not uv_executable:
        print("Error: 'uv' executable not found.", file=sys.stderr)
        print("Please install uv: https://docs.astral.sh/uv/", file=sys.stderr)
        print("Checked locations:", file=sys.stderr)
        print("  - PATH", file=sys.stderr)
        if sys.platform == "darwin":
            print("  - /opt/homebrew/bin/uv", file=sys.stderr)
            print("  - /usr/local/bin/uv", file=sys.stderr)
        print(f"  - {os.path.expanduser('~/.local/bin/uv')}", file=sys.stderr)
        print(f"  - {os.path.expanduser('~/.cargo/bin/uv')}", file=sys.stderr)
        sys.exit(1)

    # uv run handles venv creation and dependency installation automatically
    os.execv(uv_executable, [uv_executable, "run", "stock-tools-mcp-server"])


if __name__ == "__main__":
    main()