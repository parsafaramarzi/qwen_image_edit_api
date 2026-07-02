#!/usr/bin/env python3
"""Launcher for the Qwen Image Editor client GUI."""

import importlib
import sys


def main() -> int:
    print("🖼️  Qwen Image Editor — Client Launcher")
    print("=" * 40)

    missing = []
    for pkg in ("PIL", "requests", "tkinter"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        if "tkinter" in missing:
            print("   (Linux: sudo apt-get install python3-tk)")
        return 1

    from client_gui import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
