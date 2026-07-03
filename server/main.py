#!/usr/bin/env python3
"""
Qwen Image Edit - local CLI smoke test (optional).

This is a convenience script to verify the model loads and runs on the server
box WITHOUT going through the HTTP API. It uses the same memory-efficient
``ModelManager`` as the server, so if this works, the server will too.

Usage:
    python main.py [input_image] [prompt] [output_image]

Example:
    python main.py input.png "make him wear cool gaming headphones" output.png
"""

import os
import sys

from PIL import Image

from model_manager import ModelManager


def main() -> None:
    print("🖼️  Qwen Image Edit — local CLI test")
    print("=" * 50)

    if len(sys.argv) == 1:
        input_path = "input.png"
        prompt = "make him wear cool gaming headphones."
        output_path = "output_image_edit.png"
    elif len(sys.argv) == 4:
        input_path, prompt, output_path = sys.argv[1:4]
    else:
        print('Usage: python main.py [input_image] [prompt] [output_image]')
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        sys.exit(1)

    manager = ModelManager()
    manager.load()

    print(f"🎨 Editing '{input_path}' with prompt: {prompt!r}")
    image = Image.open(input_path)
    result = manager.run([image], prompt)
    result.save(output_path)
    print(f"✅ Saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
