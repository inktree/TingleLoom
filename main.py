# -*- coding: utf-8 -*-
"""Tingleloom ASMR 根目录启动器：默认直接拉起 Gradio 工作台并打开浏览器。"""
from __future__ import annotations

import sys
from typing import Sequence

from tingleloom.cli import main as cli_main

SUBCOMMANDS = {"web", "run", "parse", "tts", "match", "material", "-h", "--help"}


def main(argv: Sequence[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    if len(raw) >= 2 and raw[0] == "--tts-dir":
        prefix, rest = raw[:2], raw[2:]
    elif raw and raw[0].startswith("--tts-dir="):
        prefix, rest = raw[:1], raw[1:]
    else:
        prefix, rest = (), raw
    if not rest or rest[0] not in SUBCOMMANDS:
        return cli_main([*prefix, "web", *rest])
    return cli_main(list(raw))


if __name__ == "__main__":
    sys.exit(main())
