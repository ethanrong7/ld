"""LD solver CLI — fresh start.

Current scope: pointer stripping only.
"""
from __future__ import annotations

import argparse


def _strip_preview(args) -> None:
    from ld.debug.strip_preview import run

    run(args.input, args.out_video)


def main() -> None:
    ap = argparse.ArgumentParser(description="LD solver")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("strip-preview", help="write original | stripped side-by-side MP4")
    p.add_argument("--input", required=True)
    p.add_argument("--out-video", default=None)
    p.set_defaults(func=_strip_preview)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
