"""CLI entry point."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lie Detector solver")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", help="Probe t*_cropped_trimmed test videos (Phase 0)")
    offline = sub.add_parser("offline", help="Annotated passthrough video (Phase 1)")
    offline.add_argument("--input", "-i", type=str, default=None)
    offline.add_argument("--output", "-o", type=str, default=None)
    offline.add_argument("--max-frames", type=int, default=None)

    args, rest = parser.parse_known_args(argv)

    if args.command == "probe":
        from ld.debug.probe_videos import main as probe_main

        return probe_main(rest)

    if args.command == "offline":
        from ld.debug.run_offline import main as offline_main

        cmd = []
        if args.input:
            cmd.extend(["--input", args.input])
        if args.output:
            cmd.extend(["--output", args.output])
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])
        cmd.extend(rest)
        return offline_main(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
