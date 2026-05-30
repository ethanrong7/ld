"""Print RoundInit for a clip and save the captured template image."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ld.vision.template import analyze_round


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None, help="template PNG path")
    args = ap.parse_args()

    ri = analyze_round(args.input)
    print(f"clip            : {Path(args.input).name}")
    print(f"handoff_frame   : {ri.handoff_frame}")
    print(f"omega (deg/frm) : {ri.omega_deg_per_frame:.3f}")
    print(f"omega (deg/s)   : {ri.omega_deg_per_frame * 60:.1f}")
    print(f"center_seed     : ({ri.center_seed[0]:.1f}, {ri.center_seed[1]:.1f})")
    print(f"template_radius : {ri.template_radius:.1f}px")
    print(f"n_clean frames  : {ri.n_clean}")

    out = args.out or f"output/{Path(args.input).stem}_template.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, ri.template)
    print(f"template -> {out}")


if __name__ == "__main__":
    main()
