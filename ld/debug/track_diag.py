"""Step 7/8 diagnostic: run the solver on one clip, score vs GT, write video.

Thin wrapper over `ld.solver.track_video`. The green-cursor GT is read here (in
the debug layer) only to measure error and draw the reference marker.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import open_writer
from ld.solver import track_video
from ld.vision.cursor import find_cursor
from ld.vision.template import analyze_round


def run(
    input_path: str,
    out_video: str | None,
    *,
    strip_pointer_from_frames: bool = True,
) -> None:
    ri = analyze_round(input_path, strip_pointer_from_frames=strip_pointer_from_frames)
    errs: list[float] = []
    coast = {"n": 0}
    writer = {"w": None}
    radius = {"r": ri.template_radius}

    def on_frame(idx, frame, st):
        if writer["w"] is None and out_video:
            h, w = frame.shape[:2]
            writer["w"] = open_writer(out_video, w, h, 60.0)
        if not st.measured:
            coast["n"] += 1
        gt = find_cursor(frame)
        if gt:
            errs.append(math.hypot(st.x - gt[0], st.y - gt[1]))
        if writer["w"] is not None:
            vis = frame.copy()
            col = (0, 255, 0) if st.measured else (0, 165, 255)
            ix, iy = int(st.x), int(st.y)
            r = int(radius["r"])
            cv2.rectangle(vis, (ix - r, iy - r), (ix + r, iy + r), col, 2)
            cv2.drawMarker(vis, (ix, iy), col, cv2.MARKER_CROSS, 16, 2)
            if gt:
                cv2.circle(vis, (int(gt[0]), int(gt[1])), 6, (0, 0, 255), 2)
            tag = "TRK" if st.measured else "coast"
            for c, t in ((( 0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(vis, f"f{idx} {tag}", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, t, cv2.LINE_AA)
            writer["w"].write(vis)

    res = track_video(
        input_path,
        on_frame=on_frame,
        init=ri,
        strip_pointer_from_frames=strip_pointer_from_frames,
    )
    print(f"{Path(input_path).name}: handoff={res.init.handoff_frame} "
          f"omega={res.init.omega_deg_per_frame:.2f}/f seed={res.init.center_seed}")
    if writer["w"] is not None:
        writer["w"].release()
        print(f"wrote -> {out_video}")
    if errs:
        a = np.array(errs)
        print(f"scored={len(a)} mean={a.mean():.1f} median={np.median(a):.1f} "
              f"p90={np.percentile(a,90):.1f} max={a.max():.1f} coast_frames={coast['n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-video", default=None)
    ap.add_argument(
        "--no-strip-pointer",
        action="store_true",
        help="disable green/mouse inpaint before tracking (ablation only)",
    )
    args = ap.parse_args()
    run(args.input, args.out_video, strip_pointer_from_frames=not args.no_strip_pointer)


if __name__ == "__main__":
    main()
