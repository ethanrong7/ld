"""LD solver CLI.



Subcommands (each delegates to a module under ld/):

  probe     print metadata for clips matching a glob

  template  capture shape template + rotation rate for one clip

  analyze   per-frame white-shape + GT diagnostic (CSV/video)

  residual  motion-residual heatmap vs GT (video)

  track     run the solver on one clip, score vs GT, optional annotated video

  eval      run the solver on all clips, print table + write summary CSV



By default every solver/motion pathway strips pointer pixels (green GT on t*

clips; live mouse disk when ``mouse_at`` is wired) before grayscale. Use

``--no-strip-pointer`` only for ablation comparisons.

"""

from __future__ import annotations



import argparse

from pathlib import Path



from ld.capture.video_source import VideoSource

from ld.config import DATA_DIR





def _strip_flag(ap: argparse.ArgumentParser) -> None:

    ap.add_argument(

        "--no-strip-pointer",

        action="store_true",

        help="do not inpaint green GT / mouse before tracking (ablation only)",

    )





def _probe(args) -> None:

    for c in sorted(DATA_DIR.glob(args.glob)):

        try:

            m = VideoSource(str(c)).meta

            print(f"{c.name:<28} {m.width}x{m.height} {m.fps:.1f}fps "

                  f"frames={m.frame_count} dur={m.duration:.2f}s")

        except Exception as e:

            print(f"{c.name:<28} ERROR: {e}")





def _template(args) -> None:

    import cv2

    from ld.vision.template import analyze_round

    ri = analyze_round(

        args.input, strip_pointer_from_frames=not args.no_strip_pointer

    )

    print(f"handoff={ri.handoff_frame} omega={ri.omega_deg_per_frame:.3f}/f "

          f"({ri.omega_deg_per_frame * 60:.1f}/s) "

          f"seed=({ri.center_seed[0]:.1f},{ri.center_seed[1]:.1f}) "

          f"radius={ri.template_radius:.1f} n_clean={ri.n_clean}")

    out = args.out or f"output/{Path(args.input).stem}_template.png"

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(out, ri.template)

    print(f"template -> {out}")





def _analyze(args) -> None:

    from ld.debug.analyze import run

    run(args.input, args.out_video, args.out_csv, args.max_frames)





def _residual(args) -> None:

    from ld.debug.residual_diag import run

    run(

        args.input,

        args.out_video,

        args.start,

        strip_pointer_from_frames=not args.no_strip_pointer,

    )





def _track(args) -> None:

    from ld.debug.track_diag import run

    run(args.input, args.out_video, strip_pointer_from_frames=not args.no_strip_pointer)





def _eval(args) -> None:

    from ld.debug.eval import run

    run(args.glob, args.csv, strip_pointer_from_frames=not args.no_strip_pointer)





def main() -> None:

    ap = argparse.ArgumentParser(prog="ld")

    sub = ap.add_subparsers(dest="cmd", required=True)



    p = sub.add_parser("probe", help="print clip metadata")

    p.add_argument("--glob", default="t*_cropped_trimmed.mp4")

    p.set_defaults(func=_probe)



    p = sub.add_parser("template", help="capture template + rotation rate")

    p.add_argument("--input", required=True)

    p.add_argument("--out", default=None)

    _strip_flag(p)

    p.set_defaults(func=_template)



    p = sub.add_parser("analyze", help="white-shape + GT diagnostic")

    p.add_argument("--input", required=True)

    p.add_argument("--out-video", default=None)

    p.add_argument("--out-csv", default=None)

    p.add_argument("--max-frames", type=int, default=None)

    p.set_defaults(func=_analyze)



    p = sub.add_parser("residual", help="motion-residual heatmap vs GT")

    p.add_argument("--input", required=True)

    p.add_argument("--out-video", required=True)

    p.add_argument("--start", type=int, default=117)

    _strip_flag(p)

    p.set_defaults(func=_residual)



    p = sub.add_parser("track", help="run solver on one clip + score")

    p.add_argument("--input", required=True)

    p.add_argument("--out-video", default=None)

    _strip_flag(p)

    p.set_defaults(func=_track)



    p = sub.add_parser("eval", help="run solver on all clips + summary")

    p.add_argument("--glob", default="t*_cropped_trimmed.mp4")

    p.add_argument("--csv", default="output/eval_summary.csv")

    _strip_flag(p)

    p.set_defaults(func=_eval)



    args = ap.parse_args()

    args.func(args)





if __name__ == "__main__":

    main()


