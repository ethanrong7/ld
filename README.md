# LD Solver

Computer-vision solver for MapleStory Lie Detector clips.

## Fresh start

The previous tracker / template / distractor pipeline has been removed. The one
piece kept from that work is **pointer stripping** — inpainting the green GT
crosshair (and optionally a live mouse disk) before any vision processing.

## Pointer stripping

`ld/vision/cursor.py`:

- `strip_pointer(frame)` — inpaint green crosshair pixels (t* eval clips)
- `find_cursor(frame)` — locate green GT centroid (scoring only, never tracking input)
- `mouse_xy` kwarg — inpaint a disk at the live cursor position

Tunables in `ld/config.py`: `GREEN_*`, `POINTER_INPAINT_RADIUS`, `POINTER_RADIUS`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
set PYTHONPATH=.
```

Clips live in `data/` (gitignored). Test clips: `data/t*_cropped_trimmed.mp4`.

## CLI

Preview stripping on a clip (original | stripped, no cursor overlays):

```bash
python -m ld.main strip-preview --input data/t1_cropped_trimmed.mp4
```

Output: `output/t1_cropped_trimmed_stripped.mp4`
