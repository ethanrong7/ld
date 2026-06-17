Session Summary & Future Recommendations

---

What We Did This Session

Detection improvements (SHIPPED)

Problem: Oracle was 0.915 on old model; t1 in particular was 0.806 because the shape drifted to the top-left edge where no training examples existed.

Solution: Built a full annotation pipeline (annotate_s.py) supporting drag-to-draw boxes with full/partial class toggle, extracted 5 targeted frames from t1's miss cluster (f225–300, shape at top-left corner), migrated all labels to single-class, trained yolov8n_single_combined.

Result: Oracle improved from 0.915 → 0.974 across all 10 clips. t1 oracle improved in lock-step — confirming that better detection directly lifts identity.

Key lesson: Targeted visual coverage of the specific failure position beats label schema changes at ~55 frames. Two-class (full/partial) split was tried and regressed oracle — halves instances per class.

---

Lock bug investigation (CLOSED)

Probed t1 and t4 lock behaviour with the new detector. Both now lock correctly:
- t1: anchor at (381.6, 229.3), picked box at 7.0px from GT
- t4: anchor at (342.8, 298.9), picked box at 36.2px from GT (within_r)

The 2.04r/2.57r errors in memory were old-model artifacts. The new detector resolves them.
Note: t4's anchor drifts slightly because the white detector jumps 67px at f92-f98 (below
the 80px WHITE_TELEPORT_PX guard), leaving the anchor ~57px off before the START-text
teleport at f99 finally breaks it. This is structural fragility but does NOT cause the
current t4 regression -- the picked box is within_r. Lock bug item is closed.

---

Failure Mode Taxonomy (all four laggards, fpath mode)

Probed per-frame trace CSVs (data/detect/eval/<clip>__fpath.csv) and lock diagnostics.

t1 (0.806) -- two patterns, both post-lock, both oracle=1
  f146-168 (23 frames): slow creep -- error builds 36->92px over ~8 frames then locks in
  f500-558 (59 frames): TELEPORT -- error jumps 53->108px in one frame, stays locked 59 frames

t4 (0.729) -- two patterns, very different root causes
  f107 (1 frame): single-frame hop, self-corrects
  f383-520 (138 frames): DETECTION GAP -- oracle_hit=0 right at onset; real shape drops
    out of detector for many frames; fpath reacquires to wrong box and holds it 138 frames.
    This is the dominant t4 loss. Not fixable by emission signal -- detector coverage needed.

t5 (0.718) -- TELEPORTS dominate, NOT creep (the old "CREEP not JUMP" diagnosis was field-era)
  f328-387 (60 frames): 50->176px jump in one frame, oracle=1 throughout
  f516-556 (41 frames): 48->198px jump in one frame, oracle=1 throughout
  Root cause: a far high-saliency fake outscores the real shape on outlier mass for one
  frame. Viterbi path switches and stays switched for 40-60 frames.

t8 (0.664) -- mixed: teleports + detection gaps
  f226-253 (~28 frames): detection gap cluster (oracle=0 at f232-239); destabilises track
  f284-380 (97 frames): TELEPORT -- 10->99px in one frame, oracle=1. DOMINANT t8 loss.
  f634 (9 frames): detection gap onset (oracle=0)
  f662-719 (58 frames): TELEPORT -- 49->182px in one frame, oracle=1

Frames-lost summary:
  Sudden teleports (oracle=1): t1 f500, t5 f328, t5 f516, t8 f284, t8 f662  ->  ~313 frames
  Detection gaps (oracle=0):   t4 f383, t8 f226, t8 f634                    ->  ~165 frames
  Slow creep (oracle=1):       t1 f146                                       ->   ~23 frames

---

Fix 1 -- Coherence-weighted emission (fpath_coh)
=================================================
Target: ~313 frames of teleport loss across t1/t5/t8. Highest EV.

ROOT CAUSE

At each teleport onset a far fake has momentarily higher outlier mass than the real shape.
Viterbi's emission is purely saliency mass -- scalar magnitude, no direction. The real shape
moves independently and coherently: its residual vectors all point in roughly the same
direction (the shape's own velocity relative to the sheet). A fake's residual features are
noise: they point randomly because the fake moves with the sheet and its "outlier" features
are just LK jitter with no consistent direction.

MotionField.outlier_vectors (motion.py:45) is an (M,2) array of per-feature residual
vectors already computed in estimate_motion() (motion.py:91). saliency_map() discards
direction and uses only magnitude. The coherence signal sits unused.

HOW TO VERIFY THE SIGNAL EXISTS BEFORE BUILDING

Run this diagnostic on the two clearest teleport onsets:

  python3 <<'EOF'
  import sys; sys.path.insert(0, ".")
  import cv2, math, numpy as np
  from ld.detect.fusion import detect_fusion_clip
  from ld.vision.cursor import strip_pointer
  from ld.vision.motion import estimate_motion
  from ld.capture.video_source import VideoSource

  WEIGHTS = "data/detect/runs/yolov8n_single_combined/weights/best.pt"

  def box_coherence_raw(box, field):
      x1,y1,x2,y2 = box[:4]
      mask = ((field.outliers[:,0]>=x1)&(field.outliers[:,0]<x2)&
              (field.outliers[:,1]>=y1)&(field.outliers[:,1]<y2))
      vecs = field.outlier_vectors[mask]
      if len(vecs) < 2: return 0.0
      norms = np.linalg.norm(vecs, axis=1, keepdims=True)
      valid = norms.ravel() > 1e-6
      if valid.sum() < 2: return 0.0
      unit = vecs[valid] / norms[valid]
      return float(np.linalg.norm(unit.mean(axis=0)))

  for clip_path, onset in [("data/t5_cropped_trimmed.mp4", 328),
                            ("data/t8_cropped_trimmed.mp4", 284)]:
      packs = detect_fusion_clip(WEIGHTS, clip_path)
      src = VideoSource(clip_path)
      frames = {}
      for idx, f in src.frames():
          frames[idx] = cv2.cvtColor(strip_pointer(f, strip_green=True), cv2.COLOR_BGR2GRAY)
          if idx > onset + 1: break
      src.release()
      p = packs[onset]
      fld = estimate_motion(frames[onset-1], frames[onset])
      gt = p.gt
      print(f"\n{clip_path} f{onset}  gt=({round(gt[0])},{round(gt[1])})")
      print(f"  {'i':>3}  {'cx':>6}  {'cy':>6}  {'gt_d':>6}  {'coh':>5}")
      for i, box in enumerate(p.boxes):
          cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
          gd=math.hypot(cx-gt[0],cy-gt[1])
          print(f"  {i:>3}  {cx:>6.1f}  {cy:>6.1f}  {gd:>6.1f}  {box_coherence_raw(box,fld):>5.3f}")
  EOF

  Expected: GT-nearest box scores coherence clearly above the winning fake (delta > 0.1).
  If separation < 0.1 on both clips, coherence won't discriminate -- stop here.

IMPLEMENTATION (three files, ~40 lines total)

A. ld/vision/motion.py -- add box_coherence() after saliency_map() (after line 107)

  def box_coherence(box: tuple, field: MotionField) -> float:
      """Mean resultant length of outlier residual vectors within a YOLO box.

      Returns R in [0,1]: R=1 means all vectors aligned (real shape moves coherently),
      R~0 means random directions (fake noise). Returns 0.0 if fewer than 2 outliers.
      """
      if field.outlier_vectors is None or len(field.outliers) == 0:
          return 0.0
      x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
      mask = (
          (field.outliers[:, 0] >= x1) & (field.outliers[:, 0] < x2) &
          (field.outliers[:, 1] >= y1) & (field.outliers[:, 1] < y2)
      )
      vecs = field.outlier_vectors[mask]
      if len(vecs) < 2:
          return 0.0
      norms = np.linalg.norm(vecs, axis=1, keepdims=True)
      valid = norms.ravel() > 1e-6
      if valid.sum() < 2:
          return 0.0
      unit = vecs[valid] / norms[valid]
      return float(np.linalg.norm(unit.mean(axis=0)))  # circular mean resultant R

  Also add "box_coherence" to __all__ at line 34.

B. ld/detect/identity.py -- add coh_w to track_fused_path_identity

  1. Add constant near the FPATH_* block (~line 1423):
       FPATH_COH_W = 0.0   # coherence emission weight; 0 = pure mass (current default)

  2. Add to function signature (~line 1437):
       coh_w: float = FPATH_COH_W,

  3. Add import at top of file (with other motion imports):
       from ld.vision.motion import estimate_motion, saliency_map, box_coherence

  4. After sal = saliency_map(fld, gray.shape) (~line 1502), compute coherence:
       if coh_w > 0:
           coh = np.array([box_coherence(b, fld) for b in p.boxes], np.float32)
       else:
           coh = np.zeros(len(p.boxes), np.float32)

  5. Modify emission (~line 1517):
       # was:  emis[i] = mass[i] + prox_w * prox
       emis[i] = mass[i] * (1.0 + coh_w * coh[i]) + prox_w * prox

  The multiplicative form is intentional: coherence rescales mass rather than adding
  to it. A box with mass=0 stays at 0 even with perfect coherence -- coherence boosts
  signal, it cannot manufacture signal from silence. At coh_w=1.0 the most coherent
  box gets at most 2x mass; an incoherent box gets 1x.

C. ld/detect/identity.py -- register fpath_coh mode

  1. Add tuning constant near FPATH_* block:
       FPATH_COH_W_DEFAULT = 1.0  # confirm by sweep below; adjust if needed

  2. In _dispatch_mode (~line 1708), add before the fpath_reacq entry:
       if mode == "fpath_coh":
           return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh,
                                            coh_w=FPATH_COH_W_DEFAULT)

  3. Add "fpath_coh" to ALL_MODES (search for ALL_MODES in identity.py).

TUNING

Sweep coh_w on t5 and t8 first (fast, ~2 min per run):

  python -m ld.detect.eval_modes \
    --weights data/detect/runs/yolov8n_single_combined/weights/best.pt \
    --modes fpath_coh --clips t5 t8

  Edit FPATH_COH_W_DEFAULT and re-run for values: 0.3, 0.5, 1.0, 1.5, 2.0.
  Pick the value that maximises t5+t8 sum with no clip below fpath baseline.

Full 10-clip LOO once a good value is found:

  python -m ld.detect.loo \
    --weights data/detect/runs/yolov8n_single_combined/weights/best.pt \
    --modes fpath fpath_coh

SUCCESS CRITERIA
- LOO mean(fpath_coh) > 0.855 (current fpath LOO)
- t5 improves from 0.718
- t8 improves from 0.664
- t1 improves from 0.806 (has one teleport at f500)
- No clip regresses by more than 0.005 vs fpath
- t4 stays within +/-0.010 of 0.729 (its loss is detection gap, emission cannot fix it)

WHAT TO EXPECT
- Teleport frames (oracle=1, far fake wins mass): coherence discriminates because the
  real shape's YOLO box contains features moving coherently; the fake's box contains
  random jitter. Expected lift: +0.03-0.08 on LOO mean.
- Detection-gap frames (oracle=0): no real shape box to score, no change expected
  on t4 f383, t8 f232, t8 f634.
- Risk: if coh_w > 1.5, a fake near a genuine independent-motion region can be falsely
  boosted. Monitor t2/t7/t10 (currently 0.963/0.993/0.957) for any regression -- that
  is the signal coh_w is too aggressive.

---

Fix 2 -- Detection gap coverage (retrain) -- do after Fix 1
============================================================
Target: ~165 frames where oracle_hit=0 (t4 f383-520, t8 f226-253, t8 f634-644).

These frames have oracle_hit=0: the real shape is not in any YOLO box. No emission fix
helps. The only lever is detector coverage.

HOW TO FIND THE MISSING POSITIONS

  python3 <<'EOF'
  import csv, pathlib
  BASE = pathlib.Path("data/detect/eval")
  for clip, gaps in [("t4", [(383, 420)]), ("t8", [(232, 242), (634, 644)])]:
      path = BASE / f"{clip}_cropped_trimmed__fpath.csv"
      rows = {int(r["idx"]): r for r in csv.DictReader(open(path))}
      print(f"\n{clip} detection gap frames:")
      for start, end in gaps:
          for f in range(start, end+1):
              if f in rows and rows[f]["oracle_hit"] == "0":
                  r = rows[f]
                  print(f"  f{r['idx']}  gt=({float(r['gt_x']):.1f},{float(r['gt_y']):.1f})")
  EOF

Extract and annotate those exact frames:

  python -m ld.detect.annotate_s   # extracts new frames + opens annotator

Navigate to the frame numbers above (p/n keys). Annotate the real shape box. Full class only.

Retrain:
  python -m ld.detect.build_s_dataset \
      --labels-dir data/detect/s_labels_single \
      --out-dir data/detect/dataset_single_combined
  python -m ld.detect.train \
      --data data/detect/dataset_single_combined/dataset.yaml \
      --name yolov8n_single_combined_v2

SUCCESS CRITERIA
- oracle_hit improves at the specific gap frames (t4 f383-420, t8 f232-242, t8 f634-644)
- Oracle LOO mean improves above 0.974
- No clip's oracle regresses

---

Fix 3 -- Dense pixel differencing complementarity check (exploratory, after Fix 1)
===================================================================================

Memory entry ld-dense-differencing-promising.md says warp-and-diff beats coherence on
t8 miss frames. The question is whether it corrects different frames.

After fpath_coh is shipped:
1. Collect per-frame results for fpath_coh on all clips
2. Find frames still wrong under fpath_coh (within_r=0, oracle=1)
3. For those frames, check if warp-and-diff ranks the GT box higher
4. If >60% of residual-miss frames are corrected by dense diff and NOT by coherence:
   build as a second multiplicative emission term
5. If <30% new lift: skip

---

Recommended Execution Order

1. Run verification diagnostic (5 min) -- confirm coherence separates at t5 f328, t8 f284
2. Implement box_coherence + fpath_coh (30 min)
3. Sweep coh_w on t5/t8, pick best (10 min)
4. Full LOO to confirm no regression (15 min)
5. If t4 still lags after Fix 1: retrain with gap-frame annotation (Fix 2)
6. After Fix 1 is stable: run dense diff complementarity check (Fix 3)

Do NOT revisit:
- trans_cap / fpath_reacq -- regime-coupling confirmed, dead end
- prox_w > 0 -- hurts every clip confirmed, dead end
- Lock bug -- closed, new detector resolves it
- Two-class YOLO -- halves instances per class, dead end
- Velocity cap / confirm-gate -- wrong lever for teleport failure mode
