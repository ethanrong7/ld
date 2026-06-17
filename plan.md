Session Summary & Future Recommendations

What We Did This Session

Detection improvements (SHIPPED)

Problem: Oracle was 0.915 on old model; t1 in particular was 0.806 because the shape drifted to the top-left edge where no training examples existed.

Solution: Built a full annotation pipeline (annotate_s.py) supporting drag-to-draw boxes with full/partial class toggle, extracted 5 targeted frames from t1's miss cluster (f225–300, shape at top-left corner), migrated all labels to single-class, trained yolov8n_single_combined.

Result: Oracle improved from 0.915 → 0.974 across all 10 clips. t1 oraclemodes improved in lock-step — confirming that better detection directly lifts identity.                                                                                                                                                                         
Key lesson: Targeted visual coverage of the specific failure position beats label schema changes at ~55 frames. Two-class (full/partial) split was tried and regressed oracle — halves instances per class.
                                                                                                                                                                                  ---
Identity investigation (EXHAUSTED for fpath trellis modifications)                                                                                                                
Current best: fpath at within_r 0.855 (LOO). Laggards: t4 (0.729) and t8 (0.664).

fpath_reacq experiment (DEAD END):                                                                                                                                                - Added CV coast during detection gaps + reacquire teleport after 8 off-to limit trellis lock-in depth
- Narrow sweep on t4/t8 looked promising (t4 +0.018, t8 +0.007)                                                                                                                   - Full 10-clip run: fpath_reacq 0.845 vs fpath 0.855 (net −0.010)
- Killer: t3 −0.040, t5 −0.084. On t5 at f295, trans_cap allows path to relax and a high-saliency distant fake wins (266px teleport). Regime-coupling: the cap that rescues t4/t8 destabilises t3/t5.
- Root cause: no new signal introduced — trans_cap just changes which box wins under the same saliency signal. Without a discriminating signal on the hard frames, any relaxation of path continuity is a coin flip.

Why fpath can't be improved by trellis tuning alone:
On t4/t8 failure frames, the real shape has weaker saliency than a neighbouring fake. No Viterbi parameter changes what emission scores — they only decide how to weight path history vs current observation. Once the wrong box has higher mass, Viterbi will prefer it regardless of trans_cap.

---
Failure Mode Taxonomy (t4/t8)

┌──────┬──────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Clip │       Failure type       │                                                             Root cause                                                             │
├──────┼──────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ t4   │ Creep (f107)             │ Real shape has lower saliency than adath creeps one box over, path memory holds it off for 90+ frames │
├──────┼──────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ t8   │ Detection gap (f232–239) │ Oracle drops to 0 for 8 frames; trellg box                                                            │
└──────┴──────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

---
What's Left (Highest-EV First)

1. Coherence saliency channel (best unbuilt lever)

What: Fold directional coherence of residual optical-flow vectors into the saliency map. Currently saliency_map (in ld/vision/motion.py) uses only outlier magnitude. The real
shape's residual vectors should point coherently (it moves independently,t the same direction relative to the sheet). Fakes' residual vectors are
noise.

Why promising:
- field_coh already uses coherence as an override (far-jump trigger after 8 coherent frames) and it works — it's the reason field_coh beats field. But the override only fires on
already-large errors, not on first-creep.
- Folding coherence directly into the saliency map means fpath's emission scores would reflect coherence on every frame, not just as an escape valve.
- Memory index entry ld-coherence-override-win.md notes: "ungated challenger on-GT 50.2% of still-wrong+silent frames (t5 0.556!) → override is WRONG shape;
NEXT=coherence-weighted EMISSION integrator"

How: In saliency_map or a wrapper, compute per-box coherence score (mean cosine similarity of inlier residual vectors within each YOLO box), multiply into box mass before passing to fpath's emission. The coherence score is already computed in motion.py's MotionField.outlier_vectors — it just needs to be exposed at the box level.

Files: ld/vision/motion.py (expose outlier_vectors), ld/detect/identity.py (blend coherence into mass for fpath emission), ld/detect/fusion.py (pass coherence through FusionPack
if needed).

Expected lift: +0.05–0.08 per memory index (ld-coherence-override-win.md estimated this range). Would push toward ~0.91–0.93.

---
2. t1/t4 lock bug (orthogonal, concrete)

What: Countdown lock (compute_countdown_lock / _pick_lock_box in identity.py) lands on the wrong shape at game start for t1 and t4 (2.04r / 2.57r error). Reacquire masks it to ~0.70 but the initial lock is provably wrong.

Why: _pick_lock_box picks the highest-saliency box at lock time. On t1/t4, the real shape is not the highest-saliency box at countdown end — it may be partially obscured or near
an edge.

Fix direction: Look at what box is selected vs GT at lock time. May be a hape ranks 10–13), or a spatial bias issue (lock picks center of mass of allboxes, not GT-nearest). Low-risk change, orthogonal to the saliency signal problem.

Expected lift: ~+0.03–0.05 on t1/t4 specifically (removes a handicap that compounds through the whole clip).

---
3. Dense pixel differencing (exploratory, promising on t8)

What: Warp previous frame → current frame by affine, compute per-pixel difference, accumulate a registered heatmap, rank YOLO boxes by pixel-mass within the heat region.

Why: Memory entry ld-dense-differencing-promising.md notes this beats coherence on t8 laggard miss frames (top1 0.596 vs bar 0.70) — first signal to do so. Complementary: strong
on t8, weak on t5. If the miss frames are disjoint from coherence misses,s the single-channel ceiling.

Next step: Complementarity check — do dense-diff correct frames overlap with coherence correct frames? If largely disjoint, combine as a second emission channel in fpath.

---
Recommended Execution Order

1. Coherence-weighted emission — highest EV, builds on proven signal, no
2. Lock bug fix — concrete, low-risk, free improvement on t1/t4
3. Dense differencing complementarity check — low cost diagnostic before

Do not revisit trans_cap / fpath_reacq variants — the regime-coupling wal

The plan is saved. Here's the tl;dr of this session and what I'd prioritise

What we did:
- Built the annotation toolchain and trained yolov8n_single_combined — oracle jumped from 0.915 → 0.974 by adding 5 targeted t1 edge frames. Identity within_r followed (fpath 0.855,
up from ~0.74).
- Investigated fpath trellis improvements (fpath_reacq with CV coast + trans4/t8 (+0.018/+0.007) but kills t3/t5 (−0.040/−0.084). Regime-coupling is
structural: no trellis parameter changes what the emission scores, and on thsimply has weaker saliency than the winning fake.

Root cause of remaining failures:
- t4: real shape lower-saliency than adjacent fake at f107 → creep, then 90-
- t8: detection gap (8 frames, oracle_hit=0) → trellis reset → wrong reacqui

Recommended next steps (highest EV first):
1. Coherence-weighted emission — fold directional coherence of residual flow vectors into per-box saliency scores. field_coh already proves this signal works as an override; moving it upstream into emission is the natural next step. Estimated +0.05–0.08.
2. Lock bug fix — _pick_lock_box selects wrong shape at countdown end on t1/t4. Cheap, orthogonal.
3. Dense differencing complementarity check — warp-and-diff was strong on t8; worth checking if it's disjoint from coherence misses before building.