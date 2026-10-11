# SAS-style star-based white balance for Siril 1.4

Run `Siril_SAS_White_Balance.py` with a linear RGB image open, before stretching.
Install this entire folder as a Siril custom script folder. Keep
`sas_white_balance_core.py` beside the main script, and preserve `LICENSE` and
`NOTICE.md`. Dependencies (numpy, sep, PyQt6)
are requested through Siril's dependency installer.

1. Click **Analyze / Preview**. Auto detection uses SAS's sigma ladder
   (100, 50, 25, 12, 8, 5, 3), targeting 1200 detections before filtering.
2. Toggle **Show original** to compare. Each view uses a display-only linked
   RGB midtones auto-stretch, fitted to its own median and robust background
   spread. One black point and midtones parameter serve all three channels,
   preserving the cast rather than independently balancing them. Green circles
   show sampled stars. Drag the preview to pan.
3. If detection needs adjustment, disable Auto detection, set the sigma
   threshold, and analyze again.
4. Click **Apply** to update Siril's current image with an undo state.
   The image remains linear. The source file is not overwritten.

Analysis runs on the full-resolution image in a worker thread. Only display
is reduced. Apply refuses if the current pixels changed while the dialog was
open. Close is deferred while analysis runs.

## Algorithm and provenance

Unofficial GPL-3.0-only adaptation of Franklin Marek and SAS Pro contributors:
https://github.com/setiastro/setiastrosuitepro

Adapted from `src/setiastro/saspro/imageops/starbasedwhitebalance.py` and
`src/setiastro/saspro/backgroundneutral.py`, retrieved 2026-10-11.
The accompanying `LICENSE` contains GPL v3.

Includes whole-image per-channel pedestal subtraction, automatic 50×50
background selection, highlight-pivot background neutralization, SEP
extraction and flux-radius filtering, 3×3 spatial sampling (500 stars/cell),
and radius-3 circular median stellar samples. The standard SAS per-channel
scaling is applied around the neutralized background pivot, then clipped to
0–1. This can clip highlights, as in the source implementation.

Port changes: fixed random seed for repeatable background selection;
small-image bounds handling; explicit finite-value validation; Siril channel
layout and undo integration; linked RGB display-only preview stretch with
smooth image reduction. Its appearance is an approximation, not an exact
match to Siril's display settings. Source
caching is disabled. Advanced color-matrix mode and SAS diagnostic plots are
not included. This is empirical stellar color balance, not catalog-based
photometric calibration. HaRGB behavior depends on the stars retained in
that composite and the selected background region.

Validation: synthetic cast star field (manual and automatic detection),
neutral stellar medians within 2%, bounds and finite values, input preservation,
repeatable background selection, small-image bounds, blank-image rejection.
The user verified Apply and white-balance results on a real HaRGB stack in
Siril. The revised preview passed automated window and rendering checks.

Floating-point input outside 0–1 is rescaled with one shared transform across
RGB: `(value - low) / (high - low)`, where `low = min(0, image minimum)` and
`high = max(1, image maximum)`. This preserves extremes without independently
normalizing channels. Input already within 0–1 stays unchanged. This input
handling differs from SAS Pro's clipping. The background neutralization and
final white-balance steps still use SAS's bounded output behavior. Original
Siril pixels remain untouched until Apply.
