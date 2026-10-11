# Attribution and provenance

Unofficial standalone Siril port of **Seti Astro Suite Pro** star-based white
balance, by Franklin Marek and SAS Pro contributors. This is not an official
SAS Pro release or endorsed integration. SAS Pro itself is not bundled or required.
Upstream rights remain with their respective holders.

Upstream: https://github.com/setiastro/setiastrosuitepro
Source revision: `6e009ec2847c2c87c04b6779bcad38be6d8345fd`, retrieved 2026-10-11.

Source files adapted:
- `src/setiastro/saspro/imageops/starbasedwhitebalance.py`
- `src/setiastro/saspro/backgroundneutral.py`

Input conversion was checked against `src/setiastro/saspro/widgets/image_utils.py`.
The upstream repository provides GNU GPL version 3 in `LICENSE`, reproduced
verbatim here. The adapted algorithm files have no file-level license notices.

## Siril adaptation — 2026-10-11

Siril adaptation Copyright (C) 2026 zelazowy. Distributed under GPL-3.0-only.
Changes include Siril channel layout, undo and changed-image checks; threaded
full-resolution analysis; a repeatable background search and small-image bounds;
shared RGB input rescaling rather than clipping out-of-range input; linked RGB
preview auto-stretch and smooth resizing. SAS application integration, source
cache reuse, diagnostic plots, and advanced color-matrix correction are omitted.
Background and final output clipping retain the upstream algorithm's behavior.

Runtime dependencies (sirilpy, NumPy, SEP, and PyQt6) are installed separately,
not bundled, and retain their own licenses. PyQt6's open-source option is GPL v3;
no commercial PyQt license is supplied.
