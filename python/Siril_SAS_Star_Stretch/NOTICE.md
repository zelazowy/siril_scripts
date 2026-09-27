# Attribution and provenance

This is an unofficial standalone Siril adaptation of the Star Stretch mathematics
used in **Seti Astro Suite Pro**, authored by Franklin Marek with SAS Pro contributors.
It does not launch, bundle, or require SAS Pro, and is not presented as an official
SAS Pro release or an endorsed integration.

Upstream project: https://github.com/setiastro/setiastrosuitepro

Source files consulted on 2026-09-27:

- `src/setiastro/saspro/star_stretch.py`: processing order, mean-based colour boost,
  and control ranges/defaults.
- `src/setiastro/saspro/legacy/numba_utils.py`: `applyPixelMath_numba` and
  `applySCNR_numba`.

The consulted files contain no file-level copyright or license notices. The
upstream repository's `LICENSE` contains GNU GPL version 3. A verbatim copy is
included here. Upstream rights remain with their respective holders.

## Siril adaptation — 2026-09-27

Siril adaptation Copyright (C) 2026 zelazowy. Distributed under GPL-3.0-only.

The NumPy implementation replaces the upstream Numba loops. The interface and
Siril connection are adapted from the local `Siril_Star_Stretch.py` utility.
Changes include Siril channel layout handling, downscaled preview, undo-state
creation, and a check that the image pixels have not changed before applying.
The SAS Pro application, document system, masking, and preset support are not
included. Numerical agreement was checked for normalized float32 inputs; this is
not a claim that all SAS Pro image-loading or interface behaviour is reproduced.

Runtime dependencies (sirilpy, NumPy, OpenCV and PyQt6) are installed separately,
not bundled. They retain their respective licenses. PyQt6's open-source option
is GPL v3; no commercial PyQt license is supplied.
