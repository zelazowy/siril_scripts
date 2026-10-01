# Siril scripts

Custom and modified scripts for astronomical image processing with Siril: review exposures, preprocess and stack OSC data, align finished stacks, solve images with ASTAP, and stretch a stars layer.

## Python utilities

These utilities run from Siril's GUI with Python scripting support. Each linked guide explains installation, dependencies, controls, and limitations.

| Script | What it does | Typical use |
| --- | --- | --- |
| [Align Stacks](python/Siril_Align_Stacks/) | Aligns a few RGB or mono FITS stacks with translation, rotation and scale; provides adaptive previews and automatic or manual shared cropping; saves copies with an `_aligned` suffix. | Prepare RGB and narrowband stacks for continuum subtraction without creating a sequence or intermediate image files. |
| [ASTAP Plate Solve](python/Siril_ASTAP_PlateSolve/) | Solves the current FITS or TIFF image with ASTAP and imports the coordinate solution into Siril. | Add astrometry to an image for coordinate-aware processing. |
| [Blink / Browse / Filter / Sort](python/Blink_Browse_Filter_Sort/) | Reviews sequences or image folders, compares image quality, recommends keepers, and sorts or filters frames. | Inspect exposures and choose frames before stacking. |
| [SAS-style Star Stretch](python/Siril_SAS_Star_Stretch/) | Stretches a stars-only mono or RGB image using SAS Pro Star Stretch mathematics, with colour boost, optional SCNR, and a preview. | Adjust a separated stars layer before recombining it with the starless image. |

See [Siril's custom script installation instructions](https://siril.readthedocs.io/en/stable/scripts/Script-files.html#adding-custom-scripts-folders). Preserve each utility's license and attribution files when installing it.

## OSC preprocessing and stacking

The `.ssf` scripts convert lights, optionally build or reuse calibration masters, register with Bayer drizzle, and save a stacked result. The table describes the commands currently in each file. Choose the variant that matches your calibration data and inspect its paths and settings before running it.

| Script | Calibration | Drizzle settings |
| --- | --- | --- |
| [OSC_Preprocessing_BayerDrizzle.ssf](stacking/OSC_Preprocessing_BayerDrizzle.ssf) | Builds bias, flat and dark masters in `masters/`; calibrates lights with dark and flat, including dark-based cosmetic correction. | 1× scale, pixel fraction 1.0, square kernel. |
| [OSC_Preprocessing_BayerDrizzle_Without_dark.ssf](stacking/OSC_Preprocessing_BayerDrizzle_Without_dark.ssf) | Builds bias and flat masters in `masters_driz/`; calibrates lights with bias and flat. | 1× scale, pixel fraction 1.0, square kernel. |
| [OSC_Preprocessing_BayerDrizzle_Without_dark_2x.ssf](stacking/OSC_Preprocessing_BayerDrizzle_Without_dark_2x.ssf) | Builds bias and bias-calibrated flat masters in `masters/`; calibrates lights with flat only. | 2× scale, pixel fraction 1.0, square kernel. |
| [OSC_Preprocessing_BayerDrizzle_Without_dark_2x_v2.ssf](stacking/OSC_Preprocessing_BayerDrizzle_Without_dark_2x_v2.ssf) | Builds bias and flat masters in `masters_driz/`; calibrates lights with bias and flat. | 2× scale, pixel fraction 1.0, square kernel. |
| [OSC_Preprocessing_BayerDrizzle_Without_dark_2x_masters.ssf](stacking/OSC_Preprocessing_BayerDrizzle_Without_dark_2x_masters.ssf) | Reuses `masters/bias_stacked` and `masters/pp_flat_stacked`; calibrates lights with bias and flat. | 2× scale, pixel fraction 1.0, square kernel. |
| [OSC_Preprocessing_BayerDrizzle_wo_dbf_2x.ssf](stacking/OSC_Preprocessing_BayerDrizzle_wo_dbf_2x.ssf) | Converts and stacks lights without dark, bias or flat calibration. | **1.5× scale**, pixel fraction 0.9, square kernel, despite the `2x` filename. |

Run these from the working directory containing the input subfolders used by the chosen script, such as `lights/`, `biases/`, `flats/`, and `darks/`. The masters variant needs existing masters instead of raw calibration frames. These variants use shared `process/` and master locations, so inspect existing files before reusing a working directory.

## Keeping this catalog current

**Every new script added to this repository must also update this main README in the same commit or pull request.** Add a link, a concise purpose, and the key inputs or workflow differences in the appropriate table. For Python utilities, include a per-script README describing installation and usage. **Updates to existing scripts must also update this main README in the same commit or pull request whenever they change the documented purpose, inputs, settings, requirements, or workflow.** When a script is renamed or removed, update its catalog entry and links too. Include every runnable utility and stacking variant; supporting modules and tests do not need separate entries.

## Attribution and licensing

Some scripts are modified versions of community contributions. Preserve the copyright and license notices in each file; licenses are stated per script. The blink utility is based on Adrian Knagg-Baugh's GPL-3.0-or-later script. SAS-style Star Stretch is an unofficial GPL-3 adaptation of the SAS Pro transform; see its license and notice files. Stacking scripts retain their upstream attribution in their headers.
