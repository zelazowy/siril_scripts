# Siril scripts

Custom and modified scripts for astronomical image processing with Siril.

## Python utilities

| Script | Purpose |
| --- | --- |
| [ASTAP Plate Solve](python/Siril_ASTAP_PlateSolve/) | Solve the current FITS or TIFF image with ASTAP and import the coordinate solution into Siril. |
| [Blink / Browse / Filter / Sort](python/Blink_Browse_Filter_Sort/) | Review image sequences or folders, compare image quality, recommend keepers, and sort or filter frames. |

Use these utilities from Siril's GUI with Python scripting support. Each script's README describes its dependencies and workflow. See [Siril's custom script installation instructions](https://siril.readthedocs.io/en/stable/scripts/Script-files.html#adding-custom-scripts-folders).

## Stacking scripts

The [stacking](stacking/) folder contains OSC preprocessing and Bayer drizzle variants. Choose the variant appropriate for your calibration frames and inspect its settings before running it.

## Attribution and licensing

Some scripts are modified versions of community contributions. Preserve the copyright and license notices in each file; licenses are stated per script. The blink utility is based on Adrian Knagg-Baugh's GPL-3.0-or-later script.
