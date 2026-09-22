# Blink / Browse / Filter / Sort

A modified version of Adrian Knagg-Baugh's image review utility for Siril. Browse a loaded sequence or the images in Siril's working directory, compare frames, and select which ones to keep.

## Installation

1. Use Siril with Python scripting support (Siril 1.4).
2. Install `Blink_Browse_Filter_Sort.py` in a configured custom scripts folder, following [Siril's installation instructions](https://siril.readthedocs.io/en/stable/scripts/Script-files.html#adding-custom-scripts-folders).
3. Run it from Siril's Scripts menu. The script requests PyQt6, psutil, matplotlib and pytz through `sirilpy.ensure_installed`.

## Changes in this version

- Color previews for Bayer/CFA images, using the pattern and offsets from image metadata; preview processing leaves source pixels unchanged.
- Linked previews for RGB images and XISF support in directory browsing.
- Unrelated files remain untouched when opening a directory; color images remain available during analysis.
- **Recommend selection** ranks analysed frames by background noise, FWHM, star count and roundness, then selects the requested keeper percentage. Unanalysed frames remain included.
- **Review excluded frames only** lets you inspect proposed rejections and rescue frames before applying the filter.
- **Apply filter** asks for confirmation before moving excluded files, avoids overwriting existing rejected files, and preserves frames whose moves fail.

The script retains the upstream `2.0.0` version label; the features above describe this repository's modified edition.

## Workflow

With a sequence loaded, browse its selected frames and use **Toggle Include** or **X** to change inclusion in Siril. Quality analysis and recommendations are available in directory mode.

In directory mode:

1. Select a working folder containing FITS, XISF or supported camera RAW images.
2. Run **Analyse files** and inspect the measurements or plots.
3. Set quality thresholds or use **Recommend selection** to choose a keeper percentage.
4. Review excluded frames and adjust inclusion manually.
5. Use **Apply filter** when satisfied. It moves excluded files to a `rejected` folder in the working directory's **parent**, leaving included files in place. Moving sequence files may invalidate existing sequence references.

**Sort** is a separate operation that moves lights and calibration frames into subdirectories, groups sessions and filters, and creates calibration sequences. Use it when organizing a mixed acquisition folder.

## Attribution and license

Original script: © 2025 Adrian Knagg-Baugh, from the [Siril community scripts repository](https://gitlab.com/free-astro/siril-scripts/-/blob/master/utility/Blink_Browse_Filter_Sort.py).

Local modifications are distributed under the same **GPL-3.0-or-later** license. See [LICENSE.md](LICENSE.md). The original author and SPDX notices remain in the script.
