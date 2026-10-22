# ASTAP Plate Solve for Siril (v0.5.0)

Run `Siril_ASTAP_PlateSolve.py` from Siril's GUI with a single FITS, TIFF or JPEG (`.jpg`/`.jpeg`) image loaded. Install [ASTAP](https://www.hnsky.org/astap.htm) and [add this script to Siril](https://siril.readthedocs.io/en/latest/scripts/Script-files.html#adding-custom-scripts-folders). Select your installed ASTAP executable or application bundle, then click **Solve**. An ASTAP star database must be installed. The script uses NumPy, sirilpy and PyQt6; processed-image recovery also uses SciPy. JPEG metadata reading uses Pillow and Astropy. Extra dependencies are installed through sirilpy when first needed.

Siril exports can store FITS/WCS cards in the JPEG's EXIF `ImageDescription` tag, which SPACER also reads. The script uses these as coordinates and scale hints for a fresh ASTAP solve. It checks that the embedded dimensions, file dimensions and current loaded dimensions agree, that EXIF orientation is unrotated, and that any embedded WCS is valid. A complete current Siril WCS takes priority over an embedded one. Ordinary camera GPS EXIF is not a sky-coordinate solution. Missing, stale, rotated or malformed embedded metadata is ignored and solving continues from the loaded pixels. Even matching dimensions cannot prove that an image was never edited, so embedded astrometry is always checked by solving the stars again.

The script first tries a monochrome copy with the normal star sample, then a larger star sample. If those fail, it automatically tries a compact-star copy: it suppresses the stretched background, isolates measured stellar peaks, and places compact profiles at those same pixel positions. This helps with nonlinear, stretched final images. All solver copies retain the original dimensions and coordinate grid. Wider sky searches and automatic field-of-view searches remain available as fallbacks. Recovery cannot guarantee a solution if processing has removed too many stars or distorted their positions.

An undefined or clearly invalid `EQUINOX` value is replaced with J2000 in the temporary solver header. Valid equinox values are preserved. Only the resulting astrometry is imported; the solver's monochrome pixels and intensity metadata never replace the original image. Old PC matrices and unused SIP coefficients carried through ASTAP's output are excluded when importing its new CD/TAN solution.

**Cancel** stops the active ASTAP process and further retries, or stops proxy preparation at its next checkpoint. It leaves the dialog open so you can solve again. Escape or closing the window during a solve also requests cancellation. Cancel is briefly disabled while the completed solution is applied and saved.

The existing **Correct mirrored image if needed** option uses Siril's mirror command to transform both the image and its coordinates when necessary. Without a mirror correction, the loaded pixel array is preserved exactly, including the already-decoded JPEG pixels. FITS sources are saved in place; TIFF and JPEG sources remain unchanged and the solved image is saved beside them as `<name>_astap_solved.fit`. JPEGs are not recompressed. The log shows metadata guidance, each strategy, ASTAP's output and the successful strategy.

---

If you spot some problems - let me know! 

Clear skies!

---

Created in collaboration with Codex 🤖

