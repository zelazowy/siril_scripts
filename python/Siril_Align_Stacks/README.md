# Align Stacks 1.6.0

Star alignment is the default: add images, align, review the shared crop, and save.
Enable **Correct distortion (SIP · advanced)** when the images have different field
distortions and stars increasingly diverge toward the edges. This independently
solves and corrects every image, including the reference; it cannot repair
differences in star shape or PSF. The processing log is available on demand and
opens automatically on failure.

Run `Siril_Align_Stacks.py` from Siril's Python scripts menu. Place it in your
configured Python scripts directory. The dependency helper installs PyQt6,
Astropy, SciPy, Astroalign, SEP, Pillow and tifffile when needed.

1. **Add images**: choose two or more mono or RGB FITS, TIFF or JPEG images from any folders.
2. The first row sets the output grid. Select another row and click **Make reference**
   if needed. Remove a selected row with **Backspace** or **Delete**; its file stays intact.
3. Leave distortion correction off for normal **star alignment**. No plate solving
   or solver setup is needed. For differing field distortions, enable
   **Correct distortion (SIP · advanced)** and open **SIP settings**. Check paths
   and normally leave order **3** and **ASTAP (shared solver)** selected.
4. Click **Align images**. **Show processing log** reveals solving, validation and stellar residuals.
5. The green rectangle starts at the largest rectangle within shared valid coverage.
   Drag a smaller crop or use **Automatic common crop** to reset. Switch previews to inspect.
6. **Save aligned images** defaults to **Original format**: FITS stays FITS, TIFF
   stays TIFF and JPEG stays JPEG. The suffix is `_aligned` for similarity or
   `_aligned_sip` for SIP. Choose **FITS (float32)** to retain fractional resampling
   values in a common format. Existing files aren't overwritten.

## TIFF and JPEG inputs (version 1.4.0)

Finished/stretched `.tif`, `.tiff`, `.jpg` and `.jpeg` images are accepted alongside
FITS, including mixed input formats. **Star matching (similarity)** needs no plate
solution and is the easiest starting point. Stretched images with very saturated,
removed or distorted stars may still fail matching.

TIFF supports single-image grayscale/RGB data, including 16-bit and float samples
and planar RGB. JPEG supports grayscale/RGB. EXIF/TIFF orientation is applied once,
then raster rows are placed on the bottom-up FITS grid without interpolation.
Alpha, CMYK/palette TIFF, and multipage TIFF are rejected rather than silently
converted or flattened. TIFF compression requiring an external codec needs
`imagecodecs`; alternatively export an uncompressed TIFF. JPEG's existing compression
artifacts remain part of its decoded image.

The original sample values are converted to float32; there is no automatic stretch,
unstretch, color conversion, clipping or intensity normalization. Existing display
stretching affects previews only. The result remains nonlinear if the input was
stretched: use original linear stacks when linear continuum subtraction is needed.
For native SIP registration, private science copies outside [0,1] use a reversible
affine range encoding to prevent Siril's automatic float-FITS unit conversion.
The same encoding is applied to all channels, reversed immediately after the single
geometric resampling; final values retain their original units, including negatives.
This is not a photometric calibration or a permanent intensity normalization.

FITS and raster rows use the same internal coordinate convention; previews retain
the existing array-row-zero-at-top convention.

Output defaults to the **original format** (version 1.5.0). TIFF retains its source
sample type (for example uint16 or float32), with lossless uncompressed storage;
integer samples round to the original representable range after interpolation.
JPEG uses quality 95 without chroma subsampling and is necessarily recompressed.
Choose **FITS (float32)** to avoid JPEG loss or integer quantization. Original ICC
profiles are retained for TIFF/JPEG. Container metadata isn't copied wholesale;
outputs embed a compact, updated FITS/WCS description and normal storage orientation.

When forcing FITS output, a TIFF/JPEG pair sharing a basename gets format-qualified
names (`image_tif_aligned.fit`, `image_jpg_aligned.fit`). Original-format exports
keep their original extension. Existing files are never overwritten.

Siril-style FITS cards in TIFF ImageDescription/JPEG EXIF are reused only when their
embedded dimensions match and storage orientation is normal. Metadata is discarded
for rotated/reflected rasters to avoid applying an old WCS to a different grid.
Matching dimensions alone cannot prove that an image hasn't been edited; prefer a
fresh solve when uncertain. For distortion-aware alignment without usable embedded
astrometry, choose **SIP settings → ASTAP (shared solver)**. Native Siril solving still needs
its usual pointing/scale hints; existing-SIP mode requires valid per-image SIP.

Tests cover 16-bit/float/planar TIFF decoding, orientation, metadata screening,
alpha/multipage rejection, source preservation, and alignment/crop/export of a
synthetic nonlinear FITS/TIFF/JPEG field. Native Siril integration also covers a mono
Astro-TIFF alongside RGB FITS. No real-image TIFF/JPEG collection was supplied for
this change; synthetic and decoder checks do not guarantee matching every finished
astrophotograph.

## Independent SIP distortion correction

This mode was checked against installed Siril **1.4.4**, its corresponding official
command documentation and source, and bundled Python API **1.0.25**. It checks the
selected CLI version and required command options at runtime. Only 1.4.4 and later
1.4 patch versions are accepted; other versions need separate verification.

The existing simple alignment actually uses **Astroalign similarity transforms**,
not Siril Global Star Alignment. It estimates translation, rotation and uniform
scale; it cannot correct field-dependent distortion. SIP mode uses Siril's native
`seqplatesolve -order=N -nocache -nocrop -force`, recording astrometric registration
and **each image's own** distortion coefficients. `seqapplyreg -interp=linear
-framing=current -scale=1 -layer=1` applies those independent distortion mappings
and their projections onto a common distortion-free TAN grid. The reference is
also undistorted. No shared distortion master is used.

Each input needs accurate pointing and effective pixel scale in its own FITS
header (RA/DEC, focal length and pixel size). Siril's configured catalogues require
network access on an uncached solve. Cropped, rescaled or previously registered
stacks can carry stale WCS/scale metadata: use the original unaligned stacks for
this workflow and comparisons. **SIP settings** supports orders **2–5** and also
local Astrometry.net (`solve-field`, installed indices and Siril configuration).
**Use each input's existing SIP solution** is an explicit alternative; every input
must already contain independently valid coefficients of the selected order.
It never silently replaces a failed fresh solve.

Forward A/B and reverse AP/BP coefficients, celestial WCS, full-field inversion,
per-image sequence registration, dimensions and common output WCS are verified.
Linear WCS or zero distortion coefficients are rejected. Siril 1.4.4 evaluates
AP/BP directly; fresh cubic solves can have inaccurate inverse approximations.
When that approximation differs from A/B by more than 0.25 pixels, the script fits
an order-5 AP/BP approximation to that image's unchanged forward model and validates
it on an independent 97×97 grid, requiring maximum error below 0.05 pixels.
This is reported explicitly. Failure stops the batch and disables saving; it does
not run ordinary registration and label it distortion-corrected.

Mono inputs are replicated exactly into three planes in the private sequence and
returned to mono afterward. RGB science pixels receive one identical mapping across
all channels. There is **one linear resampling per science stack**, composing
undistortion and projection. Coverage masks use the same native transformations
separately; cropping is array slicing. Internal images and FITS exports preserve numerical units as float32 (original-format raster exports use the sample type described above):
no stretch, colour calibration, intensity normalization, background correction,
star removal or continuum subtraction is applied. Resampling redistributes pixel
values and can change sampled star peaks/aperture flux; it does not replace the
photometric scaling or PSF matching needed for continuum subtraction.

SIP mode creates a private temporary sequence, solved copies, registered images
and transformed masks, deleted when processing ends. Source files remain untouched.
Expect several times the input size in temporary disk space and substantial RAM.
Siril settings are copied to a temporary INI; its GUI image/sequence is not changed.
On macOS, `sandbox-exec` protects Siril's Python environment from its automatic
startup repair; absence of this protection stops SIP mode. Solver catalogue caches
may still be updated by Siril.

## Shared solver installation

Install [astro-platesolve](../../shared/astro-platesolve/) in the Python environment
running this script: `python -m pip install -e /path/to/siril_scripts/shared/astro-platesolve`.
Use the same source directory in SPACER's environment for one implementation.
The library is source-only for now, not published on PyPI. Its central configuration
selects ASTAP or the Skyshot CLI adapter for all consumers; the ASTAP executable
field here is a fallback when no central executable is configured. Native Siril
and existing-SIP modes remain independent of this optional library.

Skyshot schema-v1 adapter tests do not establish real solving performance or
accuracy; its runtime/index is not installed here. Results containing `local_grid`
are refused because exported FITS WCS cannot represent that extra mapping. The
alignment-specific SIP coefficient, inverse and common-grid checks remain here.
The actual solving backend is recorded in `ALNSOLVE`.

## Shared solver backend (version 1.3.0)

Select **SIP settings → Solutions → Shared solver — independent SIP with recovery**, normally set
**order 3**, and check the ASTAP executable path. Install ASTAP and its local star
database first. Siril 1.4.4 is still required to record/apply astrometric registration.
ASTAP's documented `-sip` output is cubic; a different engine must return the selected order or the batch is rejected.

This option imports the same `astro_platesolve` library as Siril Plate Solve and
SPACER; no recovery implementation is copied into this script. ASTAP is the default: same-size mono solver proxies linearly scaled to an explicit ADU range,
invalid-equinox correction, 500/2000-star attempts, compact-star recovery, wider
searches and automatic field-of-view recovery. Each stack is solved independently.
Proxy smoothing/peak isolation affects solving only; science pixels and RGB channels
remain unchanged until the single native registration resampling. Each attempt has
a maximum 180-second timeout; wider recovery attempts can take several minutes.

Old SIP is removed from solver copies before each attempt. Stale sidecars are deleted,
and both the exit status and newly written solution are checked. A linear-only WCS
is rejected even if plate solving succeeded. Only new astrometry is transferred;
reverse SIP evaluation is fitted/validated for Siril, then the usual common-grid and
coverage checks run. Failed ASTAP solving never silently switches to another backend.

ASTAP can write nonzero constant/linear A/B polynomial terms. Version 1.2.0 supports
these in both **existing SIP** and fresh ASTAP solutions: it shifts the polynomial
origin and absorbs the linear component into CRPIX/CD, preserving the sky mapping.
An 11×11 field check requires equivalent coordinates within 0.0001 pixel, and inverse
evaluation is revalidated. These terms are never simply discarded. This fixes the
previous `Invalid SIP coefficient A_0_0` rejection for valid ASTAP-produced solutions.

Tested on the supplied M39 RGB and LP stacks with installed ASTAP and Siril 1.4.4.
Normal ASTAP proxies failed; compact-star recovery solved both independently. Both
existing-SIP and fresh-ASTAP native registration/export completed. Fresh ASTAP/SIP
centre/outer RMS was 0.259/0.578 pixels; existing SIP gave 0.242/0.554 pixels, while
the similarity baseline was 0.363/0.343 pixels. Matched subsets differ, so these are
diagnostics rather than a controlled same-star comparison. SIP was not better at the
edges for this pair: inspect the log and compare modes rather than assuming that
more complex registration improves every field. No PSF or photometric matching was
applied. Sources were preserved and test results were written to temporary storage.

References: [ASTAP command-line options and cubic SIP](https://www.hnsky.org/astap.htm#astap_command_line),
[ASTAP SIP conventions](https://www.hnsky.org/sip.htm).

## M31 SIP recovery fix (version 1.5.0)

A successful sky position with linear WCS is not enough for SIP alignment. The shared
ASTAP backend now keeps trying larger star samples/recovery strategies when SIP is
required and the first candidate has no distortion coefficients. A run that never
produces valid nonzero SIP still fails explicitly; it never relabels a linear result.
An undefined EQUINOX is repaired in private Siril staging headers as well as in the
shared solver. Original files remain unchanged.

On the supplied M31 pair, ASTAP's initial 500-star candidate reported insufficient
quads for SIP; the 2000-star retry produced cubic SIP. Both images solved independently
and completed native Siril 1.4.5 registration, with a shared 5291×3512 crop. Measured
centre/outer RMS changed from 0.751/2.464 pixels (similarity) to 0.309/0.709 pixels
(shared ASTAP/SIP). Matched subsets differ; these are diagnostics, not a same-star
accuracy guarantee. The native Siril fresh solver still failed to solve this pair.
Use the default **Shared solver**, order **3**, for this tested route. Existing
order-2 models may be selected explicitly with **Existing SIP**, order **2**.

## Preview, diagnostics and output

The grayscale preview averages RGB for display, uses shared coverage and an adaptive
midtones curve placing the median at 25% brightness. Stretching is display-only and
stays fixed while adjusting crop. Array row zero is at the top.

SEP measures stellar centroids on temporary detection arrays. Logs report mutual
nearest-neighbour matches within 5 pixels, RMS and 95th-percentile offsets for the
centre (normalized radius ≤0.5) and outer field (≥0.75). Unmatched stars are excluded;
regions with fewer than five matches are flagged. For SIP, the old similarity
baseline transforms centroids only, avoiding another science resampling. These are
diagnostics, not a guarantee: crowded fields, different filters and different PSFs
can bias centroids. Geometry cannot repair differences in star shape or PSF. If
measurement fails, the log says so and makes no residual-quality claim.

The common crop includes transformed geometric bounds. Existing black borders inside
inputs aren't detected automatically; inspect and crop them manually. Source
observational headers are retained, while standard WCS/SIP keywords are replaced
by the common output grid with crop-adjusted CRPIX. `ALNMODE` distinguishes `SIP`
from `SIMILAR`; SIP outputs also record order, solver and Siril version. Auxiliary
FITS extensions aren't copied. If saving fails midway, already saved outputs remain;
use a new suffix after resolving the error.

## Verification and references

Automated tests cover preview behavior, shared crop, mixed RGB/mono registration,
SIP rejection, inverse approximation, grid mismatch and failed CLI status. An
opt-in native Siril integration test checks independent distorted RGB/mono inputs,
reference correction, channel ratios, linear background values, common saved WCS
and unchanged originals. Set `SIRIL_TEST_CLI` and `SIRIL_TEST_CONFIG` before running
`python -m unittest discover -s tests -p 'test_align*.py'`.

On the supplied original NGC 6914 RGB/L-eXtreme stacks, fresh independent order-3
solutions with refined inverse evaluation reduced measured centre RMS from 1.058
to 0.583 pixels and outer RMS from 3.608 to 0.798 pixels. There were 453 baseline
and 768 corrected mutual matches, so these are aggregate diagnostics on different
matched subsets, not a same-star accuracy guarantee. Outer 95th-percentile offsets
were 4.863 and 1.702 pixels respectively. The common output crop was 7396×6717.
Residual mismatch remains; optical tilt/collimation was not diagnosed. Online
Siril solving and existing-SIP native integration were tested; local Astrometry.net
and other operating systems were not exercised end to end.

See [Siril 1.4.4 registration](https://siril.readthedocs.io/en/stable/preprocessing/registration.html),
[sequence plate solving](https://siril.readthedocs.io/en/stable/astrometry/platesolving.html),
[commands](https://siril.readthedocs.io/en/stable/Commands.html), and
[Python API](https://siril.readthedocs.io/en/stable/Python-API.html).

Standalone: install `numpy scipy astropy astroalign sep PyQt6 Pillow tifffile`, then run the script.

## Version 1.6.0

Simplified the main window around the default star alignment workflow, with optional
SIP correction and its settings shown only when selected. File rows show names with
full paths in tooltips. Status text explains progress, crop dimensions and save results.
Alignment/reference buttons follow the file selection; custom suffixes survive mode
changes. Invalid/non-finite star matches and inconsistent export input lists are
rejected. Existing crop, preview, output formats and per-image SIP validation remain.

Offscreen interface checks cover defaults, duplicate file selection, reference changes,
mode switching, suffix preservation, busy state, deletion and failure recovery.
