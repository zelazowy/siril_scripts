# Align Stacks

Run `Siril_Align_Stacks.py` from Siril's Python scripts menu. Place it in your
configured Python scripts directory. The dependency helper installs PyQt6,
Astropy, SciPy, Astroalign and SEP when needed.

1. **Add images**: choose two or more mono or RGB FITS stacks from any folders.
2. The first row sets the output grid. Select another row and click **Make reference**
   if needed. Remove a selected row with **Backspace** or **Delete**; its file stays intact.
3. Choose **Similarity star matching** (the previous method), or **Independent SIP astrometry**.
   For the latter, open **SIP settings**, check the executable/configuration paths,
   and normally leave the polynomial order at **3** and solver at **Siril — fresh solve per stack**.
4. Click **Align / preview**. The log reports solving, validation and stellar residuals.
5. The green rectangle starts at the largest rectangle within shared valid coverage.
   Drag a smaller crop or use **Automatic common crop** to reset. Switch previews to inspect.
6. **Save aligned FITS** writes `original_name_aligned.fit` for similarity, or
   `original_name_aligned_sip.fit` for SIP. Existing files aren't overwritten.

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
separately; cropping is array slicing. Outputs preserve numerical units as float32:
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

Standalone: install `numpy scipy astropy astroalign sep PyQt6`, then run the script.
