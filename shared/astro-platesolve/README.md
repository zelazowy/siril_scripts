# astro-platesolve

Shared Python library for **Siril Plate Solve**, **Align Stacks** and **SPACER**.
The implementation lives in this directory. Consumers import `astro_platesolve`;
they do not copy solver code or import another application's GUI.

## Install once per Python environment

From this checkout, using the Python interpreter that runs the application:

```sh
python -m pip install -e /path/to/siril_scripts/shared/astro-platesolve
```

Run this with Siril's Python and with SPACER's virtual-environment Python. Both
editable installs point to the **same source directory**, so edits here apply to
all consumers after restart. Do not move/delete that checkout while it is installed.
For a release, `pip install /path/to/this/directory` installs a fixed copy instead;
each environment then needs an explicit upgrade. This package is not published on
PyPI. Dependencies are NumPy, SciPy and Astropy; no Siril or Qt import is required.

## One central engine selection

ASTAP remains the default. Existing application executable fields are ASTAP
fallbacks. A shared configuration overrides those paths for every consumer:

```sh
python -m astro_platesolve configure --backend skyshot \
  --executable /path/to/skyshot --index /path/to/compatible/index
python -m astro_platesolve show
```

To return to ASTAP:

```sh
python -m astro_platesolve configure --backend astap --executable /path/to/astap
```

The common file is `~/.config/astro-platesolve/config.json`; set
`ASTRO_PLATESOLVE_CONFIG` to use another file. All three applications must run as
the same user/use the same setting to share it. Configure never installs an engine
or downloads data. No automatic engine fallback occurs. The native Siril and
existing-WCS alignment modes continue to bypass this optional solver library.

## API

```python
from astro_platesolve import SolveOptions, solve_image

result = solve_image(pixels, fits_header, SolveOptions(), progress=print)
# result.header is astrometry; original pixels/header are left untouched.
```

`SolveOptions` supports executable/database/search hints, total timeout,
`confirm_original`, required SIP, diagnostic log and backend options. Set
`backend='astap'`/`'skyshot'` explicitly to ignore the shared default. `SolveResult`
contains FITS astrometry, backend identity, attempts, warnings and provenance.
Cancellation accepts a threading.Event-like `is_set`/`wait` object; adapters can
pump their UI there. Process cleanup runs on failure, timeout and cancellation.

A new backend implements `(pixels, header, options, progress, cancel) -> SolveResult`
and is registered through `register_backend(name, callable)`. Add/discover it here,
not in each application. Generic validation checks finite celestial WCS, dimensions,
scale and required nonzero SIP. Applications retain stricter domain checks.

ASTAP owns its command syntax, normalization and compact proxies, header-guided and
expanded recovery, stale-sidecar checks, warnings, streamed diagnostics and process
lifecycle here. SPACER sets `confirm_original=True`: seeds must pass its existing
original-pixel confirmation, parity/scale/corner and inexact-scale screens before a
project can be committed. Other adapters retain their prior no-confirmation policy.
A compact proxy is never substituted for source science pixels.

Siril retains image locking/undo, JPEG hint handling, optional mirror correction
and its existing save policy. Align Stacks retains independent SIP validation,
coordinate-preserving ASTAP compatibility, native registration and crop/export.
SPACER retains image snapshots, dates, project safety and saved-state provenance.

## Skyshot adapter status

Checked against [Skyshot CLI schema 1](https://github.com/ashtree74/skyshot-solver/blob/main/docs/CLI.md)
and source commit `76f0181` (solver 0.1.0). It uses `solve --index ... --wcs ...`,
checks process status **and** JSON `solution.status`, schema, dimensions and new
FITS output. It does not pretend to be ASTAP. `local_grid` output is rejected:
Skyshot's FITS export cannot represent that extra mapping, so silently using only
the exported WCS would discard part of the distortion correction.

Skyshot needs a separately built executable and compatible local index. No Rust
toolchain or Skyshot runtime/index was installed during this refactor. Adapter
contracts were tested with simulated CLI results, including ambiguity, schema,
linear-only SIP rejection and local-grid rejection; no real Skyshot accuracy or
performance claim is made. See upstream [installation](https://github.com/ashtree74/skyshot-solver/blob/main/docs/INSTALL.md)
and [index setup](https://github.com/ashtree74/skyshot-solver/blob/main/docs/DATA-INSTALL.md).

## Validation

Run `python -m unittest discover -s tests`. Existing SPACER solver/process tests
exercise the extracted ASTAP backend, including cancellation, streamed UTF-8,
quiet heartbeats, stale sidecars, original confirmation and refinement rejection.
The full SPACER suite, Siril adapter tests and alignment tests are separate checks.

Real M39 RGB/LP stacks solved with the shared ASTAP compact recovery and completed
native Siril registration/export. Measured RMS and crop matched the pre-extraction
workflow (centre 0.259 px, outer 0.578 px, crop 1688×2645); this checks refactoring,
not independent catalogue accuracy. Test outputs remained temporary; inputs were
preserved. Engine/index binaries and image fixtures are not included.
