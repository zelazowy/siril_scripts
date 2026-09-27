# SAS-style Star Stretch for Siril

A standalone, unofficial Siril Python utility implementing the Star Stretch
transform used by [Seti Astro Suite Pro](https://github.com/setiastro/setiastrosuitepro).
It runs inside Siril's Python scripting environment and does **not** launch or
require SAS Pro.

## Controls

- **Stretch Amount:** 0–8, default 5. Zero leaves the stretch unchanged.
- **Color Boost:** 0–2, default 1. Adjusts colour around the RGB channel mean.
- **Remove Green (SCNR):** optional average-neutral green suppression.
- Automatic/manual preview, zoom, fit, reset, and Apply.

Designed for a **stars-only image**, such as the stars layer produced by star
separation. It does not detect or separate stars, and will stretch every pixel.
Mono images are supported; colour controls are disabled for them.

## Requirements and installation

Requires Siril 1.4.x with Python scripting support. The script requests NumPy,
OpenCV (`opencv-python`) and PyQt6 through `sirilpy.ensure_installed`; initial
installation may need internet access.

1. Download this folder, keeping `LICENSE` and `NOTICE.md` with the script.
2. Put the folder in a custom script location configured in Siril.
3. Refresh Siril's script list or restart Siril if needed.

See [Siril's custom script instructions](https://siril.readthedocs.io/en/stable/scripts/Script-files.html#adding-custom-scripts-folders).

## Usage

1. Open your stars-only image in Siril.
2. Run `Siril_SAS_Star_Stretch.py` from Siril's scripts interface.
3. Adjust Stretch Amount and optionally Color Boost or SCNR.
4. Inspect the preview, then click **Apply**.
5. Inspect the result in Siril and save it under the desired filename.

Apply creates a Siril undo state and replaces the active image pixels. It does
not write an image file. Close the dialog without applying to discard the preview.
If the active pixels change while the dialog is open, reopen the script before applying.

## Processing and limitations

Each normalized channel value `x` is transformed independently:

```text
f = 3 ** amount
output = f*x / (1 + (f - 1)*x)
```

Colour boost follows the stretch, then optional SCNR. Float input is clipped to
0–1; uint8/uint16 input is scaled to 0–1. Nonfinite values are rejected.

Preview uses a downscaled image, so small stars can differ from the full-resolution
result. Full-resolution processing runs synchronously and may pause the interface
or use substantial memory on large images. SAS Pro masks and presets are not supported.

## Validation status

Syntax compilation and 24 numerical comparisons against the upstream kernels
passed, covering stretch, colour boost, and SCNR combinations. Identity stretching
and narrow RGB/mono channel-layout checks also passed. A GUI smoke test could not
run in the development Python environment because OpenCV was unavailable.
**Live testing inside Siril is still pending.**

## License and credit

GNU GPL version 3 — see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
Siril adaptation © 2026 zelazowy. Algorithm reference: Franklin Marek and SAS Pro
contributors. This is an unofficial adaptation, provided without warranty.
