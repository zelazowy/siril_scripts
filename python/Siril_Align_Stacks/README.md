# Align Stacks

Run `Siril_Align_Stacks.py` from Siril's Python scripts menu (Siril 1.4+).
Place it in your configured Python scripts directory. On first run, Siril's
dependency helper installs PyQt6, Astropy, SciPy and Astroalign if needed.

1. **Add images**: choose two or more mono or RGB FITS stacks, from any folders.
2. The first row is the reference grid. Select another row and click **Make reference** if needed.
   No explicit reference selection is required. To remove a stack from the list,
   select it and press **Backspace** or **Delete** (the source file stays intact).
3. Click **Align / preview**. The log shows matched stars, residual error, rotation and scale.
4. The green rectangle starts at the largest shared rectangular coverage.
   Drag a new rectangle for a smaller crop, or click **Automatic common crop** to reset.
   Switch between stacks in the preview dropdown to inspect alignment and framing.
5. Click **Save aligned FITS**. Each original folder receives a file named
   `original_name_aligned.fit`. Change the suffix if that name already exists.

All outputs have the same spatial dimensions, including a cropped copy of the
reference. Originals are preserved. No sequences or intermediate image files
are created. Results stay in memory until saved, so memory usage grows with
the number and size of stacks. Closing while registration runs is disabled.

Registration uses stellar triangles to estimate translation, rotation and uniform
scale. RGB is averaged only for star detection; each original channel receives
the same transform. Moving images use one linear interpolation. The reference
is only cropped. No brightness normalization, stretching, clipping or continuum
subtraction is applied. Outputs are float32 FITS in the original numerical units.
The display is stretched automatically using only shared valid coverage, so
rotated black padding doesn't wash out the preview. The stretch stays fixed
when you adjust the crop. Array row zero is shown at the top.
Each stack uses an adaptive midtones curve that places its valid-area median
at 25% display brightness, keeping bright RGB stars from hiding faint detail.

Coverage is geometric: existing black borders inside input stacks aren't detected
automatically. Crop them manually if necessary. The selected crop must lie within
every transformed image's coverage. Sparse, starless, heavily saturated or very
different fields can fail matching. Optical distortion and reflections aren't
supported. Inspect the previews and residuals before continuum subtraction;
photometric scaling and PSF matching remain separate processing steps.

Source observational headers are retained, while standard WCS/SIP keywords use
the reference grid and crop-adjusted CRPIX. Auxiliary FITS extensions aren't copied.
If saving fails halfway (e.g. disk full), already saved outputs remain and the
error is displayed; use a new suffix for a fresh batch.

Implementation references: [Astroalign](https://astroalign.quatrope.org/en/latest/examples.html)
and [Astropy FITS image handling](https://docs.astropy.org/en/stable/io/fits/usage/image.html).

Standalone use is also supported: install `numpy scipy astropy astroalign PyQt6`
in your Python environment, then run the script with that Python interpreter.
