## Siril_ASTAP_PlateSolve

Simple ASTAP integration for Siril. If you struggle with plate solving when you want to color calibrate you can try to solve file in ASTAP, save it and then result will be available in Siril.

But if you want it to work directly in Siril I made simple script that will do it for you!

You just need to:
- install ASTAP https://www.hnsky.org/astap.htm
- install this script in your custom scripts folder (see https://siril.readthedocs.io/en/latest/scripts/Script-files.html#adding-custom-scripts-folders)
- and then when running script you need to point to the ASTAP executable and voilà!

## Mirror correction (v0.3.0)

**Correct mirrored image if needed** is enabled by default, and your choice is saved between runs. After solving, the script checks the WCS matrix using Siril's parity convention. Mirrored images are flipped through Siril, which updates the coordinate solution and SIP distortion terms. Already unmirrored images are left alone.

This changes pixel positions without interpolation or cropping; pixel values and image dimensions are preserved. It does not rotate to north-up. Disable the checkbox to retain the original orientation. As before, the result overwrites FITS sources; TIFF sources get a sibling solved FITS file.

---

If you spot some problems - let me know! 

Clear skies!

---

Created in collaboration with Codex 🤖

