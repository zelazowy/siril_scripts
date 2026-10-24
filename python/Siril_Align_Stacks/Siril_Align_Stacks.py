"""Align a few mono/RGB FITS stacks; run from Siril's Python menu.

Also runs standalone when dependencies are installed. Similarity matching or
Siril 1.4.4 sequence SIP astrometry; one linear resampling of science pixels.
"""
import sys
import os
import re
import shutil
import subprocess
import tempfile
import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERSION = "1.3.0"


@dataclass
class SIPOptions:
    executable: str = ""
    config: str = ""
    order: int = 3
    solver: str = "siril"  # siril, astrometry.net, existing, astap
    timeout: int = 900
    astap_executable: str = ""


def find_siril_cli():
    candidates = [shutil.which("siril-cli"),
                  "/Applications/Siril.app/Contents/MacOS/siril-cli"]
    # Siril's bundled Python may live beside its executable on Windows.
    candidates += [str(Path(sys.executable).parent / name)
                   for name in ("siril-cli", "siril-cli.exe")]
    return next((str(p) for p in candidates if p and Path(p).is_file()), "")


def find_siril_config(configdir=None):
    roots = [Path(configdir)] if configdir else []
    roots += [Path.home()/"Library/Application Support/org.siril.Siril/siril",
              Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home()/".config")))/"siril",
              Path(os.environ.get("LOCALAPPDATA", str(Path.home())))/"siril"]
    return next((str(root/name) for root in roots
                 for name in ("config.1.4.ini", "config.ini")
                 if (root/name).is_file()), "")


def find_astap():
    try:
        from astro_platesolve.astap import find_astap as discover
        return discover()
    except ImportError:
        return ""


def canonical_sip_header(header, shape):
    """Absorb ASTAP's constant/linear SIP terms into CRPIX/CD without changing WCS.

    Normalize into the strict degree >=2 form accepted by our validator. Keep
    the full mapping by shifting the polynomial origin and factoring its
    Jacobian into CD; never discard these lower-order terms.
    """
    from astropy.wcs import WCS, Sip
    from scipy.optimize import root
    from math import comb
    old = WCS(header, naxis=2)
    if old.sip is None:
        raise ValueError("No SIP distortion model to canonicalize.")
    a, b = old.sip.a.copy(), old.sip.b.copy()
    if not any(abs(poly[i, j]) > 0 for poly in (a, b)
               for i, j in ((0, 0), (1, 0), (0, 1))):
        return header.copy(), False
    f, g = a.copy(), b.copy()
    f[1, 0] += 1; g[0, 1] += 1
    def evaluate(point):
        return [np.polynomial.polynomial.polyval2d(*point, p) for p in (f, g)]
    solved = root(evaluate, [0., 0.])
    if not solved.success or np.max(np.abs(evaluate(solved.x))) > 1e-7:
        raise ValueError("Cannot canonicalize SIP polynomial origin.")
    def shifted(poly):
        out = np.zeros_like(poly)
        for i, j in np.ndindex(poly.shape):
            for k in range(i+1):
                for l in range(j+1):
                    out[k, l] += poly[i, j]*comb(i, k)*comb(j, l)*solved.x[0]**(i-k)*solved.x[1]**(j-l)
        return out
    f, g = shifted(f), shifted(g)
    jacobian = np.array([[f[1, 0], f[0, 1]], [g[1, 0], g[0, 1]]])
    if not np.isfinite(jacobian).all() or abs(np.linalg.det(jacobian)) < 1e-10:
        raise ValueError("Singular SIP linear component.")
    a, b = np.einsum('ab,bij->aij', np.linalg.inv(jacobian), np.stack([f, g]))
    for poly in (a, b):
        poly[0, 0] = poly[1, 0] = poly[0, 1] = 0
    new = old.deepcopy()
    if new.wcs.has_pc():
        del new.wcs.pc
    new.wcs.cd = old.pixel_scale_matrix @ jacobian
    new.wcs.crpix = old.wcs.crpix+solved.x
    new.sip = Sip(a, b, None, None, new.wcs.crpix)
    h, w = shape[-2:]
    xx, yy = np.meshgrid(np.linspace(0, w-1, 11), np.linspace(0, h-1, 11))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    back = old.all_world2pix(new.all_pix2world(points, 0), 0, tolerance=1e-6, maxiter=100)
    if not np.isfinite(back).all() or np.max(np.linalg.norm(back-points, axis=1)) > 1e-4:
        raise ValueError("Canonical SIP does not preserve the original coordinate mapping.")
    result = output_header(header, new.to_header(relax=True), (0, 0, w, h))
    result, _ = refine_inverse_sip(result, shape)
    return result, True


def astap_solution_header(original, solved, shape, order=3):
    """Import only NEW astrometry: never retain old SIP from a failed fit."""
    from astropy.wcs import WCS
    if not all(str(solved.get(f"CTYPE{i}", "")).endswith("-SIP") for i in (1, 2)):
        raise ValueError("ASTAP solved only linear WCS; distortion correction rejected.")
    if solved.get("A_ORDER") != order or solved.get("B_ORDER") != order:
        raise ValueError(f"Shared solver did not return the requested order-{order} SIP model; "
                         f"got {solved.get('A_ORDER')}/{solved.get('B_ORDER')}. "
                         "ASTAP supports order 3; other engines may choose a different order.")
    # ASTAP sidecars can echo an old PC matrix alongside the newly solved CD.
    solved = solved.copy()
    if any(re.fullmatch(r"CD[12]_[12]", k) for k in solved):
        for key in list(solved):
            if re.fullmatch(r"PC[12]_[12]|CDELT[12]|CROTA[12]", key):
                del solved[key]
    clean = original.copy()
    for key in list(clean):
        if key in ("PLTSOLVD", "RA", "DEC"):
            del clean[key]
    header = output_header(clean, solved, (0, 0, shape[-1], shape[-2]))
    header, _ = canonical_sip_header(header, shape)
    # Construct inverse evaluation from the independently solved A/B model.
    header, error = refine_inverse_sip(header, shape)
    wcs = WCS(header, naxis=2)
    world = wcs.all_pix2world([[shape[-1]/2, shape[-2]/2]], 0)[0]
    header['RA'], header['DEC'], header['PLTSOLVD'] = float(world[0]), float(world[1]), True
    return header, error


def solve_astap_stack(data, header, options, path, report):
    """Shared solver returns only WCS; registration-specific SIP checks stay here."""
    try:
        from astro_platesolve import SolveOptions, solve_image
    except ImportError as exc:
        raise RuntimeError("Install the shared astro-platesolve library in this Python environment; see README.") from exc
    result = solve_image(data, header, SolveOptions(
        executable=options.astap_executable, timeout=options.timeout,
        require_sip=True, log_path=str(path.with_suffix('.log'))), report)
    report(f"Shared solver backend: {result.backend}; strategy: {result.strategy}")
    accepted, error = astap_solution_header(header, result.header, data.shape, options.order)
    accepted['ALNBACK'] = result.backend
    report(f"Independent SIP verified; inverse maximum error {error:.5f} px. Science pixels unchanged.")
    return accepted


def run_siril(options, directory, commands, report=lambda message: None):
    """Run in an isolated CLI, never changing the GUI's current image/sequence."""
    script = Path(directory)/"align_commands.ssf"
    script.write_text("requires 1.4.4\n" + "\n".join(commands) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["LANGUAGE"] = "C"
    # Linux respects these locations. macOS Siril overrides XDG variables;
    # protect its eager Python environment repair with an OS write sandbox.
    environment["XDG_DATA_HOME"] = str(Path(options.config).parent/"runtime-data")
    environment["XDG_CACHE_HOME"] = str(Path(options.config).parent/"runtime-cache")
    command = [options.executable, "-i", options.config,
               "-d", str(directory), "-s", str(script)]
    if sys.platform == "darwin":
        import json
        sandbox = shutil.which("sandbox-exec")
        if not sandbox:
            raise RuntimeError("macOS SIP mode needs sandbox-exec to protect Siril's Python environment.")
        protected = Path.home()/"Library/Application Support/org.siril.Siril/siril/venv"
        profile = '(version 1)(allow default)(deny file-write* (subpath ' + json.dumps(str(protected)) + '))'
        command = [sandbox, "-p", profile, *command]
    try:
        result = subprocess.run(command,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", timeout=options.timeout,
                                env=environment)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Siril timed out; no corrected results were accepted.") from exc
    log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", result.stdout)
    for line in log.splitlines():
        if line.startswith("log:") and not any(s in line for s in
            ("Running command:", "Reading configuration", "Setting CWD", "Supported file types:")):
            report(line[4:].strip())
    # Some failed Siril scripts return exit code 0: check the terminal status too.
    if result.returncode or "Script execution finished successfully." not in log:
        raise RuntimeError("Siril command batch failed; no fallback was used.\n" +
                           "\n".join(log.splitlines()[-14:]))
    return log


def check_siril(options, directory, report):
    if not Path(options.executable).is_file():
        raise ValueError("Choose the installed siril-cli executable in SIP settings.")
    version = subprocess.run([options.executable, "--version"], capture_output=True,
                             text=True, timeout=15)
    match = re.search(r"\bsiril\s+(\d+)\.(\d+)\.(\d+)", version.stdout+version.stderr, re.I)
    if not match or version.returncode:
        raise ValueError("Could not determine the selected Siril CLI version.")
    number = tuple(map(int, match.groups()))
    if number < (1, 4, 4) or number[:2] != (1, 4):
        raise ValueError("SIP mode requires Siril 1.4.4 (or a later 1.4 patch). "
                         "Other command/API versions have not been verified.")
    report(f"Siril CLI {'.'.join(match.groups())}; checking supported command options.")
    help_log = run_siril(options, directory, ["help seqplatesolve", "help seqapplyreg",
                                            "help convert", "help setref"], lambda _: None)
    for token in ("-order=", "-force", "-nocache", "-nocrop", "-localasnet",
                  "-framing=", "-interp=", "-layer=", "-out=", "setref sequencename"):
        if token not in help_log:
            raise ValueError(f"Selected Siril lacks the verified command option {token}.")
    return '.'.join(match.groups())


def validate_sip(header, shape, order):
    """Reject linear, zero, malformed or non-invertible distortion solutions."""
    from astropy.wcs import WCS
    if not 2 <= order <= 5:
        raise ValueError("SIP order must be between 2 and 5.")
    for key in ("A_ORDER", "B_ORDER"):
        if header.get(key) != order:
            raise ValueError(f"Expected {key}={order}; got {header.get(key)!r}. "
                             "A linear or lower-order solve is not accepted.")
    if not all(str(header.get(f"CTYPE{i}", "")).endswith("-SIP") for i in (1, 2)):
        raise ValueError("Missing celestial TAN-SIP axis declarations.")
    coefficients = []
    for prefix in ("A", "B"):
        for key in header:
            match = re.fullmatch(prefix+r"_(\d+)_(\d+)", key)
            if match:
                i, j = map(int, match.groups())
                value = float(header[key])
                if not np.isfinite(value) or i+j > order or (i+j < 2 and value != 0):
                    raise ValueError(f"Invalid SIP coefficient {key}.")
                coefficients.append(value)
    if not coefficients or not np.any(np.asarray(coefficients) != 0):
        raise ValueError("No nonzero SIP distortion coefficients; correction not verified.")
    # Siril 1.4.4's interpolation path uses the reverse AP/BP polynomials.
    # Forward-only SIP is legal FITS WCS, but is not sufficient for this backend.
    for prefix in ("AP", "BP"):
        inverse_order = header.get(prefix+"_ORDER")
        if not isinstance(inverse_order, int) or not 2 <= inverse_order <= 5:
            raise ValueError(f"Missing valid {prefix}_ORDER inverse SIP model; "
                             "re-solve this stack in Siril instead of reusing it.")
        terms = [float(header[key]) for key in header
                 if re.fullmatch(prefix+r"_\d+_\d+", key)]
        if not terms or not np.isfinite(terms).all():
            raise ValueError(f"Invalid or missing {prefix} inverse coefficients.")
    wcs = WCS(header, naxis=2)
    if not wcs.has_celestial or wcs.sip is None:
        raise ValueError("Not a valid celestial SIP solution.")
    h, w = shape[-2:]
    xx, yy = np.meshgrid(np.linspace(0, w-1, 9), np.linspace(0, h-1, 9))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    world = wcs.all_pix2world(points, 0)
    inverse = wcs.all_world2pix(world, 0, tolerance=1e-5, maxiter=100,
                               adaptive=True, detect_divergence=True)
    if not np.isfinite(world).all() or not np.isfinite(inverse).all() or \
            np.max(np.linalg.norm(inverse-points, axis=1)) > .01:
        raise ValueError("SIP solution fails a full-field inverse round-trip check.")
    reverse = wcs.sip_foc2pix(wcs.sip_pix2foc(points, 0), 0)
    error = float(np.max(np.linalg.norm(reverse-points, axis=1)))
    if not np.isfinite(reverse).all() or error > .25:
        raise ValueError(f"SIP reverse polynomials differ from forward model by {error:.3f} px "
                         "(limit 0.25 px); re-solve or lower the polynomial order.")
    return wcs


def refine_inverse_sip(header, shape):
    """Fit inverse evaluation polynomials, keeping the solved forward WCS fixed.

    Siril uses AP/BP directly rather than numerically inverting A/B. Its cubic
    inverse can be too approximate on large stacks; degree five is supported
    by 1.4.4. This changes coordinate evaluation only, never science pixels.
    """
    from astropy.wcs import WCS
    result = header.copy()
    wcs = WCS(header, naxis=2)
    h, w = shape[-2:]
    scale = float(max(h, w))
    terms = [(i, j) for i in range(6) for j in range(6-i)]
    xx, yy = np.meshgrid(np.linspace(-2, w+1, 65), np.linspace(-2, h+1, 65))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    uv = points+1-wcs.wcs.crpix
    forward = wcs.sip_pix2foc(points, 0)+1  # SIP focal API honours origin=0
    design = np.column_stack([(forward[:, 0]/scale)**i *
                              (forward[:, 1]/scale)**j for i, j in terms])
    coefficients = np.linalg.lstsq(design, uv-forward, rcond=None)[0]
    for key in list(result):
        if re.fullmatch(r"(?:AP|BP)_(?:ORDER|\d+_\d+)", key):
            del result[key]
    for axis, prefix in enumerate(("AP", "BP")):
        result[prefix+"_ORDER"] = 5
        for (i, j), values in zip(terms, coefficients):
            result[f"{prefix}_{i}_{j}"] = float(values[axis]/scale**(i+j))
    refined = WCS(result, naxis=2)
    # Independent denser validation points, including every image boundary.
    xx, yy = np.meshgrid(np.linspace(0, w-1, 97), np.linspace(0, h-1, 97))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    reverse = refined.sip_foc2pix(refined.sip_pix2foc(points, 0), 0)
    error = float(np.max(np.linalg.norm(reverse-points, axis=1)))
    if not np.isfinite(error) or error > .05:
        raise ValueError(f"Refined inverse SIP is inaccurate ({error:.3f} px); solve rejected.")
    validate_sip(result, shape, int(result["A_ORDER"]))
    return result, error


def check_registration_file(path, count):
    """Check the native v6 sequence: every frame included, per-image distortion."""
    lines = path.read_text().splitlines()
    if "D1 4" not in lines:  # DISTO_FILES in Siril 1.4.4, not a shared master
        raise ValueError("Siril did not record per-image astrometric distortion data.")
    included = [line.split() for line in lines if line.startswith("I ")]
    if len(included) != count or any(row[2] != "1" for row in included):
        raise ValueError("Siril excluded one or more unsolved stacks; batch rejected.")
    rows = [line for line in lines if line.startswith("R1 ")]
    if len(rows) != count:
        raise ValueError("Missing astrometric registration entries.")
    for row in rows:
        if " H " not in row:
            raise ValueError("Missing astrometric projection matrix.")
        matrix = np.array(list(map(float, row.split(" H ")[1].split()))).reshape(3, 3)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("Invalid astrometric projection matrix.")


def same_output_grid(headers, shape):
    from astropy.wcs import WCS
    reference = WCS(headers[0], naxis=2)
    if not reference.has_celestial or reference.sip is not None:
        raise ValueError("Siril output is not a distortion-free celestial projection.")
    h, w = shape
    points = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1], [w/2, h/2]])
    world = reference.all_pix2world(points, 0)
    for header in headers[1:]:
        current = WCS(header, naxis=2)
        if current.sip is not None or not current.has_celestial or \
                not np.allclose(current.all_world2pix(world, 0), points, atol=1e-4, rtol=0):
            raise ValueError("Exported stacks do not share a common undistorted WCS grid.")


def stellar_centroids(data):
    import sep
    image = np.ascontiguousarray(mono(data), dtype=np.float32)
    # Large stacks can exceed SEP's default 300k active-pixel buffer.
    sep.set_extract_pixstack(max(300000, min(5000000, image.size//10)))
    padding = np.ascontiguousarray(image == 0)
    background = sep.Background(image, mask=padding)
    objects = sep.extract(image-background.back(), 5, err=background.rms(),
                          minarea=5, mask=padding)
    objects = objects[(objects['flag'] == 0) & np.isfinite(objects['x']) & np.isfinite(objects['y'])]
    objects = objects[np.argsort(objects['flux'])[-2000:]]
    return np.column_stack((objects['x'], objects['y']))


def centroid_residuals(reference, moving, shape, mask=None, max_distance=5):
    """Mutual nearest neighbours, without fitting away the residual mismatch."""
    from scipy.spatial import cKDTree
    if mask is not None:
        def inside(points):
            x, y = np.rint(points).astype(int).T
            valid = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
            indexes = np.flatnonzero(valid)
            valid[indexes] &= mask[y[indexes], x[indexes]]
            return points[valid]
        reference, moving = inside(reference), inside(moving)
    if not len(reference) or not len(moving):
        raise ValueError("No usable stellar centroids in common coverage.")
    distance, indexes = cKDTree(reference).query(moving)
    _, reverse = cKDTree(moving).query(reference)
    keep = (distance <= max_distance) & (reverse[indexes] == np.arange(len(moving)))
    positions = reference[indexes[keep]]
    radius = np.linalg.norm((positions-(np.array(shape[::-1])-1)/2) /
                            (np.array(shape[::-1])/2), axis=1)
    residual = distance[keep]
    stats = {"matched": len(residual), "reference": len(reference), "moving": len(moving),
             "radius_limit": max_distance}
    for name, region in (("all", np.ones(len(residual), bool)),
                         ("centre", radius <= .5), ("outer", radius >= .75)):
        values = residual[region]
        stats[name] = {"count": len(values),
                       "rms": float(np.sqrt(np.mean(values**2))) if len(values) else None,
                       "p95": float(np.percentile(values, 95)) if len(values) else None}
    return stats


def report_residuals(label, stats, report):
    report(f"{label}: {stats['matched']} mutual matches from {stats['moving']} moving / "
           f"{stats['reference']} reference stars (search <= {stats['radius_limit']} px; "
           "unmatched stars excluded).")
    for region in ("all", "centre", "outer"):
        entry = stats[region]
        if entry['count'] >= 5:
            report(f"  {region}: n={entry['count']}, RMS {entry['rms']:.3f} px, "
                   f"95th percentile {entry['p95']:.3f} px")
        else:
            report(f"  {region}: insufficient matches ({entry['count']}); residual not reliable.")


def validate_alignment(images, common, report, originals=None):
    """Diagnostics only; no changes to science intensities or fitted output grid."""
    try:
        reference = stellar_centroids(images[0][0])
        baseline_ref = stellar_centroids(originals[0][0]) if originals else None
        for index in range(1, len(images)):
            if originals:
                try:
                    import astroalign
                    transform, _ = astroalign.find_transform(mono(originals[index][0]),
                                                             mono(originals[0][0]),
                                                             max_control_points=100)
                    old = transform(stellar_centroids(originals[index][0]))
                    report_residuals(f"Stack {index+1}, old similarity (centroids only)",
                                     centroid_residuals(baseline_ref, old, originals[0][0].shape[-2:]), report)
                except Exception as exc:
                    report(f"Old similarity comparison unavailable for stack {index+1}: {exc}")
            report_residuals(f"Stack {index+1}, output grid",
                             centroid_residuals(reference, stellar_centroids(images[index][0]),
                                                common.shape, common), report)
        report("Centroid residuals measure position, not PSF/shape agreement. "
               "Geometry cannot repair different star shapes or PSFs.")
    except Exception as exc:
        report(f"Stellar validation unavailable: {exc}. Inspect alignment manually; "
               "no measured residual quality is claimed.")


def align_sip(paths, options, report=lambda message: None):
    """Native sequence solving and applyreg, independently undistorting all inputs."""
    from astropy.io import fits
    if len(paths) < 2 or options.solver not in ("siril", "astrometry.net", "existing", "astap"):
        raise ValueError("Choose at least two stacks and a supported SIP solver.")
    if not 2 <= options.order <= 5:
        raise ValueError("SIP order must be between 2 and 5.")
    if not Path(options.config).is_file():
        raise ValueError("Choose your Siril 1.4 configuration file in SIP settings.")
    originals = [read_stack(path) for path in paths]
    if options.solver == "existing":
        for index, (path, (data, header)) in enumerate(zip(paths, originals)):
            try:
                header, changed = canonical_sip_header(header, data.shape)
                if changed:
                    report(f"{Path(path).name}: absorbed ASTAP constant/linear SIP terms into CRPIX/CD; "
                           "coordinate mapping preserved, inverse revalidated.")
                    originals[index] = (data, header)
                validate_sip(header, data.shape, options.order)
            except Exception as exc:
                raise ValueError(f"{Path(path).name}: invalid existing SIP solution: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix="siril-align-sip-") as temporary:
        root = Path(temporary)
        source, work = root/"inputs", root/"work"
        source.mkdir(); work.mkdir()
        config = configparser.RawConfigParser(strict=False)
        config.read(options.config, encoding="utf-8")
        if not config.has_section("core"):
            raise ValueError("Not a Siril INI configuration file.")
        config.set("core", "force_16bit", "false")
        config.set("core", "extension", ".fit")
        config.set("core", "wd", str(work))
        config.set("core", "swap_dir", str(root))
        if not config.has_section("astrometry"):
            config.add_section("astrometry")
        config.set("astrometry", "update_default_scale", "false")
        config.set("astrometry", "sip_order", str(options.order))
        config_path = root/"config.ini"
        with config_path.open("w", encoding="utf-8") as stream:
            config.write(stream, space_around_delimiters=False)
        local = SIPOptions(options.executable, str(config_path), options.order,
                           options.solver, options.timeout)
        version = check_siril(local, work, report)
        shared_backends = []
        for i, (data, header) in enumerate(originals):
            # Native sequences require a common channel count. Replicating mono
            # into RGB is exact; it neither interpolates nor changes its values.
            staged = data if data.ndim == 3 and data.shape[0] == 3 else \
                np.repeat(mono(data)[None], 3, axis=0)
            if options.solver == "astap":
                report(f"Solving independently with ASTAP: {Path(paths[i]).name}")
                header = solve_astap_stack(data, header, options, root/f"solver_{i:05d}.fit", report)
            shared_backends.append(header.get('ALNBACK', options.solver))
            fits.PrimaryHDU(staged, header).writeto(source/f"input_{i:05d}.fit")
        report("Creating private sequence copies; originals stay untouched.")
        run_siril(local, source, ["convert stack -out=../work"], report)
        solve = f"seqplatesolve stack -order={options.order} -nocache -nocrop"
        if options.solver not in ("existing", "astap"):
            solve += " -force"
        if options.solver == "astrometry.net":
            solve += " -localasnet"
        report("Recording independent SIP solutions and astrometric registration "
               f"(order {options.order}, solver {options.solver}).")
        run_siril(local, work, ["setref stack 1", solve], report)
        solved_headers = []
        for i, path in enumerate(paths):
            staged = work/f"stack_{i+1:05d}.fit"
            if not staged.is_file():
                raise ValueError(f"Siril did not produce solved input {Path(path).name}.")
            header = fits.getheader(staged)
            try:
                try:
                    validate_sip(header, originals[i][0].shape, options.order)
                except ValueError as exc:
                    if options.solver == "existing" or not str(exc).startswith("SIP reverse polynomials differ"):
                        raise
                    report(f"{Path(path).name}: {exc}")
                    header, error = refine_inverse_sip(header, originals[i][0].shape)
                    report(f"Refined this stack's AP/BP inverse approximation to order 5: "
                           f"maximum round-trip {error:.5f} px. Forward solve stays order {options.order}.")
                    with fits.open(staged, mode="update") as hdus:
                        hdus[0].header = header
            except Exception as exc:
                raise ValueError(f"{Path(path).name}: SIP solve rejected: {exc}. "
                                 "No uncorrected fallback was used.") from exc
            solved_headers.append(header)
            report(f"{Path(path).name}: nonzero SIP order {options.order} verified, "
                   "including full-field inverse round-trip.")
        check_registration_file(work/"stack_.seq", len(paths))
        apply = "seqapplyreg stack -interp=linear -framing=current -scale=1 -layer=1"
        report("Undistorting ALL stacks, including reference, onto its common TAN grid.")
        applied_log = run_siril(local, work, [apply+" -prefix=aligned_"], report)
        if "undistortion will be applied" not in applied_log:
            raise ValueError("Siril did not confirm applying distortion correction; batch rejected.")
        outputs = [read_stack(work/f"aligned_stack_{i+1:05d}.fit") for i in range(len(paths))]
        shape = outputs[0][0].shape[-2:]
        if any(data.shape != (3, *shape) for data, _ in outputs):
            raise ValueError("Native output dimensions or RGB channel counts differ.")
        same_output_grid([header for _, header in outputs], shape)
        # Transform geometric masks through the SAME native registration. Each
        # science stack has already been resampled once; mask export doesn't touch it.
        for i, ((data, _), header) in enumerate(zip(originals, solved_headers)):
            fits.PrimaryHDU(np.ones((3, *data.shape[-2:]), np.float32), header).writeto(
                work/f"stack_{i+1:05d}.fit", overwrite=True)
        report("Computing shared valid coverage through the native distortion mappings.")
        run_siril(local, work, [apply+" -prefix=coverage_"], lambda _: None)
        common = np.ones(shape, bool)
        for i in range(len(paths)):
            mask, mask_header = read_stack(work/f"coverage_stack_{i+1:05d}.fit")
            if mask.shape != (3, *shape):
                raise ValueError("Coverage grid dimensions differ from science grid.")
            same_output_grid([outputs[0][1], mask_header], shape)
            common &= np.all(mask >= 1-1e-6, axis=0)
        if not common.any():
            raise ValueError("No shared coverage remains after distortion correction.")
        images = []
        grid_header = outputs[0][1]
        for (output, _), (original, header), backend in zip(outputs, originals, shared_backends):
            if original.ndim == 2:
                output = output[0]
            elif original.shape[0] == 1:
                output = output[:1]
            updated = output_header(header, grid_header, (0, 0, shape[1], shape[0]))
            updated["ALNMODE"] = ("SIP", "Independent per-stack SIP undistortion applied")
            updated["ALNORDER"] = options.order
            updated["ALNSOLVE"] = backend
            updated["ALNSIRIL"] = version
            images.append((output, updated))
        validate_alignment(images, common, report, originals)
        report("STATUS: SIP correction applied; common undistorted grid verified. "
               "No fallback. Science data resampled once; no photometric normalization.")
        return images, common


def read_stack(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdus:
        hdu = next((h for h in hdus if h.data is not None
                    and h.data.ndim in (2, 3)), None)
        if hdu is None:
            raise ValueError(f"No image in {path}")
        data = np.array(hdu.data, dtype=np.float32)
        header = hdu.header.copy()
    if data.ndim == 3 and data.shape[0] not in (1, 3):
        raise ValueError("Expected mono or channel-first RGB FITS data.")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite pixels in {path}; repair them first.")
    return data, header


def mono(data):
    return data.mean(axis=0) if data.ndim == 3 else data


def preview_pixels(data, common, step=1):
    """Stretch for display using shared coverage, excluding padded borders."""
    sample = mono(data)[::step, ::step]
    valid = common[::step, ::step] & np.isfinite(sample)
    values = sample[valid].astype(np.float64)
    if not values.size:
        raise ValueError("No valid pixels for the preview stretch.")
    lo, hi = np.percentile(values, (1, 99.8))
    if hi <= lo:
        return np.zeros(sample.shape, dtype=np.uint8)
    normalized = np.clip((sample.astype(np.float64)-lo)/(hi-lo), 0, 1)
    # Solve the midtones curve so the valid-area median displays at 25%.
    # Unlike a fixed asinh curve, bright stars cannot darken the background.
    median = np.clip((np.median(values)-lo)/(hi-lo), 1e-6, 1-1e-6)
    target = .25
    midtone = median * (target-1) / (2*target*median-target-median)
    stretched = ((midtone-1)*normalized /
                 ((2*midtone-1)*normalized-midtone))
    return np.ascontiguousarray(stretched * 255, dtype=np.uint8)


def warp(data, matrix, shape):
    from scipy.ndimage import affine_transform
    inv = np.linalg.inv(matrix)
    # Astroalign uses (x,y); scipy uses (row,column).
    linear = inv[:2, :2][::-1, ::-1]
    offset = inv[:2, 2][::-1]
    planes = data[None] if data.ndim == 2 else data
    result = np.stack([affine_transform(p, linear, offset, output_shape=shape,
                        order=1, mode="constant", cval=0, prefilter=False)
                       for p in planes])
    # Geometry, rather than pixel brightness, defines valid coverage.
    coverage = affine_transform(np.ones(planes.shape[1:], np.uint8), linear,
                                offset, output_shape=shape, order=1,
                                mode="constant", cval=0, prefilter=False) == 1
    return (result[0] if data.ndim == 2 else result), coverage


def largest_rectangle(mask):
    """Largest axis-aligned all-valid rectangle, as x,y,width,height."""
    heights = np.zeros(mask.shape[1], dtype=int)
    best = (0, 0, 0, 0)
    area = 0
    for y, row in enumerate(mask):
        heights = np.where(row, heights + 1, 0)
        stack = []
        for x in range(len(heights) + 1):
            height = int(heights[x]) if x < len(heights) else 0
            start = x
            while stack and stack[-1][1] > height:
                left, h = stack.pop()
                if h * (x - left) > area:
                    area = h * (x - left)
                    best = (left, y - h + 1, x - left, h)
                start = left
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    if not area:
        raise ValueError("Images have no common rectangular coverage.")
    return best


def align(paths, report=lambda message: None):
    import astroalign
    report("MODE: similarity star matching; distortion correction is OFF.")
    reference, ref_header = read_stack(paths[0])
    target = mono(reference)
    for key in ("ALNORDER", "ALNSOLVE", "ALNSIRIL"):
        ref_header.remove(key, ignore_missing=True)
    ref_header["ALNMODE"] = "SIMILAR"
    images = [(reference, ref_header)]
    common = np.ones(target.shape, bool)
    for path in paths[1:]:
        report(f"Matching stars: {Path(path).name}")
        data, header = read_stack(path)
        transform, (source_stars, target_stars) = astroalign.find_transform(
            mono(data), target, max_control_points=100)
        residual = np.linalg.norm(transform(source_stars) - target_stars, axis=1)
        rms = float(np.sqrt(np.mean(residual ** 2)))
        if len(residual) < 3 or rms > 2:
            raise ValueError(f"Unreliable match for {path}: RMS {rms:.2f} px.")
        report(f"{len(residual)} matched stars; RMS {rms:.3f} px; "
               f"rotation {np.degrees(transform.rotation):.3f}°; "
               f"scale {transform.scale:.6f}")
        output, valid = warp(data, transform.params, target.shape)
        common &= valid
        header["ALNMODE"] = "SIMILAR"
        for key in ("ALNORDER", "ALNSOLVE", "ALNSIRIL"):
            header.remove(key, ignore_missing=True)
        images.append((output, header))
    validate_alignment(images, common, report)
    return images, common


def output_header(original, reference, crop):
    import re
    header = original.copy()
    # Drop stale astrometry from the moving frame; copy the reference grid.
    pattern = re.compile(r"^(WCSAXES|CTYPE\d+[A-Z]?|CUNIT\d+[A-Z]?|CRPIX\d+[A-Z]?|"
                         r"CRVAL\d+[A-Z]?|CDELT\d+[A-Z]?|CROTA\d+[A-Z]?|"
                         r"(?:CD|PC|PV|PS)\d+_\d+[A-Z]?|LONPOLE[A-Z]?|LATPOLE[A-Z]?|"
                         r"RADESYS[A-Z]?|EQUINOX[A-Z]?|[AB]P?_(?:ORDER|DMAX)|[AB]P?_\d+_\d+)$")
    for key in list(header):
        if pattern.match(key) or key in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM"):
            del header[key]
    for card in reference.cards:
        if pattern.match(card.keyword):
            header[card.keyword] = (card.value, card.comment)
    x, y, _, _ = crop
    for key in list(header):
        if re.fullmatch(r"CRPIX1[A-Z]?", key):
            header[key] -= x
        elif re.fullmatch(r"CRPIX2[A-Z]?", key):
            header[key] -= y
    header.add_history(f"Align Stacks {VERSION}: reference grid; linear interpolation.")
    header.add_history(f"Common crop (zero-based FITS array): {crop}")
    return header


def save_outputs(paths, images, crop, suffix):
    from astropy.io import fits
    if not suffix or any(c in suffix for c in '/\\'):
        raise ValueError("Enter a non-empty suffix without path separators.")
    outputs = [Path(p).with_name(Path(p).stem + suffix + ".fit") for p in paths]
    if len(set(outputs)) != len(outputs) or any(p.exists() for p in outputs):
        raise ValueError("An output already exists or output names collide. Change the suffix.")
    x, y, w, h = crop
    if min(x, y) < 0 or min(w, h) <= 0:
        raise ValueError("Invalid crop.")
    for data, _ in images:
        if x + w > data.shape[-1] or y + h > data.shape[-2]:
            raise ValueError("Crop exceeds image dimensions.")
    for path, (data, header) in zip(outputs, images):
        fits.PrimaryHDU(data=data[..., y:y+h, x:x+w],
                        header=output_header(header, images[0][1], crop)).writeto(path)
    return outputs


def main():
    try:
        import sirilpy as s
    except ImportError:
        s = None
    if s is not None:
        for package in ("PyQt6", "astropy", "scipy", "astroalign", "sep"):
            s.ensure_installed(package)
    from PyQt6 import QtCore, QtGui, QtWidgets as Q
    app = Q.QApplication.instance() or Q.QApplication(sys.argv)
    configdir = None
    if s is not None:
        interface = s.SirilInterface()
        try:
            interface.connect()
            configdir = interface.get_siril_configdir()
        except Exception:
            pass  # File picker still allows explicit configuration selection.
        finally:
            if interface.connected:
                interface.disconnect()

    class Preview(Q.QGraphicsView):
        def __init__(self):
            super().__init__()
            self.scene_ = Q.QGraphicsScene(self)
            self.setScene(self.scene_)
            self.setMinimumSize(650, 400)
            self.origin = None
            self.box = None
            self.crop = None

        def show_image(self, data, crop, common):
            image = mono(data)
            step = max(1, int(np.ceil(max(image.shape) / 1800)))
            pixels = preview_pixels(data, common, step)
            qimage = QtGui.QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                                 pixels.strides[0], QtGui.QImage.Format.Format_Grayscale8).copy()
            self.scene_.clear()
            item = self.scene_.addPixmap(QtGui.QPixmap.fromImage(qimage))
            item.setScale(step)
            self.scene_.setSceneRect(0, 0, image.shape[1], image.shape[0])
            self.box = self.scene_.addRect(QtCore.QRectF(), QtGui.QPen(QtCore.Qt.GlobalColor.green, 0))
            self.set_crop(crop)
            self.fitInView(self.scene_.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        def set_crop(self, crop):
            self.crop = crop
            if self.box:
                self.box.setRect(QtCore.QRectF(*crop))

        def mousePressEvent(self, event):
            if self.box and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.origin = self.mapToScene(event.position().toPoint())

        def mouseMoveEvent(self, event):
            if self.origin is not None:
                rect = QtCore.QRectF(self.origin, self.mapToScene(event.position().toPoint())).normalized()
                rect = rect.intersected(self.scene_.sceneRect())
                x, y = int(np.ceil(rect.x())), int(np.ceil(rect.y()))
                self.set_crop((x, y, max(0, int(rect.right())-x), max(0, int(rect.bottom())-y)))

        def mouseReleaseEvent(self, event):
            self.origin = None

    class Worker(QtCore.QThread):
        message = QtCore.pyqtSignal(str)
        result = QtCore.pyqtSignal(object)
        error = QtCore.pyqtSignal(str)

        def __init__(self, paths, options=None):
            super().__init__()
            self.paths = paths
            self.options = options

        def run(self):
            try:
                if self.options is None:
                    result = align(self.paths, self.message.emit)
                else:
                    result = align_sip(self.paths, self.options, self.message.emit)
                self.result.emit(result)
            except Exception as exc:
                self.error.emit(str(exc))

    class Dialog(Q.QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Align Stacks — {VERSION}")
            self.images = None
            self.worker = None
            layout = Q.QVBoxLayout(self)
            layout.addWidget(Q.QLabel("Add FITS stacks. The first image is the reference.\n"
                                     "Optionally use Make reference to change it. Delete/Backspace removes a selected stack."))
            self.files = Q.QListWidget()
            self.files.setMaximumHeight(110)
            self.delete_shortcuts = []
            for key in (QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete):
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.files)
                shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
                shortcut.activated.connect(self.remove_file)
                self.delete_shortcuts.append(shortcut)
            layout.addWidget(self.files)
            row = Q.QHBoxLayout()
            self.add = Q.QPushButton("Add images…")
            self.reference = Q.QPushButton("Make reference")
            self.run = Q.QPushButton("Align / preview")
            for button in (self.add, self.reference, self.run):
                row.addWidget(button)
            layout.addLayout(row)
            registration = Q.QHBoxLayout()
            registration.addWidget(Q.QLabel("Alignment:"))
            self.mode = Q.QComboBox()
            self.mode.addItems(["Star matching (similarity)", "Astrometric + SIP distortion correction"])
            registration.addWidget(self.mode)
            self.settings_button = Q.QPushButton("SIP settings…")
            self.settings_button.setEnabled(False)
            registration.addWidget(self.settings_button)
            layout.addLayout(registration)
            self.sip_options = SIPOptions(find_siril_cli(), find_siril_config(configdir))
            self.mode.currentIndexChanged.connect(self.change_mode)
            self.settings_button.clicked.connect(self.sip_settings)
            self.preview = Preview()
            layout.addWidget(self.preview)
            layout.addWidget(Q.QLabel("Drag a rectangle for a custom shared crop. Preview stretch is display only."))
            row = Q.QHBoxLayout()
            self.auto = Q.QPushButton("Automatic common crop")
            self.choice = Q.QComboBox()
            self.choice.currentIndexChanged.connect(self.display)
            self.suffix = Q.QLineEdit("_aligned")
            self.suffix.setMaximumWidth(140)
            self.save = Q.QPushButton("Save aligned FITS")
            for widget in (self.auto, self.choice, Q.QLabel("Suffix:"), self.suffix, self.save):
                row.addWidget(widget)
            layout.addLayout(row)
            self.log = Q.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(100)
            layout.addWidget(self.log)
            self.add.clicked.connect(self.add_files)
            self.reference.clicked.connect(self.make_reference)
            self.run.clicked.connect(self.start)
            self.auto.clicked.connect(lambda: self.preview.set_crop(self.auto_crop))
            self.save.clicked.connect(self.write)
            self.invalidate()

        def change_mode(self, index):
            self.settings_button.setEnabled(index == 1)
            self.suffix.setText("_aligned_sip" if index == 1 else "_aligned")
            self.invalidate()

        def sip_settings(self):
            dialog = Q.QDialog(self)
            dialog.setWindowTitle("Independent SIP solving — Siril 1.4.4")
            form = Q.QFormLayout(dialog)
            order = Q.QSpinBox()
            order.setRange(2, 5)
            order.setValue(self.sip_options.order)
            solver = Q.QComboBox()
            solver.addItems(["Siril solver — fresh solve per stack",
                             "Local Astrometry.net — fresh solve per stack",
                             "Use each input's existing SIP solution",
                             "Shared solver — independent SIP with recovery"])
            solvers = ["siril", "astrometry.net", "existing", "astap"]
            solver.setCurrentIndex(solvers.index(self.sip_options.solver))
            form.addRow("SIP polynomial order:", order)
            form.addRow("Solutions:", solver)
            fields = []
            for title, value in (("Siril CLI executable:", self.sip_options.executable),
                                 ("Siril 1.4 INI config:", self.sip_options.config),
                                 ("ASTAP executable:", self.sip_options.astap_executable or find_astap())):
                row = Q.QHBoxLayout()
                field = Q.QLineEdit(value)
                browse = Q.QPushButton("Browse…")
                def choose(_, target=field):
                    path, _ = Q.QFileDialog.getOpenFileName(dialog, "Choose file", target.text())
                    if path:
                        target.setText(path)
                browse.clicked.connect(choose)
                row.addWidget(field); row.addWidget(browse)
                form.addRow(title, row)
                fields.append(field)
            note = Q.QLabel("Each input needs its own accurate coordinates and effective pixel scale "
                            "in its FITS header. Fresh solves use Siril's configured catalogues "
                            "or local Astrometry.net indices.\n"
                            "No shared distortion master and no automatic uncorrected fallback. "
                            "Existing solutions must contain nonzero SIP coefficients of the selected order. "
                            "The reference will also be undistorted. The shared solver must provide order-3 SIP; ASTAP needs its star database; "
                            "recovery changes solver proxies only, never output science pixels.")
            note.setWordWrap(True)
            form.addRow(note)
            buttons = Q.QDialogButtonBox(Q.QDialogButtonBox.StandardButton.Ok |
                                         Q.QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            form.addRow(buttons)
            if dialog.exec() == Q.QDialog.DialogCode.Accepted:
                self.sip_options = SIPOptions(fields[0].text(), fields[1].text(),
                                              order.value(), solvers[solver.currentIndex()],
                                              astap_executable=fields[2].text())
                self.invalidate()

        def invalidate(self):
            self.images = None
            self.save.setEnabled(False)
            self.auto.setEnabled(False)
            self.choice.clear()
            self.preview.scene_.clear()
            self.preview.box = None

        def add_files(self):
            paths, _ = Q.QFileDialog.getOpenFileNames(self, "Choose stacks", "", "FITS (*.fit *.fits *.fts)")
            existing = {self.files.item(i).text() for i in range(self.files.count())}
            for path in paths:
                if path not in existing:
                    self.files.addItem(path)
                    existing.add(path)
            self.invalidate()

        def remove_file(self):
            index = self.files.currentRow()
            if index >= 0 and self.files.isEnabled():
                self.files.takeItem(index)
                self.invalidate()

        def make_reference(self):
            index = self.files.currentRow()
            if index >= 0:
                self.files.insertItem(0, self.files.takeItem(index))
                self.files.setCurrentRow(0)
                self.invalidate()

        def busy(self, value):
            for widget in (self.files, self.add, self.reference, self.run, self.mode):
                widget.setEnabled(not value)
            self.settings_button.setEnabled(not value and self.mode.currentIndex() == 1)

        def start(self):
            self.paths = [self.files.item(i).text() for i in range(self.files.count())]
            if len(self.paths) < 2:
                Q.QMessageBox.warning(self, "Align", "Choose at least two stacks.")
                return
            self.invalidate()
            self.busy(True)
            self.log.clear()
            options = self.sip_options if self.mode.currentIndex() == 1 else None
            self.worker = Worker(self.paths, options)
            self.worker.message.connect(self.log.appendPlainText)
            self.worker.result.connect(self.ready)
            self.worker.error.connect(self.failed)
            self.worker.finished.connect(lambda: self.busy(False))
            self.worker.start()

        def ready(self, result):
            try:
                self.images, self.common = result
                self.auto_crop = largest_rectangle(self.common)
                self.preview.crop = self.auto_crop
                self.choice.addItems([Path(p).name for p in self.paths])
                self.save.setEnabled(True)
                self.auto.setEnabled(True)
                self.log.appendPlainText(f"Ready. Automatic crop: {self.auto_crop}")
            except Exception as exc:
                self.failed(str(exc))

        def display(self, index):
            if self.images is not None and index >= 0:
                self.preview.show_image(self.images[index][0], self.preview.crop, self.common)

        def failed(self, message):
            self.invalidate()
            self.log.appendPlainText("FAILED: " + message + "\nNo fallback outputs accepted.")
            Q.QMessageBox.critical(self, "Alignment failed", message)

        def write(self):
            try:
                x, y, w, h = self.preview.crop
                if w <= 0 or h <= 0 or not self.common[y:y+h, x:x+w].all():
                    raise ValueError("Crop includes uncovered edges. Use Automatic common crop or draw a smaller rectangle.")
                paths = save_outputs(self.paths, self.images, self.preview.crop, self.suffix.text())
                self.log.appendPlainText("Saved:\n" + "\n".join(map(str, paths)))
                Q.QMessageBox.information(self, "Saved", "\n".join(map(str, paths)))
            except Exception as exc:
                Q.QMessageBox.critical(self, "Save failed", str(exc))

        def reject(self):
            if self.worker is None or not self.worker.isRunning():
                super().reject()

        def closeEvent(self, event):
            if self.worker is not None and self.worker.isRunning():
                event.ignore()
            else:
                event.accept()

    dialog = Dialog()
    dialog.exec()


if __name__ == "__main__":
    main()
