from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import os
import threading
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

class Diagnostic(str):
    detail = True

@dataclass
class SolveOptions:
    backend: str = 'default'
    executable: str = ''
    database: str = ''
    target: str = ''
    fov: float = 0
    radius: float = 180
    binning: int = 0
    timeout: float = 600
    confirm_original: bool = False
    require_sip: bool = False
    log_path: str = ''
    backend_options: dict = field(default_factory=dict)

@dataclass
class SolveResult:
    header: fits.Header
    backend: str
    attempts: list = field(default_factory=list)
    strategy: str = ''
    maximum_stars: int = 0
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

_BACKENDS = {}
def register_backend(name, solver):
    """Register a callable (pixels, header, options, progress, cancel) -> SolveResult."""
    if name == 'default' or not name:
        raise ValueError('A backend needs an explicit name.')
    _BACKENDS[name] = solver

def settings_path():
    return Path(os.environ.get('ASTRO_PLATESOLVE_CONFIG',
        str(Path.home()/'.config'/'astro-platesolve'/'config.json'))).expanduser()

def read_settings():
    path = settings_path()
    if not path.exists(): return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict): raise ValueError('Expected an object.')
        return value
    except (ValueError, OSError) as exc:
        raise ValueError(f'Invalid shared plate-solver configuration {path}: {exc}') from exc

def validate_solution(header, shape, require_sip=False):
    h, w = shape[-2:]
    for key, value in (('NAXIS1', w), ('NAXIS2', h)):
        if key in header and header[key] != value:
            raise ValueError('Solver changed image dimensions; WCS rejected.')
    if not {'CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1','CRVAL2'}.issubset(header):
        raise ValueError('Solver returned no valid celestial solution.')
    wcs = WCS(header, naxis=2)
    scale = wcs.pixel_scale_matrix
    points = [[0,0],[w-1,0],[0,h-1],[w-1,h-1],[w/2,h/2]]
    if (not wcs.has_celestial or not np.isfinite(scale).all() or
        abs(np.linalg.det(scale)) < 1e-15 or
        not np.isfinite(wcs.all_pix2world(points,0)).all()):
        raise ValueError('Solver returned an invalid celestial coordinate grid.')
    if require_sip and (wcs.sip is None or not
        any(np.any(matrix != 0) for matrix in (wcs.sip.a,wcs.sip.b))):
        raise ValueError('Solver returned linear WCS without nonzero SIP distortion.')
    return wcs

def resolve_options(options=None):
    options = options or SolveOptions()
    settings = read_settings() if options.backend == 'default' else {}
    backend = settings.get('backend', 'astap') if options.backend == 'default' else options.backend
    selected = replace(options, backend=backend)
    if options.backend == 'default':
        selected.executable = settings.get('executable', options.executable)
        selected.database = settings.get('database', options.database)
        selected.backend_options = {**options.backend_options, **settings.get('options', {})}
    return selected

def validate_backend(options=None):
    from .astap import resolve_astap, find_astap
    selected = resolve_options(options)
    if selected.backend == 'astap':
        resolve_astap(selected.executable or find_astap())
    elif selected.backend == 'skyshot':
        import shutil
        executable = selected.executable or shutil.which('skyshot')
        index = selected.backend_options.get('index') or selected.database
        if not executable or not Path(executable).is_file() or not index or not Path(index).is_dir():
            raise ValueError('Configure Skyshot executable and local index in shared solver settings.')
    elif selected.backend not in _BACKENDS:
        raise ValueError('Unknown plate-solving backend: '+selected.backend)
    return selected

def solve_image(pixels, header, options=None, progress=lambda message: None, cancel=None):
    """Return coordinates/provenance only; never return or commit science pixels.

    Applications keep their own file/undo/project policies. The shared default
    backend is selected centrally; an explicit backend overrides that default.
    """
    from . import astap, skyshot
    options = options or SolveOptions()
    selected = resolve_options(options)
    backend = selected.backend
    solver = _BACKENDS.get(backend) or {'astap':astap.solve, 'skyshot':skyshot.solve}.get(backend)
    if solver is None: raise ValueError(f'Unknown plate-solving backend: {backend}')
    cancel = cancel or threading.Event()
    if cancel.is_set(): raise InterruptedError('Plate solving cancelled.')
    data = np.array(pixels, dtype=np.float32, copy=True)
    if data.ndim not in (2,3) or (data.ndim == 3 and data.shape[0] not in (1,3)):
        raise ValueError('Expected mono or channel-first RGB pixels.')
    if not np.isfinite(data).all(): raise ValueError('Nonfinite solver input pixels.')
    if not np.isfinite(options.timeout) or options.timeout <= 0:
        raise RuntimeError('Plate solver timed out: timeout must be positive.')
    result = solver(data, header.copy(), selected, progress, cancel)
    if cancel.is_set(): raise InterruptedError('Plate solving cancelled.')
    validate_solution(result.header, data.shape, options.require_sip)
    result.backend = backend
    for warning in result.warnings:
        progress(Diagnostic('Solver warning: '+str(warning)))
    return result
