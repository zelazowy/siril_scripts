"""Skyshot CLI schema-v1 adapter. Never emulate ASTAP flags or drop local_grid."""
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from astropy.io import fits
from .api import SolveResult, Diagnostic

def solve(pixels, header, options, progress, cancel):
    executable=options.executable or shutil.which('skyshot')
    index=options.backend_options.get('index') or options.database
    if not executable or not Path(executable).is_file():
        raise ValueError('Configure an installed skyshot executable in the shared solver settings.')
    if not index or not Path(index).is_dir():
        raise ValueError('Skyshot requires a compatible local index directory.')
    mode=options.backend_options.get('mode','auto')
    if mode not in ('auto','standard','deep'): raise ValueError('Invalid Skyshot mode.')
    deadline=options.backend_options.get('deadline_ms',2000)
    if not isinstance(deadline,(int,float)) or not 0<deadline<=options.timeout*1000:
        raise ValueError('Skyshot deadline must fit within the total process timeout.')
    with tempfile.TemporaryDirectory(prefix='astro_skyshot_') as temporary:
        root=Path(temporary);source=root/'input.fits';wcs_path=root/'solution.wcs'
        # FITS preserves native pixels/axis order; no PNG/JPEG flipping or resizing.
        fits.writeto(source,pixels,header)
        command=[str(executable),'solve',str(source),'--index',str(index),
                 '--mode',mode,'--deadline-ms',str(deadline),'--wcs',str(wcs_path)]
        progress('Solving sky position with Skyshot…')
        with (root/'stdout.json').open('w+') as stdout, (root/'stderr.log').open('w+') as stderr:
            started=time.monotonic()
            process=subprocess.Popen(command,stdout=stdout,stderr=stderr)
            try:
                while process.poll() is None:
                    if cancel.wait(.05): raise InterruptedError('Plate solving cancelled.')
                    if time.monotonic()-started>options.timeout:
                        raise RuntimeError('Skyshot process timed out.')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill();process.wait()
            stdout.seek(0);stderr.seek(0)
            raw=stdout.read();errors=stderr.read()
        if errors: progress(Diagnostic(errors))
        if options.log_path:
            log=Path(options.log_path);log.parent.mkdir(parents=True,exist_ok=True)
            log.write_text(errors+'\n'+raw)
        if process.returncode != 0:
            raise RuntimeError(f'Skyshot failed (exit {process.returncode}): {errors[-2000:]} {raw[-2000:]}')
        try: envelope=json.loads(raw)
        except ValueError as exc: raise ValueError('Skyshot returned invalid JSON.') from exc
        if not isinstance(envelope,dict) or not isinstance(envelope.get('solution'),dict):
            raise ValueError('Skyshot returned an invalid result envelope.')
        solution=envelope['solution']
        if envelope.get('schema_version') != 1 or solution.get('status') != 'solved':
            raise ValueError('Skyshot did not return a solved schema-v1 result.')
        h,w=pixels.shape[-2:]
        if solution.get('width') != w or solution.get('height') != h:
            raise ValueError('Skyshot result dimensions differ from input.')
        coordinates=solution.get('wcs')
        if not isinstance(coordinates,dict): raise ValueError('Skyshot returned no WCS.')
        if coordinates.get('local_grid') is not None:
            raise ValueError('Skyshot local_grid is not representable in FITS WCS; refusing to discard distortion correction.')
        if not wcs_path.is_file(): raise ValueError('Skyshot returned no new FITS WCS file.')
        text=wcs_path.read_text(encoding='ascii')
        solved=fits.Header.fromstring(text,sep='\n' if '\n' in text else '')
        return SolveResult(solved,'skyshot',strategy='native',warnings=solution.get('warnings',[]),
            metadata={'schema_version':1,'rms_px':solution.get('rms_px'),
                      'timings':envelope.get('timings',{}),'executable':str(executable)})
