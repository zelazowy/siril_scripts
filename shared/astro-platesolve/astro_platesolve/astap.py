"""ASTAP backend and recovery/confirmation policies shared by all consumers."""
import os, codecs, copy, re, shlex, shutil, subprocess, tempfile, threading, time
from pathlib import Path
import numpy as np
from scipy import ndimage
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from .api import Diagnostic, SolveResult, validate_solution

def has_solution(header):
    try:
        validate_solution(header, (int(header.get('NAXIS2',1)),int(header.get('NAXIS1',1))))
        return True
    except (ValueError, TypeError): return False

def find_astap():
    for value in (shutil.which('astap'),shutil.which('astap_cli'),
                  '/Applications/ASTAP.app/Contents/MacOS/astap','/Applications/astap.app/Contents/MacOS/astap','/usr/local/bin/astap'):
        if value and Path(value).is_file(): return str(Path(value).resolve())
    return ''


def resolve_astap(value):
    path=Path(value).expanduser()
    if path.is_dir(): path=path/'Contents'/'MacOS'/'astap' if path.suffix.lower()=='.app' else path/'astap'
    if not path.is_file() or not os.access(path,os.X_OK):
        raise ValueError('Choose an installed ASTAP executable.')
    return path.resolve()


class SolverLogTail:
    """Read appended solver output without blocking or losing partial lines."""
    def __init__(self,reader,progress):
        self.reader=reader;self.progress=progress;self.pending=''
        self.decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')

    def drain(self,final=False):
        data=self.reader.read()
        if isinstance(data,bytes): text=self.decoder.decode(data,final=final)
        else: text=data
        lines=(self.pending+text).replace('\r','\n').split('\n')
        self.pending=lines.pop()
        if final and self.pending: lines.append(self.pending);self.pending=''
        for line in lines:
            if line.strip(): self.progress(Diagnostic('ASTAP | '+line.rstrip()))
        return bool(data)


def solver_header(header):
    """Repair undefined equinox only on the private solver copy."""
    result=header.copy()
    if 'EQUINOX' in result:
        try: value=float(result['EQUINOX'])
        except (ValueError,TypeError): value=float('nan')
        if not np.isfinite(value) or not 1000<=value<=3000:
            result['EQUINOX']=(2000.,'Undefined input equinox; use J2000')
    return result


def search_header(header):
    result = solver_header(header)
    for key in list(result):
        if re.fullmatch(r'(?:A|B|AP|BP)_(?:ORDER|DMAX|\d+_\d+)',key): del result[key]
    for key in ('CTYPE1','CTYPE2'):
        if key in result: result[key] = str(result[key]).removesuffix('-SIP')
    result.remove('PLTSOLVD',ignore_missing=True)
    return result


def clean_solution_header(header):
    """ASTAP sidecars may retain obsolete input PC and SIP coefficients."""
    result=solver_header(header)
    if any(re.fullmatch(r'CD[12]_[12]',key) for key in result):
        for key in list(result):
            if re.fullmatch(r'PC[12]_[12]',key):del result[key]
    if not any('-SIP' in str(result.get(key,'')) for key in ('CTYPE1','CTYPE2')):
        for key in list(result):
            if re.fullmatch(r'(?:A|B|AP|BP)_(?:ORDER|DMAX|\d+_\d+)',key):del result[key]
    return result


def has_pointing_hints(header):
    return any(all(header.get(key) not in (None,'') for key in pair)
        for pair in (('OBJCTRA','OBJCTDEC'),('RA','DEC'),('CRVAL1','CRVAL2')))


def has_search_hints(header):
    """ASTAP can interpret pointing/optics metadata without replacing pixels."""
    pointing=has_pointing_hints(header)
    def positive(key):
        try:return np.isfinite(float(header.get(key,0))) and float(header.get(key,0))>0
        except (ValueError,TypeError):return False
    optics=positive('FOCALLEN') and (positive('YPIXSZ') or positive('XPIXSZ'))
    return pointing or optics


def normalize_pixels(pixels):
    image = np.array(pixels, dtype=np.float32, copy=True)
    if image.ndim == 3 and image.shape[0] in (1,3):
        image = image.mean(axis=0,dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Expected finite mono or channel-first RGB solver pixels.')
    low, high = float(image.min()), float(image.max())
    if high <= low: raise ValueError('This image is constant; no stars can be detected.')
    return np.ascontiguousarray((image-low)*(65535./(high-low)))


def compact_star_proxy(mono,cancel=None):
    """Measured star peaks only, in the unchanged pixel grid; not a render layer."""
    if cancel and cancel.is_set(): raise InterruptedError('Plate solving cancelled.')
    smooth=ndimage.gaussian_filter(mono,1)
    if cancel and cancel.is_set(): raise InterruptedError('Plate solving cancelled.')
    light=np.maximum(smooth-ndimage.gaussian_filter(mono,10),0)
    if cancel and cancel.is_set(): raise InterruptedError('Plate solving cancelled.')
    peaks=(light==ndimage.maximum_filter(light,9))&(light>np.percentile(light,99))
    yy,xx=np.where(peaks);strength=light[yy,xx]
    order=np.argsort(strength)[-1500:]
    if len(order)<12: return None
    result=np.zeros_like(mono)
    # Integer measured peaks preserve the verified proxy strategy. No resampling,
    # cropping, rotation or coordinate remapping is needed for the solved WCS.
    result[yy[order],xx[order]]=strength[order]
    result=ndimage.gaussian_filter(result,2)
    if cancel and cancel.is_set(): raise InterruptedError('Plate solving cancelled.')
    return result


def run_attempt(executable,proxy,options,ra,dec,stars,log_path,progress,cancel,budget):
    for suffix in ('.ini','.wcs'):
        sidecar=proxy.with_suffix(suffix)
        if sidecar.exists():sidecar.unlink()
    argv=[str(executable),'-f',str(proxy),'-r',str(options.radius),'-z',str(options.binning),
          '-s',str(stars),'-sip','-wcs','-progress']
    if options.fov is not None:argv+=['-fov',str(options.fov)]
    if ra is not None and dec is not None:argv+=['-ra',str(ra),'-spd',str(dec+90)]
    if options.database: argv+=['-d',str(Path(options.database).expanduser().resolve())]
    progress(Diagnostic('ASTAP command: '+shlex.join(argv)))
    started=time.monotonic();last_output=started;last_heartbeat=started;timed_out=False
    with log_path.open('a') as log:
        log.write('\nCommand: '+shlex.join(argv)+'\n');log.flush()
        with log_path.open('rb') as reader:
            reader.seek(log.tell());tail=SolverLogTail(reader,progress)
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=str(proxy.parent))
            try:
                while process.poll() is None:
                    if tail.drain(): last_output=time.monotonic()
                    if cancel.wait(.2): raise InterruptedError('Plate solving cancelled. Diagnostic log: '+str(log_path))
                    now=time.monotonic()
                    if now-last_heartbeat>=10 and now-last_output>=10:
                        progress(f'ASTAP still working — {now-started:.0f} s in this attempt, {now-last_output:.0f} s since last output. Cancel is available.')
                        last_heartbeat=now
                    if now-started>budget: timed_out=True;break
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill();process.wait()
                tail.drain(final=True)
    if timed_out:
        progress('Attempt time limit reached; trying the next available strategy.')
        return None,'Attempt time limit reached.'
    progress(Diagnostic(f'ASTAP exited with code {process.returncode} after {time.monotonic()-started:.1f} s.'))
    if cancel.is_set(): raise InterruptedError('Plate solving cancelled.')
    values={};ini=proxy.with_suffix('.ini')
    if ini.exists():
        for line in ini.read_text(errors='replace').splitlines():
            if '=' in line:
                key,value=line.split('=',1);values[key.strip().upper()]=value.split('//')[0].strip().strip(chr(34)).strip(chr(39))
    if values.get('WARNING'):progress(Diagnostic('ASTAP warning: '+values['WARNING']))
    wcs_path=proxy.with_suffix('.wcs')
    if process.returncode not in (0,1,2,16,32,33):
        raise RuntimeError(f'ASTAP exited unexpectedly with code {process.returncode}. Diagnostic log: {log_path}')
    if process.returncode==0 and values.get('PLTSOLVD','').upper()=='T' and wcs_path.exists():
        text=wcs_path.read_text(encoding='ascii').rstrip('\x00')
        result=clean_solution_header(fits.Header.fromstring(text,sep='\n' if '\n' in text else ''))
        if values.get('WARNING'):result['WARNING']=values['WARNING']
        return result,''
    reasons={1:'No matching star pattern found.',2:'Not enough usable stars detected.',16:'Could not read the solver image.',32:'ASTAP star database not found.',33:'Could not read the ASTAP star database.'}
    reason=values.get('ERROR') or reasons.get(process.returncode,f'Exit code {process.returncode}.')
    if process.returncode in (16,32,33):
        raise RuntimeError(reason+' Diagnostic log: '+str(log_path))
    return None,reason


def refinement_consistent(seed,refined,width,height):
    """Reject gross shifts, parity changes and unstable distortion at image edges."""
    try:
        before=clean_solution_header(seed);after=clean_solution_header(refined);before['NAXIS']=2;after['NAXIS']=2
        a=WCS(before,naxis=2);b=WCS(after,naxis=2)
        grid=np.array([[width/2,height/2],[0,0],[width-1,0],[0,height-1],[width-1,height-1]])
        errors=a.pixel_to_world(grid[:,0],grid[:,1]).separation(b.pixel_to_world(grid[:,0],grid[:,1])).arcsec
        scale=float(np.linalg.norm(a.pixel_scale_matrix,axis=0).mean()*3600)
        coarse='inexact scale' in str(seed.get('WARNING','')).lower()
        edge_limit=max(5*scale,float(np.linalg.norm(a.pixel_scale_matrix[:,1])*height*3600)*(.02 if coarse else .005))
        same_parity=np.linalg.det(a.pixel_scale_matrix)*np.linalg.det(b.pixel_scale_matrix)>0
        good=bool(np.isfinite(errors).all() and errors[0]<=max(2,3*scale) and errors[1:].max()<=edge_limit and same_parity)
        return good,f'center shift {errors[0]:.2f} arcsec, maximum corner shift {errors[1:].max():.2f} arcsec'
    except (ValueError,TypeError): return False,'invalid refined coordinate grid'


def solve(pixels, header, options, progress, cancel):
    executable=resolve_astap(options.executable or find_astap())
    if options.database and not Path(options.database).expanduser().is_dir():
        raise ValueError('ASTAP star database folder does not exist.')
    if not 0<=options.fov<=180 or not 0<options.radius<=180 or not 0<=options.binning<=8:
        raise ValueError('Invalid field height, search radius or downsampling.')
    ra=0.;dec=0.
    if options.target.strip():
        progress('Finding the approximate sky position…')
        try: center=SkyCoord(options.target,unit=(u.hourangle,u.deg),frame='icrs')
        except ValueError: center=SkyCoord.from_name(options.target)
        ra=float(center.icrs.ra.hour);dec=float(center.icrs.dec.deg)
    if cancel.is_set(): raise InterruptedError('Plate solving cancelled.')
    mono=normalize_pixels(pixels)
    progress(Diagnostic(f'ASTAP search: target {options.target or "blind/all sky"}; radius {options.radius:g}°; field height {format(options.fov,"g")+"°" if options.fov else "auto (about 0.3–10°)"}; downsampling {options.binning or "auto"}.'))
    progress(Diagnostic('ASTAP database: '+(options.database or 'ASTAP default location')))
    progress(Diagnostic('Preparing the same-size monochrome solver copy…'))
    log_path=Path(options.log_path) if options.log_path else Path(tempfile.mkdtemp(prefix='astro-solve-log-'))/'solver.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    log_path.write_text('astro-platesolve ASTAP backend\n')
    progress('Solver diagnostic log: '+str(log_path))
    search_header=globals()['search_header'](header)
    if search_header.get('EQUINOX')!=header.get('EQUINOX'):
        progress(Diagnostic('Undefined/invalid EQUINOX in source metadata; using J2000 for the solver.'))
    strategies=[]
    if has_search_hints(search_header):
        strategies += [('header-guided',500),('header-guided-expanded',2000),('header-guided-compact',500)]
    strategies += [('expanded-star',2000),('compact-peaks',500),('standard',500)]
    compact=None;compact_prepared=False;insufficient_original=False
    deadline=time.monotonic()+getattr(options,'timeout',600)
    solved=None;error='No matching star pattern found.';attempts=[]
    with tempfile.TemporaryDirectory(prefix='astro_astap_') as work:
        for index,(strategy,stars) in enumerate(strategies):
            if cancel.is_set(): raise InterruptedError('Plate solving cancelled.')
            use_compact=strategy in ('compact-peaks','header-guided-compact')
            if insufficient_original and not use_compact:
                progress(Diagnostic('Skipping ordinary-image retry after insufficient star detection.'));continue
            remaining=deadline-time.monotonic()
            if remaining<=0: raise RuntimeError('ASTAP timed out. Diagnostic log: '+str(log_path))
            proxy=Path(work)/f'proxy_{index+1}.fits'
            data=mono
            if use_compact:
                if not compact_prepared:
                    progress('Building a compact-star proxy from measured peaks; keeping the original coordinate grid.')
                    compact=compact_star_proxy(mono,cancel);compact_prepared=True
                data=compact
                if data is None:
                    progress('Too few distinct peaks for a compact-star retry; skipping it.');continue
                data=data/max(float(data.max()),1e-10)*65535
            fits.PrimaryHDU(data,search_header).writeto(proxy)
            progress(f'Solving sky position: attempt {index+1}/{len(strategies)} — {strategy}, up to {stars} stars.')
            remaining=deadline-time.monotonic()
            if remaining<=0: raise RuntimeError('ASTAP timed out. Diagnostic log: '+str(log_path))
            budget=remaining/(len(strategies)-index)
            search=copy.copy(options);search_ra,search_dec=ra,dec
            if strategy.startswith('header-guided'):
                search.fov=options.fov if options.fov else None
                if options.target.strip():search.radius=min(options.radius,5)
                elif has_pointing_hints(search_header):search.radius=min(options.radius,30)
                if not options.target.strip():search_ra=search_dec=None
            elif options.target.strip() and strategy!='standard':search.radius=min(options.radius,5)
            progress(Diagnostic(f'Attempt search radius: {search.radius:g}°.'))
            solved,error=run_attempt(executable,proxy,search,search_ra,search_dec,stars,log_path,progress,cancel,budget)
            if not use_compact and any(text in error.lower() for text in ('not enough usable stars','not enough stars','no stars')):
                insufficient_original=True
                progress(Diagnostic('Too few detected stars; trying processed-image recovery before wider searches.'))
            if solved is not None:solved=clean_solution_header(solved)
            attempts.append(dict(strategy=strategy,stars=stars,fov=search.fov,radius=search.radius,solved=solved is not None,error=error))
            if solved is not None and options.confirm_original:
                # Every first match is a seed. Confirm the measured scale and
                # refine with more real photographed stars before committing WCS.
                progress('Candidate found; refining coordinates on original photographic pixels, up to 5000 stars.')
                seed=solved.copy();seed['NAXIS']=2
                if not has_solution(seed):
                    raise RuntimeError('ASTAP returned no valid celestial solution.')
                else:
                    seed_wcs=WCS(seed,naxis=2)
                    center=seed_wcs.pixel_to_world((pixels.shape[-1]-1)/2,(pixels.shape[-2]-1)/2)
                    guided=copy.copy(options)
                    guided.fov=float(np.linalg.norm(seed_wcs.pixel_scale_matrix[:,1])*pixels.shape[-2])
                    guided.radius=min(options.radius,5)
                    if not np.isfinite(guided.fov) or not 0<guided.fov<=180:
                        raise RuntimeError('ASTAP returned an invalid sky scale.')
                    original_proxy=Path(work)/f'original_confirmation_{index}.fits'
                    fits.PrimaryHDU(mono,search_header).writeto(original_proxy)
                    solved=None
                    measured_fov=guided.fov
                    for refinement_index,(refinement_stars,scale_factor) in enumerate(((5000,1),(2000,1),(3000,1.01))):
                        # A slightly wider scale bracket helps processed fields whose
                        # first estimate lands on an unsuccessful quad selection.
                        guided.fov=measured_fov*scale_factor
                        remaining=deadline-time.monotonic()
                        if remaining<=0: raise RuntimeError('ASTAP timed out. Diagnostic log: '+str(log_path))
                        # Fresh output names prevent reusing a stale successful sidecar.
                        refinement_proxy=Path(work)/f'refine_{index}_{refinement_index}.fits'
                        shutil.copyfile(original_proxy,refinement_proxy)
                        candidate,error=run_attempt(executable,refinement_proxy,guided,float(center.ra.hour),float(center.dec.deg),
                            refinement_stars,log_path,progress,cancel,remaining/2)
                        accepted=False
                        if candidate is not None:
                            candidate=clean_solution_header(candidate)
                            accepted,comparison=refinement_consistent(seed,candidate,pixels.shape[-1],pixels.shape[-2])
                            if 'inexact scale' in str(candidate.get('WARNING','')).lower(): accepted=False
                            progress(Diagnostic(('Refinement accepted: ' if accepted else 'Refinement rejected: ')+comparison))
                            if not accepted: error='Refined WCS failed consistency/scale checks.'
                        attempts.append(dict(strategy='original-refinement',stars=refinement_stars,fov=guided.fov,radius=guided.radius,solved=candidate is not None,accepted=accepted,error=error))
                        if accepted:
                            solved=candidate;strategy=strategy+'+original-refinement';stars=refinement_stars;break
                        progress('Retrying refinement with another star sample/scale bracket.' if refinement_index<2 else 'No reliable refinement; trying the next search strategy.')
            if solved is not None:
                progress(Diagnostic('Successful ASTAP strategy: '+strategy));break
            progress('Attempt did not solve: '+error)
        if solved is None:
            if time.monotonic()>=deadline: error='ASTAP timed out.'
            raise RuntimeError('ASTAP could not solve this image: '+error+'\nCheck its star database, or provide an object name or set the field height in Advanced settings.\nDiagnostic log: '+str(log_path))
        output=log_path.read_text(errors='replace')
        for key,size in (('NAXIS1',pixels.shape[-1]),('NAXIS2',pixels.shape[-2])):
            if key in solved and solved[key]!=size: raise RuntimeError('ASTAP changed image dimensions; solution rejected.')
        required={'CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1','CRVAL2'}
        if not required.issubset(solved): raise RuntimeError('ASTAP returned no valid celestial solution.')
        solved['NAXIS']=2;solved['NAXIS1']=pixels.shape[-1];solved['NAXIS2']=pixels.shape[-2]
        if not has_solution(solved): raise RuntimeError('ASTAP returned no valid celestial solution.')
        wcs=WCS(solved,naxis=2)
        if not np.isfinite(wcs.pixel_scale_matrix).all() or abs(np.linalg.det(wcs.pixel_scale_matrix))<1e-15:
            raise RuntimeError('ASTAP returned an invalid sky scale.')
        if not np.isfinite(wcs.all_pix2world([[0,0],[pixels.shape[-1]-1,pixels.shape[-2]-1]],0)).all():
            raise RuntimeError('ASTAP returned invalid sky coordinates.')
        return SolveResult(solved, 'astap', attempts, strategy, stars,
            [str(solved['WARNING'])] if solved.get('WARNING') else [],
            {'diagnostic_log':str(log_path),'executable':str(executable)})
