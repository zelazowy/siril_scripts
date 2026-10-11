import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sas_white_balance_core import apply_star_based_white_balance, auto_rect_50x50
rng=np.random.default_rng(42)
h=w=512
y,x=np.mgrid[:h,:w]
image=np.full((h,w,3),.012,dtype=np.float32)
for _ in range(140):
 cx,cy=rng.integers(15,497,size=2)
 star=.05*np.exp(-((x-cx)**2+(y-cy)**2)/8)
 image+=star[...,None]*np.array([1.8,1.,.7],np.float32)
image+=rng.normal(0,.0001,image.shape).astype(np.float32)
original=image.copy()
out,count,_=apply_star_based_white_balance(image,threshold=5,autostretch=False)
assert count>50, count
assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1
assert np.array_equal(image,original)
import sas_white_balance_core as core
excursions = np.array([-0.001, 0.012, 0.5, 1.001], dtype=np.float32)
saved = excursions.copy()
assert np.allclose(core._to_float01(excursions),
                   (excursions.astype(np.float64)-float(saved.min())) /
                   (float(saved.max())-float(saved.min())))
assert np.array_equal(excursions, saved)
assert np.array_equal(core._to_float01(np.array([0.1, 0.6], np.float32)),
                      np.array([0.1, 0.6], np.float32))
assert np.allclose(core._to_float01(np.array([0.2, 1., 2.], np.float32)), [0.1, 0.5, 1.])
assert np.allclose(core._to_float01(np.array([-1., 0., 0.5], np.float32)), [0., 0.5, 0.75])
rgb = np.array([[[-0.2, 0.4, 1.6], [0., 0.8, 1.2]]], np.float32)
assert np.allclose(core._to_float01(rgb), (rgb.astype(np.float64)+0.2)/1.8)
assert np.allclose(core._to_float01(core._to_float01(rgb)), core._to_float01(rgb))
for invalid in (np.nan, np.inf):
 try: core._to_float01(np.array([invalid], dtype=np.float32))
 except ValueError: pass
 else: raise AssertionError('non-finite input accepted')
samples,_=core._sample_star_circle_medians(out,core.cached_star_sources)
med=np.median(samples,axis=0)
assert (np.max(med)-np.min(med))/np.mean(med)<0.02,med
out2,count2,_=apply_star_based_white_balance(image,auto_sigma=True)
assert count2>50
assert auto_rect_50x50(image)==auto_rect_50x50(image)
for size in (1,8,20,60):
 rect=auto_rect_50x50(np.zeros((size,size,3),np.float32))
 a,b,c,d=rect
 assert 0<=a and 0<=b and a+c<=size and b+d<=size,(size,rect)
try: apply_star_based_white_balance(np.zeros((64,64,3),np.float32))
except ValueError: pass
else: raise AssertionError('blank field accepted')
print(f'PASS: {count} manual / {count2} auto stars; neutral median {med}; bounded output, unchanged input, deterministic background, small images, blank rejection')
# Preview must adapt to pedestal changes without inventing a new color cast.
reference = image.copy()
preview = core.preview_rgb(reference)
shifted_preview = core.preview_rgb(0.1 + 0.9 * reference)
assert np.max(np.abs(preview.astype(int)-shifted_preview.astype(int))) <= 1
assert np.array_equal(reference, image)
assert preview.dtype == np.uint8 and preview.shape == image.shape
neutral = np.repeat(image[..., 1:2], 3, axis=2)
neutral_preview = core.preview_rgb(neutral)
assert np.array_equal(neutral_preview[..., 0], neutral_preview[..., 1])
assert np.array_equal(neutral_preview[..., 1], neutral_preview[..., 2])
assert 55 < np.median(neutral_preview) < 75
assert core.preview_rgb(np.zeros((8,8,3),np.float32)).max() == 0
print('PASS: display stretch adapts to pedestal, preserves neutral RGB and input; median near 25%')
