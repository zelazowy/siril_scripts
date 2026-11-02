import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image, ImageOps
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from skimage.transform import SimilarityTransform
from Siril_Align_Stacks import read_stack, align, largest_rectangle, save_outputs, warp

class RasterTests(unittest.TestCase):
    def test_tiff_16bit_float_and_planar_rgb_preserve_decoded_values(self):
        rgb=np.arange(3*24*32,dtype=np.uint16).reshape(3,24,32)*17
        with tempfile.TemporaryDirectory() as folder:
            for name, raster, kwargs in (
                ('rgb.tif',np.moveaxis(rgb,0,-1),{'photometric':'rgb'}),
                ('planar.tiff',rgb,{'photometric':'rgb','planarconfig':'separate'}),
                ('mono.tif',rgb[0],{}),
                ('float.tif',(rgb[0].astype(np.float32)-200)/500,{})):
                p=Path(folder)/name;tifffile.imwrite(p,raster,**kwargs);before=p.read_bytes()
                result,h=read_stack(p)
                expected=rgb if raster.ndim==3 else raster
                np.testing.assert_array_equal(result,expected[...,::-1,:].astype(np.float32))
                self.assertEqual(result.dtype,np.float32)
                self.assertEqual(h['ROWORDER'],'BOTTOM-UP')
                self.assertEqual(p.read_bytes(),before)

    def test_jpeg_exif_orientation_and_embedded_wcs(self):
        rgb=np.arange(24*32*3,dtype=np.uint8).reshape(24,32,3)
        w=WCS(naxis=2);w.wcs.crpix=[16,12];w.wcs.crval=[56,24]
        w.wcs.ctype=['RA---TAN','DEC--TAN'];w.wcs.cdelt=[-.001,.001]
        header=fits.PrimaryHDU(np.zeros((3,24,32),np.float32)).header
        header.update(w.to_header())
        with tempfile.TemporaryDirectory() as folder:
            for orientation in (1,6):
                exif=Image.Exif();exif[274]=orientation;exif[270]=header.tostring(sep='\n',padding=False)
                p=Path(folder)/f'photo{orientation}.JPG';Image.fromarray(rgb).save(p,exif=exif)
                before=p.read_bytes()
                with Image.open(p) as image:expected=np.array(ImageOps.exif_transpose(image))
                data,actual=read_stack(p)
                np.testing.assert_array_equal(data,np.moveaxis(expected[::-1],-1,0))
                self.assertEqual('CTYPE1' in actual,orientation==1)
                self.assertEqual(p.read_bytes(),before)

    def test_tiff_metadata_orientation_and_invalid_inputs(self):
        data=np.arange(24*32,dtype=np.uint16).reshape(24,32)
        header=fits.PrimaryHDU(data).header;header['CTYPE1']='RA---TAN'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'embedded.tif'
            tifffile.imwrite(p,data,description=header.tostring(sep='\n',padding=False),metadata=None)
            result,h=read_stack(p);self.assertEqual(h['CTYPE1'],'RA---TAN')
            tifffile.imwrite(p,data,extratags=[(274,'H',1,6,False)],description=header.tostring(sep='\n',padding=False),metadata=None)
            result,h=read_stack(p)
            np.testing.assert_array_equal(result,np.rot90(data,-1)[::-1])
            self.assertNotIn('CTYPE1',h)
            tifffile.imwrite(root/'alpha.tif',np.zeros((24,32,4),np.uint8),photometric='rgb')
            with self.assertRaisesRegex(ValueError,'alpha'):read_stack(root/'alpha.tif')
            with tifffile.TiffWriter(root/'pages.tif') as image:
                image.write(data);image.write(data)
            with self.assertRaisesRegex(ValueError,'multipage'):read_stack(root/'pages.tif')

    def test_original_format_export_preserves_tiff_type_and_jpeg_orientation(self):
        rng=np.random.default_rng(8)
        raster=rng.integers(0,65535,(30,40,3),dtype=np.uint16)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=[root/'image.tif',root/'image.jpg',root/'float.tiff']
            tifffile.imwrite(paths[0],raster,photometric='rgb')
            exif=Image.Exif();exif[274]=6
            Image.fromarray((raster/257).astype(np.uint8)).save(paths[1],quality=95,exif=exif)
            tifffile.imwrite(paths[2],raster.astype(np.float32)/65535,photometric='rgb')
            originals=[p.read_bytes() for p in paths]
            images=[read_stack(p) for p in paths]
            crop=(2,3,20,18);x,y,w,h=crop
            outputs=save_outputs(paths,images,crop,'_aligned')
            self.assertEqual([p.suffix for p in outputs],['.tif','.jpg','.tiff'])
            for index in (0,2):
                exported=tifffile.imread(outputs[index])
                self.assertEqual(exported.dtype,np.uint16 if index==0 else np.float32)
                expected=np.moveaxis(images[index][0][...,y:y+h,x:x+w][:,::-1,:],0,-1)
                np.testing.assert_array_equal(exported,expected.astype(exported.dtype))
                np.testing.assert_array_equal(read_stack(outputs[index])[0],images[index][0][...,y:y+h,x:x+w])
            with Image.open(outputs[1]) as image:
                self.assertEqual(image.size,(w,h));self.assertEqual(image.getexif().get(274),1)
            self.assertEqual([p.read_bytes() for p in paths],originals)
            with self.assertRaises(ValueError):save_outputs(paths,images,crop,'_aligned')

    def test_stretched_mixed_inputs_register_crop_and_export_losslessly(self):
        rng=np.random.default_rng(42)
        stars=np.zeros((256,280),np.float32)
        for y,x in rng.integers([25,25],[230,255],size=(80,2)):stars[y,x]+=rng.uniform(10,50)
        blurred=gaussian_filter(stars,1.5)
        # Deliberately nonlinear finished image, in 16-bit digital units.
        reference=(500+60000*np.sqrt(blurred/blurred.max())).astype(np.uint16)
        reference_rgb=np.stack([reference,(reference*.8).astype(np.uint16),(reference*.6).astype(np.uint16)])
        transform=SimilarityTransform(translation=(6,-4))
        moving,_=warp(reference_rgb,transform.params,reference.shape)
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/name for name in ('reference.fit','moving.tif','moving.jpg')]
            fits.writeto(paths[0],reference_rgb)
            tifffile.imwrite(paths[1],np.moveaxis(moving[:,::-1,:].astype(np.uint16),0,-1),photometric='rgb')
            jpeg=np.moveaxis((moving[:,::-1,:]/257).astype(np.uint8),0,-1)
            Image.fromarray(jpeg).save(paths[2],quality=95)
            originals=[p.read_bytes() for p in paths]
            images,common=align(paths)
            crop=largest_rectangle(common);x,y,w,h=crop
            outputs=save_outputs(paths,images,crop,'_aligned',output_format='fits')
            # Distinct source formats sharing a stem must not overwrite each other.
            for output,(aligned,_) in zip(outputs,images):
                np.testing.assert_array_equal(fits.getdata(output),aligned[...,y:y+h,x:x+w])
            np.testing.assert_array_equal(fits.getdata(outputs[0]),reference_rgb[...,y:y+h,x:x+w])
            for i,divisor in ((1,1),(2,257)):
                error=np.sqrt(np.mean((images[i][0][...,y:y+h,x:x+w]*divisor-reference_rgb[...,y:y+h,x:x+w])**2))
                self.assertLess(error,1200)
            self.assertEqual([p.read_bytes() for p in paths],originals)

if __name__=='__main__':unittest.main()
