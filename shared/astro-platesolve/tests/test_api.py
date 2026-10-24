import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from astropy.wcs import WCS
from astro_platesolve import SolveOptions, SolveResult, solve_image, register_backend
from astro_platesolve import skyshot


def header():
    w=WCS(naxis=2);w.wcs.crpix=[8,6];w.wcs.crval=[56,24]
    w.wcs.ctype=['RA---TAN','DEC--TAN'];w.wcs.cdelt=[-.01,.01]
    return w.to_header()

class SharedTests(unittest.TestCase):
    def test_one_default_selects_registered_engine_and_preserves_inputs(self):
        data=np.arange(192,dtype=np.float32).reshape(12,16);before=data.copy();h=header()
        def fake(pixels,source,options,progress,cancel):
            self.assertEqual(options.executable,'/central/engine')
            pixels[:]=0;source['OBJECT']='private'
            return SolveResult(header(),'fake')
        register_backend('fake',fake)
        with tempfile.TemporaryDirectory() as folder:
            config=Path(folder)/'config.json';config.write_text(json.dumps({'backend':'fake','executable':'/central/engine'}))
            with patch.dict('os.environ',{'ASTRO_PLATESOLVE_CONFIG':str(config)}):
                result=solve_image(data,h,SolveOptions(executable='/old/astap'))
        self.assertEqual(result.backend,'fake');np.testing.assert_array_equal(data,before)
        self.assertNotIn('OBJECT',h)

    def test_required_sip_and_dimension_validation_apply_to_all_backends(self):
        register_backend('linear',lambda *args:SolveResult(header(),'linear'))
        with self.assertRaisesRegex(ValueError,'nonzero SIP'):
            solve_image(np.ones((12,16)),header(),SolveOptions(backend='linear',require_sip=True))
        bad=header();bad['NAXIS1']=99
        register_backend('wrong-size',lambda *args:SolveResult(bad,'wrong-size'))
        with self.assertRaisesRegex(ValueError,'dimensions'):
            solve_image(np.ones((12,16)),header(),SolveOptions(backend='wrong-size'))

    def test_cancelled_request_does_not_start_backend(self):
        called=[];register_backend('never',lambda *args:called.append(True))
        cancel=threading.Event();cancel.set()
        with self.assertRaises(InterruptedError):
            solve_image(np.ones((12,16)),header(),SolveOptions(backend='never'),cancel=cancel)
        self.assertEqual(called,[])

    def skyshot_case(self,grid=None,status='solved',schema=1,require_sip=False,returncode=0):
        with tempfile.TemporaryDirectory() as folder:
            options=SolveOptions(backend='skyshot',executable=__file__,database=folder,require_sip=require_sip)
            def start(command,stdout,stderr):
                self.assertEqual(command[1],'solve')
                self.assertIn('--index',command);self.assertNotIn('-sip',command)
                Path(command[command.index('--wcs')+1]).write_text(header().tostring(sep='\n'))
                json.dump({'schema_version':schema,'solution':{'status':status,'width':16,'height':12,
                          'wcs':{'local_grid':grid},'warnings':[]}},stdout);stdout.flush()
                return SimpleNamespace(poll=lambda:returncode,returncode=returncode)
            with patch('astro_platesolve.skyshot.subprocess.Popen',side_effect=start):
                return solve_image(np.ones((12,16)),header(),options)

    def test_skyshot_schema_and_native_fits_contract(self):
        self.assertEqual(self.skyshot_case().backend,'skyshot')
        with self.assertRaisesRegex(ValueError,'local_grid'):self.skyshot_case(grid={'dx':[.2]})
        with self.assertRaisesRegex(ValueError,'schema-v1'):self.skyshot_case(schema=2)
        with self.assertRaisesRegex(ValueError,'schema-v1'):self.skyshot_case(status='ambiguous')
        with self.assertRaisesRegex(ValueError,'nonzero SIP'):self.skyshot_case(require_sip=True)
        with self.assertRaisesRegex(RuntimeError,'exit 1'):self.skyshot_case(returncode=1)

if __name__=='__main__':unittest.main()
