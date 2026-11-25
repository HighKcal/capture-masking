import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from coord_common import HERE,digest,save_masked
from refine import align_tokens,refine_candidates,candidate_mask
from evaluate import metrics,verify_pixels,load_mask

def fixture(text='김민수와 이영희🙂'):
    rgb=Image.new('RGB',(120,30),(245,240,220))
    box=[[0,0],[110,0],[110,25],[0,25]]
    u=dict(id='r:i:u0000',text=text,polygons=[box],location_status='valid')
    ocr=dict(image_id='i',run_id='r',input=dict(sha256='file'),units=[u],status='ok')
    identity=dict(image_id='i',ocr_run_id='r',file_sha256='file',rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),width=120,height=30,ocr_sha256=digest(ocr))
    tokens=[]
    for i,ch in enumerate(text):
        if ch.isspace():continue
        x=3+10*i
        tokens.append(dict(text=ch,polygon=[[x,4],[x+8,4],[x+8,20],[x,20]],score=.9,score_status='recognition_not_location'))
    details=dict(identity=identity,status='ok',units=[dict(unit_id=u['id'],text=text,polygons=[box],status='ok',tokens=tokens)])
    def candidate(pid,a,b):return dict(id=pid,type='NAME',segments=[dict(block_id=u['id'],start=a,end=b,text=text[a:b])])
    return rgb,ocr,identity,details,candidate

class AlignmentTests(unittest.TestCase):
    def test_spaces_repeated_emoji(self):
        r=align_tokens(' 🙂 김김  A1@ 김',['🙂','김','김','A','1','@','김'])
        self.assertEqual(r['spans'],[[1,2],[3,4],[4,5],[7,8],[8,9],[9,10],[11,12]])
    def test_no_normalization(self):
        self.assertEqual(align_tokens('e\u0301',['é'])['status'],'unavailable')
        self.assertEqual(align_tokens('Ａ',['A'])['status'],'unavailable')
    def test_no_global_repeat_search(self):
        self.assertEqual(align_tokens('김 김',['김'])['status'],'unavailable')
    def test_ambiguous_whitespace(self):
        self.assertEqual(align_tokens('  ',[' '])['status'],'uncertain')
    def test_multicodepoint(self):
        self.assertEqual(align_tokens('가 AB 나',['가','AB','나'])['spans'],[[0,1],[2,4],[5,6]])
    def test_empty_and_invalid(self):
        self.assertEqual(align_tokens(' ',[])['spans'],[])
        self.assertEqual(align_tokens('가',[''])['status'],'unavailable')
        self.assertEqual(align_tokens('가',[])['status'],'unavailable')

class RefinementTests(unittest.TestCase):
    def run_fixture(self,change=None,span=(0,3)):
        rgb,ocr,ident,detail,make=fixture()
        if change:change(detail)
        return rgb,refine_candidates(rgb,ocr,ident,[make('a',*span)],detail)
    def test_identity_rejects_each_mixing(self):
        for field in ('image_id','ocr_run_id','file_sha256','rgb_sha256','ocr_sha256','width','height'):
            rgb,ocr,identity,details,make=fixture(); identity=copy.deepcopy(identity)
            identity[field]='other' if isinstance(identity[field],str) else identity[field]+1
            with self.assertRaises(ValueError):refine_candidates(rgb,ocr,identity,[make('a',0,3)],details)
    def test_detail_bound_to_run(self):
        rgb,ocr,identity,details,make=fixture();details=copy.deepcopy(details);details['identity']['ocr_run_id']='other'
        with self.assertRaises(ValueError):refine_candidates(rgb,ocr,identity,[make('a',0,3)],details)
    def test_candidate_slice_not_repaired(self):
        rgb,ocr,identity,details,make=fixture();p=make('a',0,3);p['segments'][0]['text']='다른값'
        with self.assertRaises(ValueError):refine_candidates(rgb,ocr,identity,[p],details)
    def test_indivisible_boundary(self):
        def change(d):
            d['units'][0]['tokens'][0]['text']='김민'
            del d['units'][0]['tokens'][1]
        _,r=self.run_fixture(change,span=(1,3))
        self.assertEqual(r['candidates'][0]['status'],'uncertain')
        self.assertEqual(r['candidates'][0]['segment_results'][0]['detail_polygons'],[])
    def test_invalid_polygon(self):
        def change(d):d['units'][0]['tokens'][0]['polygon']=[[0,0],[20,20],[20,0],[0,20]]
        _,r=self.run_fixture(change)
        self.assertEqual(r['candidates'][0]['status'],'unavailable')
    def test_parent_mismatch(self):
        def change(d):d['units'][0]['text']='changed'
        _,r=self.run_fixture(change)
        self.assertEqual(r['candidates'][0]['status'],'uncertain')
    def test_missing_location_and_fallback_separate(self):
        rgb,r=self.run_fixture(lambda d:d['units'].clear())
        p=r['candidates'][0]
        self.assertFalse(candidate_mask(p,rgb.size,'detail')[0].any())
        coarse,_=candidate_mask(p,rgb.size,'coarse')
        fallback,subs=candidate_mask(p,rgb.size,'detail',True)
        self.assertTrue(np.array_equal(coarse,fallback));self.assertEqual(len(subs),1)
        self.assertEqual(p['status'],'unavailable')
    def test_same_parent_four_selections(self):
        rgb,ocr,ident,d,make=fixture();r=refine_candidates(rgb,ocr,ident,[make('a',0,3),make('b',5,8)],d)
        a,_=candidate_mask(r['candidates'][0],rgb.size);b,_=candidate_mask(r['candidates'][1],rgb.size)
        self.assertFalse((a&b).any());self.assertTrue(a.any() and b.any())
        both=a|b;none=np.zeros_like(a)
        self.assertEqual(both.sum(),a.sum()+b.sum());self.assertFalse(none.any())
    def test_multisegment_does_not_fill_gap(self):
        rgb,ocr,ident,d,make=fixture();p=make('a',0,3);p['segments']+=make('b',5,8)['segments']
        r=refine_candidates(rgb,ocr,ident,[p],d);m,_=candidate_mask(r['candidates'][0],rgb.size)
        self.assertFalse(m[:,35:50].any());self.assertEqual(len(r['candidates'][0]['segment_results']),2)
    def test_multisegment_other_parent(self):
        rgb,ocr,ident,d,make=fixture();u=copy.deepcopy(ocr['units'][0]);u['id']='r:i:u0001';ocr['units'].append(u)
        child=copy.deepcopy(d['units'][0]);child['unit_id']=u['id'];d['units'].append(child)
        ident['ocr_sha256']=digest(ocr)
        p=make('a',0,3);s=make('b',5,8)['segments'][0];s['block_id']=u['id'];p['segments'].append(s)
        r=refine_candidates(rgb,ocr,ident,[p],d)
        self.assertEqual([s['unit_id'] for s in r['candidates'][0]['segment_results']],['r:i:u0000','r:i:u0001'])
    def test_unchanged_inputs_and_no_truth_access(self):
        rgb,ocr,ident,d,make=fixture();p=[make('a',0,3)];before=copy.deepcopy((ocr,ident,d,p))
        with patch('builtins.open',side_effect=AssertionError('No algorithm filesystem reads')):
            r=refine_candidates(rgb,ocr,ident,p,d)
        self.assertEqual(before,(ocr,ident,d,p));self.assertEqual(r['candidates'][0]['segments'],p[0]['segments'])
    def test_zero_is_distinct_from_failure(self):
        rgb,ocr,ident,d,make=fixture();r=refine_candidates(rgb,ocr,ident,[],d)
        self.assertEqual(r['state'],'normal_zero_candidates')
        d['status']='failure';r=refine_candidates(rgb,ocr,ident,[],d)
        self.assertEqual(r['state'],'incomplete')
    def test_fixed_padding_and_no_coarse_expansion(self):
        rgb,r=self.run_fixture();p=r['candidates'][0]
        raw,_=candidate_mask(p,rgb.size,'detail');pad,_=candidate_mask(p,rgb.size,'detail_padded')
        self.assertTrue(np.all(pad[raw]));self.assertGreater(pad.sum(),raw.sum())
        self.assertFalse(pad[:2].any());self.assertTrue(pad[2].any())
        rgb,r=self.run_fixture(lambda d:d['units'].clear());p=r['candidates'][0]
        self.assertTrue(np.array_equal(candidate_mask(p,rgb.size,'coarse')[0],candidate_mask(p,rgb.size,'detail_padded',True)[0]))
    def test_png_round_trip_and_solo_metrics(self):
        rgb,r=self.run_fixture();mask,_=candidate_mask(r['candidates'][0],rgb.size)
        base=HERE/'test-artifacts';base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            dest=Path(temp);save_masked(rgb,mask,dest)
            self.assertTrue(np.array_equal(load_mask(dest/'mask.png',rgb.size),mask))
            verify_pixels(rgb,mask,dest/'masked.png')
            target=mask.copy();target[0,0]=True
            self.assertEqual(metrics(mask,target,target)['missed_target_pixels'],1)
            bad=np.array(rgb);bad[0,0]=0;Image.fromarray(bad).save(dest/'bad.png')
            with self.assertRaises(ValueError):verify_pixels(rgb,mask,dest/'bad.png')

if __name__=='__main__':unittest.main()
