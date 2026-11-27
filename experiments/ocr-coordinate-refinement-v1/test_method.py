"""Contract/metamorphic tests; production inputs never include foreground truth."""
import copy
import unittest
import numpy as np
import mask_method as mm

def quad(x,y,w=6,h=10):return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
def segment(index,polys,status='possible'):
    return dict(segment_index=index,segment=dict(block_id='u'+str(index),start=0,end=len(polys),text='가'*len(polys)),
                status=status,reason=None if status=='possible' else 'unproved_transform',
                detail_polygons=polys if status=='possible' else [],coarse_polygons=[quad(1,1,35,12)],
                token_mapping=[dict(span=[i,i+1],text='가') for i in range(len(polys))])
def candidate(*parts):return dict(id='field1',type='NAME',segment_results=list(parts))

class MaskContract(unittest.TestCase):
    def test_internal_gap_is_filled(self):
        p=candidate(segment(0,[quad(2,4),quad(15,4)]))
        old,_=mm.refine.candidate_mask(p,(40,30),'detail_padded')
        new,_=mm.candidate_mask(p,(40,30))
        self.assertFalse(old[9,11]);self.assertTrue(new[9,11])
        self.assertFalse((old&~new).any())

    def test_separate_segments_do_not_bridge(self):
        p=candidate(segment(0,[quad(3,4)]),segment(1,[quad(50,4)]))
        m,_=mm.candidate_mask(p,(70,30),mm.METHODS[1])
        self.assertFalse(m[:,25:35].any())
        self.assertEqual([s['segment_index'] for s in mm.geometry(p,(70,30))],[0,1])

    def test_valid_coincident_token_centers_do_not_fail(self):
        p=candidate(segment(0,[quad(8,4),quad(8,4)]))
        for method in mm.METHODS:
            m,_=mm.candidate_mask(p,(40,30),method)
            self.assertTrue(m[8,10])
            self.assertEqual(mm.geometry(p,(40,30),method)[0]['direction_basis'],'coincident-centers first-token edge')

    def test_separate_lines_do_not_bridge(self):
        p=candidate(segment(0,[quad(3,3,8,10)]),segment(1,[quad(3,45,8,10)]))
        m,_=mm.candidate_mask(p,(40,70),mm.METHODS[1])
        self.assertFalse(m[25:35,:].any())

    def test_independent_candidates(self):
        a=candidate(segment(0,[quad(3,4)]));b=candidate(segment(0,[quad(30,4)]))
        x,_=mm.candidate_mask(a,(50,30),mm.METHODS[1]);y,_=mm.candidate_mask(b,(50,30),mm.METHODS[1])
        self.assertFalse((x&y).any())

    def test_boundary_clipped(self):
        p=candidate(segment(0,[quad(0,0,5,12)]))
        m,_=mm.candidate_mask(p,(10,14),mm.METHODS[1])
        self.assertEqual(m.shape,(14,10));self.assertTrue(m[0,0]);self.assertFalse(m[:,-1].any())

    def test_rotated_axis_caps_follow_line(self):
        # The token reading order runs vertically after a 90-degree transform.
        polys=[[[30-y,x+5] for x,y in quad(v,3,6,20)] for v in (2,14)]
        p=candidate(segment(0,polys));g=mm.geometry(p,(40,40),mm.METHODS[1])[0]
        self.assertAlmostEqual(g['reading_direction'][0],0)
        self.assertAlmostEqual(g['reading_direction'][1],1)
        m,_=mm.candidate_mask(p,(40,40),mm.METHODS[1])
        old,_=mm.refine.candidate_mask(p,(40,40),'detail_padded')
        self.assertFalse((old&~m).any());self.assertTrue(m[16,15])

    def test_skewed_quads_no_axis_aligned_bounding_rect(self):
        p=candidate(segment(0,[[[4,4],[10,8],[8,18],[2,14]],[[14,10],[20,14],[18,24],[12,20]]]))
        m,_=mm.candidate_mask(p,(30,32),mm.METHODS[0])
        self.assertFalse(m[4,19]);self.assertTrue(m[14,10])

    def test_perspective_preserves_original_polygons(self):
        p=candidate(segment(0,[[[4,4],[12,5],[11,18],[3,17]],[[14,5],[22,6],[20,19],[12,18]]]))
        before=copy.deepcopy(p);m,_=mm.candidate_mask(p,(32,30),mm.METHODS[1])
        old,_=mm.refine.candidate_mask(p,(32,30),'detail_padded')
        self.assertEqual(p,before);self.assertFalse((old&~m).any())

    def test_uncertainty_is_not_resolved(self):
        p=candidate(segment(0,[quad(5,4)],'uncertain'))
        m,subs=mm.candidate_mask(p,(50,30),mm.METHODS[1],False)
        self.assertFalse(m.any());self.assertEqual(subs,[])
        m,subs=mm.candidate_mask(p,(50,30),mm.METHODS[1],True)
        old,old_subs=mm.refine.candidate_mask(p,(50,30),'detail_padded',True)
        self.assertTrue(np.array_equal(m,old));self.assertEqual(subs,old_subs)

    def test_unavailable_and_mixed_segments(self):
        p=candidate(segment(0,[quad(5,4)]),segment(1,[],'unavailable'))
        p['segment_results'][1]['coarse_polygons']=[]
        m,subs=mm.candidate_mask(p,(50,30),mm.METHODS[1],True)
        self.assertTrue(m.any());self.assertFalse(subs[0]['available'])

    def test_bad_geometry_rejected(self):
        for bad in ([],[quad(-1,0)],[[[1,1],[2,2],[3,3],[4,4]]],[[[1,1],[3,3],[1,3],[3,1]]]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                mm.candidate_mask(candidate(segment(0,bad)),(40,30))

    def test_invalid_method_and_fallback_rejected(self):
        p=candidate(segment(0,[quad(4,4)]))
        with self.assertRaises(ValueError):mm.candidate_mask(p,(40,30),'detail_padded')
        with self.assertRaises(ValueError):mm.candidate_mask(p,(40,30),fallback='yes')

    def test_cap_is_bounded_and_height_scaled(self):
        for height,extra in ((10,0),(15,1),(25,3),(30,4),(90,4)):
            p=candidate(segment(0,[quad(20,2,10,height)]))
            g=mm.geometry(p,(60,100),mm.METHODS[1])[0]
            self.assertEqual(g['extra_tangent_pixels'],extra)

    def test_source_candidate_unicode_fields_untouched(self):
        p=candidate(segment(0,[quad(3,3),quad(15,3)]))
        p['id']='opaque-표시😀';p['segment_results'][0]['segment'].update(start=3,end=7,text='가😀 가')
        before=copy.deepcopy(p)
        mm.candidate_mask(p,(40,30),mm.METHODS[1])
        self.assertEqual(p,before)

if __name__=='__main__':unittest.main()
