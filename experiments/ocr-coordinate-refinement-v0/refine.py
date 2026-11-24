"""Prediction-only exact code-point mapping. No gold, ledger or mask inputs."""
import copy
import hashlib
import time
from functools import lru_cache
import numpy as np
from coord_common import identity_of, digest, polygon_valid, raster

METHODS = ('coarse','detail','detail_padded')
PADDING_PIXELS = 2

def align_tokens(text, tokens):
    """Unique ordered exact-token alignment; only original whitespace may be skipped.

    Complete sequence consumption makes repeated text occurrence-safe. Tokens can
    contain multiple code points. No normalization, correction, search or zip.
    At most two distinct complete alignments are retained to detect ambiguity.
    """
    if not isinstance(text,str) or any(not isinstance(t,str) or not t for t in tokens):
        return {'status':'unavailable','reason':'invalid_or_empty_token','spans':[]}
    @lru_cache(None)
    def visit(i,j):
        if j == len(tokens):
            return ((),) if all(c.isspace() for c in text[i:]) else ()
        token = tokens[j]
        answers=[]
        k=i
        while True:
            if text.startswith(token,k):
                for tail in visit(k+len(token),j+1):
                    path=((k,k+len(token)),)+tail
                    if path not in answers: answers.append(path)
                    if len(answers)==2:return tuple(answers)
            if k >= len(text) or not text[k].isspace():break
            k+=1
        return tuple(answers)
    paths = visit(0,0)
    if len(paths) != 1:
        return {'status':'uncertain' if paths else 'unavailable',
                'reason':'ambiguous_whitespace_alignment' if paths else 'token_sequence_mismatch','spans':[]}
    return {'status':'possible','reason':None,'spans':[list(p) for p in paths[0]],
            'rule':'unique full ordered exact alignment; whitespace-only omissions'}

def validate_binding(rgb, ocr, identity, details):
    if (identity['image_id'] != ocr['image_id'] or identity['ocr_run_id'] != ocr['run_id']
        or identity['ocr_sha256'] != digest(ocr) or identity['file_sha256'] != ocr['input']['sha256']
        or identity['width'] != rgb.width or identity['height'] != rgb.height
        or identity['rgb_sha256'] != hashlib.sha256(rgb.tobytes()).hexdigest()
        or details['identity'] != identity):
        raise ValueError('Image/run/RGB/OCR/detail identity mismatch')
    ids = [u['id'] for u in ocr['units']]
    if len(set(ids)) != len(ids):raise ValueError('Duplicate source unit ID')
    detail_ids=[u['unit_id'] for u in details['units']]
    if len(set(detail_ids)) != len(detail_ids) or set(detail_ids)-set(ids):
        raise ValueError('Unknown or duplicate detail unit ID')

def refine_candidates(rgb, ocr, identity, candidates, details):
    """Return immutable candidate identity/segments and per-segment detail/coarse areas.

    Candidates only need id/type/segments. Extra original metadata is copied to
    original_candidate. Caller must bind them to this identity (not a global text).
    Identity is explicitly supplied and verified against source pixels and OCR.
    """
    started=time.perf_counter()
    identity=identity_of(identity)
    validate_binding(rgb,ocr,identity,details)
    units={u['id']:u for u in ocr['units']}
    children={u['unit_id']:u for u in details['units']}
    output=[]; seen=set()
    for candidate in candidates:
        if candidate['id'] in seen or not candidate['segments'] or candidate['type'] not in {'NAME','PHONE','EMAIL','ACCOUNT_ID'}:
            raise ValueError('Invalid candidate identity/type/segments')
        seen.add(candidate['id']); segments=[]
        for si,s in enumerate(candidate['segments']):
            u=units[s['block_id']]; a,b=s['start'],s['end']
            if type(a) is not int or type(b) is not int or not 0<=a<b<=len(u['text']) or u['text'][a:b]!=s['text']:
                raise ValueError('Segment must be an exact original code-point slice')
            coarse=copy.deepcopy(u['polygons']) if u['location_status']=='valid' else []
            if any(not polygon_valid(p,rgb.width,rgb.height) for p in coarse):
                raise ValueError('Invalid source polygon')
            child=children.get(u['id']); polys=[]; selected=[]; alignment=None
            status='unavailable'; reason='missing_detail_unit'
            if details['status'] in ('failure','indeterminate'):
                reason='detail_extraction_'+details['status']
            elif child:
                if child.get('text') != u['text'] or child.get('polygons') != u['polygons']:
                    status='uncertain'; reason='detail_parent_mismatch'
                elif child.get('status') not in ('ok','possible'):
                    status='uncertain' if child.get('status')=='uncertain' else 'unavailable'
                    reason='detail_unit_'+child.get('status','unknown')
                else:
                    tokens=child['tokens']
                    alignment=align_tokens(u['text'],[t['text'] for t in tokens])
                    status=alignment['status']; reason=alignment['reason']
                    if status=='possible':
                        for ti,(x,y) in enumerate(alignment['spans']):
                            if x>=b or y<=a:continue
                            if not a<=x<y<=b:
                                status='uncertain'; reason='indivisible_token_crosses_candidate_boundary';break
                            token=tokens[ti]; poly=token.get('polygon')
                            if not polygon_valid(poly,rgb.width,rgb.height):
                                status='unavailable'; reason='invalid_or_missing_token_polygon';break
                            selected.append(dict(token_index=ti,span=[x,y],text=u['text'][x:y],
                                                 score=token.get('score'),score_status=token.get('score_status')))
                            polys.append(copy.deepcopy(poly))
                        if status=='possible' and not polys:
                            status='unavailable'; reason='no_nonspace_token_for_segment'
            if status!='possible':polys=[]
            segments.append(dict(segment_index=si,segment=copy.deepcopy(s),unit_id=u['id'],
                status=status,reason=reason,detail_reasons=copy.deepcopy(child.get('reasons',[])) if child else [],
                detail_polygons=polys,coarse_polygons=coarse,
                token_mapping=selected,full_unit_alignment=alignment,
                coordinate_source='RapidOCR CTC estimate restored to original pixel boundaries',
                glyph_coverage='unproven; status describes correspondence only'))
        statuses=[s['status'] for s in segments]
        status='possible' if all(s=='possible' for s in statuses) else 'uncertain' if 'uncertain' in statuses or 'possible' in statuses else 'unavailable'
        output.append(dict(id=candidate['id'],type=candidate['type'],segments=copy.deepcopy(candidate['segments']),
            original_candidate=copy.deepcopy(candidate),status=status,segment_results=segments))
    return dict(schema='ocr-coordinate-refinement-v0',identity=identity,source_status=ocr['status'],
        detail_status=details['status'],state='normal_zero_candidates' if not candidates and ocr['status']=='ok' and details['status']=='ok' else 'candidates' if candidates else 'incomplete',
        candidates=output,methods=list(METHODS),
        settings=dict(detail_padded=dict(operation='binary raster Chebyshev dilation',
                      radius_original_pixels=PADDING_PIXELS,border='clip to original frame',
                      scope='each supported segment; no expansion of coarse fallback')),
        mapping_seconds=time.perf_counter()-started)

def candidate_mask(candidate, size, method='detail', fallback=False):
    if method not in METHODS:raise ValueError('Unsupported method')
    mask=np.zeros(size[::-1],bool); substituted=[]
    for s in candidate['segment_results']:
        use_coarse=method=='coarse' or (fallback and s['status']!='possible')
        polygons=s['coarse_polygons'] if use_coarse else s['detail_polygons']
        part=raster(polygons,size)
        if method=='detail_padded' and not use_coarse:
            import cv2
            part=cv2.dilate(part.astype(np.uint8),np.ones((2*PADDING_PIXELS+1,2*PADDING_PIXELS+1),np.uint8),
                            borderType=cv2.BORDER_CONSTANT,borderValue=0)!=0
        mask |= part
        if method!='coarse' and use_coarse:
            substituted.append(dict(segment_index=s['segment_index'],reason=s['reason'],available=bool(polygons)))
    return mask,substituted
