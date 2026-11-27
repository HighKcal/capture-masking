"""OCR-COORD02: per-segment convex envelope, optionally bounded end caps.

Only verified original-coordinate token polygons enter this module. No RGB,
truth, text recognition, case identifiers or candidate-boundary edits are used.
"""
import hashlib
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.dont_write_bytecode = True
_UPSTREAM = Path(__file__).resolve().parent.parent / 'ocr-integration-v0'
if str(_UPSTREAM) not in sys.path:sys.path.insert(0,str(_UPSTREAM))
from dependencies import refine, support

METHODS = ('segment_hull_2px_v1', 'segment_hull_caps_v1')
VERSION = 'ocr-coord02-mask-1'
PADDING = 2

def settings(method):
    if method not in METHODS:
        raise ValueError('Unsupported COORD02 method')
    return dict(method=method, version=VERSION,
                polygon_operation='convex hull of tokens within each possible original segment',
                cap_fraction_of_median_token_height=0.2 if method == METHODS[1] else 0.0,
                cap_total_min_pixels=2, cap_total_max_pixels=6 if method == METHODS[1] else 2,
                cap_formula='extra=max(0,min(6,max(2,ceil(0.2*h)))-2)' if method == METHODS[1] else 'extra=0',
                cap_direction='first-to-last token center; first token edge when single or coincident centers',
                raster='convex half planes at pixel centers, boundary inclusive at 1e-9',
                raster_dilation=dict(metric='Chebyshev', radius_original_pixels=PADDING, border='clip'),
                scope='each original segment separately; never bridge separate segments or candidates',
                fallback='unchanged: non-possible segment uses undilated original coarse only when requested',
                glyph_coverage='unproven; correspondence status is preserved')

def method_source():
    path = Path(__file__).resolve()
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}

def convex_hull(points):
    """Deterministic monotonic-chain hull in original double-precision coordinates."""
    points = sorted(set(tuple(map(float, p)) for p in points))
    if len(points) < 3:
        raise ValueError('Polygon has no area')
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    result=lower[:-1]+upper[:-1]
    if len(result)<3:raise ValueError('Degenerate polygon')
    return [list(p) for p in result]

def raster_convex(polygon, size):
    """Raster polygon boundaries, not integer vertex fill; clip only to input frame."""
    out=np.zeros(size[::-1],bool)
    p=np.asarray(polygon,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():
        raise ValueError('Invalid convex polygon')
    x0,y0=np.maximum(np.floor(p.min(0)).astype(int),0)
    x1,y1=np.minimum(np.ceil(p.max(0)).astype(int),size)
    if x0>=x1 or y0>=y1:return out
    y,x=np.mgrid[y0:y1,x0:x1];x=x+0.5;y=y+0.5
    positive=np.ones(x.shape,bool);negative=positive.copy()
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        cross=(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])
        positive&=cross>=-1e-9;negative&=cross<=1e-9
    out[y0:y1,x0:x1]=positive|negative
    return out

def _quads(polygons,size):
    if not polygons or any(not refine.polygon_valid(p,*size) for p in polygons):
        raise ValueError('Possible segment must contain valid original-coordinate token quads')
    return np.asarray(polygons,float)

def geometry(candidate,size,method=METHODS[0],fallback=False):
    settings(method)
    if type(fallback) is not bool or len(size)!=2 or any(type(n)is not int or n<=0 for n in size):
        raise ValueError('Expected bool fallback and positive integer original dimensions')
    rows=[]
    for s in candidate['segment_results']:
        row=dict(segment_index=s['segment_index'],segment=s['segment'],status=s['status'],reason=s['reason'],
                 coordinate_source='saved verified original-coordinate CTC token quads',
                 source_polygons=s['detail_polygons'],coarse_polygons=s['coarse_polygons'],
                 token_mapping=s['token_mapping'],used_fallback=fallback and s['status']!='possible',
                 polygons=[],raster_dilation_pixels=0)
        if s['status']!='possible':
            if row['used_fallback']:row['polygons']=s['coarse_polygons']
            rows.append(row);continue
        q=_quads(s['detail_polygons'],size)
        direction=(q[-1].mean(0)-q[0].mean(0)) if len(q)>1 else q[0,1]-q[0,0]
        length=float(np.linalg.norm(direction))
        direction_basis='first-to-last token center' if len(q)>1 else 'single-token first edge'
        if length<=1e-12:
            # Valid CTC token quads may overlap completely despite distinct spans.
            direction=q[0,1]-q[0,0]
            length=float(np.linalg.norm(direction))
            direction_basis='coincident-centers first-token edge'
        if length<=1e-12:raise ValueError('Invalid token edge')
        direction/=length
        # The native token vertical edges span the recognition crop height; their
        # mean lengths retain original pixel scale under accepted transformations.
        height=float(np.median((np.linalg.norm(q[:,3]-q[:,0],axis=1)+np.linalg.norm(q[:,2]-q[:,1],axis=1))/2))
        total=min(6,max(2,math.ceil(0.2*height))) if method==METHODS[1] else 2
        extra=total-2
        points=q.reshape(-1,2)
        base=convex_hull(points)
        expanded=convex_hull(np.concatenate([points-extra*direction,points+extra*direction]))
        row.update(polygons=[expanded],base_hull=base,reading_direction=direction.tolist(),direction_basis=direction_basis,
                   median_token_height=height,extra_tangent_pixels=extra,
                   cap_total_nominal_pixels=total,raster_dilation_pixels=PADDING,
                   polygon_meaning='continuous envelope with tangent cap before raster dilation')
        rows.append(row)
    return rows

def candidate_mask(candidate,size,method=METHODS[0],fallback=False):
    mask=np.zeros(size[::-1],bool);substitutions=[]
    for row in geometry(candidate,size,method,fallback):
        if row['used_fallback']:
            part=support.raster(row['polygons'],size)
            substitutions.append(dict(segment_index=row['segment_index'],reason=row['reason'],available=bool(row['polygons'])))
        elif row['status']=='possible':
            part=raster_convex(row['polygons'][0],size)
            part=cv2.dilate(part.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)!=0
        else:
            part=np.zeros(size[::-1],bool)
        mask|=part
    return mask,substitutions
