"""Bounded, conservative field relationships. No gold, case-name or NER access."""
import copy
import math
import re
import time
import numpy as np
from bridge import digest,polygon_valid,improved_contacts

CONFIG=dict(version='ocr-field-relations-v1',max_units=300,max_paths=256,max_chain=3,max_branches=4,
    max_angle_deg=8.,height_ratio_min=.55,height_ratio_max=1.8,
    vertical_left_tolerance=.65,vertical_gap=1.25,horizontal_gap=1.7,
    label_horizontal_gap=5.,label_vertical_gap=1.6,label_left_tolerance=.8,
    same_row_center=.45,boundary_contrast=45,boundary_fraction=.72)
ACCOUNT_LABEL=re.compile(r'^\s*(?P<label>계정\s*아이디|계정\s*ID|사용자\s*ID|로그인\s*ID|아이디|계정)\s*(?:[:：=＝]|\t)?\s*$',re.I)
INLINE_ACCOUNT=re.compile(r'^\s*(?P<label>계정\s*아이디|계정\s*ID|사용자\s*ID|로그인\s*ID|아이디|계정)(?:[ \t]*[:：=＝][ \t]*|[ \t]+)(?P<value>@?\w+(?:[.\-]\w+)*)[ \t]*$',re.I)
ACCOUNT_VALUE=re.compile(r'@?\w+(?:[.\-]\w+)*')
PHONE_START=re.compile(r'(?<![\w+])(?:01[016789]|02|0(?:3[1-3]|4[1-4]|5[1-5]|6[1-4]|70)|\+82[- ]?(?:10|1[16789]|2|3[1-3]|4[1-4]|5[1-5]|6[1-4]|70))[0-9 \t-]*$')
PHONE_PART=re.compile(r'[0-9]+(?:[ \t-]*[0-9]+)*[ \t-]*')
MAIL_TOKEN=re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~@-]+")
MAIL_SUFFIX=re.compile(r"(?<![\w])([A-Za-z0-9.!#$%&'*+/=?^_`{|}~@-]+)[ \t]*$")
CONTACT_LABEL=re.compile(r'^\s*(전화(?:번호)?|연락처|연락|휴대(?:전화|폰)?|팩스|이메일|전자\s*우편|email|e-mail)(?:\s*[:：=＝]?\s*|\s*[（(][^）)]*[）)]\s*)$',re.I)
NEGATIVE_LABEL=re.compile(r'(주문|파일|제품|상품|재고|규격|승인|지점|참조|송장|운송장|모델|로트|자재|묶음|수량|품번|사번|고객\s*번호|코드)',re.I)
FIELDISH=re.compile(r'[:：=＝]|(?:이름|성명|담당자|연락처|전화|이메일|아이디|\bID\b|코드|번호|재고|규격|승인|파일|제품)',re.I)

def signature(p):
    return p['type'],tuple((s['block_id'],s['start'],s['end'],s['text']) for s in p['segments'])

def refs(segments):
    return [(s['block_id'],i) for s in segments for i in range(s['start'],s['end'])]

def span(u,a=0,z=None):
    if z is None:z=len(u['text'])
    if type(a) is not int or type(z) is not int or not 0<=a<z<=len(u['text']):raise ValueError('Invalid source span')
    return dict(block_id=u['id'],start=a,end=z,text=u['text'][a:z])

def trimmed_span(u):
    text=u['text'];a=len(text)-len(text.lstrip(' \t'));z=len(text.rstrip(' \t'))
    return span(u,a,z) if a<z else None

def geometry(u):
    coords=[p for poly in u['polygons'] for p in poly]
    x=[p[0] for p in coords];y=[p[1] for p in coords]
    box=[min(x),min(y),max(x),max(y)]
    angles=[]
    for poly in u['polygons']:
        a,b=poly[0],poly[1];angles.append(abs(math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))))
    return dict(box=box,h=box[3]-box[1],cy=(box[1]+box[3])/2,angle=max(angles))

def orient(a,b,label=False):
    ga,gb=a['_g'],b['_g'];h=max(ga['h'],gb['h']);ratio=gb['h']/ga['h']
    if not CONFIG['height_ratio_min']<=ratio<=CONFIG['height_ratio_max']:return None,'incompatible_line_heights',{}
    al,at,ar,ab=ga['box'];bl,bt,br,bb=gb['box']
    if abs(ga['cy']-gb['cy'])<=CONFIG['same_row_center']*min(ga['h'],gb['h']) and bl>=ar-.15*h:
        gap=(bl-ar)/h;limit=CONFIG['label_horizontal_gap'] if label else CONFIG['horizontal_gap']
        return ('right' if gap<=limit else None),'horizontal_gap',dict(gap_heights=gap,height_ratio=ratio)
    gap=(bt-ab)/h;left=abs(bl-al)/h
    if gb['cy']>ga['cy']+.55*h and -.15<=gap<=(CONFIG['label_vertical_gap'] if label else CONFIG['vertical_gap']) and left<=(CONFIG['label_left_tolerance'] if label else CONFIG['vertical_left_tolerance']):
        return 'below','aligned_next_line',dict(gap_heights=gap,left_delta_heights=left,height_ratio=ratio)
    return None,'different_column_or_distant_row',dict(gap_heights=gap,left_delta_heights=left,height_ratio=ratio)

def blocking(a,b,units,direction,rgb):
    al,at,ar,ab=a['_g']['box'];bl,bt,br,bb=b['_g']['box'];h=max(a['_g']['h'],b['_g']['h'])
    for u in units:
        if u['id'] in (a['id'],b['id']):continue
        l,t,r,bot=u['_g']['box']
        if direction=='below':
            if ab-.1*h<t<bt and bot<=bt+.1*h and min(ar,br,r)>max(al,bl,l):return 'intervening_ocr_unit'
        elif ar<l<bl and r<=bl+.1*h and min(ab,bb,bot)>max(at,bt,t):return 'intervening_ocr_unit'
    if rgb is None:return None
    image=np.asarray(rgb);height,width=image.shape[:2]
    # Sample background just outside text crops; compare independent panels.
    def background(u):
        l,t,r,bot=u['_g']['box'];hh=u['_g']['h']
        xx=max(0,min(width-1,int(l-.25*hh)));y0=max(0,int(t));y1=min(height,int(bot)+1)
        return np.median(image[y0:y1,xx,:3],axis=0)
    bg1,bg2=background(a),background(b)
    if np.max(np.abs(bg1-bg2))>CONFIG['boundary_contrast']:return 'different_background_panels'
    bg=(bg1+bg2)/2
    if direction=='below' and bt-ab>=2:
        x0=max(0,int(min(al,bl)));x1=min(width,int(max(ar,br)))
        y0=max(0,int(math.ceil(ab)));y1=min(height,int(math.floor(bt)))
        if x1>x0 and y1>y0:
            contrast=np.max(np.abs(image[y0:y1,x0:x1,:3].astype(float)-bg),axis=2)>CONFIG['boundary_contrast']
            if np.any(contrast.mean(axis=1)>=CONFIG['boundary_fraction']):return 'horizontal_visual_boundary'
    if direction=='right' and bl-ar>=2:
        x0=max(0,int(math.ceil(ar)));x1=min(width,int(math.floor(bl)))
        y0=max(0,int(max(at,bt)));y1=min(height,int(min(ab,bb)))
        if x1>x0 and y1>y0:
            contrast=np.max(np.abs(image[y0:y1,x0:x1,:3].astype(float)-bg),axis=2)>CONFIG['boundary_contrast']
            if np.any(contrast.mean(axis=0)>=CONFIG['boundary_fraction']):return 'vertical_visual_boundary'
    return None

def owner_label(u,units):
    possible=[]
    for label in units:
        if label['id']==u['id'] or not FIELDISH.search(label['text']):continue
        d,reason,e=orient(label,u,label=True)
        if d and not blocking(label,u,units,d,None):
            distance=abs(u['_g']['cy']-label['_g']['cy'])+max(0,u['_g']['box'][0]-label['_g']['box'][2])
            possible.append((distance,label))
    possible.sort(key=lambda p:(p[0],p[1]['id']))
    if not possible:return None,[]
    # A distance does not choose between contact/noncontact owners.
    nearby=[p[1] for p in possible if p[0]<=possible[0][0]+u['_g']['h']]
    roles={bool(NEGATIVE_LABEL.search(p['text'])) for p in nearby}
    return (None,nearby) if len(roles)>1 else (possible[0][1],[])

def project_source(ocr):
    required={'image_id','run_id','input','units','status','schema_version'}
    if not required<=ocr.keys() or ocr['schema_version']!='ocr-output-v0':raise ValueError('OCR schema mismatch')
    if {'gold','occurrences','expected','gold_relations'}&ocr.keys():raise ValueError('Gold/expected data prohibited in relation prediction')
    w,h=ocr['input']['width'],ocr['input']['height']
    if type(w) is not int or type(h) is not int or w<=0 or h<=0:raise ValueError('Invalid original size')
    if len(ocr['units'])>CONFIG['max_units']:raise ValueError('Relation input unit limit exceeded')
    seen=set();good=[];skipped=[]
    prefix=f"{ocr['run_id']}:{ocr['image_id']}:u"
    for u in ocr['units']:
        if {'gold','expected','field_id','gold_type'}&u.keys():raise ValueError('Gold unit metadata prohibited')
        if u['id'] in seen or not u['id'].startswith(prefix) or not isinstance(u['text'],str):raise ValueError('Mixed/duplicate OCR unit')
        seen.add(u['id'])
        if u['location_status']!='valid' or not u['polygons']:
            skipped.append(dict(unit_id=u['id'],reason='location_unavailable'));continue
        if not all(polygon_valid(p,w,h) for p in u['polygons']):raise ValueError('Invalid original polygon')
        p=dict(id=u['id'],text=u['text'],polygons=copy.deepcopy(u['polygons']))
        p['_g']=geometry(p)
        if p['_g']['angle']>CONFIG['max_angle_deg']:
            skipped.append(dict(unit_id=u['id'],reason='unsupported_rotation'));continue
        good.append(p)
    return sorted(good,key=lambda u:(u['_g']['box'][1],u['_g']['box'][0],u['id'])),skipped

def make_view(value_segments,label_segments,kind,units):
    byid={u['id']:u for u in units}
    text='';mapping=[]
    if kind=='label_value':
        # Semantic label recognized separately. Canonical grammar header is virtual.
        text='아이디: ';mapping=[None]*len(text)
    for i,s in enumerate(value_segments):
        u=byid[s['block_id']]
        if u['text'][s['start']:s['end']]!=s['text']:raise ValueError('Source exact slice mismatch')
        if kind=='value_continuation' and i==0:
            # Preserve first-unit original context, including exclusions before value.
            text+=u['text'][:s['start']]
            mapping.extend([dict(unit_id=u['id'],offset=j) for j in range(s['start'])])
        text+=s['text'];mapping.extend(dict(unit_id=u['id'],offset=j) for j in range(s['start'],s['end']))
    return text,mapping

def map_virtual(segments,mapping,units):
    byid={u['id']:u for u in units};groups=[]
    for s in segments:
        for i in range(s['start'],s['end']):
            ref=mapping[i]
            if ref is None:raise ValueError('Candidate contains virtual grammar characters')
            bid,a=ref['unit_id'],ref['offset']
            if groups and groups[-1][0]==bid and groups[-1][2]==a:groups[-1][2]=a+1
            else:groups.append([bid,a,a+1])
    return [span(byid[bid],a,z) for bid,a,z in groups]

def consume(kind,typ,values,labels,units):
    text,mapping=make_view(values,labels,kind,units)
    candidates,audit=improved_contacts(dict(id='field-view',blocks=[dict(id='virtual',text=text)],links=[]))
    wanted=[]
    expected=refs(values)
    for p in candidates:
        if p['type']!=typ:continue
        segments=map_virtual(p['segments'],mapping,units)
        if refs(segments)==expected:
            wanted.append(dict(p,segments=segments,sources=p['sources']+['field_relation_v1']))
    return wanted,dict(text=text,mapping=mapping,contact_audit=audit,
                      synthetic_character_count=sum(r is None for r in mapping))

def infer_relations(ocr,rgb=None):
    started=time.perf_counter();units,skipped=project_source(ocr)
    if rgb is not None and (rgb.size if hasattr(rgb,'size') and not isinstance(rgb,np.ndarray) else (rgb.shape[1],rgb.shape[0]))!=(ocr['input']['width'],ocr['input']['height']):
        raise ValueError('RGB dimensions differ from original')
    relations=[];trials=0
    def record(kind,typ,values,labels,status,reason,evidence=None,proposed=None,view=None):
        identity=dict(image_id=ocr['image_id'],ocr_run_id=ocr['run_id'],kind=kind,type=typ,value_segments=values,label_segments=labels)
        rid='r-'+digest(identity)[:20]
        rel=dict(identity,id=rid,status=status,reason=reason,evidence=evidence or {},candidate_ids=[],
                 proposed_candidates=proposed or [],view=view)
        if not any(r['id']==rid for r in relations):relations.append(rel)
    # Label/value semantic relation is never a value concatenation edge.
    for label in units:
        standalone=ACCOUNT_LABEL.fullmatch(label['text'])
        inline=None if standalone else INLINE_ACCOUNT.fullmatch(label['text'])
        if inline:
            values=[span(label,*inline.span('value'))];labels=[span(label,*inline.span('label'))]
            proposed,view=consume('label_value','ACCOUNT_ID',values,labels,units)
            record('label_value','ACCOUNT_ID',values,labels,'accepted' if proposed else 'rejected','inline_explicit_account_label',
                dict(label_in_value_fragment=True,region_includes_label=True),proposed,view)
            continue
        match=standalone
        if not match:continue
        for value in units:
            if value['id']==label['id']:continue
            s=trimmed_span(value)
            if not s or not ACCOUNT_VALUE.fullmatch(s['text']):continue
            direction,reason,evidence=orient(label,value,label=True)
            if not direction:continue
            why=blocking(label,value,units,direction,rgb)
            labels=[span(label,*match.span('label'))]
            if why:
                record('label_value','ACCOUNT_ID',[s],labels,'rejected',why,evidence);continue
            value_owner,value_conflicts=owner_label(value,units)
            if value_conflicts or (value_owner and value_owner['id']!=label['id'] and not ACCOUNT_LABEL.fullmatch(value_owner['text'])):
                record('label_value','ACCOUNT_ID',[s],labels,'deferred','competing_other_field_owner',
                       dict(evidence,owner_ids=[u['id'] for u in value_conflicts] or [value_owner['id']]));continue
            # Distinct label/value quads must not overlap, even at crop margins.
            l,t,r,b=label['_g']['box'];vl,vt,vr,vb=value['_g']['box']
            if min(r,vr)>max(l,vl) and min(b,vb)>max(t,vt):
                record('label_value','ACCOUNT_ID',[s],labels,'deferred','label_value_regions_overlap',evidence);continue
            proposed,view=consume('label_value','ACCOUNT_ID',[s],labels,units)
            record('label_value','ACCOUNT_ID',[s],labels,'accepted' if proposed else 'rejected',
                   'aligned_explicit_account_label' if proposed else 'existing_contact_policy_rejected',
                   dict(evidence,direction=direction,label_in_value_fragment=False),proposed,view)
    # A continuation needs aligned geometry, incomplete syntax and no boundary/context conflict.
    for start in units:
        owner,owner_conflicts=owner_label(start,units)
        for typ in ('PHONE','EMAIL'):
            if typ=='PHONE':
                match=PHONE_START.search(start['text'].rstrip(' \t'))
                initial=span(start,*match.span()) if match else None
            else:
                match=MAIL_SUFFIX.search(start['text']);initial=span(start,*match.span(1)) if match else None
                if initial and '@' not in initial['text'] and not (owner and re.search(r'이메일|전자\s*우편|e-?mail',owner['text'],re.I)):
                    initial=None
            if initial is None:continue
            existing,_=consume('value_continuation',typ,[initial],[],units)
            if existing:continue # Complete nearby contacts are not continuation anchors.
            if typ=='EMAIL' and initial['text'].count('@')>1:continue
            before=start['text'][:initial['start']]
            rejected_owner=bool((owner and NEGATIVE_LABEL.search(owner['text'])) or NEGATIVE_LABEL.search(before))
            phone_context=bool(re.search(r'전화|연락|휴대|팩스|회신|직통',before) or
                               (owner and CONTACT_LABEL.fullmatch(owner['text']) and re.search(r'전화|연락|휴대|팩스|회신|직통',owner['text'])))
            stack=[([start],[initial],[])]
            while stack:
                chain,values,edges=stack.pop()
                if len(chain)>=CONFIG['max_chain']:continue
                nexts=[]
                for next_unit in units:
                    if next_unit['id'] in {u['id'] for u in chain}:continue
                    s=trimmed_span(next_unit)
                    if not s or not (PHONE_PART.fullmatch(s['text']) if typ=='PHONE' else MAIL_TOKEN.fullmatch(s['text'])):continue
                    direction,reason,evidence=orient(chain[-1],next_unit)
                    if direction:nexts.append((next_unit,s,direction,evidence))
                if len(nexts)>CONFIG['max_branches']:
                    record('value_continuation',typ,values,[],'deferred','branch_limit',dict(branches=len(nexts)));continue
                for nxt,s,direction,evidence in reversed(nexts):
                    trials+=1
                    if trials>CONFIG['max_paths']:
                        raise ValueError('Relation path exploration limit exceeded')
                    val=values+[s];edge=edges+[dict(from_unit=chain[-1]['id'],to_unit=nxt['id'],direction=direction,**evidence)]
                    why=blocking(chain[-1],nxt,units,direction,rgb)
                    if rejected_owner:why='noncontact_owner_label'
                    if why:
                        record('value_continuation',typ,val,[],'rejected',why,dict(edges=edge,owner_label=owner['id'] if owner else None));continue
                    proposed,view=consume('value_continuation',typ,val,[],units)
                    if proposed:
                        hold='competing_owner_labels' if owner_conflicts else ('phone_continuation_without_contact_context' if typ=='PHONE' and not phone_context else None)
                        record('value_continuation',typ,val,[],'deferred' if hold else 'accepted',hold or 'aligned_incomplete_value_completed',
                               dict(edges=edge,owner_label=owner['id'] if owner else None,competing_owner_ids=[u['id'] for u in owner_conflicts]),proposed,view)
                    elif len(chain)+1<CONFIG['max_chain']:
                        record('value_continuation',typ,val,[],'rejected','not_complete_at_this_chain_length',dict(edges=edge),view=view)
                        stack.append((chain+[nxt],val,edge))
                    else:
                        record('value_continuation',typ,val,[],'rejected','existing_contact_policy_not_complete',dict(edges=edge),view=view)
    accepted=[r for r in relations if r['status']=='accepted']
    # No nearest-neighbor winner: competing interpretations sharing values/labels defer.
    conflicts=set()
    for index,a in enumerate(accepted):
        for b in accepted[index+1:]:
            if a['id']==b['id']:continue
            shared_value=set(refs(a['value_segments']))&set(refs(b['value_segments']))
            shared_label=set(refs(a['label_segments']))&set(refs(b['label_segments']))
            if shared_value or shared_label:
                if [signature(p) for p in a['proposed_candidates']]!=[signature(p) for p in b['proposed_candidates']]:
                    conflicts.update((a['id'],b['id']))
                elif a['label_segments']!=b['label_segments']:
                    conflicts.update((a['id'],b['id']))
    for r in relations:
        if r['id'] in conflicts:r.update(status='deferred',reason='competing_relationships')
    return dict(schema_version='ocr-field-relations-v1',image_id=ocr['image_id'],ocr_run_id=ocr['run_id'],
        ocr_sha256=digest(ocr),status='ok',relations=relations,errors=[],skipped_units=skipped,
        limits=copy.deepcopy(CONFIG),explored_paths=trials,seconds=time.perf_counter()-started,
        boundary_evidence='RGB and OCR' if rgb is not None else 'OCR geometry/text only; image boundaries unverified')

def supplement(ocr,baseline,rgb=None,relation_fn=None):
    source_case=dict(id=baseline['id'],blocks=[dict(id=u['id'],text=u['text']) for u in ocr['units']],links=[])
    if baseline.get('input_case_sha256')!=digest(source_case):raise ValueError('Baseline belongs to different OCR blocks/run')
    result=copy.deepcopy(baseline);before_names=[p for p in baseline['configs']['FULL'] if p['type']=='NAME']
    actions=[]
    try:
        relation=(relation_fn or infer_relations)(ocr,rgb)
        if relation['image_id']!=ocr['image_id'] or relation['ocr_run_id']!=ocr['run_id'] or relation['ocr_sha256']!=digest(ocr):
            raise ValueError('Foreign relation result')
        if relation['status']!='ok':raise ValueError('Relationship stage did not complete successfully')
        final=copy.deepcopy(baseline['configs']['FULL']);original_ids={p['id'] for p in final};serial=0
        for rel in relation['relations']:
            if rel['status']!='accepted':continue
            for proposal in rel['proposed_candidates']:
                same=next((p for p in final if signature(p)==signature(proposal)),None)
                if same:
                    rel['candidate_ids'].append(same['id']);actions.append(dict(action='retained_duplicate',candidate_id=same['id'],relation_id=rel['id']));continue
                serial+=1;pid=f'field{serial}'
                while pid in original_ids:serial+=1;pid=f'field{serial}'
                p=dict(copy.deepcopy(proposal),id=pid,field_relation_ids=[rel['id']],field_action='added')
                newrefs=set(refs(p['segments']));removed=[]
                for old in final:
                    oldrefs=set(refs(old['segments']))
                    compatible=old['type']==p['type'] or (p['type']=='EMAIL' and old['type']=='ACCOUNT_ID')
                    if old['type']!='NAME' and compatible and oldrefs<newrefs:
                        removed.append(old)
                for old in removed:
                    final.remove(old);actions.append(dict(action='replaced_partial',candidate=old,replacement_id=pid,relation_id=rel['id'],reason='strict_original_span_subset_of_completed_contact'))
                final.append(p);rel['candidate_ids'].append(pid)
                actions.append(dict(action='added',candidate_id=pid,relation_id=rel['id']))
        result['configs']['FULL']=final
        result['configs']['C1']=copy.deepcopy(baseline['configs']['C1'])
        result['configs']['FIELD_CONTACTS']=[copy.deepcopy(p) for p in final if p['type']!='NAME']
        result['config_statuses']['FIELD']='ok'
        actions.extend(dict(action='retained',candidate_id=p['id']) for p in final if p['id'] in original_ids)
    except Exception as ex:
        relation=dict(schema_version='ocr-field-relations-v1',image_id=ocr['image_id'],ocr_run_id=ocr['run_id'],ocr_sha256=digest(ocr),
            status='failure',relations=[],errors=[dict(stage='field_relation',type=type(ex).__name__,message=str(ex))])
        result=copy.deepcopy(baseline);result['config_statuses']['FIELD']='failure'
        result['status']='failure' if baseline['status']=='failure' else 'partial_failure'
        result['config_statuses']['FULL']=result['status']
        result['errors']=result['errors']+copy.deepcopy(relation['errors'])
        actions=[dict(action='retained_on_relation_failure',candidate_id=p['id']) for p in baseline['configs']['FULL']]
    after_names=[p for p in result['configs']['FULL'] if p['type']=='NAME']
    if before_names!=after_names or baseline['config_statuses']['N1']!=result['config_statuses']['N1']:
        raise AssertionError('NAME contract changed')
    result['field_actions']=actions;result['field_relation_status']=relation['status']
    return result,relation
