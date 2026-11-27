"""NAME-ACCOUNT01 v0.1: conservative, source-preserving candidate experiments.

No gold, filenames, image IDs, network, inference, or global patches. The model's
saved USER_ID diagnostics are corroborating evidence, never a new model run.
Only FULL is changed. Older stage configs/audits remain historical snapshots.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re

VERSION = 'name-account-recall-v0.1.2'
FEATURES = ('mention', 'account_boundary', 'account_context')
HANDLE = re.compile(r'(?=.*[A-Za-z])(?=.*[0-9_.\-])[A-Za-z0-9_][A-Za-z0-9_.\-]{1,63}')
KOREAN_MENTION = re.compile(r'@[가-힣]+')
TIME = re.compile(r'(?:오전|오후)?\s*\d{1,2}\s*:\s*\d{2}')
CLOCK = re.compile(r'(?:(?:오전|오후)\s*(?:[1-9]|1[0-2])|(?:[01]?\d|2[0-3]))\s*:\s*[0-5]\d')
PLATFORM = re.compile(r'(?:인스타(?:그램)?|instagram)(?:로|에서|는|도|를|을)?', re.I)
QUESTION = re.compile(r'(?:네|너의|당신의)?\s*(?:아이디|계정)\s*(?:뭐였지|뭐야|뭔가요|알려줄래)[?？!！.。]*')
INTRO = re.compile(r'(?:(?:아|네|넵|응)\s*){0,2}(?:제|내)\s*(?:아이디|계정)(?:입니다|이에요|예요|이야)[!！.。]*')
ANNOUNCE = re.compile(r'(?:계정\s*)?(?:아이디|계정)\s*(?:알려줄게|알려드릴게요|보내줄게)[!！.。]*')
DEICTIC = re.compile(r'(?:이건|이게)\s*(?:제|내)\s*(?:아이디|계정)(?:입니다|이에요|예요|이야)[!！.。]*')
ENDING = re.compile(r'(?:야|이야|예요|이에요|입니다)')

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def key(candidate):
    return candidate['type'], tuple((s['block_id'], s['start'], s['end']) for s in candidate['segments'])

def refs(segments):
    return {(s['block_id'], i) for s in segments for i in range(s['start'], s['end'])}

def span(block, start, end):
    return dict(block_id=block['id'], start=start, end=end, text=block['text'][start:end])

def validate(segments, texts):
    seen = set()
    for s in segments:
        a,z = s['start'],s['end']
        if type(a) is not int or type(z) is not int or not 0 <= a < z <= len(texts[s['block_id']]):
            raise ValueError('Invalid original code-point interval')
        if texts[s['block_id']][a:z] != s['text'] or seen & refs([s]):
            raise ValueError('Non-exact or repeated source interval')
        seen |= refs([s])

def raw_user_spans(baseline, block):
    """Require contiguous original tokens with an actual B followed by I labels."""
    tokens = baseline.get('raw_model', {}).get(block['id'], [])
    result = []
    current = None
    for t in tokens:
        a,z = t['start'],t['end']
        valid = (not t.get('special') and not t.get('unk') and type(a) is int
                 and type(z) is int and 0 <= a < z <= len(block['text'])
                 and block['text'][a:z] == t['text'])
        if valid and t['label'] == 'B-USER_ID':
            if current: result.append(current)
            current = [a,z]
        elif valid and t['label'] == 'I-USER_ID' and current and a == current[1]:
            current[1] = z
        else:
            if current: result.append(current)
            current = None
    if current: result.append(current)
    return result

def box(unit):
    points = [p for poly in unit.get('polygons', []) for p in poly]
    if not points or unit.get('location_status') != 'valid': return None
    x,y,z,w = min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)
    return (x,y,z,w) if z>x and w>y else None

def context_lines(ocr, full, baseline):
    """Local OCR rows; a decision view only. Original units/offsets are untouched.

    Timestamps, nonword decorations and independently accepted bare raw NAME
    labels are auxiliary. Rows join only close, similarly sized original boxes.
    """
    rawkeys = {key(p) for p in baseline.get('raw_names', [])}
    bare_names = {s['block_id'] for p in full if p['type']=='NAME' and key(p) in rawkeys
                  and len(p['segments'])==1 for s in p['segments']
                  if s['start']==0 and any(u['id']==s['block_id'] and u['text']==s['text'] for u in ocr['units'])}
    units = []
    for u in ocr['units']:
        b = box(u); text = u['text'].strip()
        if not b or u['id'] in bare_names or TIME.fullmatch(text): continue
        if not re.search(r'[가-힣A-Za-z]',text): continue
        units.append((u,b))
    rows = []
    for u,b in sorted(units, key=lambda pair: ((pair[1][1]+pair[1][3])/2,pair[1][0])):
        h=b[3]-b[1]; cy=(b[1]+b[3])/2
        compatible=[]
        for row in rows:
            rb=row['box']; rh=rb[3]-rb[1]
            gap=max(b[0]-rb[2],rb[0]-b[2],0)
            if abs(cy-(rb[1]+rb[3])/2)<=.35*min(h,rh) and max(h,rh)<=1.8*min(h,rh) and gap<=1.0*max(h,rh):
                compatible.append(row)
        if len(compatible)==1:
            row=compatible[0];row['units'].append(u)
            rb=row['box'];row['box']=(min(rb[0],b[0]),min(rb[1],b[1]),max(rb[2],b[2]),max(rb[3],b[3]))
        else:
            rows.append(dict(units=[u],box=b))
    for row in rows:
        row['units'].sort(key=lambda u:box(u)[0])
        row['text']=' '.join(u['text'].strip() for u in row['units'])
        row['segments']=[span(u,0,len(u['text'])) for u in row['units']]
    return sorted(rows,key=lambda r:(r['box'][1],r['box'][0]))

def local_pair(a,b,aligned=True):
    aa,bb=a['box'],b['box']; h=max(aa[3]-aa[1],bb[3]-bb[1])
    gap=max(aa[1],bb[1])-min(aa[3],bb[3])
    if not 0 <= gap <= 3*h: return False
    overlap=max(0,min(aa[2],bb[2])-max(aa[0],bb[0]))
    return not aligned or overlap>=.5*min(aa[2]-aa[0],bb[2]-bb[0]) or abs(aa[0]-bb[0])<=h

def chat_timestamps(ocr, platform, question, answer):
    """Three adjacent messages in opposing columns share a timestamp corridor.

    This rejects merely nearby rows in separate panes. It is a deliberately
    narrow layout proof, not a general screen/dialogue classifier.
    """
    q,a=question['box'],answer['box']
    if q[2]<a[0]: question_side='right'
    elif a[2]<q[0]: question_side='left'
    else: return None
    answer_side='left' if question_side=='right' else 'right'
    p=platform['box']
    if question_side=='right' and not q[2]<p[0]: return None
    if question_side=='left' and not p[2]<q[0]: return None
    anchors=[]
    for row,side in ((platform,answer_side),(question,question_side),(answer,answer_side)):
        b=row['box'];h=b[3]-b[1];cy=(b[1]+b[3])/2;matches=[]
        for u in ocr['units']:
            if not CLOCK.fullmatch(u['text'].strip()):continue
            t=box(u)
            if t is None:continue
            th=t[3]-t[1];tc=(t[1]+t[3])/2
            gap=t[0]-b[2] if side=='right' else b[0]-t[2]
            if .4*h<=th<=1.2*h and cy-.35*h<=tc<=b[3]+.5*h and 0<=gap<=2.5*h:
                matches.append((u,t))
        if len(matches)!=1:return None
        anchors.append(matches[0])
    if len({u['id'] for u,t in anchors})!=3:return None
    for i,(_,aa) in enumerate(anchors):
        for _,bb in anchors[i+1:]:
            overlap=max(0,min(aa[2],bb[2])-max(aa[0],bb[0]))
            if overlap<.4*min(aa[2]-aa[0],bb[2]-bb[0]):return None
    return [span(u,0,len(u['text'])) for u,t in anchors]

def context_evidence(rows,index,baseline,unit,ocr,*,boundary=False):
    row=rows[index]; prior=rows[index-1] if index else None
    after=rows[index+1] if index+1<len(rows) else None
    exact_raw=[0,len(unit['text'])] in raw_user_spans(baseline,unit)
    if prior and local_pair(prior,row) and (INTRO.fullmatch(prior['text']) or ANNOUNCE.fullmatch(prior['text'])):
        if exact_raw or boundary:
            return dict(kind='adjacent_account_introduction',context_segments=prior['segments'],raw_user_id_exact=exact_raw)
    if after and local_pair(row,after) and DEICTIC.fullmatch(after['text']) and exact_raw:
        return dict(kind='adjacent_deictic_account_declaration',context_segments=after['segments'],raw_user_id_exact=True)
    if not boundary and prior and local_pair(prior,row,False) and QUESTION.fullmatch(prior['text']):
        # A bounded platform question/answer pair, not all machine-like strings
        # anywhere in an image containing the word "account".
        earlier=rows[max(0,index-3):index-1]
        platform=[p for p in earlier if PLATFORM.search(p['text']) and local_pair(p,prior,False)]
        timestamps=chat_timestamps(ocr,platform[0],prior,row) if len(platform)==1 else None
        if timestamps and any(a==0 for a,z in raw_user_spans(baseline,unit)):
            return dict(kind='platform_account_question_next_answer',context_segments=prior['segments'],
                        platform_segments=platform[0]['segments'],timestamp_segments=timestamps,raw_user_id_prefix=True)
    return None

def improve(ocr, case, baseline, detection, relations, *, features=FEATURES):
    """Return (new_detection, new_relations). Invalid provenance raises, no mutation.

    Caller must rerun refine_candidates and regenerate the candidate version,
    selection and mask. Failure/partial stage statuses cause an unchanged return.
    """
    if set(features)-set(FEATURES): raise ValueError('Unknown feature')
    if 'name_account_audit' in detection or 'name_account_stage' in relations:
        raise ValueError('Already processed; replay from the frozen pre-experiment FULL')
    expected=dict(id=case['id'],blocks=[dict(id=u['id'],text=u['text']) for u in ocr['units']],links=[])
    if case!=expected or any(d.get('id')!=case['id'] or d.get('input_case_sha256')!=digest(case) for d in (baseline,detection)):
        raise ValueError('Foreign case/detection')
    if (relations.get('image_id')!=ocr['image_id'] or relations.get('ocr_run_id')!=ocr['run_id']
            or relations.get('ocr_sha256')!=digest(ocr)):
        raise ValueError('Foreign relations')
    result,updated=copy.deepcopy((detection,relations))
    if (ocr['status']!='ok' or baseline['raw_names_status']!='ok' or detection['status']!='ok'
            or any(v!='ok' for v in detection['config_statuses'].values()) or relations['status']!='ok'):
        return result,updated
    units={u['id']:u for u in ocr['units']}; texts={k:u['text'] for k,u in units.items()}
    full=result['configs']['FULL']; before=copy.deepcopy(full); actions=[]
    for p in full: validate(p['segments'],texts)
    # Do not resolve a mention against a strong account field. Ambiguity remains.
    account_fields=[r for r in relations['relations'] if r['type']=='ACCOUNT_ID' and r['status'] in ('accepted','deferred')]
    rawkeys={key(p) for p in baseline.get('raw_names',[])}
    name_support=[p for p in full if p['type']=='NAME' and key(p) in rawkeys and len(p['segments'])==1]
    rows=context_lines(ocr,full,baseline)
    replacements={}

    def replace(old, typ, segments, reason, evidence, existing=None):
        validate(segments,texts)
        candidate=existing
        if candidate is None:
            candidate=dict(id='na1-'+digest(dict(identity=case['id'],type=typ,segments=segments))[:20],
                           type=typ,segments=copy.deepcopy(segments),sources=[VERSION],
                           name_account_evidence=dict(reason=reason,**copy.deepcopy(evidence)))
            same=next((p for p in full if key(p)==key(candidate) and (old is None or p['id']!=old['id'])),None)
            if same is not None and same['id']==candidate['id']:
                candidate=same
            else:
                if same is not None:
                    # Metadata/backlinks will change: do not keep an old identity
                    # merely because an exact-span duplicate already existed.
                    full.remove(same);replacements[same['id']]=candidate['id']
                    actions.append(dict(action='replace',reason='deduplicate_into_updated_candidate',
                        original_candidate=copy.deepcopy(same),replacement_id=candidate['id'],
                        candidate=copy.deepcopy(candidate),evidence=dict(trigger=reason)))
                full.append(candidate)
        if old:
            full.remove(old);replacements[old['id']]=candidate['id']
        actions.append(dict(action='replace' if old else 'add',reason=reason,
                            original_candidate=copy.deepcopy(old),replacement_id=candidate['id'],
                            candidate=copy.deepcopy(candidate),evidence=copy.deepcopy(evidence)))

    if 'mention' in features:
        for p in before:
            if p['type']!='ACCOUNT_ID' or len(p['segments'])!=1: continue
            s=p['segments'][0]
            if not KOREAN_MENTION.fullmatch(s['text']) or 'account_at_v1' not in p.get('sources',[]): continue
            if any(refs(r['value_segments']) & refs([s]) for r in account_fields): continue
            target=span(units[s['block_id']],s['start']+1,s['end'])
            witnesses=[q for q in name_support if q['segments'][0]['text']==target['text']]
            same=next((q for q in witnesses if q['segments']==[target]),None)
            if not witnesses: continue
            # Bare other-occurrence evidence requires the entire separate block;
            # a NAME prefix embedded in an ordinary word is not corroboration.
            witnesses=[q for q in witnesses if q==same or q['segments'][0]['text']==texts[q['segments'][0]['block_id']]]
            if not witnesses: continue
            replace(p,'NAME',[target],'raw_backed_exact_person_mention',
                    dict(witness_candidates=witnesses),existing=same)

    if 'account_boundary' in features:
        for p in list(full):
            if p['type']!='ACCOUNT_ID' or len(p['segments'])!=1: continue
            s=p['segments'][0];u=units[s['block_id']]
            if s['start']!=0 or s['end']!=len(u['text']): continue
            if not any(r['status']=='accepted' and p['id'] in r['candidate_ids'] for r in account_fields): continue
            # A literal inline field is authoritative even when the model splits
            # a real mixed-script ID into USER_ID and a Korean-looking ending.
            if any(re.search(r'[:：=＝]',texts[t['block_id']]) for r in account_fields
                   if p['id'] in r['candidate_ids'] for t in r['label_segments']): continue
            raw=raw_user_spans(baseline,u)
            if len(raw)!=1 or raw[0][0]!=0: continue
            end=raw[0][1]
            if not HANDLE.fullmatch(u['text'][:end]) or not ENDING.fullmatch(u['text'][end:]): continue
            tail=[t for t in baseline.get('raw_model',{}).get(u['id'],[]) if t['start']>=end]
            if (not tail or any(t['label']!='O' or t.get('special') or t.get('unk')
                                or t.get('special_token_id') or t.get('text')!=u['text'][t['start']:t['end']] for t in tail)
                    or [(t['start'],t['end']) for t in tail]!=[(end,len(u['text']))]): continue
            matches=[(j,r) for j,r in enumerate(rows) if len(r['units'])==1 and r['units'][0]['id']==u['id']]
            if len(matches)!=1: continue
            evidence=context_evidence(rows,matches[0][0],baseline,u,ocr,boundary=True)
            if evidence:
                replace(p,'ACCOUNT_ID',[span(u,0,end)],'raw_user_id_boundary_with_account_introduction',dict(evidence,raw_user_span=raw[0],outside_token=tail))

    if 'account_context' in features:
        for j,row in enumerate(rows):
            if len(row['units'])!=1: continue
            u=row['units'][0]
            if not HANDLE.fullmatch(u['text']): continue
            segment=span(u,0,len(u['text']))
            # Never take over an existing candidate, including PHONE/EMAIL.
            if any(refs(p['segments']) & refs([segment]) for p in full): continue
            if any(refs(r['value_segments']) & refs([segment]) for r in account_fields): continue
            evidence=context_evidence(rows,j,baseline,u,ocr)
            if evidence: replace(None,'ACCOUNT_ID',[segment],evidence['kind'],evidence)

    # Keep historical stage snapshots intact; rebuild the sole current relation
    # list. A changed relation gets a new ID and records its entire predecessor.
    current=[];relation_actions=[];byid={p['id']:p for p in full}
    for rel in updated['relations']:
        oldrefs=rel.get('candidate_ids',[])+rel.get('supported_candidate_ids',[])
        if not set(oldrefs)&replacements.keys(): current.append(rel);continue
        revised=copy.deepcopy(rel)
        for field in ('candidate_ids','supported_candidate_ids'):
            revised[field]=list(dict.fromkeys(replacements.get(cid,cid) for cid in rel.get(field,[])
                                if byid[replacements.get(cid,cid)]['type']==rel['type']))
        ids=revised['candidate_ids']+revised['supported_candidate_ids']
        if ids:
            segments=[byid[cid]['segments'] for cid in ids]
            if any(s!=segments[0] for s in segments): raise ValueError('Ambiguous relation replacement')
            revised['value_segments']=copy.deepcopy(segments[0])
            revised['proposed_candidates']=[copy.deepcopy(byid[cid]) for cid in ids]
            revised.pop('view',None)  # Old synthetic view belongs only to audit.
            revised['reason']='name_account_boundary_replacement'
        else:
            revised.update(status='rejected',reason='name_account_type_conflict_resolved',proposed_candidates=[])
        revised['id']='na1-r-'+digest(dict(original_id=rel['id'],segments=revised['value_segments'],ids=ids))[:20]
        revised['source']=VERSION
        revised['evidence']=dict(kind='experiment_candidate_replacement',
                                previous_relation_id=rel['id'],previous_source=rel.get('source'),
                                inherited_label_geometry=copy.deepcopy(rel.get('evidence',{})))
        relation_actions.append(dict(original_relation=copy.deepcopy(rel),replacement_relation_id=revised['id']))
        current.append(revised)
    updated['relations']=current
    for p in full:
        if p['id'].startswith('na1-'):
            linked=[r['id'] for r in current if p['id'] in r.get('candidate_ids',[])+r.get('supported_candidate_ids',[])]
            if linked: p['field_relation_ids']=linked
    # References in the live proposal agree with final candidate objects. Old
    # stage copies and predecessor views exist only in the explicit audit.
    rewritten_ids={row['replacement_relation_id'] for row in relation_actions}
    for rel in current:
        if rel['id'] in rewritten_ids:
            rel['proposed_candidates']=[copy.deepcopy(byid[cid]) for cid in
                rel.get('candidate_ids',[])+rel.get('supported_candidate_ids',[])]
    for action in actions:
        action['candidate']=copy.deepcopy(byid[action['replacement_id']])
    # Exact PHONE/EMAIL objects, current relations and upstream statuses preserved.
    protected=lambda ps:[p for p in ps if p['type'] in ('PHONE','EMAIL')]
    if protected(before)!=protected(full) or protected(relations['relations'])!=protected(current):
        raise ValueError('Protected candidate/relation changed')
    if len({p['id'] for p in full})!=len(full) or len({key(p) for p in full})!=len(full):
        raise ValueError('Duplicate final candidate')
    for p in full: validate(p['segments'],texts)
    for rel in current:
        for field in ('candidate_ids','supported_candidate_ids'):
            for cid in rel.get(field,[]):
                if cid not in byid or byid[cid]['type']!=rel['type']: raise ValueError('Stale relationship consumer')
    audit=dict(version=VERSION,features=list(features),case_id=case['id'],input_case_sha256=digest(case),
               before_candidate_version=digest(before),after_candidate_version=digest(full),actions=actions,
               relation_actions=relation_actions,replacements=replacements,
               historical_fields='All preexisting configs except FULL and nested stage audits are historical, not current consumers.')
    result['name_account_audit']=audit
    updated['name_account_stage']=copy.deepcopy(audit)
    return result,updated
