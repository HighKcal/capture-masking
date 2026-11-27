"""Two bounded postprocessing changes; OCR/NER/FIELD inputs remain immutable.

No model inference, gold access, global monkeypatch or cross-field search.
Original Unicode code-point intervals are the only output coordinates.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
TXT = ROOT.parent / 'text-detection-ab-v0'
sys.dont_write_bytecode = True
sys.path.insert(0, str(TXT))
import improved as legacy
from detectors import key

VERSION = 'pii-recall-v1'
SETTINGS = dict(version=VERSION, names='raw_backed_coordinated_boundary',
                email='explicit_mail_label_and_plus_horizontal_gap',
                max_plus_gap=2, new_ocr=False, new_ner=False, n2=False)
CONJUNCTION = re.compile(r'(?:와|과|랑|이랑|하고)')
MAIL_LABEL = re.compile(r'(?P<label>(?<![\w])(?:이메일|전자[ \t]*우편|메일))(?=[A-Za-z0-9.!#$%&\x27*+/=?^_`{|}~-]+@)')
LOCAL_CHARS = r"A-Za-z0-9.!#$%&'*+/=?^_`{|}~-"
PLUS_GAP = re.compile(r'(?<=[' + LOCAL_CHARS + r'])[ \t]{1,2}(?=\+[' + LOCAL_CHARS + r']+@)')
PREFIX_BEFORE_PLUS = re.compile(r'([' + LOCAL_CHARS + r']+)[ \t]{1,2}$')
MAIL_CONTEXT = re.compile(r'(?<![\w])(?:(?:이메일|전자[ \t]*우편|메일)[ \t]*[:：=＝]?[ \t]*|e-?mail(?:[ \t]*[:：=＝][ \t]*|[ \t]+))[' + LOCAL_CHARS + r']+$', re.I)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def refs(segments):
    return [(s['block_id'], i) for s in segments for i in range(s['start'], s['end'])]


def validate_segments(segments, texts):
    for s in segments:
        a,z=s['start'],s['end']
        if type(a) is not int or type(z) is not int or not 0 <= a < z <= len(texts[s['block_id']]):
            raise ValueError('Invalid original code-point interval')
        if texts[s['block_id']][a:z] != s['text']:
            raise ValueError('Original exact slice mismatch')
    if len(refs(segments)) != len(set(refs(segments))):
        raise ValueError('Overlapping/repeated source interval')


def name_additions(baseline, case):
    """Only rescue an existing raw model NAME rejected solely at a left join."""
    texts={b['id']:b['text'] for b in case['blocks']}
    accepted=baseline['configs']['N1']
    additions,audit=[],[]
    for row in baseline.get('policy_audit',{}).get('N1',[]):
        if row.get('action')!='reject' or row.get('reason')!='candidate_inside_left_lexeme':
            continue
        p=row['candidate']
        if p['type']!='NAME' or len(p['segments'])!=1:
            continue
        s=p['segments'][0];text=texts[s['block_id']]
        validate_segments(p['segments'],texts)
        # Require an independently accepted predecessor in this exact OCR block.
        predecessors=[]
        for previous in accepted:
            if len(previous['segments'])!=1:continue
            q=previous['segments'][0]
            if q['block_id']==s['block_id'] and q['end']<s['start'] and CONJUNCTION.fullmatch(text[q['end']:s['start']]):
                predecessors.append(q)
        if len(predecessors)!=1:continue
        q=predecessors[0]
        if baseline.get('block_statuses',{}).get(s['block_id'])!='ok':continue
        # Same-length decision view: only the conjunction outside both candidates
        # becomes whitespace. Original candidate text/offsets cannot change.
        view=text[:q['end']]+' '*(s['start']-q['end'])+text[s['start']:]
        kept,checks=legacy.improved_names([copy.deepcopy(p)],[dict(id=s['block_id'],text=view)])
        event=dict(kind='name_coordinated_boundary',original_candidate_id=p.get('id'),
                   preceding_segment=q,decision_only_interval=[q['end'],s['start']],
                   source_block_id=s['block_id'],mapping='identity outside decision-only conjunction',
                   legacy_decision=checks,accepted=False)
        if len(kept)==1 and kept[0]['segments']==p['segments']:
            candidate=kept[0]
            candidate['sources']=list(dict.fromkeys(candidate['sources']+['recall_name_boundary_v1']))
            candidate['recall_evidence']=dict(previous_name=q,raw_candidate_id=p.get('id'))
            additions.append(candidate);event['accepted']=True
        audit.append(event)
    return additions,audit


def mapped_segments(mapping, a, z, texts):
    groups=[]
    for ref in mapping[a:z]:
        if ref is None:raise ValueError('Candidate contains virtual character')
        bid,i=ref
        if groups and groups[-1][0]==bid and groups[-1][2]==i:
            groups[-1][2]=i+1
        else:groups.append([bid,i,i+1])
    segments=[dict(block_id=bid,start=a,end=z,text=texts[bid][a:z]) for bid,a,z in groups]
    validate_segments(segments,texts)
    return segments


def normalize_view(text, mapping, accepted_email_relation=False):
    """Only explicit mail-label boundary and 1–2 spaces immediately before +."""
    inserts={m.end():m.group('label') for m in MAIL_LABEL.finditer(text)}
    # A normal word followed by a valid +tag address is not evidence of one value.
    # Demand an explicit mail label or an already accepted FIELD email relation.
    removed={i for m in PLUS_GAP.finditer(text)
             if accepted_email_relation or MAIL_CONTEXT.search(text[:m.start()])
             for i in range(*m.span())}
    result=[];mapped=[];edits=[]
    for i,c in enumerate(text):
        if i in inserts:
            result.append(' ');mapped.append(None)
            edits.append(dict(kind='virtual_mail_label_separator',at=i,label=inserts[i]))
        if i in removed:
            edits.append(dict(kind='omit_horizontal_gap_before_plus',at=i,source=mapping[i]));continue
        result.append(c);mapped.append(mapping[i])
    return ''.join(result),mapped,edits


def email_additions(case,current_relations):
    texts={b['id']:b['text'] for b in case['blocks']}
    views=[]
    for b in case['blocks']:
        views.append((b['text'],[(b['id'],i) for i in range(len(b['text']))],None))
    # Existing FIELD decided geometry and boundaries. No new edge is proposed.
    for relation in current_relations.get('relations',[]):
        if relation.get('status')!='accepted' or relation.get('type')!='EMAIL':continue
        values=copy.deepcopy(relation['value_segments'])
        if not values:continue
        first=values[0];prefix=texts[first['block_id']][:first['start']]
        before=PREFIX_BEFORE_PLUS.search(prefix)
        if first['text'].startswith('+') and before:
            first['start']=before.start(1)
            first['text']=texts[first['block_id']][first['start']:first['end']]
        validate_segments(values,texts)
        text=''.join(s['text'] for s in values)
        mapping=refs(values)
        views.append((text,mapping,relation['id']))
    additions,audit=[],[]
    for original,mapping,relation_id in views:
        normalized,source_map,edits=normalize_view(original,mapping,relation_id is not None)
        if not edits:continue
        predictions,contact_audit=legacy.improved_contacts(dict(id='recall-decision-view',
            blocks=[dict(id='virtual',text=normalized)],links=[]))
        event=dict(kind='email_boundary_normalization',source_relation_id=relation_id,
                   original_view=original,decision_view=normalized,
                   source_map=source_map,edits=edits,contact_audit=contact_audit,accepted=[])
        for p in predictions:
            if p['type']!='EMAIL':continue
            segments=[]
            for s in p['segments']:
                segments+=mapped_segments(source_map,s['start'],s['end'],texts)
            validate_segments(segments,texts)
            p=copy.deepcopy(p);p['segments']=segments
            p['sources']=list(dict.fromkeys(p['sources']+['recall_email_boundary_v1']))
            p['recall_evidence']=dict(source_relation_id=relation_id,normalization=edits)
            additions.append(p);event['accepted'].append(segments)
        audit.append(event)
    return additions,audit


def improve(ocr,case,baseline,current_detection,current_relations,rgb=None,context=None,
            enable_names=True,enable_email=True):
    if case['id']!=baseline['id'] or baseline.get('input_case_sha256')!=digest(case):
        raise ValueError('Foreign baseline/case')
    expected=dict(id=case['id'],blocks=[dict(id=u['id'],text=u['text']) for u in ocr['units']],links=[])
    if case!=expected or current_detection.get('input_case_sha256')!=digest(case):
        raise ValueError('Foreign OCR/detection')
    if (current_relations.get('image_id')!=ocr['image_id'] or
        current_relations.get('ocr_run_id')!=ocr['run_id'] or
        current_relations.get('ocr_sha256')!=digest(ocr)):
        raise ValueError('Foreign FIELD relations')
    result=copy.deepcopy(current_detection);relations=copy.deepcopy(current_relations)
    before=digest(dict(ocr=ocr,case=case,baseline=baseline,detection=current_detection,relations=current_relations))
    proposals=[];audit=[];actions=[]
    texts={b['id']:b['text'] for b in case['blocks']}
    if enable_names:
        new,events=name_additions(baseline,case);proposals+=new;audit+=events
    if enable_email and ocr['status']=='ok' and current_relations['status']=='ok' and baseline['config_statuses']['C1']=='ok':
        new,events=email_additions(case,current_relations);proposals+=new;audit+=events
    final=result['configs']['FULL'];existing_ids={p['id'] for p in final}
    for p in proposals:
        validate_segments(p['segments'],texts)
        same=next((q for q in final if key(q)==key(p)),None)
        if same:
            actions.append(dict(action='duplicate_preserved',candidate_id=same['id']));continue
        if p['type']=='EMAIL' and any(q['type']=='EMAIL' and set(refs(p['segments']))<set(refs(q['segments'])) for q in final):
            actions.append(dict(action='contained_email_proposal_discarded',segments=p['segments']));continue
        p=copy.deepcopy(p);p['id']='recall-'+digest(dict(type=p['type'],segments=p['segments']))[:20]
        if p['id'] in existing_ids:raise ValueError('Candidate ID collision')
        new_refs=set(refs(p['segments']));removed=[]
        if p['type']=='EMAIL':
            for old in final:
                if old['type'] in ('EMAIL','ACCOUNT_ID') and set(refs(old['segments']))<new_refs:
                    removed.append(old)
        for old in removed:
            final.remove(old)
            actions.append(dict(action='replaced_partial',candidate=old,replacement_id=p['id'],
                                reason='strict_original_span_subset_of_normalized_email'))
        final.append(p);existing_ids.add(p['id'])
        actions.append(dict(action='added',candidate_id=p['id'],type=p['type'],segments=p['segments']))
    replacements={a['candidate']['id']:a['replacement_id'] for a in actions if a['action']=='replaced_partial'}
    final_by_id={p['id']:p for p in final}
    for rel in relations.get('relations',[]):
        old=list(rel.get('candidate_ids',[]))
        new=[];superseded=[]
        for candidate_id in old:
            replacement=replacements.get(candidate_id,candidate_id)
            if replacement!=candidate_id and final_by_id[replacement]['type']!=rel['type']:
                superseded.append(dict(candidate_id=candidate_id,replacement_id=replacement,
                    reason='superseded_by_different_contact_type_not_a_relation_consumer'))
            elif replacement not in new:new.append(replacement)
        if new!=old:rel['pre_recall_candidate_ids']=old;rel['candidate_ids']=new
        if superseded:rel['recall_superseded_consumers']=superseded
    result['recall']=dict(version=VERSION,settings=SETTINGS,enable_names=enable_names,
        enable_email=enable_email,audit=audit,actions=actions,source_candidate_version=digest(current_detection['configs']['FULL']))
    result['config_statuses']['RECALL']='ok'
    result['configs']['RECALL_NAMES']=[copy.deepcopy(p) for p in final if p['type']=='NAME']
    relations['recall_candidate_replacements']=replacements
    if before!=digest(dict(ocr=ocr,case=case,baseline=baseline,detection=current_detection,relations=current_relations)):
        raise AssertionError('Shared source mutation')
    return result,relations
