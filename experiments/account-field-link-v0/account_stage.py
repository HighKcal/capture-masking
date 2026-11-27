"""Account-only post-FIELD augmentation with source intervals and conservative roles."""
import copy
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'ocr-integration-v0'))
import dependencies as dep
if Path(dep.__file__).resolve() != ROOT.parent / 'ocr-integration-v0/dependencies.py':
    raise ImportError('Foreign dependency loader')
f = dep.field
VERSION = 'account-field-link-v0.1.0'
SETTINGS = dict(version=VERSION, labels='semantic_head_v1', roles='corroborated_field_pair_v1',
                geometry='unchanged_FIELD', candidate_policy='append_exact_nonoverlap', max_relations=256,
                max_pair_checks=4096,max_role_comparisons=8192)
# Meaning-bearing heads, not screen/case names or values. Only whole field syntax.
HEAD = r'(?:(?:로그인|사용자|온라인|프로필)[ \t]*(?:아이디|ID|계정)|계정[ \t]*(?:아이디|ID)|아이디|계정)'
STANDALONE = re.compile(r'^[ \t]*(?P<label>' + HEAD + r')[ \t]*(?:[:：=＝])?[ \t]*$', re.I)
INLINE = re.compile(r'^[ \t]*(?P<label>' + HEAD + r')[ \t]*[:：=＝][ \t]*(?P<value>@?\w+(?:[.\-]\w+)*)[ \t]*$', re.I)
OTHER_HEADING = re.compile(r'^[가-힣A-Za-z][가-힣A-Za-z \t]{0,19}?(?:코드|번호|ID)[ \t]*$', re.I)
DELIMITED_HEADING = re.compile(r'^[가-힣A-Za-z][가-힣A-Za-z \t]{0,23}[:：=＝][ \t]*$')
DATE_HEADING = re.compile(r'^[가-힣]{1,10}(?:일자|일)[ \t]*$')
DATE_VALUE = re.compile(r'(?:19|20)[0-9]{2}[-./](?:0?[1-9]|1[0-2])[-./](?:0?[1-9]|[12][0-9]|3[01])')


def parse_label(unit, expanded=True):
    standalone = (STANDALONE if expanded else f.ACCOUNT_LABEL).fullmatch(unit['text'])
    inline = None if standalone else (INLINE if expanded else f.INLINE_ACCOUNT).fullmatch(unit['text'])
    m = standalone or inline
    if not m:
        return None
    return dict(label_segments=[f.span(unit, *m.span('label'))], inline=bool(inline),
                value_segments=[f.span(unit, *m.span('value'))] if inline else [],
                semantic=re.sub(r'[ \t]+', '', m['label']).lower())


def pair(a, b, units, rgb):
    direction, reason, evidence = f.orient(a, b, label=True)
    if not direction:
        return None
    blocked = f.blocking(a, b, units, direction, rgb)
    l,t,r,z = a['_g']['box']; ll,tt,rr,zz = b['_g']['box']
    if min(r,rr)>max(l,ll) and min(z,zz)>max(t,tt):
        blocked = blocked or 'label_value_regions_overlap'
    return dict(direction=direction, geometry_reason=reason, **evidence, blocked=blocked)


def repeated_layout(a, b, c, d, direction):
    """Two independently different pairs share label/value columns or rows."""
    h = max(u['_g']['h'] for u in (a,b,c,d))
    if len({u['id'] for u in (a,b,c,d)}) != 4:
        return False
    if direction == 'right':
        return (abs(a['_g']['box'][0]-c['_g']['box'][0]) <= .8*h and
                abs(b['_g']['box'][0]-d['_g']['box'][0]) <= .8*h and
                abs(a['_g']['cy']-c['_g']['cy']) > .55*h)
    return (abs(a['_g']['cy']-c['_g']['cy']) <= .45*h and
            abs(b['_g']['cy']-d['_g']['cy']) <= .45*h and
            abs(a['_g']['box'][0]-c['_g']['box'][0]) > h)


def infer(ocr, rgb, labels, roles):
    units, skipped = f.project_source(ocr)
    if ocr['status'] not in ('ok', 'partial_failure'):
        raise ValueError('OCR stage not usable')
    if rgb is not None and rgb.size != (ocr['input']['width'], ocr['input']['height']):
        raise ValueError('Original RGB size mismatch')
    parsed = {u['id']: parse_label(u, labels) for u in units}
    parsed = {k:v for k,v in parsed.items() if v}
    pair_cache={}
    def checked_pair(a,b):
        key=(a['id'],b['id'])
        if key not in pair_cache:
            if len(pair_cache)>=SETTINGS['max_pair_checks']:
                raise ValueError('Account geometry exploration limit exceeded')
            pair_cache[key]=pair(a,b,units,rgb)
        return pair_cache[key]
    # Every complete other-field role needs its own value. No nearest selection.
    role_proofs = {}
    if roles:
        possible_headings = {u['id'] for u in units if u['id'] not in parsed and
                             (OTHER_HEADING.fullmatch(u['text'].strip()) or DATE_HEADING.fullmatch(u['text'].strip()) or DELIMITED_HEADING.fullmatch(u['text'].strip()))}
        account_pairs=[]
        for anchor in units:
            if anchor['id'] not in parsed or parsed[anchor['id']]['inline']:
                continue
            for av in units:
                if av['id'] in parsed or av['id'] in possible_headings:
                    continue
                sv=f.trimmed_span(av)
                if not sv or not f.ACCOUNT_VALUE.fullmatch(sv['text']):
                    continue
                ae=checked_pair(anchor,av)
                if ae and not ae['blocked']:
                    account_pairs.append((anchor,av,sv,ae))
        comparisons=0
        for title in units:
            if title['id'] not in possible_headings:
                continue
            options=[]
            for value in units:
                if value['id'] == title['id'] or value['id'] in parsed or value['id'] in possible_headings:
                    continue
                s=f.trimmed_span(value)
                if not s or not f.ACCOUNT_VALUE.fullmatch(s['text']):
                    continue
                if DATE_HEADING.fullmatch(title['text'].strip()) and not DATE_VALUE.fullmatch(s['text']):
                    continue
                e=checked_pair(title,value)
                if not e or e['blocked']:
                    continue
                witnesses=[]
                for anchor,av,sv,ae in account_pairs:
                    comparisons+=1
                    if comparisons>SETTINGS['max_role_comparisons']:
                        raise ValueError('Account role proof exploration limit exceeded')
                    if ae['direction']==e['direction'] and repeated_layout(title,value,anchor,av,e['direction']):
                        witnesses.append(dict(label_segments=parsed[anchor['id']]['label_segments'],value_segments=[sv]))
                explicit=bool(DELIMITED_HEADING.fullmatch(title['text'].strip()))
                if explicit or witnesses:
                    options.append(dict(label_segments=[f.trimmed_span(title)],value_segments=[s],
                                        evidence=e,explicit_delimiter=explicit,parallel_account_pairs=witnesses))
                if len(options)>1:
                    break
            if len(options)==1:
                role_proofs[title['id']]=options[0]
    relations=[]
    def record(label, values, status, reason, evidence=None):
        identity=dict(image_id=ocr['image_id'],ocr_run_id=ocr['run_id'],kind='label_value',type='ACCOUNT_ID',
                      value_segments=values,label_segments=label['label_segments'])
        rel=dict(identity,id='af-r-'+f.digest(identity)[:20],status=status,reason=reason,
                 source=VERSION,evidence=evidence or {},candidate_ids=[],proposed_candidates=[])
        if status=='accepted':
            proposals,view=f.consume('label_value','ACCOUNT_ID',values,label['label_segments'],units)
            rel.update(proposed_candidates=proposals,view=view)
            if not proposals:
                rel.update(status='rejected',reason='existing_contact_policy_rejected')
        relations.append(rel)
        if len(relations)>SETTINGS['max_relations']:
            raise ValueError('Account relation limit exceeded')
    for u in units:
        label=parsed.get(u['id'])
        if not label:
            continue
        if label['inline']:
            record(label,label['value_segments'],'accepted','inline_explicit_account_label')
            continue
        for v in units:
            if v['id']==u['id']:
                continue
            s=f.trimmed_span(v)
            if not s or not f.ACCOUNT_VALUE.fullmatch(s['text']):
                continue
            e=checked_pair(u,v)
            if not e:
                continue
            if e['blocked']:
                record(label,[s],'rejected',e['blocked'],e);continue
            if v['id'] in role_proofs:
                record(label,[s],'rejected','independent_other_field_heading',dict(e,role_proof=role_proofs[v['id']]));continue
            owners=[proof for proof in role_proofs.values() if proof['value_segments']==[s]]
            if owners:
                record(label,[s],'deferred','independent_other_field_value',dict(e,other_owners=owners));continue
            if v['id'] in parsed:
                record(label,[s],'deferred','value_is_explicit_account_label',e);continue
            if (labels and not parse_label(u,False) and e['direction']=='right' and
                not re.search(r'[:：=＝]',u['text']) and re.fullmatch(r'[가-힣]+',s['text']) and
                not any(w['label_segments']==label['label_segments'] and w['value_segments']==[s]
                        for proof in role_proofs.values() for w in proof['parallel_account_pairs'])):
                record(label,[s],'deferred','unpunctuated_new_label_hangul_title_ambiguity',e);continue
            # Preserve old broad owner cautions except semantic labels newly recognized by A.
            owner, conflicts=f.owner_label(v,units)
            other=[x for x in conflicts if x['id'] not in parsed]
            if other or (owner and owner['id']!=u['id'] and owner['id'] not in parsed):
                record(label,[s],'deferred','competing_other_field_owner',dict(e,owner_ids=[x['id'] for x in other] or [owner['id']]));continue
            record(label,[s],'accepted','aligned_explicit_account_label',e)
    # Real alternate values and alternate owners both remain ambiguous.
    accepted=[r for r in relations if r['status']=='accepted' or
              r['reason']=='unpunctuated_new_label_hangul_title_ambiguity']
    conflicts=set()
    for i,a in enumerate(accepted):
        for b in accepted[i+1:]:
            if ((set(f.refs(a['label_segments'])) & set(f.refs(b['label_segments'])) or
                 set(f.refs(a['value_segments'])) & set(f.refs(b['value_segments']))) and
                (a['value_segments']!=b['value_segments'] or a['label_segments']!=b['label_segments'])):
                conflicts.update((a['id'],b['id']))
    for r in relations:
        if r['id'] in conflicts:
            r.update(prior_reason=r['reason'],status='deferred',reason='competing_relationships')
    return dict(version=VERSION,status='ok',labels=list(parsed.values()),role_evidence=list(role_proofs.values()),
                relations=relations,skipped_units=skipped,settings=copy.deepcopy(SETTINGS),
                ablation=dict(labels=labels,roles=roles))


def supplement(ocr, baseline_detection, baseline_relations, rgb=None, *, labels=True, roles=True):
    started=time.perf_counter()
    result=copy.deepcopy(baseline_detection); rel=copy.deepcopy(baseline_relations)
    actions=[]
    try:
        source_case=dict(id=result['id'],blocks=[dict(id=u['id'],text=u['text']) for u in ocr['units']],links=[])
        if result.get('input_case_sha256')!=f.digest(source_case):
            raise ValueError('Detection belongs to a different OCR source')
        if (rel['image_id']!=ocr['image_id'] or rel['ocr_run_id']!=ocr['run_id'] or rel['ocr_sha256']!=f.digest(ocr)):
            raise ValueError('Relations belong to a different OCR source')
        audit=infer(ocr,rgb,labels,roles)
        final=result['configs']['FULL']
        for r in audit['relations']:
            if r['status']!='accepted':
                continue
            for proposal in r['proposed_candidates']:
                same=next((p for p in final if f.signature(p)==f.signature(proposal)),None)
                if same:
                    r['candidate_ids'].append(same['id']);actions.append(dict(action='retained_duplicate',candidate_id=same['id'],relation_id=r['id']));continue
                rr=set(f.refs(proposal['segments']))
                overlaps=[p for p in final if p['type'] in ('EMAIL','ACCOUNT_ID') and rr&set(f.refs(p['segments']))]
                if overlaps:
                    r.update(status='deferred',reason='preserve_existing_email_or_account_overlap')
                    r['evidence']['overlap_candidate_ids']=[p['id'] for p in overlaps];continue
                p=copy.deepcopy(proposal);p.update(id='af-c-'+f.digest(f.signature(p))[:20],
                    field_relation_ids=[r['id']],account_stage_version=VERSION)
                if p['id'] in {x['id'] for x in final}:
                    raise ValueError('Candidate ID collision')
                final.append(p);r['candidate_ids'].append(p['id'])
                actions.append(dict(action='added',candidate_id=p['id'],relation_id=r['id']))
        audit['upstream_account_relations']=copy.deepcopy([r for r in rel.get('relations',[]) if r['type']=='ACCOUNT_ID'])
        rel['relations'].extend(copy.deepcopy(audit['relations']))
        # Preserve old configs, including FIELD_CONTACTS. This producer consumes final FULL.
        result['config_statuses']['ACCOUNT']='ok'
    except Exception as ex:
        result=copy.deepcopy(baseline_detection);rel=copy.deepcopy(baseline_relations)
        result['config_statuses']['ACCOUNT']='failure'
        audit=dict(version=VERSION,status='failure',relations=[],errors=[dict(type=type(ex).__name__,message=str(ex))])
        actions=[dict(action='retained_on_account_failure',candidate_id=p['id']) for p in result['configs']['FULL']]
    audit.update(actions=actions,seconds=time.perf_counter()-started)
    rel['account_stage']=audit
    result['account_stage']=dict(version=VERSION,status=audit['status'],actions=actions)
    before=[p for p in baseline_detection['configs']['FULL'] if p['type']!='ACCOUNT_ID']
    after=[p for p in result['configs']['FULL'] if p['type']!='ACCOUNT_ID']
    if before!=after:
        raise AssertionError('Non-account candidates changed')
    valid={p['id'] for p in result['configs']['FULL']}
    if any(set(r.get('candidate_ids',[]))-valid for r in rel.get('relations',[])):
        raise ValueError('Relationship references absent final candidate')
    return result,rel
