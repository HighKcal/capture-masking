"""ACCOUNT-02 v1.0.1: omit new role search when no account pair can witness it."""
import copy
from pathlib import Path
import re
import sys
import time


sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'ocr-integration-v0'))
import dependencies as dep
if Path(dep.__file__).resolve()!=ROOT.parent/'ocr-integration-v0/dependencies.py':
    raise ImportError('Foreign dependency loader')
f=dep.field
old=dep.load('account01_joint_dependency',ROOT.parent/'account-field-link-v0/account_stage.py')
geo=dep.load('account02_joint_geometry',ROOT.parent/'account-field-link-v1/joint_geometry.py')
VERSION='account-field-joint-v1.0.1'
SETTINGS=dict(version=VERSION,labels='bounded_head_and_modifier_structure',roles='paired_fields_and_closed_cells',
              correction='counterevidence_only',max_pairs=4096,max_edges=256,
              max_group_gap_heights=3.0,max_role_comparisons=8192)
HEAD=re.compile(r'(?P<prefix>[가-힣A-Za-z][가-힣A-Za-z \t]{0,29}?)?(?P<head>계정(?:[ \t]*(?:아이디|ID))?|아이디|ID)$',re.I)
ROLE_END=re.compile(r'(?:자|인|원|회원|담당)$')
VALUE=re.compile(r'@?\w+(?:[.\-]\w+)*')
MACHINE=re.compile(r'(?=.*[A-Za-z0-9])(?=.*[0-9_.@-])@?\w+(?:[.\-]\w+)*$')
SHORT_TITLE=re.compile(r'[가-힣A-Za-z][가-힣A-Za-z \t]{0,23}[:：=＝]?')


def semantic_label(text):
    m=HEAD.fullmatch(text.strip(' \t'))
    if not m:
        return None
    prefix=(m['prefix'] or '').strip();head=re.sub(r'[ \t]','',m['head']).lower()
    if len(prefix.split())>3:
        return None
    if head.startswith('계정'):
        return 'account_head'
    if not prefix and head=='아이디':
        return 'identifier_head'
    legacy=old.parse_label(dict(id='semantic',text=text),True)
    if legacy:
        return 'established_authentication_head'
    if head=='아이디' and ROLE_END.search(prefix):
        return 'role_or_purpose_identifier'
    return None


def parse(u, expanded):
    if not expanded:
        p=old.parse_label(u,True)
        if p:
            return dict(p,decision='recognized',explicit=bool(re.search(r'[:：=＝]',u['text'])),origin='ACCOUNT01')
        return None
    text=u['text'];left=len(text)-len(text.lstrip(' \t'));trim=text.strip(' \t')
    # Delimiter has one unambiguous original position. No normalization changes offsets.
    delimiter=re.fullmatch(r'(?P<label>[^:：=＝\n]+?)[ \t]*(?P<sep>[:：=＝])[ \t]*(?P<value>[^\n]*)',trim)
    if delimiter:
        label=delimiter['label'].rstrip(' \t');kind=semantic_label(label)
        if not kind:
            return None
        a=left+delimiter.start('label');z=a+len(label);val=delimiter['value'].strip(' \t')
        if not val:
            return dict(label_segments=[f.span(u,a,z)],value_segments=[],inline=False,explicit=True,semantic=kind,decision='recognized')
        if VALUE.fullmatch(val):
            va=left+delimiter.start('value')+len(delimiter['value'])-len(delimiter['value'].lstrip(' \t'))
            return dict(label_segments=[f.span(u,a,z)],value_segments=[f.span(u,va,va+len(val))],inline=True,explicit=True,semantic=kind,decision='recognized')
        return None
    kind=semantic_label(trim)
    if kind:
        return dict(label_segments=[f.span(u,left,left+len(trim))],value_segments=[],inline=False,explicit=False,semantic=kind,decision='recognized')
    # Unpunctuated inline needs a complete machine-like value, never arbitrary prose.
    for space in re.finditer(r'[ \t]+',trim):
        label=trim[:space.start()];val=trim[space.end():]
        kind=semantic_label(label)
        if kind and MACHINE.fullmatch(val):
            return dict(label_segments=[f.span(u,left,left+len(label))],value_segments=[f.span(u,left+space.end(),left+len(trim))],
                        inline=True,explicit=False,semantic=kind,decision='recognized')
    return None


def _overlap(a,b):
    l,t,r,z=a;ll,tt,rr,zz=b
    return min(r,rr)>max(l,ll) and min(z,zz)>max(t,tt)


def infer(ocr,rgb,baseline_relations,labels=True,roles=True):
    units,skipped=f.project_source(ocr);byid={u['id']:u for u in units}
    if rgb is not None and rgb.size!=(ocr['input']['width'],ocr['input']['height']):
        raise ValueError('Original RGB dimensions mismatch')
    if ocr['status'] not in ('ok','partial_failure'):
        raise ValueError('OCR stage unavailable')
    enclosed,lines=geo.analyze(rgb,units) if roles else ({},[])
    parsed={u['id']:p for u in units if (p:=parse(u,labels))}
    pair_cache={}
    def geometry(a,b):
        key=(a['id'],b['id'])
        if key in pair_cache:return pair_cache[key]
        if len(pair_cache)>=SETTINGS['max_pairs']:raise ValueError('Account pair exploration limit exceeded')
        direction,reason,e=f.orient(a,b,label=True)
        if not direction and roles and a['id'] in enclosed and b['id'] in enclosed and b['_g']['cy']>a['_g']['cy']:
            gap=(b['_g']['box'][1]-a['_g']['box'][3])/max(a['_g']['h'],b['_g']['h'])
            if 0<=gap<=SETTINGS['max_group_gap_heights'] and geo.connector(rgb,enclosed[a['id']],enclosed[b['id']]):
                direction,reason,e='below','connected_closed_cells',dict(gap_heights=gap)
        if not direction:
            pair_cache[key]=None;return None
        blocked=f.blocking(a,b,units,direction,rgb)
        if _overlap(a['_g']['box'],b['_g']['box']):blocked='label_value_regions_overlap'
        evidence=dict(direction=direction,geometry_reason=reason,**e,original_boundary_reason=blocked)
        if roles and blocked:
            proof=geo.boundary_evidence(a,b,units,rgb,enclosed,lines,direction,blocked)
            if proof:
                blocked=None;evidence['cell_boundary_exception']=proof
        evidence['blocked']=blocked;pair_cache[key]=evidence;return evidence
    # A semantic-looking string in an input cell can be the value of an outside label.
    suppress=[]
    for vid in parsed:
        if vid not in enclosed:continue
        for lid,p in parsed.items():
            if lid==vid or p['inline']:continue
            e=geometry(byid[lid],byid[vid])
            if e and not e['blocked'] and e['direction']=='right' and lid not in enclosed:
                suppress.append(vid);break
    label_audit=[]
    for key,p in parsed.items():
        label_audit.append(dict(p,decision='value_cell' if key in suppress else 'recognized'))
    active={key:p for key,p in parsed.items() if key not in suppress}
    # Potential account pairs, before role conflicts. Used as witnesses, never nearest choice.
    account_pairs=[]
    for lid,p in active.items():
        if p['inline']:continue
        for v in units:
            if v['id']==lid or (v['id'] in active and v['id'] not in enclosed):continue
            s=f.trimmed_span(v)
            if not s or not VALUE.fullmatch(s['text']):continue
            e=geometry(byid[lid],v)
            if e and not e['blocked']:account_pairs.append((byid[lid],v,s,e))
    proofs={}
    if roles:
        # Carry only verified same-source prior role evidence, not gold or label word lists.
        for p in baseline_relations.get('account_stage',{}).get('role_evidence',[]):
            refs=p['label_segments']+p['value_segments']
            if all(s['block_id'] in byid and byid[s['block_id']]['text'][s['start']:s['end']]==s['text'] for s in refs):
                proofs[p['label_segments'][0]['block_id']]=dict(p,origin='verified_ACCOUNT01_role')
        comparisons=0
        # New role proofs require a witness from account_pairs. The verified
        # ACCOUNT01 proofs above still matter when this list is empty.
        for title in (units if account_pairs else ()):
            if title['id'] in active or title['id'] in enclosed or not SHORT_TITLE.fullmatch(title['text'].strip()):continue
            possibilities=[]
            for val in units:
                if val['id']==title['id'] or val['id'] in active:continue
                sv=f.trimmed_span(val)
                if not sv or len(sv['text'])>40:continue
                e=geometry(title,val)
                if not e or e['blocked']:continue
                witnesses=[]
                for anchor,av,ss,ae in account_pairs:
                    comparisons+=1
                    if comparisons>SETTINGS['max_role_comparisons']:raise ValueError('Account role comparison limit exceeded')
                    if ae['direction']==e['direction'] and old.repeated_layout(title,val,anchor,av,e['direction']):
                        # Generic words require explicit value cells in both rows/columns.
                        typed=bool(old.OTHER_HEADING.fullmatch(title['text'].strip()) or
                                   (old.DATE_HEADING.fullmatch(title['text'].strip()) and old.DATE_VALUE.fullmatch(sv['text'])))
                        if (val['id'] in enclosed and av['id'] in enclosed) or typed:
                            witnesses.append(dict(label_segments=active[anchor['id']]['label_segments'],value_segments=[ss]))
                if witnesses:
                    possibilities.append(dict(label_segments=[f.trimmed_span(title)],value_segments=[sv],parallel_account_pairs=witnesses,
                                              evidence=e,value_cell=enclosed.get(val['id']),origin='joint_role_pair'))
                if len(possibilities)>1:break
            if len(possibilities)==1:proofs[title['id']]=possibilities[0]
    edges=[]
    def record(label,values,status,reason,evidence):
        identity=dict(image_id=ocr['image_id'],ocr_run_id=ocr['run_id'],kind='label_value',type='ACCOUNT_ID',
                      label_segments=label['label_segments'],value_segments=values)
        rid='af2-r-'+f.digest(identity)[:20]
        existing=next((r for r in edges if r['id']==rid),None)
        if existing:
            if evidence.get('group'):
                existing['evidence'].update(evidence)
                existing.setdefault('prior_reason',existing['reason'])
                existing.update(status=status,reason=reason)
            return
        edges.append(dict(identity,id=rid,source=VERSION,status=status,reason=reason,evidence=evidence,candidate_ids=[],supported_candidate_ids=[],proposed_candidates=[]))
        if len(edges)>SETTINGS['max_edges']:raise ValueError('Account relation limit exceeded')
    for lid,p in active.items():
        if p['inline']:
            record(p,p['value_segments'],'accepted','inline_account_head',dict(explicit=p['explicit']));continue
        for v in units:
            if v['id']==lid:continue
            s=f.trimmed_span(v)
            if not s or not VALUE.fullmatch(s['text']):continue
            e=geometry(byid[lid],v)
            if not e:continue
            if e['blocked']:
                record(p,[s],'rejected',e['blocked'],e);continue
            if v['id'] in proofs:
                record(p,[s],'rejected','independent_other_field_heading',dict(e,role_proof=proofs[v['id']]));continue
            owners=[proof for proof in proofs.values() if proof['value_segments']==[s]]
            if owners:
                record(p,[s],'deferred','independent_other_field_value',dict(e,other_owners=owners));continue
            if v['id'] in active:
                record(p,[s],'deferred','semantic_label_value_ambiguity',e);continue
            if (not p['explicit'] and e['direction']=='right' and re.fullmatch('[가-힣]+',s['text']) and
                v['id'] not in enclosed and not any(w['label_segments']==p['label_segments'] and w['value_segments']==[s]
                    for proof in proofs.values() for w in proof.get('parallel_account_pairs',[]))):
                record(p,[s],'deferred','unpunctuated_title_value_ambiguity',e);continue
            record(p,[s],'accepted','joint_account_field',e)
    # A centered field label over a pair of closed choices supports both values, not ownership.
    if roles:
        for lid,p in active.items():
            if p['inline']:continue
            label=byid[lid];l,t,r,b=label['_g']['box'];lh=label['_g']['h'];cx=(l+r)/2
            options=[]
            for v in units:
                s=f.trimmed_span(v)
                if v['id'] in active or v['id'] in proofs or v['id'] not in enclosed or not s or not VALUE.fullmatch(s['text']):continue
                if any(proof['value_segments']==[s] for proof in proofs.values()):continue
                vl,vt,vr,vb=v['_g']['box'];h=max(lh,v['_g']['h']);gap=(vt-b)/h
                if .0<=gap<=SETTINGS['max_group_gap_heights'] and abs((vl+vr)/2-cx)<=8*h:
                    options.append((v,s,gap))
            if len(options)==2:
                (a,sa,ga),(c,sc,gc)=sorted(options,key=lambda x:x[0]['_g']['box'][0])
                ac=sum(a['_g']['box'][::2])/2;cc=sum(c['_g']['box'][::2])/2;h=max(lh,a['_g']['h'],c['_g']['h'])
                if ac<cx<cc and abs((ac+cc)/2-cx)<=h and abs(a['_g']['cy']-c['_g']['cy'])<=.45*h:
                    proof=geo.choice_group(label,[a,c],units,rgb,enclosed,lines)
                    if not proof:continue
                    ev=dict(group='centered_closed_choices',members=[sa,sc],cells=[enclosed[a['id']],enclosed[c['id']]],connector_proof=proof)
                    record(p,[sa],'deferred','competing_relationships',ev);record(p,[sc],'deferred','competing_relationships',ev)
        # A value between two aligned account owners is an explicit ownership ambiguity.
        for upper in list(edges):
            if upper['status']!='accepted' or upper['evidence'].get('direction')!='below':continue
            vs=upper['value_segments'][0];v=byid[vs['block_id']]
            if v['id'] not in enclosed:continue
            for lid,p in active.items():
                if p['inline'] or p['label_segments']==upper['label_segments']:continue
                lower=byid[lid];h=max(v['_g']['h'],lower['_g']['h']);gap=(lower['_g']['box'][1]-v['_g']['box'][3])/h
                if (0<=gap<=SETTINGS['max_group_gap_heights'] and abs(lower['_g']['box'][0]-byid[upper['label_segments'][0]['block_id']]['_g']['box'][0])<=.8*h and
                    geo.connector(rgb,enclosed.get(lid),enclosed[v['id']])):
                    record(p,[vs],'deferred','competing_relationships',dict(group='opposed_account_owners',value_cell=enclosed[v['id']]))
                    upper.update(prior_reason=upper['reason'],status='deferred',reason='competing_relationships')
                    upper['evidence']['group']='opposed_account_owners'
    potential=[r for r in edges if r['status']=='accepted' or r['reason'] in ('competing_relationships','unpunctuated_title_value_ambiguity','semantic_label_value_ambiguity')]
    conflicts=set()
    for i,a in enumerate(potential):
        for b in potential[i+1:]:
            if ((set(f.refs(a['label_segments']))&set(f.refs(b['label_segments'])) or set(f.refs(a['value_segments']))&set(f.refs(b['value_segments']))) and
                (a['label_segments']!=b['label_segments'] or a['value_segments']!=b['value_segments'])):
                conflicts.update((a['id'],b['id']))
    for r in edges:
        if r['id'] in conflicts:r.update(prior_reason=r['reason'],status='deferred',reason='competing_relationships')
        independent=(r['reason']=='competing_relationships' and r['evidence'].get('group') in ('centered_closed_choices','opposed_account_owners') and
                     r['value_segments'][0]['block_id'] in enclosed and
                     r.get('prior_reason') not in ('unpunctuated_title_value_ambiguity','semantic_label_value_ambiguity') and
                     r['value_segments'][0]['block_id'] not in active)
        if r['status']=='accepted' or independent:
            proposed,view=f.consume('label_value','ACCOUNT_ID',r['value_segments'],r['label_segments'],units)
            r.update(proposed_candidates=proposed,view=view,independent_value_support=independent)
            if not proposed and r['status']=='accepted':r.update(status='rejected',reason='existing_contact_policy_rejected')
    return dict(version=VERSION,status='ok',labels=label_audit,role_evidence=list(proofs.values()),cells=list(enclosed.values()),
                relations=edges,skipped_units=skipped,settings=copy.deepcopy(SETTINGS),pair_checks=len(pair_cache))


def supplement(ocr,baseline_detection,baseline_relations,rgb=None,*,labels=True,roles=True,corrections=True):
    started=time.perf_counter();result=copy.deepcopy(baseline_detection);rel=copy.deepcopy(baseline_relations)
    try:
        case=dict(id=result['id'],blocks=[dict(id=u['id'],text=u['text']) for u in ocr['units']],links=[])
        if result['input_case_sha256']!=f.digest(case):raise ValueError('Foreign detection source')
        if rel['image_id']!=ocr['image_id'] or rel['ocr_run_id']!=ocr['run_id'] or rel['ocr_sha256']!=f.digest(ocr):raise ValueError('Foreign relationship source')
        audit=infer(ocr,rgb,rel,labels,roles);audit['ablation']=dict(labels=labels,roles=roles,corrections=corrections)
        final=copy.deepcopy(result['configs']['FULL']);actions=[];removed=set()
        if corrections:
            for candidate in list(final):
                if candidate['type']!='ACCOUNT_ID' or any(s['text'].startswith('@') for s in candidate['segments']):continue
                rr=set(f.refs(candidate['segments']))
                proof=next((p for p in audit['role_evidence'] if rr and rr<=set(f.refs(p['label_segments']))),None)
                if proof:
                    final.remove(candidate);removed.add(candidate['id'])
                    actions.append(dict(action='excluded',candidate=copy.deepcopy(candidate),reason='independent_other_field_heading',proof=proof,replacement_ids=[]))
        for r in audit['relations']:
            if r['status']!='accepted' and not r.get('independent_value_support'):continue
            for proposed in r['proposed_candidates']:
                same=next((p for p in final if f.signature(p)==f.signature(proposed)),None)
                if same:
                    pid=same['id'];action='retained_duplicate'
                else:
                    overlap=[p for p in final if p['type'] in ('EMAIL','ACCOUNT_ID') and set(f.refs(p['segments']))&set(f.refs(proposed['segments']))]
                    if overlap:
                        r.update(prior_reason=r['reason'],status='deferred',reason='preserve_existing_email_or_partial_account_overlap',independent_value_support=False)
                        r['evidence']['overlap_candidate_ids']=[p['id'] for p in overlap];continue
                    candidate=copy.deepcopy(proposed);pid='af2-c-'+f.digest(f.signature(candidate))[:20]
                    if pid in {p['id'] for p in final}:raise ValueError('Candidate ID collision')
                    candidate.update(id=pid,account_joint_version=VERSION,account_support_relation_ids=[r['id']],
                                     relationship_status=r['status'])
                    final.append(candidate);action='added'
                r['candidate_ids' if r['status']=='accepted' else 'supported_candidate_ids'].append(pid)
                actions.append(dict(action=action,candidate_id=pid,relation_id=r['id'],relationship_status=r['status']))
        # Original account relations are an audit trail, not a second consumer graph.
        previous=[r for r in rel['relations'] if r['type']=='ACCOUNT_ID']
        for action in actions:
            if action['action']!='excluded':continue
            old_labels=[set(f.refs(r['label_segments'])) for r in previous if action['candidate']['id'] in r.get('candidate_ids',[])]
            action['replacement_ids']=sorted({pid for r in audit['relations'] if r['status']=='accepted' and
                any(set(f.refs(r['label_segments']))==old_label for old_label in old_labels) for pid in r['candidate_ids']})
        addressed={tuple(f.refs(r['label_segments'])) for r in audit['relations']}
        current=[r for r in rel['relations'] if r['type']!='ACCOUNT_ID']
        for r in previous:
            if not set(r.get('candidate_ids',[]))&removed and tuple(f.refs(r['label_segments'])) not in addressed:
                current.append(r)
        current.extend(copy.deepcopy(audit['relations']))
        ids={p['id'] for p in final}
        if any(set(r.get('candidate_ids',[]))-ids for r in current):raise ValueError('Dangling active relationship consumer')
        audit.update(actions=actions,previous_account_relations=copy.deepcopy(previous),removed_candidate_ids=sorted(removed))
        for p in baseline_detection['configs']['FULL']:
            if p['type']=='ACCOUNT_ID' and p['id'] not in removed:
                actions.append(dict(action='retained',candidate_id=p['id'],reason='no_specific_counterevidence'))
        result['configs']['FULL']=final;result['config_statuses']['ACCOUNT2']='ok';rel['relations']=current
    except Exception as ex:
        result=copy.deepcopy(baseline_detection);rel=copy.deepcopy(baseline_relations)
        result['config_statuses']['ACCOUNT2']='failure'
        audit=dict(version=VERSION,status='failure',errors=[dict(type=type(ex).__name__,message=str(ex))],relations=[],actions=[dict(action='retained_on_failure',candidate_id=p['id']) for p in result['configs']['FULL']])
    audit['seconds']=time.perf_counter()-started;rel['account_joint']=audit
    result['account_joint']=dict(version=VERSION,status=audit['status'],actions=copy.deepcopy(audit['actions']))
    if [p for p in result['configs']['FULL'] if p['type']!='ACCOUNT_ID']!=[p for p in baseline_detection['configs']['FULL'] if p['type']!='ACCOUNT_ID']:
        raise AssertionError('Non-account candidate mutation')
    return result,rel
