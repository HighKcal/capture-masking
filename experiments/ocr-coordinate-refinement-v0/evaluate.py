"""Post-seal scoring only. Foreground never enters refinement or inference."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from coord_common import HERE, LINK, DATA, read, write, sha, jsonl, confined, verify_files, module
from refine import candidate_mask, METHODS

legacy=module('readonly_link_score',LINK/'score.py')

def load_mask(path,size):
    with Image.open(path) as im:
        arr=np.array(im)
        if im.mode!='L' or im.size!=size or np.any((arr!=0)&(arr!=255)):
            raise ValueError('Invalid saved mask')
    return arr!=0

def verify_pixels(rgb,mask,path):
    with Image.open(path) as im:
        saved=np.array(im)
        if im.mode!='RGB' or im.size!=rgb.size:raise ValueError('Invalid saved RGB')
    source=np.array(rgb)
    if np.any(saved[mask]!=0) or not np.array_equal(source[~mask],saved[~mask]):
        raise ValueError('Saved PNG does not implement mask')

def metrics(mask,target,text):
    return dict(target_pixels=int(target.sum()),covered_target_pixels=int((mask&target).sum()),
                missed_target_pixels=int((target&~mask).sum()),
                non_target_text_pixels=int((mask&text&~target).sum()),region_area_pixels=int(mask.sum()))

def verify_seal(root):
    seal=read(root/'predictions-sealed.json');verify_files(seal['files'],root)
    run=read(root/'run.json')
    if 'run.json' not in seal['files'] or run['phase']!=seal['phase']:
        raise ValueError('Missing sealed run or phase mismatch')
    if len(set(seal['image_ids']))!=len(seal['image_ids']):raise ValueError('Duplicate sealed image IDs')
    if len(set(run['methods']))!=len(run['methods']) or not {'coarse','detail'}<=set(run['methods'])<=set(METHODS):
        raise ValueError('Unsupported method set')
    if run['phase']=='new' and tuple(run['methods'])!=METHODS:raise ValueError('Incomplete frozen methods')
    expected_methods={k for m in run['methods'] for k in ([m] if m=='coarse' else [m,m+'-fallback'])}
    if [r['image_id'] for r in run['records']]!=seal['image_ids']:raise ValueError('Seal/run mismatch')
    for iid in seal['image_ids']:
        if not iid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in iid):
            raise ValueError('Unsafe sealed image ID')
        required=[f'{iid}/{n}.json' for n in ('ocr','linked','detection','source','refined','detail','render-index')]
        if any(n not in seal['files'] for n in required):raise ValueError('Incomplete prediction seal')
        ix=read(root/iid/'render-index.json')
        if set(ix)!=expected_methods:raise ValueError('Missing or extra rendered method')
        refined=read(root/iid/'refined.json');linked=read(root/iid/'linked.json')
        if refined['identity']['image_id']!=iid:raise ValueError('Mixed sealed image')
        signature=lambda ps:[(p['id'],p['type'],p['segments']) for p in ps]
        if signature(refined['candidates'])!=signature(linked['candidates']):raise ValueError('Changed candidate semantics')
        ids=[p['id'] for p in refined['candidates']]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate candidate ID')
        names=[p['id'] for p in refined['candidates'] if p['type']=='NAME']
        expected_pairs=[(a,b) for i,a in enumerate(names) for b in names[i+1:]]
        for key,group in ix.items():
            if key!=group['method']+('-fallback' if group['fallback'] else ''):raise ValueError('Method key mismatch')
            if [r['candidate_id'] for r in group['candidates']]!=ids:raise ValueError('Missing/duplicate solo candidate')
            if [(p['a'],p['b']) for p in group['pairs']]!=expected_pairs:raise ValueError('Missing/duplicate name pair')
            dirs=[r['directory'] for r in group['candidates']]+[group['union_directory'],f'{key}/none']
            for pair in group['pairs']:dirs.extend(pair[k] for k in ('first','second','both','neither'))
            for d in dirs:
                if any(f'{iid}/{d}/{n}.png' not in seal['files'] for n in ('mask','masked')):
                    raise ValueError('Missing selected-pixel output in seal')
    return seal,run

def evaluate(predictions,output):
    root=Path(predictions).resolve();seal,run=verify_seal(root)
    out=confined(output);out.mkdir(parents=True,exist_ok=False)
    # Gold is opened only after all prediction files and selection masks verify.
    if run['phase']=='new':
        goldroot=HERE/'heldout';golds=jsonl(goldroot/'ground_truth/eval.jsonl')
    else:
        goldroot=DATA
        golds=sum((jsonl(DATA/'ground_truth'/f'{split}.jsonl') for split in ('dev','eval','check')),[])
    gold_by_id={g['image_id']:g for g in golds}
    legacy.FIXTURES=goldroot
    summaries={}; details=[]
    for rec in run['records']:
        iid=rec['image_id'];d=root/iid
        result=read(d/'refined.json');linked=read(d/'linked.json');ocr=read(d/'ocr.json')
        det=read(d/'detection.json');source=read(d/'source.json');index=read(d/'render-index.json')
        with Image.open(source['input_path']) as im:rgb=im.convert('RGB').copy()
        if sha(source['input_path'])!=result['identity']['file_sha256']:raise ValueError('Original PNG changed')
        size=rgb.size; candidates={p['id']:p for p in result['candidates']}
        if [(p['id'],p['type'],p['segments']) for p in result['candidates']] != [(p['id'],p['type'],p['segments']) for p in linked['candidates']]:
            raise ValueError('Refinement changed candidate identity or semantics')
        masks={}; unions={}
        for method,group in index.items():
            masks[method]={}; union=np.zeros(size[::-1],bool)
            for row in group['candidates']:
                mask=load_mask(d/row['directory']/'mask.png',size)
                expected,subs=candidate_mask(candidates[row['candidate_id']],size,group['method'],group['fallback'])
                if not np.array_equal(mask,expected) or subs!=row['fallback_segments']:raise ValueError('Saved solo mask differs')
                verify_pixels(rgb,mask,d/row['directory']/'masked.png')
                masks[method][row['candidate_id']]=mask;union|=mask
            actual=load_mask(d/group['union_directory']/'mask.png',size)
            if not np.array_equal(union,actual):raise ValueError('All/solo union mismatch')
            verify_pixels(rgb,actual,d/group['union_directory']/'masked.png');unions[method]=actual
            for pair in group['pairs']:
                for key,expected in [('both',masks[method][pair['a']]|masks[method][pair['b']]),('neither',np.zeros(size[::-1],bool))]:
                    actual=load_mask(d/pair[key]/'mask.png',size)
                    if not np.array_equal(actual,expected):raise ValueError('Pair selection mismatch')
                    verify_pixels(rgb,actual,d/pair[key]/'masked.png')
        statuses=Counter(p['status'] for p in candidates.values())
        reasons=Counter(s['reason'] for p in candidates.values() for s in p['segment_results'] if s['reason'])
        operations={method:dict(candidates=len(group['candidates']),region_area_pixels=int(unions[method].sum()),
                    fallback_candidates=sum(bool(r['fallback_segments']) for r in group['candidates']),
                    fallback_segments=sum(len(r['fallback_segments']) for r in group['candidates']),
                    fallback_reasons=dict(Counter(s['reason'] for r in group['candidates'] for s in r['fallback_segments'])))
                    for method,group in index.items()}
        row=dict(image_id=iid,group=rec['group'],candidate_statuses=dict(statuses),segment_reasons=dict(reasons),methods={},operations=operations)
        if iid not in gold_by_id:
            row['scope']='reference photo: no quantitative foreground truth'
            write(out/(iid+'.json'),row);details.append(row);continue
        gold=gold_by_id[iid]
        base=legacy.score_image(ocr,det,linked,gold,unions['coarse'])
        write(out/(iid+'-baseline-association.json'),base)
        targets={o['id']:legacy._load_foreground(o['foreground_mask'],iid,size) for o in gold['occurrences']}
        text=legacy._load_foreground(gold['masks']['text_foreground'],iid,size)
        pii=legacy._load_foreground(gold['masks']['pii_foreground'],iid,size)
        if targets and not np.array_equal(np.logical_or.reduce(list(targets.values())),pii):raise ValueError('Gold union mismatch')
        row['type_counts']=base['type_counts']
        row['actual_detection_exact']=sum(o['text_match']=='exact' for o in base['occurrences'])
        row['gold_occurrences']=len(base['occurrences'])
        for method,union in unions.items():
            occurrences=[]
            for occ in base['occurrences']:
                oid=occ['id'];pid=occ['candidate_id'];target=targets[oid]
                solo=masks[method][pid] if pid else np.zeros_like(union)
                coarse=masks['coarse'][pid] if pid else np.zeros_like(union)
                one=metrics(solo,target,text);allm=metrics(union,target,text)
                occurrences.append(dict(id=oid,type=occ['type'],text_match=occ['text_match'],candidate_id=pid,
                    solo=one,union=allm,solo_complete=not one['missed_target_pixels'],
                    union_complete=not allm['missed_target_pixels'],
                    newly_exposed_solo_pixels=int((target&coarse&~solo).sum()),
                    baseline_full_to_incomplete=bool(pid and not (target&~coarse).any() and (target&~solo).any()),
                    counterpart_position_overlap_zero=bool(pid and not (target&solo).any())))
            pairs=[]
            for pair in base['name_pairs']:
                aid,bid=pair['a'],pair['b'];ap,bp=pair['a_candidate_id'],pair['b_candidate_id']
                a=masks[method].get(ap,np.zeros_like(union));b=masks[method].get(bp,np.zeros_like(union))
                exact=all(next(o for o in base['occurrences'] if o['id']==oid)['text_match']=='exact' for oid in (aid,bid))
                ab=int((a&targets[bid]).sum());ba=int((b&targets[aid]).sum())
                amiss=int((targets[aid]&~a).sum());bmiss=int((targets[bid]&~b).sum())
                overlap=int((a&b).sum());independent=bool(exact and not ab and not ba and not amiss and not bmiss)
                pairs.append(dict(a=aid,b=bid,a_candidate_id=ap,b_candidate_id=bp,
                    parent_units_shared=pair['shared_unit_ids'],mask_overlap_pixels=overlap,
                    first_invades_second_pixels=ab,second_invades_first_pixels=ba,
                    first_missing_pixels=amiss,second_missing_pixels=bmiss,exact_candidates=exact,
                    independent_full=independent,strict_disjoint_full=bool(independent and overlap==0)))
            stats=metrics(union,pii,text)
            exactrows=[o for o in occurrences if o['text_match']=='exact']
            stats.update(full_occurrences=sum(o['union_complete'] for o in occurrences),
                exact_solo_complete=sum(o['solo_complete'] for o in exactrows),exact_candidates=len(exactrows),
                exact_solo_missed_pixels=sum(o['solo']['missed_target_pixels'] for o in exactrows),
                exact_solo_other_text_pixels=sum(o['solo']['non_target_text_pixels'] for o in exactrows),
                solo_regression_occurrences=sum(o['baseline_full_to_incomplete'] for o in occurrences),
                newly_exposed_solo_pixels=sum(o['newly_exposed_solo_pixels'] for o in occurrences),
                newly_exposed_union_pixels=int((pii&unions['coarse']&~union).sum()),
                name_pairs=len(pairs),independent_full_pairs=sum(p['independent_full'] for p in pairs),
                shared_parent_pairs=sum(bool(p['parent_units_shared']) for p in pairs),
                shared_parent_independent_full=sum(bool(p['parent_units_shared']) and p['independent_full'] for p in pairs),
                fallback_candidates=sum(bool(r['fallback_segments']) for r in index[method]['candidates']),
                fallback_segments=sum(len(r['fallback_segments']) for r in index[method]['candidates']))
            row['methods'][method]=dict(metrics=stats,occurrences=occurrences,name_pairs=pairs)
        write(out/(iid+'.json'),row);details.append(row)
    for group in sorted({r['group'] for r in details}):
        rows=[r for r in details if r['group']==group]
        stats=dict(images=len(rows),quantitative_images=sum('gold_occurrences' in r for r in rows),
                   candidate_statuses={s:sum(r['candidate_statuses'].get(s,0) for r in rows) for s in ('possible','uncertain','unavailable')},
                   segment_reasons=dict(sum((Counter(r['segment_reasons']) for r in rows),Counter())),methods={},operations={})
        for method in run['methods']:
            for key in ([method] if method=='coarse' else [method,method+'-fallback']):
                values=[r['methods'][key]['metrics'] for r in rows if key in r['methods']]
                stats['methods'][key]={k:sum(v.get(k,0) for v in values) for k in sorted({k for v in values for k in v})}
                stats['operations'][key]={k:sum(r['operations'][key][k] for r in rows) for k in ('candidates','region_area_pixels','fallback_candidates','fallback_segments')}
        summaries[group]=stats
    write(out/'summary.json',dict(predictions=str(root),seal_sha256=sha(root/'predictions-sealed.json'),groups=summaries,
        independent_definition='both exact candidates individually cover own foreground completely with zero partner foreground invasion; parent sharing and mask intersection separately reported',
        score_code_sha256=sha(Path(__file__))))
    print(__import__('json').dumps(summaries,ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();evaluate(a.predictions,a.output)
