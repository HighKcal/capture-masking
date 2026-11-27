"""Authorized saved/live OCR -> existing NER -> fragment pixel masks, local only."""
import argparse
import copy
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
from PIL import ImageDraw
from support import (HERE, OCR, DATA, ASSETS, NER_PY, SAVED, read, write, sha, digest,
                     load_manifest, decode, finalize, overlay, validate_output,
                     ocr_runner, confined_output, eval_rows, verify_freeze)
from link import bind_ocr, map_candidates, save_masked

def existing_inputs():
    items = []
    for runpath in SAVED:
        run = read(runpath)
        if run['preparation_status'] != 'ready' or sha(run['manifest']) != run['manifest_sha256']:
            raise ValueError('Saved run/manifest mismatch')
        rows = {r['image_id']: r for r in load_manifest(run['manifest'])}
        seen = set()
        for record in run['records']:
            iid = record['image_id']
            if iid in seen or iid not in rows:
                raise ValueError('Saved output identity mismatch')
            seen.add(iid)
            path = (runpath.parent / record['output']).resolve()
            if not path.is_relative_to(runpath.parent):
                raise ValueError('Unsafe stored output path')
            ocr = read(path)
            if ocr['status'] != record['status'] or len(ocr['units']) != record['units']:
                raise ValueError('Saved record mismatch')
            if ocr['engine'] != run['metadata']:
                raise ValueError('Saved engine mismatch')
            rgb, _ = decode(rows[iid])
            transforms = read(path.parent / 'transforms.json')
            identity, case = bind_ocr(ocr, rows[iid], transforms, rgb, run['run_id'])
            items.append(dict(row=rows[iid], ocr=ocr, transforms=transforms, rgb=rgb,
                native=read(path.parent / 'native.json'), identity=identity, case=case,
                source=dict(run=str(runpath), run_sha256=sha(runpath), output=str(path),
                    output_sha256=sha(path), transforms_sha256=sha(path.parent/'transforms.json'),
                    native_sha256=sha(path.parent/'native.json'), source_sha256=run['source_sha256'])))
        if seen != set(rows):
            raise ValueError('Missing saved image output')
    if len(items) != 7 or len({i['row']['image_id'] for i in items}) != 7:
        raise ValueError('Expected four development, one blank, two reference images')
    return items

def live_inputs(phase, target):
    if phase == 'eval':
        rows = eval_rows(DATA / 'eval/manifest.jsonl')
    else:
        rows = [next(r for r in load_manifest(DATA/'dev/manifest.jsonl') if r['image_id']=='dev-01'),
                next(r for r in load_manifest(DATA/'checks/manifest.jsonl') if r['image_id']=='check-01')]
    started = time.perf_counter()
    from adapters.rapid import Adapter
    reference_metadata = read(SAVED[0])['metadata']
    adapter = Adapter(ASSETS)
    if adapter.metadata != reference_metadata:
        write(target / 'unexpected-ocr-metadata.json', adapter.metadata)
        raise ValueError('Live OCR configuration differs from pinned saved baseline')
    run_id = 'rapid-link-' + uuid.uuid4().hex
    meta = dict(run_id=run_id, phase=phase, metadata=adapter.metadata,
        load_seconds=time.perf_counter()-started, python=sys.version, executable=sys.executable,
        platform=platform.platform(), records=[])
    write(target / 'ocr-run.json', meta)
    items = []
    for row in rows:
        image_target = target / row['image_id']
        image_target.mkdir()
        started = time.perf_counter()
        rgb, transforms = decode(row)
        decode_seconds = time.perf_counter()-started
        started = time.perf_counter()
        try:
            result = adapter.infer(rgb)
        except Exception as ex:
            result = dict(status='failure', units=[], errors=[dict(stage='inference',type=type(ex).__name__,message=str(ex))],native={},transforms=[])
        inference_seconds = time.perf_counter()-started
        write(image_target / 'native.json', result.get('native', result))
        transforms += result.get('transforms', [])
        write(image_target / 'transforms.json', transforms)
        try:
            ocr = finalize(result, row, run_id, adapter.metadata)
        except Exception as ex:
            ocr = finalize(dict(status='failure',units=[],errors=[dict(stage='adapter_validation',type=type(ex).__name__,message=str(ex))]), row, run_id, adapter.metadata)
        ocr['metrics'].update(inference_seconds=inference_seconds, decode_seconds=decode_seconds)
        write(image_target/'ocr.json', ocr)
        identity, case = bind_ocr(ocr, row, transforms, rgb, run_id)
        items.append(dict(row=row, ocr=ocr, transforms=transforms, rgb=rgb,
            native=result.get('native',result), identity=identity, case=case,
            source=dict(mode='fresh', run_id=run_id)))
        meta['records'].append(dict(image_id=row['image_id'], status=ocr['status'], units=len(ocr['units']),
            inference_seconds=inference_seconds, decode_seconds=decode_seconds, ocr_sha256=sha(image_target/'ocr.json')))
        write(target / 'ocr-run.json', meta)
    return items

def ner(cases, output):
    job = output / 'detector-input.json'
    write(job, dict(schema='ocr-link-ner-input-v1', cases=cases))
    command = [str(NER_PY), '-B', '-X', 'utf8', str(HERE/'ner_worker.py'), '--input',str(job),'--output',str(output/'ner')]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    started = time.perf_counter()
    with (output/'ner-process.log').open('w',encoding='utf-8') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    write(output/'ner-process.json',dict(command=command,exit_code=completed.returncode,seconds=time.perf_counter()-started))
    if completed.returncode:
        raise RuntimeError('NER worker failed; preserved local log')
    run = read(output/'ner/run.json')
    if run['input_sha256'] != sha(job) or len(run['records']) != len(cases):
        raise ValueError('NER response job mismatch')
    byid = {}
    for rec in run['records']:
        path = output/'ner'/rec['output']
        result = read(path)
        if sha(path) != rec['sha256'] or result['id'] != rec['id'] or result['id'] in byid:
            raise ValueError('NER result integrity mismatch')
        byid[result['id']] = result
    for case in cases:
        if byid[case['id']]['input_case_sha256'] != digest(case):
            raise ValueError('NER case content mismatch')
    return byid

def process(mode, phase, output):
    if phase == 'eval':
        freeze = verify_freeze()
        if mode != 'live':
            raise ValueError('Evaluation requires live inference')
    output = confined_output(output)
    output.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    with ocr_runner.python_network_guard():
        items = existing_inputs() if mode == 'reuse' else live_inputs(phase, output)
    for item in items:
        dst = output/item['row']['image_id']
        dst.mkdir(exist_ok=True)
        write(dst/'ocr.json', item['ocr'])
        write(dst/'native.json', item['native'])
        write(dst/'transforms.json', item['transforms'])
        write(dst/'source.json',dict(**item['source'], input_path=str(item['row']['_path']),
             input_manifest=item['row']['_manifest'], identity=item['identity']))
    results = ner([i['case'] for i in items], output)
    run = dict(mode=mode, phase=phase, records=[], timing_scope='saved OCR excluded' if mode=='reuse' else 'fresh OCR + separate NER process',
               freeze_sha256=sha(HERE/'freeze.json') if phase=='eval' else None)
    for item in items:
        started = time.perf_counter()
        dst = output/item['row']['image_id']
        detected = results[item['case']['id']]
        linked, mask = map_candidates(item['ocr'],item['identity'],item['case'],detected)
        write(dst/'detection.json',detected)
        linked['pixel_verification'] = save_masked(item['rgb'],mask,dst)
        linked['mask_png_sha256'] = sha(dst/'mask.png')
        linked['masked_png_sha256'] = sha(dst/'masked.png')
        write(dst/'linked.json',linked)
        overlay(item['rgb'],item['ocr'],dst/'ocr-overlay.png')
        preview = item['rgb'].copy()
        draw = ImageDraw.Draw(preview)
        for i,p in enumerate(linked['candidates']):
            for r in p['regions']:
                for poly in r['polygons']:
                    pts = [tuple(x) for x in poly]
                    draw.line(pts+[pts[0]],fill=(0,220,100),width=2)
                    draw.text(pts[0],str(i+1),fill=(255,70,0))
        preview.save(dst/'candidate-regions.png')
        run['records'].append(dict(image_id=item['row']['image_id'], case_id=item['case']['id'],
            status=linked['status'], state=linked['state'], candidates=len(linked['candidates']),
            shared_pairs=len(linked['shared_regions']), detection_ms=detected['timing']['single_path_ms'],
            link_mask_seconds=linked['link_mask_seconds'], link_render_record_seconds=time.perf_counter()-started))
        write(output/'run.json',run)
    # Written only after every prediction and actual pixel image has been saved.
    files = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'predictions-sealed.json',dict(phase=phase, image_ids=[i['row']['image_id'] for i in items],files=files))
    print(__import__('json').dumps(dict(output=str(output),records=run['records'])))

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['reuse','live'],required=True)
    ap.add_argument('--phase',choices=['development','smoke','eval'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if (a.mode,a.phase) not in {('reuse','development'),('live','smoke'),('live','eval')}:
        ap.error('Unsupported mode/phase pair')
    existed=a.output.exists()
    try:
        process(a.mode,a.phase,a.output)
    except Exception as ex:
        # A batch/identity/preparation failure invalidates the run; partial files stay.
        if not existed and a.output.exists():
            dst=confined_output(a.output)
            write(dst/'execution-error.json',dict(status='failure',stage='execution_or_integrity',
                exception=type(ex).__name__,message=str(ex),predictions_sealed=False))
        raise
