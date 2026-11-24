"""Local-only OCR experiment runner; never executes the downstream NER."""
import argparse
import importlib
import json
import os
import platform
import socket
import sys
import time
import uuid
from pathlib import Path
from contextlib import contextmanager
from common import ASSET_ROOT, load_manifest, decode, error, finalize, overlay, write_json, validate_output, sha256, no_eval_path

@contextmanager
def python_network_guard():
    """Defense in depth, not a whole-OS network isolation claim."""
    old_connect,old_connect_ex,old_create=socket.socket.connect,socket.socket.connect_ex,socket.create_connection
    def blocked(*args,**kw):
        raise RuntimeError('network access prohibited during local inference')
    socket.socket.connect=blocked; socket.socket.connect_ex=blocked; socket.create_connection=blocked
    try: yield
    finally:
        socket.socket.connect=old_connect; socket.socket.connect_ex=old_connect_ex; socket.create_connection=old_create

def memory():
    try:
        import psutil
        m=psutil.Process().memory_info()
        return m._asdict()
    except ImportError: return {'unavailable':'psutil not installed'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--engine',required=True,choices=['rapid','easy','windows','vl'])
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--asset-root',type=Path,default=ASSET_ROOT)
    args=ap.parse_args()
    rows=load_manifest(args.manifest)
    args.output=no_eval_path(args.output)
    args.asset_root=no_eval_path(args.asset_root)
    args.output.mkdir(parents=True,exist_ok=False)
    run_id=args.engine+'-'+time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    log=dict(run_id=run_id,engine=args.engine,manifest=str(args.manifest.resolve()),
        manifest_sha256=sha256(args.manifest),python=sys.version,platform=platform.platform(),
        asset_root=str(args.asset_root),network_policy='local assets + Python socket guard; not whole-system proof',
        records=[],preparation_status='not_run')
    source_root=Path(__file__).resolve().parent
    log['source_sha256']={str(p.relative_to(source_root)):sha256(p) for p in [source_root/'run.py',source_root/'common.py',source_root/'adapters'/f'{args.engine}.py']}
    if args.engine=='windows':
        helper=source_root/'windows-helper/Invoke-Ocr.ps1'
        log['source_sha256'][str(helper.relative_to(source_root))]=sha256(helper)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    with python_network_guard():
        start=time.perf_counter()
        try:
            module=importlib.import_module('adapters.'+args.engine)
            adapter=module.Adapter(args.asset_root)
            log.update(preparation_status='ready',load_seconds=time.perf_counter()-start,metadata=adapter.metadata,memory_after_load=memory())
        except Exception as ex:
            log.update(preparation_status='failure',error=error('prepare',ex),load_seconds=time.perf_counter()-start)
            write_json(args.output/'run.json',log)
            print(json.dumps({'engine':args.engine,'preparation_status':'failure','error_type':type(ex).__name__}))
            return 2
        for i,row in enumerate(rows):
            target=args.output/f'{i:03d}';target.mkdir()
            started=time.perf_counter(); rgb=None; transforms=[]
            try:
                rgb,transforms=decode(row)
                decode_time=time.perf_counter()-started
                before=time.perf_counter()
                result=adapter.infer(rgb)
                elapsed=time.perf_counter()-before
            except Exception as ex:
                result=dict(status='failure',units=[],details=[],errors=[error('decode' if rgb is None else 'inference',ex)],native={'exception':error('execution',ex)},transforms=[])
                elapsed=time.perf_counter()-started;decode_time=None
            # Preserve raw evidence before common validation. Malformed adapter
            # return values fail this image, without dropping subsequent inputs.
            try:
                native=result.get('native',result) if isinstance(result,dict) else result
                json.dumps(native,ensure_ascii=False,allow_nan=False).encode('utf-8')
                native_transforms=result.get('transforms',[]) if isinstance(result,dict) else []
                if not isinstance(native_transforms,list) or any(not isinstance(t,dict) for t in native_transforms):
                    raise ValueError('adapter transforms must be a list of objects')
                json.dumps(native_transforms,ensure_ascii=False,allow_nan=False).encode('utf-8')
            except Exception as ex:
                native={'evidence_serialization_error':error('adapter_evidence',ex),'raw_result_repr':repr(result)}
                native_transforms=[]
                result=dict(status='failure',units=[],errors=[error('adapter_evidence',ex)])
            write_json(target/'native.json',native)
            write_json(target/'transforms.json',transforms+native_transforms)
            try:
                obj=finalize(result,row,run_id,adapter.metadata)
                json.dumps(obj,ensure_ascii=False,allow_nan=False).encode('utf-8')
            except Exception as ex:
                obj=finalize(dict(status='failure',units=[],errors=[error('adapter_validation',ex)]),row,run_id,adapter.metadata)
            obj['metrics'].update(inference_seconds=elapsed,decode_seconds=decode_time,memory_after=memory(),request_index=i)
            if rgb is not None: overlay(rgb,obj,target/'overlay.png')
            else: obj['artifacts']['overlay']=None
            write_json(target/'output.json',obj)
            saved=json.loads((target/'output.json').read_text(encoding='utf-8'))
            validate_output(saved)
            log['records'].append(dict(image_id=row['image_id'],status=saved['status'],units=len(saved['units']),details=len(saved['details']),output=f'{i:03d}/output.json',inference_seconds=elapsed))
            write_json(args.output/'run.json',log)
            print(json.dumps(log['records'][-1],ensure_ascii=True))
    write_json(args.output/'run.json',log)
    return 0 if all(r['status']=='ok' for r in log['records']) else 1

if __name__=='__main__': raise SystemExit(main())
