"""Pinned RapidOCR baseline, with non-mutating evidence hooks around native steps."""
import copy
import hashlib
import importlib.metadata as md
import logging
import os
import math
from pathlib import Path
import numpy as np
from common import ROOT,sha256,error

MODELS = [('ch_PP-OCRv5_det_server.onnx','0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad'),('korean_PP-OCRv5_rec_mobile.onnx','cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b')]

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def validate_parallel(boxes,texts,scores):
    if not isinstance(texts,(list,tuple)) or not isinstance(scores,(list,tuple,np.ndarray)) or not isinstance(boxes,(list,tuple,np.ndarray)):
        raise ValueError('RapidOCR full return fields have invalid types')
    if len(boxes)!=len(texts) or len(texts)!=len(scores):
        raise ValueError('RapidOCR text/box/score count mismatch')
    if any(not isinstance(t,str) for t in texts):raise ValueError('non-string OCR text')
    if any(not isinstance(s,(int,float,np.number)) or not math.isfinite(float(s)) for s in scores):raise ValueError('nonfinite recognizer score')

class Capture(logging.Handler):
    def __init__(self):super().__init__(logging.WARNING);self.messages=[]
    def emit(self,record):self.messages.append(dict(level=record.levelname,message=record.getMessage()))

class Proxy:
    def __init__(self,obj,callback):self.obj=obj;self.callback=callback
    def __getattr__(self,key):return getattr(self.obj,key)
    def __call__(self,*args,**kwargs):
        out=self.obj(*args,**kwargs);self.callback(args,out);return out

def empty_status(evidence,warnings):
    # Public RapidOCR converts both no boxes and hidden RapidOCRError to an empty
    # result. Only completed detector inference + a witnessed empty postprocess is
    # enough to call this a normal blank. Unknown routes remain indeterminate.
    if evidence.get('det_inference') and evidence.get('det_postprocess_count')==0 and all(w['message']=='The text detection result is empty' for w in warnings):
        return 'ok'
    if evidence.get('pre_filter') is not None and not warnings:
        before=evidence['pre_filter']
        validate_parallel(before['boxes'],before['texts'],before['scores'])
        if not any(t.strip() and s >= 0.5 for t,s in zip(before['texts'],before['scores'])):
            return 'ok'  # every observed candidate removed by pinned native filters
    return 'indeterminate'

class Adapter:
    def __init__(self,asset_root):
        if md.version('rapidocr')!='3.10.0' or md.version('onnxruntime')!='1.31.1':raise RuntimeError('RapidOCR/ORT versions differ from pin')
        import onnxruntime as ort
        ort.disable_telemetry_events()
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import EngineType,LangDet,LangRec,ModelType,OCRVersion
        dest=Path(asset_root)/'assets/rapid'
        for name,digest in MODELS:
            if sha256(dest/name)!=digest:raise ValueError('model SHA-256 mismatch')
        params={
            'Global.use_cls':False,'Global.return_word_box':False,'Global.return_single_char_box':False,
            'Global.model_root_dir':str(dest),'Global.font_path':str(Path(os.environ['WINDIR'])/'Fonts/malgun.ttf'),
            'Global.log_level':'warning','Global.text_score':0.5,
            'Det.engine_type':EngineType.ONNXRUNTIME,'Det.lang_type':LangDet.CH,'Det.model_type':ModelType.SERVER,'Det.ocr_version':OCRVersion.PPOCRV5,'Det.model_path':str(dest/MODELS[0][0]),
            'Rec.engine_type':EngineType.ONNXRUNTIME,'Rec.lang_type':LangRec.KOREAN,'Rec.model_type':ModelType.MOBILE,'Rec.ocr_version':OCRVersion.PPOCRV5,'Rec.model_path':str(dest/MODELS[1][0]),
            'Rec.rec_keys_path':str(dest/'embedded-character.txt'),
            'EngineConfig.onnxruntime.use_cuda':False,'EngineConfig.onnxruntime.use_dml':False,
            'EngineConfig.onnxruntime.use_cann':False,'EngineConfig.onnxruntime.use_coreml':False,
            'EngineConfig.onnxruntime.intra_op_num_threads':6,'EngineConfig.onnxruntime.inter_op_num_threads':1}
        owner=self
        class TracedRapid(RapidOCR):
            def preprocess_img(self,img):
                output,record=super().preprocess_img(img)
                owner.evidence['preprocess_shape']=list(output.shape)
                return output,record
            def build_final_output(self,ori,det,cls,rec,crops,record):
                owner.evidence['op_record']=copy.deepcopy(record)
                owner.evidence['crop_shapes']=[list(x.shape) for x in crops]
                if rec.txts is not None:
                    owner.evidence['pre_filter']=plain(dict(boxes=det.boxes,texts=rec.txts,scores=rec.scores))
                    validate_parallel(det.boxes,rec.txts,rec.scores)
                return super().build_final_output(ori,det,cls,rec,crops,record)
        self.evidence={}
        self.engine=TracedRapid(params=params)
        det=self.engine._load_det_model();rec=self.engine._load_rec_model()
        for model in (det,rec):
            if model.session.session.get_providers()!=['CPUExecutionProvider']:raise ValueError('unexpected ORT provider')
        embedded=rec.session.session.get_modelmeta().custom_metadata_map['character']
        if embedded != (dest/'embedded-character.txt').read_text(encoding='utf-8'):raise ValueError('dictionary differs from ONNX metadata')
        self.dictionary_sha256=hashlib.sha256(embedded.encode()).hexdigest()
        def infer_event(args,out):
            self.evidence['det_inference']=dict(input_shape=list(args[0].shape),output_shape=list(out.shape))
        def post_event(args,out):self.evidence['det_postprocess_count']=len(out[0])
        det.session=Proxy(det.session,infer_event)
        det.postprocess_op=Proxy(det.postprocess_op,post_event)
        # Mutable thresholds remain on the original postprocessor; every baseline
        # call below explicitly reassigns them to avoid retained call options.
        self.det_postprocess=det.postprocess_op.obj
        from omegaconf import OmegaConf
        self.metadata=dict(engine='rapid',packages={p:md.version(p) for p in ['rapidocr','onnxruntime','numpy','opencv-python-headless','Pillow']},
            models=[dict(path=str(dest/n),sha256=h) for n,h in MODELS],device='CPUExecutionProvider',
            dictionary=dict(source='ONNX character metadata, UTF-8 bytes',sha256=self.dictionary_sha256,
                extracted_file_path=str(dest/'embedded-character.txt'),extracted_file_sha256=sha256(dest/'embedded-character.txt'),
                extracted_file_note='Windows text writer newline translation; Unicode contents verified equal to embedded metadata',entries=len(embedded.splitlines()),language='korean'),
            settings=OmegaConf.to_container(self.engine.cfg,resolve=True,enum_to_str=True),
            evidence_policy='read-only native stage hooks; no changes to OCR predictions; baseline flags reset every call')
        # Some config objects contain Paths; round-trip through string fallback for metadata only.
        import json
        self.metadata=json.loads(json.dumps(self.metadata,default=str))

    def infer(self,rgb):
        from rapidocr.utils.output import RapidOCROutput
        from rapidocr.utils.log import logger
        self.evidence={};capture=Capture();logger.addHandler(capture)
        raw={};units=[];errors=[]
        try:
            self.det_postprocess.box_thresh=0.5;self.det_postprocess.unclip_ratio=1.6
            out=self.engine(np.asarray(rgb)[:,:,::-1].copy(),use_det=True,use_cls=False,use_rec=True,return_word_box=False,return_single_char_box=False,text_score=0.5,box_thresh=0.5,unclip_ratio=1.6)
            raw=dict(return_type=type(out).__name__,boxes=plain(getattr(out,'boxes',None)),texts=plain(getattr(out,'txts',None)),scores=plain(getattr(out,'scores',None)),score_source='recognizer CTC native score; not detector score',word_results=plain(getattr(out,'word_results',None)),elapse_list=plain(getattr(out,'elapse_list',None)))
            if not isinstance(out,RapidOCROutput):raise ValueError('not full RapidOCROutput')
            if out.txts is None:
                status=empty_status(self.evidence,capture.messages)
                if status!='ok':errors.append(dict(stage='native_empty',type='UnobservableFailure',message='empty result lacks completed native no-text/filter evidence'))
            else:
                validate_parallel(out.boxes,out.txts,out.scores)
                units=[dict(text=t,unit_type='fragment',polygons=[plain(b)],location_status='valid',confidence=float(s),geometry_source='native detector quad restored by RapidOCR') for b,t,s in zip(out.boxes,out.txts,out.scores)]
                status='partial_failure' if capture.messages else 'ok'
                if capture.messages:errors.append(dict(stage='native_warning',type='RapidWarning',message='native warnings captured; inspect native.json'))
        except Exception as ex:
            status='failure';errors.append(error('rapid_inference',ex))
        finally:logger.removeHandler(capture)
        return dict(status=status,units=units,details=[],errors=errors,native=dict(output=raw,evidence=plain(self.evidence),warnings=capture.messages),
            transforms=[dict(stage='rapid_input',conversion='RGB to BGR',width=rgb.width,height=rgb.height),
                dict(stage='rapid_internal',op_record=plain(self.evidence.get('op_record')),det_tensor=self.evidence.get('det_inference'),crop_shapes=self.evidence.get('crop_shapes'),
                geometry='native restoration includes resize/padding inversion and clipping; per-crop perspective/possible90deg used only for recognition',character_boxes=False)])
