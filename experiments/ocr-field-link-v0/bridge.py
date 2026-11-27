"""Read-only dependencies and file helpers for OCR-FIELD01."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'ocr-ner-link-v0'
TXT=HERE.parent/'text-detection-ab-v0'
sys.path.insert(0,str(OLD))
from support import read,write,sha,digest,jsonl,decode,finalize,overlay,validate_output,polygon_valid,raster,ASSETS,NER_PY,RAPID_PY,ocr_runner
from link import bind_ocr,map_candidates,save_masked
sys.path.insert(0,str(TXT))
from improved import improved_contacts
# Restore this experiment's namespace priority after loading read-only dependencies.
sys.path.insert(0,str(HERE))

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def output_path(path):
    path=Path(path).resolve()
    if not path.is_relative_to(HERE) or path==HERE:raise ValueError('Output outside experiment')
    return path
