"""Shared TXT-AB01 adapter derived from pinned TXT-B01; policies stay separate.
See models.json and results/provenance-before.json for copied origins.
"""
import hashlib
import json
import os
from pathlib import Path
import time

# Set before importing either HF library. These are process-local settings.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
DEFAULT_ROOT = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'capture-masking' / 'TXT-AB01'
os.environ['HF_HOME'] = str(DEFAULT_ROOT / 'cache')
os.environ['HF_HUB_CACHE'] = str(DEFAULT_ROOT / 'cache' / 'hub')

SPECS = json.loads((Path(__file__).parent / 'models.json').read_text(encoding='utf-8'))
NAME_ENTITIES = {'A': 'PERSON_NAME', 'B': 'PS'}
WINDOW = 256
OVERLAP = 64


def segment(block_id, text, start, end):
    if not 0 <= start < end <= len(text):
        raise ValueError('Invalid nonempty code-point span')
    if '\n' in text[start:end] or '\r' in text[start:end]:
        raise ValueError('NAME segment contains newline')
    return dict(block_id=block_id, start=start, end=end, text=text[start:end])


def candidate(kind, segs, source, **extra):
    return dict(type=kind, segments=segs, sources=[source], **extra)


def key(p):
    return p['type'], tuple((s['block_id'], s['start'], s['end']) for s in p['segments'])


def dedup(preds):
    result = {}
    for p in preds:
        k = key(p)
        if k not in result:
            result[k] = dict(p, sources=list(p['sources']))
        else:
            result[k]['sources'] = sorted(set(result[k]['sources'] + p['sources']))
            result[k].setdefault('merged_evidence', []).append(
                {field: value for field, value in p.items() if field not in ('segments', 'type')})
    out = list(result.values())
    for i, p in enumerate(out):
        p['id'] = f'p{i+1}'
    return out


def validate_blocks(blocks):
    seen = set()
    for b in blocks:
        if not isinstance(b, dict) or not isinstance(b.get('id'), str) or not b['id']:
            raise ValueError('Each block needs a nonempty string id')
        if not isinstance(b.get('text'), str) or b['id'] in seen:
            raise ValueError('Block text must be str and block ids must be unique')
        seen.add(b['id'])


def assemble(bid, text, records, issues, spec):
    """Reconstruct BIO after selecting windows; orphan I starts a candidate."""
    names, active = [], None

    def flush():
        nonlocal active
        if active is not None and active['kind'] == spec['name_entity']:
            names.append(candidate('NAME', [segment(bid, text, active['a'], active['z'])],
                                   'model', score=sum(active['scores']) / len(active['scores']),
                                   evidence=active['evidence'], model_id=spec['model_id'],
                                   revision=spec['revision'], original_entity=spec['name_entity'],
                                   token_indices=active['indices'], original_labels=active['labels']))
        active = None

    for rec in records:
        label = rec['label']
        if rec['unk']:
            issues.append(dict(block_id=bid, index=rec['index'], kind='UNK',
                               start=rec['start'], end=rec['end']))
        if rec['end'] <= rec['start'] or label == 'O':
            flush()
            continue
        if '\n' in rec['text'] or '\r' in rec['text']:
            # An indivisible token spanning lines has no exact single-line mapping.
            # Preserve the raw token and report failure instead of inventing offsets.
            raise ValueError('Token crosses a newline; cannot reconstruct exact NAME segment')
        prefix, kind = label.split('-', 1)
        gap = text[active['z']:rec['start']] if active else ''
        contiguous = (active is not None and active['kind'] == kind and not gap.strip()
                      and '\n' not in gap and '\r' not in gap)
        if active is not None and ('\n' in gap or '\r' in gap):
            issues.append(dict(block_id=bid, index=rec['index'], kind='BIO_line_break_split'))
        if prefix == 'B' or not contiguous:
            if prefix == 'I':
                issues.append(dict(block_id=bid, index=rec['index'], kind='BIO_orphan_I', label=label))
            flush()
            active = dict(kind=kind, a=rec['start'], z=rec['end'], scores=[rec['score']],
                          evidence=['argmax_' + label] + (['orphan_I_repair'] if prefix == 'I' else []),
                          indices=[rec['index']], labels=[label])
        else:
            active['z'] = max(active['z'], rec['end'])
            active['scores'].append(rec['score'])
            active['indices'].append(rec['index'])
            active['labels'].append(label)
            if 'argmax_' + label not in active['evidence']:
                active['evidence'].append('argmax_' + label)
    flush()
    return names


def verify_files(spec):
    path = Path(spec['path'])
    entries = {item['name']: item for item in spec['files']}
    for name, item in entries.items():
        file = path / name
        if not file.is_file():
            raise FileNotFoundError(f'Missing local model file: {file}')
        with file.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if file.stat().st_size != item['size'] or digest != item['sha256']:
            raise ValueError(f'Local model file does not match frozen manifest: {name}')
    return entries


class NameModel:
    def __init__(self, name='A', device='cuda', legacy_probability_argmax=False):
        if name not in SPECS:
            raise ValueError('Model must be A or B')
        self.name = name
        self.spec = dict(SPECS[name], name_entity=NAME_ENTITIES[name])
        self.labels = {int(k): v for k, v in self.spec['id2label'].items()}
        self.num_labels = len(self.labels)
        self.legacy_probability_argmax = legacy_probability_argmax
        path = Path(self.spec['path'])
        self.device = device
        self.forward_ms = self.event_ms = 0.0
        start = time.perf_counter()
        files = verify_files(self.spec)
        self.verification_s = time.perf_counter() - start
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
        self.torch = torch
        torch.set_num_threads(6)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.manual_seed(1729)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        start = time.perf_counter()
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if (config.architectures != self.spec['architecture'] or config.model_type != self.spec['model_type']
                or config.num_labels != self.num_labels or config.id2label != self.labels
                or config.label2id != {v: k for k, v in self.labels.items()}):
            raise ValueError('Unexpected architecture or label mapping')
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True,
                                                       trust_remote_code=False)
        if not self.tokenizer.is_fast:
            raise ValueError('Original offsets require a fast tokenizer')
        self.special_count = self.tokenizer.num_special_tokens_to_add(pair=False)
        probe = self.tokenizer('가 [CLS] [SEP] [UNK]', add_special_tokens=False)['input_ids']
        self._wrap(probe)  # Verify B template, including literal special IDs, before inference.
        if self.special_count != 2 or WINDOW > config.max_position_embeddings:
            raise ValueError('Unsupported special-token or position layout')
        self.capacity = WINDOW - self.special_count
        self.model, loading = AutoModelForTokenClassification.from_pretrained(
            path, config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=(name == 'A'), weights_only=True, attn_implementation='eager',
            torch_dtype=torch.float32, output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError(f'Checkpoint not loaded exactly: {loading}')
        if self.model.classifier.out_features != self.num_labels:
            raise ValueError('Full original classifier required')
        self.model.to(device).eval()
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('FP32 required')
        self.sync()
        self.load_s = time.perf_counter() - start
        self.metadata = dict(model=name, model_id=self.spec['model_id'], revision=self.spec['revision'], label_map=self.labels,
                             weights_sha256=files['model.safetensors' if name == 'A' else 'pytorch_model.bin']['sha256'],
                             loading_info=loading, weights_only=True, policy_applied=False,
                             tokenizer_class=type(self.tokenizer).__name__,
                             special_count=self.special_count, special_tokens=self.tokenizer.special_tokens_map,
                             special_ids=self.tokenizer.all_special_ids,
                             tokenizer_backend=json.loads(self.tokenizer.backend_tokenizer.to_str()),
                             device=device, batch=1, dtype='float32', attention='eager',
                             window=WINDOW, overlap=OVERLAP, cpu_threads=torch.get_num_threads(),
                             interop_threads=torch.get_num_interop_threads(), threshold=None,
                             argmax_basis='legacy_softmax' if legacy_probability_argmax else 'full_head_logits',
                             verification_s=self.verification_s, load_s=self.load_s)
        # Full vocabulary is already preserved in the pinned tokenizer artifact.
        self.metadata['tokenizer_backend'].pop('model', None)

    def sync(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize(self.device)

    def _wrap(self, content):
        full = self.tokenizer.build_inputs_with_special_tokens(content)
        if full != [self.tokenizer.cls_token_id] + content + [self.tokenizer.sep_token_id]:
            raise ValueError('Pinned tokenizer template is not CLS + content + SEP')
        return full, [True] + [False] * len(content) + [True]

    def _predict(self, full):
        torch = self.torch
        input_ids = torch.tensor([full], dtype=torch.long, device=self.device)
        mask = torch.ones_like(input_ids)
        self.sync()
        start = time.perf_counter()
        ev1 = ev2 = None
        if self.device.startswith('cuda'):
            ev1, ev2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev1.record()
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=mask).logits
        if ev2 is not None:
            ev2.record()
        self.sync()
        self.forward_ms += (time.perf_counter() - start) * 1000
        if ev1 is not None:
            self.event_ms += ev1.elapsed_time(ev2)
        if tuple(logits.shape) != (1, len(full), self.num_labels) or not torch.isfinite(logits).all().item():
            raise ValueError('Invalid full-head logits')
        # Softmax can round unequal FP32 logits to equal probabilities.
        # Keep the actual full-head logits argmax independently from scores.
        probs = logits.softmax(-1)
        labels = probs.argmax(-1) if self.legacy_probability_argmax else logits.argmax(-1)
        return probs[0].cpu().tolist(), labels[0].cpu().tolist()

    def _infer_block(self, b, result):
        text, bid = b['text'], b['id']
        tok = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        ids, offsets = tok['input_ids'], tok['offset_mapping']
        result['tokenization_evidence'][bid] = dict(input_ids=ids, offsets=offsets)
        if len(ids) != len(offsets):
            raise ValueError('Token/offset length mismatch')
        previous = (0, 0)
        for i, (a, z) in enumerate(offsets):
            if not 0 <= a <= z <= len(text) or a < previous[0] or z < previous[1]:
                raise ValueError(f'Invalid/nonmonotonic original offset at token {i}')
            if a < previous[1]:
                result['model_issues'].append(dict(block_id=bid, index=i, kind='overlapping_token_offsets'))
            if a == z:
                result['model_issues'].append(dict(block_id=bid, index=i, kind='empty_token_offset'))
            previous = (a, z)
        result['token_boundaries'][bid] = dict(starts=sorted({a for a, z in offsets if a < z}),
                                               ends=sorted({z for a, z in offsets if a < z}))
        best, start, wi = {}, 0, 0
        result['raw_windows'][bid] = []
        while start < len(ids):
            end = min(start + self.capacity, len(ids))
            content = ids[start:end]
            full, special = self._wrap(content)
            if len(full) > WINDOW or len(full) != len(special):
                raise ValueError('Invalid window template')
            probs, label_ids = self._predict(full)
            if (len(probs) != len(full) or any(len(p) != self.num_labels for p in probs)
                    or len(label_ids) != len(full) or any(i not in self.labels for i in label_ids)):
                raise ValueError('Unexpected probability shape')
            records, j = [], 0
            for position, (tid, sp, prob, label_id) in enumerate(zip(full, special, probs, label_ids)):
                rec = dict(token_id=tid, token=self.tokenizer.convert_ids_to_tokens(tid),
                           label_id=label_id, label=self.labels[label_id], score=prob[label_id],
                           probabilities=prob, special=sp, window=wi, position=position,
                           index=None, start=None, end=None, text=None, unk=False,
                           special_token_id=tid in self.tokenizer.all_special_ids)
                if not sp:
                    idx = start + j
                    a, z = offsets[idx]
                    priority = min(j, len(content) - 1 - j)
                    rec.update(index=idx, start=a, end=z, text=text[a:z],
                               unk=tid == self.tokenizer.unk_token_id, edge_distance=priority)
                    if idx not in best or priority > best[idx][0]:
                        best[idx] = (priority, rec)
                    j += 1
                records.append(rec)
            if j != len(content):
                raise ValueError('Special-token alignment failure')
            result['raw_windows'][bid].append(dict(start_token=start, end_token=end, tokens=records))
            wi += 1
            if end == len(ids):
                break
            start = end - OVERLAP
        if sorted(best) != list(range(len(ids))):
            raise ValueError('Window coverage failure')
        final = [best[i][1] for i in range(len(ids))]
        result['raw_model'][bid] = final
        result['block_diagnostics'][bid] = dict(code_points=len(text), tokens=len(ids), windows=wi,
                                              coverage_complete=True, selected_tokens=len(final),
                                              unk_tokens=sum(r['unk'] for r in final),
                                              last_token_end=offsets[-1][1] if offsets else None)
        # Commit candidates only after the complete block succeeds.
        return assemble(bid, text, final, result['model_issues'], self.spec)

    def infer(self, blocks):
        validate_blocks(blocks)
        started = time.perf_counter()
        self.forward_ms = self.event_ms = 0.0
        result = dict(raw_names=[], raw_model={}, raw_windows={}, token_boundaries={},
                      model_issues=[], timing={}, status='ok', errors=[], block_statuses={},
                      block_diagnostics={}, tokenization_evidence={}, model_metadata=dict(self.metadata))
        for b in blocks:
            bid = b['id']
            try:
                result['raw_names'].extend(self._infer_block(b, result))
                result['block_statuses'][bid] = 'ok'
            except Exception as error:
                result['block_statuses'][bid] = 'failure'
                detail = dict(block_id=bid, stage='model_block', exception=type(error).__name__,
                              message=str(error))
                result['errors'].append(detail)
                result['model_issues'].append(dict(detail, kind='block_failure'))
        failures = len(result['errors'])
        if failures:
            result['status'] = 'failure' if failures == len(blocks) else 'partial_failure'
        order = {b['id']: i for i, b in enumerate(blocks)}
        result['raw_names'].sort(key=lambda p: (order[p['segments'][0]['block_id']],
                                                 p['segments'][0]['start'], p['segments'][0]['end']))
        result['raw_names'] = dedup(result['raw_names'])
        result['timing'] = dict(model_total_ms=(time.perf_counter() - started) * 1000,
                                forward_wall_ms=self.forward_ms,
                                forward_cuda_event_ms=self.event_ms if self.device.startswith('cuda') else None)
        return result


def infer_case(model, case):
    # Whitelist; gold/excluded/tags/expected values never reach model.infer().
    try:
        blocks = [dict(id=b['id'], text=b['text']) for b in case['blocks']]
        result = model.infer(blocks)
    except Exception as error:
        result = dict(raw_names=[], raw_model={}, raw_windows={}, token_boundaries={}, model_issues=[],
                      timing={}, status='failure', errors=[dict(stage='case_input_or_model',
                      exception=type(error).__name__, message=str(error))],
                      block_statuses={}, block_diagnostics={}, model_metadata=dict(model.metadata))
    return dict(id=case['id'], **result)
