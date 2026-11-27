"""Explainable baselines. No case IDs, gold spans or name dictionary are used."""
import re
import unicodedata
import time

NAME_FIELDS = r'(?:성명|이름|담당자|작성자|신청자|수신자|보내는\s*사람)'
ROLE_WORDS = set('엄마 아빠 어머니 아버지 할아버지 할머니 형 누나 언니 오빠 동생 팀장 팀장님 부장 부장님 과장 과장님 사장님 선생님 연구원 교수 기자 작가 배우 관리자 운영자 진행자 사용자 담당자 고객 고객님 친구 본인 이름 성명 없음 익명 비공개 미입력 미정 담당 미등록 확인 필요 그녀 그분 여러분 우리 모두'.split())
ORG_END = r'(?:기념재단|재단|주식회사|대학교|학교|병원|회사|협회|센터|박물관|문학관|기념관|아파트|거리|공원|대로|로|길|전)'
HONOR = r'(?:선생님|교수님|과장님|부장님|팀장님|대표님|님|씨)'

def segment(block_id, text, start, end):
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
            result[k].setdefault('merged_evidence', []).append({k: v for k, v in p.items() if k not in ('segments', 'type')})
    out = list(result.values())
    for i, p in enumerate(out):
        p['id'] = f'p{i+1}'
    return out

def name_policy(preds, blocks):
    """Same policy for rule/model candidates; rejection and boundary edits are auditable."""
    texts = {b['id']: b['text'] for b in blocks}
    accepted, audit = [], []
    for p in preds:
        q = dict(p, segments=[dict(s) for s in p['segments']])
        s = q['segments'][0]
        t = texts[s['block_id']]
        start, end = s['start'], s['end']
        value = t[start:end]
        reason = None
        explicit_field = bool(re.search(NAME_FIELDS+r'\s*[:：]\s*[^가-힣A-Za-z0-9\n]*$',t[max(0,start-30):start]))
        # Remove clear honorifics and multi-letter particles only, with contextual evidence.
        m = re.fullmatch(r'([가-힣]{2,4}?)'+HONOR+r'(?:께서|에게|은|는|이|가|도)?', value)
        if m:
            end = start + len(m[1])
        elif not explicit_field and re.search(r'[,!！]', t[end:end+1]) and re.fullmatch(r'[가-힣]{2,4}[야아]', value):
            end -= 1
        elif re.fullmatch(r'[가-힣]{2,4}(?:에게|한테|께서)', value):
            end = start + len(re.sub(r'(?:에게|한테|께서)$', '', value))
        elif not explicit_field and re.fullmatch(r'[가-힣]{2,4}[이가은는도]', value) and re.match(r'\s+(?:말했다|도착했다|확인했다|답했다|보냈다|참석했다|서명했다)',t[end:]):
            end -= 1
        value = t[start:end]
        if not re.fullmatch(r'[가-힣]{2,4}', value):
            reason = 'not_2_to_4_hangul_name_surface'
        elif value in ROLE_WORDS:
            reason = 'role_relationship_or_nonvalue'
        line_start = t.rfind('\n', 0, start)+1
        prefix = t[line_start:start]
        # Nearest explicit field controls its value, rather than every earlier label.
        labels = list(re.finditer(r'([가-힣 ]+)\s*[:：]\s*', prefix))
        label = labels[-1][1].strip() if labels else ''
        if re.search(r'(?:닉네임|기관명|조직명|회사명|재료명|상품명|책 제목|작품명|주소)$', label):
            reason = 'excluded_field_context'
        for title in re.finditer(r'(?:도서|책|작품|소설 제목)\s*[《「〈]([^》」〉]+)[》」〉]',t):
            if title.start(1)<=start and end<=title.end(1):
                reason = 'embedded_work_title'
        lexical_start = start
        while lexical_start > 0 and re.match('[가-힣]', t[lexical_start-1]):
            lexical_start -= 1
        lexical_end = end
        while lexical_end < len(t) and re.match('[가-힣]', t[lexical_end]):
            lexical_end += 1
        tail = t[end:lexical_end]
        if lexical_start < start:
            reason = 'embedded_left_hangul'
        if re.match(ORG_END, tail):
            reason = 'embedded_organization_work_address'
        elif tail and not re.fullmatch(HONOR+r'?(?:이|가|은|는|을|를|도|와|과|랑|야|아|에게|한테|께서|의|이랑|라고|라는|입니다|이다)?', tail):
            reason = 'embedded_unknown_suffix'
        if re.match(r'\s+씨(?:\W|$)', t[end:]) and len(value) == 1:
            reason = 'surname_only'
        q['segments'] = [segment(s['block_id'], t, start, end)]
        if end != s['end']:
            q['boundary_edit'] = dict(before=s, reason='contextual_honorific_or_particle')
        if reason:
            audit.append(dict(candidate=p, action='reject', reason=reason))
        else:
            accepted.append(q)
            if 'boundary_edit' in q:
                audit.append(dict(candidate=p, action='trim', result=q))
    return dedup(accepted), audit

def raw_rule_names(blocks):
    out = []
    patterns = [
        ('name_field', re.compile(NAME_FIELDS+r'\s*[:：]\s*[^가-힣A-Za-z0-9\n]*([가-힣]{2,8})(?![가-힣])')),
        ('name_context', re.compile(r'(?:담당자|인물|친구|학생|교수|작가|기자|배우|회원|참가자)\s+([가-힣]{2,4}?)(?=(?:가|이|는|은|도|에게)\s|\s|[.,!?]|$)')),
        ('name_separate_title', re.compile(r'(?<![가-힣])([가-힣]{2,4})\s+(?:씨|연구원|선생님|교수|부장|과장|팀장)(?=\s|[.,!?]|$|(?:만|에게|님|께서|은|는|이|가))')),
        ('name_honorific', re.compile(r'(?<![가-힣])([가-힣]{2,4}?)(?:선생님|교수님|과장님|부장님|팀장님|대표님|님|씨)(?=\s|[.,!?]|$|(?:이|가|은|는|께서|에게))')),
        ('name_recipient', re.compile(r'(?<![가-힣])([가-힣]{2,4})(?:에게|한테|께서)(?=\s|$)')),
        ('name_vocative', re.compile(r'(?<![가-힣])([가-힣]{2,4}?)[야아](?=\s*[,!！])')),
        ('name_action', re.compile(r'(?<![가-힣])([가-힣]{2,4}?)(?:이|가|는|은|도)\s+(?:말했다|도착했다|확인했다|답했다|보냈다|참석했다|서명했다)')),
    ]
    for b in blocks:
        for source, pattern in patterns:
            for m in pattern.finditer(b['text']):
                if source=='name_vocative' and re.search(NAME_FIELDS+r'\s*[:：]\s*[^가-힣A-Za-z0-9\n]*$',b['text'][max(0,m.start(1)-30):m.start(1)]):
                    continue
                out.append(candidate('NAME', [segment(b['id'], b['text'], *m.span(1))], source,
                                     evidence='explicit_field' if source == 'name_field' else 'human_context'))
    return dedup(out)

def views(case):
    """Identity mappings plus explicit synthetic-layout joins; no inferred column stitching."""
    by_id = {b['id']: b['text'] for b in case['blocks']}
    for b in case['blocks']:
        yield b['text'], [(b['id'], i) for i in range(len(b['text']))], 'original'
    for link in case.get('links', []):
        if link.get('relation') != 'same_field':
            continue
        text, mapping = '', []
        for bid in link['blocks']:
            text += by_id[bid]
            mapping += [(bid, i) for i in range(len(by_id[bid]))]
        yield text, mapping, 'explicit_same_field'

def mapped_segments(text, mapping, start, end, blocks):
    by_id = {b['id']: b['text'] for b in blocks}
    points = [(mapping[i][0], mapping[i][1]) for i in range(start,end) if text[i] not in '\r\n']
    groups = []
    for bid, idx in points:
        if groups and groups[-1][0] == bid and groups[-1][2] == idx:
            groups[-1][2] = idx+1
        else:
            groups.append([bid,idx,idx+1])
    out = []
    for bid, a, z in groups:
        while a < z and by_id[bid][a] in ' \t': a += 1
        while z > a and by_id[bid][z-1] in ' \t': z -= 1
        if z > a: out.append(segment(bid, by_id[bid], a,z))
    order={b['id']:i for i,b in enumerate(blocks)}
    return sorted(out,key=lambda s:(order[s['block_id']],s['start']))

LOCAL = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
DOMAIN = r'[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+'
EMAIL = re.compile(r'(?<![\w.@+\-])'+LOCAL+r'@(?:[ \t]*\r?\n[ \t]*)?'+DOMAIN+r'(?![\w@\-])')
SEP = r'(?:[- ]|[ \t]*\r?\n[ \t]*)?'
PREFIX = r'(?:01[016789]|02|0(?:3[1-3]|4[1-4]|5[1-5]|6[1-4]|70))'
PHONE = re.compile(r'(?<![\w+])(?:'+PREFIX+SEP+r'\d{3,4}'+SEP+r'\d{4}|\+82[- ]?(?:10|1[16789]|2|3[1-3]|4[1-4]|5[1-5]|6[1-4]|70)'+SEP+r'\d{3,4}'+SEP+r'\d{4})(?![\d-])')
EXT = re.compile(r'[ \t,;]*(?:내선|ext\.?|extension)[ \t:：]*(\d{1,6})(?![\w])', re.I)
ACCOUNT_AT = re.compile(r'(?<![\w@.])@[A-Za-z0-9_][A-Za-z0-9_.-]*(?<![.-])')
ACCOUNT_FIELD = re.compile(r'(?:계정(?:\s*ID)?|사용자\s*ID|아이디|로그인\s*ID)\s*[:：]\s*(@?[A-Za-z0-9_][A-Za-z0-9_.-]*)(?<![.-])', re.I)

def contact_rules(case):
    result = []
    for text, mapping, view in views(case):
        occupied = []
        # Suppress account extraction inside ALL email-like words, including excluded EAI.
        nonascii = []
        for m in re.finditer(r"[\w.!#$%&'*+/=?^_`{|}~\-]+@[\w.\-]+", text):
            occupied.append(m.span())
            if not m[0].isascii(): nonascii.append(m.span())
        for m in EMAIL.finditer(text):
            if any(m.start()<y and m.end()>x for x,y in nonascii): continue
            if m.start()>0 and unicodedata.category(text[m.start()-1]).startswith('M'): continue
            if m.end()<len(text) and unicodedata.category(text[m.end()]).startswith('M'): continue
            result.append(candidate('EMAIL', mapped_segments(text,mapping,*m.span(),case['blocks']), 'email_rule', evidence=view))
            occupied.append(m.span())
        for m in PHONE.finditer(text):
            raw = m[0]
            prefix = text[max(0,m.start()-24):m.start()]
            if re.search(r'(?:주문번호|상품코드|상자코드|참조번호)\s*[:：]?\s*$',prefix):
                continue
            if '\n' in raw and (raw.count('\n') > 1 or not re.search(r'(?:전화|연락처|연락|휴대폰|휴대전화|팩스)\s*[:：]?\s*$', prefix)):
                continue
            segs = mapped_segments(text,mapping,*m.span(),case['blocks'])
            ext = EXT.match(text, m.end())
            if ext:
                segs += mapped_segments(text,mapping,*ext.span(1),case['blocks'])
            order={b['id']:i for i,b in enumerate(case['blocks'])}
            segs.sort(key=lambda s:(order[s['block_id']],s['start']))
            result.append(candidate('PHONE', segs, 'phone_rule', evidence=view))
        for regex, group, source in [(ACCOUNT_AT,0,'account_at_rule'), (ACCOUNT_FIELD,1,'account_field_rule')]:
            for m in regex.finditer(text):
                a,z=m.span(group)
                if any(a < y and z > x for x,y in occupied): continue
                result.append(candidate('ACCOUNT_ID',mapped_segments(text,mapping,a,z,case['blocks']),source,evidence=view))
    return dedup(result)

def assemble(case, raw_names):
    started=time.perf_counter()
    rule_raw = raw_rule_names(case['blocks'])
    r, ra = name_policy(rule_raw,case['blocks'])
    common = contact_rules(case)
    baseline = dedup(r+common)
    rule_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter()
    m, ma = name_policy(raw_names,case['blocks'])
    rm = dedup(r+m)
    combined=dedup(rm+common)
    return dict(configs=dict(R=r,M=m,RM=rm,FULL_R=baseline,FULL_RM=combined),
                raw_rules=rule_raw,policy_audit=dict(rules=ra,model=ma),
                rule_timing_ms=rule_ms,model_postprocess_ms=(time.perf_counter()-started)*1000)
