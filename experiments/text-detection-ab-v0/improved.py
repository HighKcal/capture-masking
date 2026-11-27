"""TXT-02 contextual postprocessing and explicit-field/contact grammars.

No case IDs, gold labels, example name dictionaries, model changes or Unicode
normalization are used. Baseline detector and model helper code stay unchanged.
"""
import copy
import re
import time
import unicodedata
import detectors as baseline
from detectors import candidate, segment, dedup, key, views, mapped_segments

NAME_LABEL = r'(?:표시\s*이름|프로필\s*이름|등록\s*이름|법정\s*성명|실명|본명|성명|이름|닉네임|별명|활동명)'
ALIAS_LABELS=('닉네임','별명','활동명')
ACCOUNT_LABEL = r'(?:사용자\s*ID|로그인\s*ID|계정\s*ID|계정|아이디)'
FIELD_RE = re.compile(r'(?<![\w])(?P<label>'+NAME_LABEL+'|'+ACCOUNT_LABEL+r')(?P<close>[ \t]*[\]】〕〉>）)]?)(?P<sep>[ \t]*[:：=＝][ \t]*|\t+)',re.I)
ROLE = baseline.ROLE_WORDS | set('고모 이모 삼촌 숙모 조부 조모 조카 친척 부모 자녀 가족 사람 인원 직원 선임 후임 연구자 강사 강연자 수강생 방문객 보호자 운영진 당직자 야간당직자 보조관리자 게스트 고객상담원 상담원 사회자 접수자 관람객 안내원 책임자'.split())
HONOR = r'(?:선생님|교수님|과장님|부장님|팀장님|대표님|원장님|박사님|강사님|님|씨)'
PERSON_END = r'(?:께서|께서는|에게|에게는|한테|으로부터|로부터|으로|로|만|만은|조차|마저|까지|부터|밖에|보다|처럼|하고|이랑|랑|과|와|도|의|은|는|이|가|을|를|야|아|라고|이라고|이라는|라는|입니다|이다|였다|다)?'
STRONG_ORG = re.compile(r'(?:기념재단|기념관|문학관|박물관|생활문화연구소|연구소|재단|대학교|협회|주식회사|유한회사|아파트|센터|병원|학교|공원)')
WORK = re.compile(r'(?:도서|책|작품|소설|연극|영화|뮤지컬|드라마|앨범|공연|동화)(?:\s*(?:제목|이름|명))?\s*[:：=＝]?\s*[《「〈『“"\']([^》」〉』”"\']+)[》」〉』”"\']')
NONPERSON_FIELD = re.compile(r'(?:기관명|기관 이름|단체명|조직명|회사명|상품명|제품명|재료명|재료|색조|색상|색조 항목|캐릭터명|역할|직함|관계|주소|소재지|작품명|책 제목|연극 제목|영화 제목)\s*[:：=＝]\s*([^\n/|;]+)')
VALUE_PLACEHOLDERS = set('없음 미정 익명 비공개 미등록 미입력 성명 이름 값 내용 입력 예정 선택 이름값 입력값 이름없음 해당없음'.split())
SURNAME_CHARS = set('김이박최정강조윤장임한오서신권황안송류유홍전고문양손배백허남심노하곽성차주우구민진지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용')

def _is_hangul(ch):
    return '가'<=ch<='힣'

def _decor_skip(text,pos):
    while pos<len(text) and (text[pos] in ' \t' or unicodedata.category(text[pos])[0] in 'PS'):
        if text[pos]=='@': break
        pos+=1
    return pos

def fields(text):
    out=[]
    for match in FIELD_RE.finditer(text):
        label=re.sub(r'\s+','',match['label']).lower()
        prefix=text[max(0,match.start()-25):match.start()]
        nonperson=label.endswith('이름') and re.search(r'(?:파일|변수|폴더|함수|상품|기기|도구|객체|문서|기관|회사|단체|팀|그룹|동아리|프로젝트|조직|작품|책|영화|연극|제품)(?:의)?\s*$',prefix)
        pos=_decor_skip(text,match.end())
        kind='account' if re.fullmatch(ACCOUNT_LABEL,match['label'],re.I) else 'nonperson' if nonperson else 'name'
        pattern=r'@?[\w]+(?:[.\-][\w]+)*' if kind=='account' else r'[가-힣]+'
        value=re.match(pattern,text[pos:])
        if not value: continue
        end=pos+len(value[0])
        if kind=='name' and end<len(text) and (text[end].isalnum() or text[end]=='_' or unicodedata.category(text[end]).startswith('M')):
            continue
        out.append(dict(label=label,kind=kind,start=pos,end=end,label_start=match.start(),text=text[pos:end]))
    return out

def _field_for(fs,start,end):
    return next((f for f in fs if f['kind']=='name' and f['start']<=start and end<=f['end']),None)

def _nickname_support(text,field,all_fields):
    if field['label'] not in ALIAS_LABELS: return True
    value=field['text']
    context=text[max(0,field['label_start']-30):min(len(text),field['end']+70)]
    if re.search(r'(?:아니|아닌|않|미확인|불명|불확실|여부|다르|자동\s*생성|게임\s*별명)',context):return False
    if any(f['kind']=='name' and f['label'] not in ALIAS_LABELS and f['text']==value for f in all_fields):return True
    return bool(re.search(r'(?:본명|실명|실제\s*이름|법정\s*성명|등록된\s*이름).{0,20}(?:동일|같|그대로|사용|올|등록)',context))

def _name_decision(pred,text,fs):
    """Return corrected original interval, decision, reason(s). No length cap."""
    seg=pred['segments'][0]
    start,end=seg['start'],seg['end']
    field=_field_for(fs,start,end)
    before=text[start:end]
    reasons=[]
    # Written/legal-name evidence protects genuine final syllables, including 님.
    literal_name=bool(field and field['start']==start and field['end']==end and field['label'] not in ALIAS_LABELS)
    if not literal_name:
        honor=re.fullmatch(r'([가-힣]+?)'+HONOR+PERSON_END,before)
        if honor:
            end=start+len(honor[1]);reasons.append('explicit_honorific_boundary')
        elif not field:
            multi=re.fullmatch(r'([가-힣]+?)(께서는|께서|에게는|에게|한테)',before)
            if multi:
                end=start+len(multi[1]);reasons.append('multi_syllable_particle_boundary')
            elif re.fullmatch(r'[가-힣]{2,}[야아]',before) and re.match(r'\s*[,!！]',text[end:]):
                end-=1;reasons.append('vocative_punctuation_boundary')
            elif re.fullmatch(r'[가-힣]{2,}[이가은는도]',before) and re.match(r'\s+(?:말했다|도착했다|확인했다|답했다|보냈다|참석했다|서명했다)',text[end:]):
                end-=1;reasons.append('sentence_particle_boundary')
    value=text[start:end]
    if not value or not all(_is_hangul(c) for c in value):
        return start,end,'reject',['not_korean_name_surface']
    if value in ROLE or value in VALUE_PLACEHOLDERS:
        return start,end,'reject',['role_relationship_or_placeholder']
    if not literal_name and any(value.startswith(r) and value!=r and re.fullmatch(PERSON_END,value[len(r):]) for r in ROLE):
        return start,end,'reject',['role_relationship_with_grammatical_suffix']
    if len(value)==1 and value in SURNAME_CHARS:
        return start,end,'reject',['isolated_surname_ambiguous']
    if re.search(r'(?:^|\s)(?:성씨|성)(?:은|는|이|가)?\s*[:：=＝]?\s*$',text[max(0,start-25):start]):
        return start,end,'reject',['explicit_surname_context']
    if any(f['kind']=='nonperson' and f['start']<=start and end<=f['end'] for f in fs):
        return start,end,'reject',['nonperson_named_field']
    if re.match(r'\s*씨(?:\W|에게|$)',text[end:]) and len(value)==1:
        return start,end,'reject',['surname_only']
    for m in NONPERSON_FIELD.finditer(text):
        if m.start(1)<=start and end<=m.end(1):
            return start,end,'reject',['explicit_nonperson_field']
    for m in WORK.finditer(text):
        if m.start(1)<=start and end<=m.end(1):
            return start,end,'reject',['embedded_work_title']
    a,z=start,end
    while a>0 and _is_hangul(text[a-1]):a-=1
    while z<len(text) and _is_hangul(text[z]):z+=1
    word=text[a:z]
    suffix=text[end:z]
    strong_name_context=field and field['label'] not in ALIAS_LABELS
    if not strong_name_context:
        if STRONG_ORG.search(word) and re.search(r'(?:후원|기관|단체|재단|회사|학교|소속|명칭|로고|간판|배지|조직)',text[max(0,a-45):z+20]):
            return start,end,'reject',['embedded_organization_context']
        if (suffix.startswith(('로','길','동')) or re.search(r'(?:로|길|동)$',value)) and re.match(r'\s*\d+',text[z:]) and re.search(r'(?:주소|소재|위치|읍|면|동|우편)',text[max(0,a-45):z+25]):
            return start,end,'reject',['embedded_address_context']
        if suffix.startswith('전') and re.search(r'(?:소설|도서|책|작품|제목)',text[max(0,a-30):a]):
            return start,end,'reject',['embedded_unquoted_work']
    if a<start:
        return start,end,'reject',['candidate_inside_left_lexeme']
    if field and field['label'] in ALIAS_LABELS:
        if _nickname_support(text,field,fs):
            reasons.append('nickname_corroborated_as_person_name')
        elif re.search(r'(?:자동\s*생성|게임\s*별명|사물\s*조합|동물\s*캐릭터|실명이\s*아닌|(?:실제\s*이름|본명|실명).{0,15}(?:않|아니|아닌))',text[field['label_start']:field['end']+70]):
            return start,end,'reject',['explicit_nonpersonal_alias']
        else:
            reasons.append('nickname_semantics_uncertain_model_evidence_retained')
    if suffix and not re.fullmatch(HONOR+r'?'+PERSON_END,suffix):
        reasons.append('unlisted_suffix_not_a_rejection_reason')
    elif suffix:
        reasons.append('grammatical_suffix_outside_candidate')
    if literal_name:reasons.append('explicit_written_name_boundary')
    return start,end,('trim' if end!=seg['end'] else 'keep'),reasons or ['model_name_evidence_no_positive_exclusion']

def improved_names(raw_names,blocks):
    texts={b['id']:b['text'] for b in blocks}
    parsed={bid:fields(t) for bid,t in texts.items()}
    result,audit=[],[]
    for p in raw_names:
        s=p['segments'][0];t=texts[s['block_id']]
        start,end,action,reasons=_name_decision(p,t,parsed[s['block_id']])
        q=copy.deepcopy(p);q['segments']=[segment(s['block_id'],t,start,end)]
        q['policy_v1_reasons']=reasons
        row=dict(candidate=copy.deepcopy(p),action=action,reason=';'.join(reasons))
        if action!='reject':
            result.append(q);row['result']=q
        audit.append(row)
    return dedup(result),audit

def field_names(blocks):
    result,audit=[],[]
    for b in blocks:
        text=b['text'];fs=fields(text)
        for f in fs:
            if f['kind']!='name':continue
            p=candidate('NAME',[segment(b['id'],text,f['start'],f['end'])],'name_field_v1',evidence=dict(label=f['label'],label_start=f['label_start']))
            if f['label'] in ALIAS_LABELS and not _nickname_support(text,f,fs):
                audit.append(dict(candidate=p,action='no_supplement',reason='nickname_field_without_independent_name_evidence'));continue
            start,end,action,reasons=_name_decision(p,text,fs)
            # Field labels establish a candidate hypothesis; explicit non-person evidence still vetoes.
            q=copy.deepcopy(p);q['segments']=[segment(b['id'],text,start,end)];q['policy_v1_reasons']=reasons+['explicit_name_field']
            audit.append(dict(candidate=p,action=action,reason=';'.join(reasons),result=q if action!='reject' else None))
            if action!='reject': result.append(q)
    return dedup(result),audit

GAP=r'[ \t]*(?:-[ \t]*)?(?:\r?\n[ \t]*)?'
PHONE=re.compile(r'(?<![\w+])(?:'+baseline.PREFIX+GAP+r'\d{3,4}'+GAP+r'\d{4}|\+82[- ]?(?:10|1[16789]|2|3[1-3]|4[1-4]|5[1-5]|6[1-4]|70)'+GAP+r'\d{3,4}'+GAP+r'\d{4})(?![\d-])')
EXT=re.compile(r'[ \t]*(?:[,;:/][ \t]*)?(?:[（(\[][ \t]*)?(?:\r?\n[ \t]*)?(?:내선(?:\s*번호)?|extension|ext\.?)[ \t]*[:：=#-]?[ \t]*(\d{1,6})(?![\w])',re.I)
ORDER=re.compile(r'(?:주문\s*번호|주문\s*ID|송장\s*번호|운송장\s*번호|운송\s*식별\s*(?:번호|코드)|(?:상품|물품|제품|제작|재고|보관)\s*(?:코드|품번|번호|키)|모델\s*번호|제작\s*로트|로트\s*번호|참조\s*번호)(?:[은는이가의]|라는)?[ \t]*[:：=＝]?[ \t]*[「〈《(（]?[ \t]*$')
EMAIL=re.compile(r'(?<![\w.@+\-])'+baseline.LOCAL+r'@(?:[ \t]*\r?\n[ \t]*)?'+baseline.DOMAIN+r'(?![A-Za-z0-9_@\-])')
KOREAN_TAIL=re.compile(r'(?:에게|께|에서|으로|로|에|은|는|을|를|과|와|도|만|부터|까지)(?:도|만|는|부터|까지)?$')
COPULA=re.compile(r'(?:입니다|이다|이었다|였습니다|이에요|예요|이라고|라는|이란)$')
SENTENCE_USE=re.compile(r'\s+(?:[가-힣]+\s+){0,3}(?:보내|보낼|보내면|보내고|보냈|전달|발송|회신|답장|문의|연락|전송|적어|입력|읽|등록|접속|확인|사용|받|설정|남겨|신청|기록|수정|알려)')
AT_ACCOUNT=re.compile(r'(?<![\w@.])@[\w]+(?:[.\-][\w]+)*')

def mailbox_regions(text):
    allowed=set(".!#$%&'*+/=?^_`{|}~-\u200d\u200c")
    is_local=lambda c:unicodedata.category(c)[0] in 'LNM' or c in allowed
    out=[]
    for m in re.finditer('@',text):
        a,z=m.start(),m.end()
        while a>0 and is_local(text[a-1]):a-=1
        if a==m.start():continue
        while z<len(text) and (unicodedata.category(text[z])[0] in 'LNM' or text[z] in '.-_'):z+=1
        out.append((a,z))
    return out

def improved_contacts(case):
    preds,audit=[],[]
    for text,mapping,view in views(case):
        email_regions=mailbox_regions(text)
        # Broad mailbox occupancy suppresses account fragments even for excluded EAI.
        for m in EMAIL.finditer(text):
            a,z=m.span();reason=None;tail=''
            if a and unicodedata.category(text[a-1]).startswith('M'): reason='nonascii_local_combining_mark'
            if z<len(text) and unicodedata.category(text[z]).startswith('M'): reason='nonascii_domain_combining_mark'
            if z<len(text) and not text[z].isascii() and text[z].isalnum() and not _is_hangul(text[z]): reason='nonascii_domain_constituent'
            end=z
            while end<len(text) and (_is_hangul(text[end]) or unicodedata.category(text[end]).startswith('M')):end+=1
            tail=text[z:end]
            if text[z:z+1]=='.' and z+1<len(text) and (not text[z+1].isascii() and (text[z+1].isalnum() or unicodedata.category(text[z+1]).startswith('M'))):
                reason='nonascii_domain_continuation'
            if tail:
                grammatical=(bool(KOREAN_TAIL.fullmatch(tail)) and bool(SENTENCE_USE.match(text[end:]))) or (bool(COPULA.fullmatch(tail)) and bool(re.match(r'[.!?。！\s]|$',text[end:])))
                explicit_eai=bool(re.search(r'(?:국제화|비ASCII|비아스키|유니코드|EAI)',text[max(0,a-65):a],re.I))
                if not grammatical or explicit_eai:reason='nonascii_or_ambiguous_domain_tail'
            if reason:
                audit.append(dict(type='EMAIL',view=view,start=a,end=z,action='reject',reason=reason));continue
            segs=mapped_segments(text,mapping,a,z,case['blocks'])
            preds.append(candidate('EMAIL',segs,'email_rule_v1',evidence=dict(view=view,korean_tail=tail or None)))
            email_regions.append((a,end))
        for m in PHONE.finditer(text):
            prefix=text[max(0,m.start()-65):m.start()]
            pieces=re.split(r'[/|;,\r\n]',prefix)
            current=next((x for x in reversed(pieces) if x.strip()),'')
            if ORDER.search(current):
                audit.append(dict(type='PHONE',view=view,start=m.start(),end=m.end(),action='reject',reason='explicit_order_identifier_context'));continue
            if '\n' in m[0] and view!='explicit_same_field' and not re.search(r'(?:전화|연락|휴대|팩스|회신|직통)',current):
                audit.append(dict(type='PHONE',view=view,start=m.start(),end=m.end(),action='reject',reason='wrapped_number_lacks_contact_relation'));continue
            segs=mapped_segments(text,mapping,*m.span(),case['blocks'])
            ext=EXT.match(text,m.end())
            if ext: segs+=mapped_segments(text,mapping,*ext.span(1),case['blocks'])
            order={b['id']:i for i,b in enumerate(case['blocks'])}
            segs.sort(key=lambda s:(order[s['block_id']],s['start']))
            preds.append(candidate('PHONE',segs,'phone_rule_v1',evidence=dict(view=view,extension=bool(ext))))
        for m in AT_ACCOUNT.finditer(text):
            a,z=m.span()
            if any(a<end and z>start for start,end in email_regions):continue
            # Keep ASCII handles outside Korean sentence suffixes; Unicode handles require explicit account field.
            raw=m[0]
            korean=re.search(r'[가-힣]',raw)
            if korean and raw[1:korean.start()].isascii() and korean.start()>1:
                suffix=raw[korean.start():]
                if re.fullmatch(PERSON_END,suffix): z=a+korean.start()
            preds.append(candidate('ACCOUNT_ID',mapped_segments(text,mapping,a,z,case['blocks']),'account_at_v1',evidence=view))
        for f in fields(text):
            if f['kind']!='account':continue
            a,z=f['start'],f['end']
            if any(a<end and z>start for start,end in email_regions):continue
            preds.append(candidate('ACCOUNT_ID',mapped_segments(text,mapping,a,z,case['blocks']),'account_field_v1',evidence=dict(view=view,label=f['label'])))
    return dedup(preds),audit

def assemble(case,raw_names):
    times={}
    t=time.perf_counter();n0,a0=baseline.name_policy(copy.deepcopy(raw_names),case['blocks']);times['N0_ms']=(time.perf_counter()-t)*1000
    t=time.perf_counter();n1,a1=improved_names(raw_names,case['blocks']);times['N1_ms']=(time.perf_counter()-t)*1000
    t=time.perf_counter();f,af=field_names(case['blocks']);n2=dedup(copy.deepcopy(n1)+f);times['field_addition_ms']=(time.perf_counter()-t)*1000
    t=time.perf_counter();c0=baseline.contact_rules(case);times['C0_ms']=(time.perf_counter()-t)*1000
    t=time.perf_counter();c1,ac=improved_contacts(case);times['C1_ms']=(time.perf_counter()-t)*1000
    configs=dict(N0=n0,N1=n1,N2=n2,C0=c0,C1=c1,BASE=dedup(copy.deepcopy(n0+c0)),FIXED=dedup(copy.deepcopy(n1+c1)),FIELD=dedup(copy.deepcopy(n2+c1)))
    return dict(configs=configs,field_candidates=copy.deepcopy(f),policy_audit=dict(N0=a0,N1=a1,field=af),contact_audit=ac,postprocess_timings=times)
