"""Audit only original English V1 evaluation files. Never train, filter or split.

HF_TOKEN is read only from environment variables; it is never printed or saved.
Run --download-only to inspect actual schemas before the full audit.
"""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import unicodedata
import zlib

os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import numpy as np
import pandas as pd
from datasets import load_dataset, DownloadConfig
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
EVIDENCE=REPORTS/'bhashabench_evidence'
DATA=ROOT/'data/bhashabench'
SAMPLES=ROOT/'analysis_samples'
REPOS={'ayur':'bharatgenai/BhashaBench-Ayur','legal':'bharatgenai/BhashaBench-Legal',
       'finance':'bharatgenai/BhashaBench-Finance','krishi':'bharatgenai/BhashaBench-Krishi'}
SEED=42
for p in [REPORTS,EVIDENCE,DATA,SAMPLES]:p.mkdir(parents=True,exist_ok=True)

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def get_token():
    value=os.environ.get('HF_TOKEN')
    if value:return value,'process environment'
    # These are Windows environment-variable stores, not Hugging Face cached login.
    if os.name=='nt':
        import winreg
        for hive,key,label in [(winreg.HKEY_CURRENT_USER,'Environment','user environment'),
                               (winreg.HKEY_LOCAL_MACHINE,r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment','machine environment')]:
            try:
                with winreg.OpenKey(hive,key) as handle:
                    value,_=winreg.QueryValueEx(handle,'HF_TOKEN')
                if value:return value,label
            except FileNotFoundError:
                pass
    return None,'missing'

def repositories():
    result={}
    for domain,repo in REPOS.items():
        dest=EVIDENCE/(domain+'_repository.json')
        if dest.exists():
            metadata=load(dest)
        else:
            info=HfApi(token=False).dataset_info(repo,files_metadata=True)
            metadata={'repo_id':repo,'revision':info.sha,'gated':info.gated,
                      'card':info.card_data.to_dict() if info.card_data else {},
                      'files':[{'path':x.rfilename,'bytes':x.size} for x in info.siblings]}
            save(dest,metadata)
        if metadata['repo_id']!=repo:
            raise ValueError('Repository identity mismatch; never substitute Multi or mirrors')
        result[domain]=metadata
    return result

def download_all():
    token,auth_source=get_token()
    metadata=repositories()
    if not token:
        save(EVIDENCE/'access_status.json',{'utc':datetime.now(timezone.utc).isoformat(),
              'status':'BLOCKED_MISSING_HF_TOKEN','token_present':False,
              'sources':{d:{'repo':v['repo_id'],'revision':v['revision'],'gated':v.get('gated')} for d,v in metadata.items()},
              'actual_data_downloaded':False,'note':'Public metadata only; no README counts are used as actual statistics.'})
        text=['# BhashaBench V1 审计状态：尚未下载数据',
              '当前状态：**BLOCKED_MISSING_HF_TOKEN**。此进程及 Windows 用户/系统环境变量均读不到 HF_TOKEN。四个指定仓库均标记 gated=auto，尚未读取受限的数据文件。',
              '已核验四个原始 V1 仓库的公开文件清单，均提供 English/test；未使用 Multi、Hindi 或翻译版本。',
              '| domain | 原始仓库 | 固定 revision | English 评测文件 |',
              '| --- | --- | --- | --- |']
        for domain,meta in metadata.items():
            paths=[f['path'] for f in meta['files'] if f['path'].startswith('English/')]
            text.append(f"| {domain} | {meta['repo_id']} | {meta['revision']} | {', '.join(paths)} |")
        text.extend(['数据统计、200 条样本及 A–E 结论均**未计算**。不以 README 的双语总数填入英文统计。',
                     '可复现脚本：`scripts/audit_bhashabench_v1.py`。在本机设置获准访问四个仓库的用户环境变量 HF_TOKEN 后重跑；token 不写入文件，也不要发送到聊天。',
                     "```powershell\n& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/audit_bhashabench_v1.py\n```",
                     '本状态报告会在完成真实数据审计后被完整报告替换。'])
        (REPORTS/'bhashabench_audit.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
        raise RuntimeError('HF_TOKEN is absent from process/user/machine environment. Set an authorized HF_TOKEN locally; never paste it into chat.')
    print('HF_TOKEN available from',auth_source,'(value not printed)',flush=True)
    for domain,meta in metadata.items():
        # Determine the actual English evaluation configuration from repository metadata.
        configs=meta.get('card',{}).get('configs',[])
        english=[c for c in configs if c.get('config_name','').casefold()=='english']
        if len(english)!=1:
            raise ValueError(f'{domain}: cannot uniquely identify original English config from real metadata')
        eval_splits={x['split'] for x in english[0].get('data_files',[]) if x.get('split') in ['test','evaluation','eval']}
        if len(eval_splits)!=1:
            raise ValueError(f'{domain}: original evaluation split ambiguous: {eval_splits}')
        split=next(iter(eval_splits));folder=DATA/domain;folder.mkdir(parents=True,exist_ok=True)
        dataset=load_dataset(meta['repo_id'],name=english[0]['config_name'],split=split,
                             revision=meta['revision'],token=token,
                             cache_dir=str(ROOT/'data/.hf_cache/bhashabench_v1'),
                             download_config=DownloadConfig(token=token,max_retries=5))
        if len(dataset)==0:raise ValueError(f'{domain}: unexpectedly empty English evaluation set')
        dataset.to_parquet(str(folder/(split+'.parquet')))
        dataset.to_json(str(folder/(split+'.jsonl')),force_ascii=False)
        save(folder/'metadata.json',{'repo':meta['repo_id'],'revision':meta['revision'],
             'config':english[0]['config_name'],'split':split,'rows':len(dataset),
             'utc':datetime.now(timezone.utc).isoformat(),'license':meta['card'].get('license'),
             'auth_source':auth_source,'fields_unchanged':True,'translation_data_used':False})
        save(EVIDENCE/(domain+'_schema.json'),{'columns':dataset.column_names,
             'features':dataset.features.to_dict(),'first_five':dataset.select(range(min(5,len(dataset)))).to_list()})
        print('SCHEMA',domain,dataset.features,flush=True)
        for row in dataset.select(range(min(5,len(dataset)))).to_list():
            print('SAMPLE',json.dumps(row,ensure_ascii=False,default=str)[:4000],flush=True)
        print('DOWNLOADED',domain,split,len(dataset),flush=True)


def import_local():
    """Use the user's explicitly supplied filename-to-domain mapping, never sort heuristics."""
    names={'ayur':'test-00000-of-00001.parquet','legal':'test-00000-of-00001 (1).parquet',
           'finance':'test-00000-of-00001 (2).parquet','krishi':'test-00000-of-00001 (3).parquet'}
    metadata=repositories()
    for domain,name in names.items():
        source=ROOT/'data'/name
        dataset=load_dataset('parquet',data_files={'test':str(source)},split='test',cache_dir=str(ROOT/'data/.hf_cache/bhashabench_v1_local'))
        if set(dataset['language'])!={'en'}:
            raise ValueError(f'{domain}: local file is not exclusively native English')
        folder=DATA/domain;folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,folder/'test.parquet')
        dataset.to_json(str(folder/'test.jsonl'),force_ascii=False)
        meta=metadata[domain]
        expected=[f.get('bytes') for f in meta['files'] if f['path'].startswith('English/') and f['path'].endswith('.parquet')]
        save(folder/'metadata.json',{'repo':REPOS[domain],'reference_revision':meta['revision'],
            'revision_verification':'Reference public metadata only; manually downloaded revision not independently verified',
            'config':'English','split':'test','rows':len(dataset),'utc':datetime.now(timezone.utc).isoformat(),
            'license':meta['card'].get('license'),'source':'user downloaded original repository file',
            'local_input':source.relative_to(ROOT).as_posix(),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'input_bytes':source.stat().st_size,'reference_file_bytes':expected,
            'fields_unchanged':True,'translation_data_used':False})
        save(EVIDENCE/(domain+'_schema.json'),{'columns':dataset.column_names,
             'features':dataset.features.to_dict(),'first_five':dataset.select(range(min(5,len(dataset)))).to_list()})
        print('LOCAL IMPORT',domain,len(dataset),'SCHEMA',dataset.features,flush=True)


def normalized(value):
    # Required definition: lowercase, remove Unicode punctuation, collapse spaces.
    text=str(value or '').lower()
    text=''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    return ' '.join(text.split())

def keyname(value):
    return re.sub(r'[^a-z0-9]','',value.casefold())

def choose_column(columns,aliases,required=False):
    matches=[c for a in aliases for c in columns if keyname(c)==a]
    matches=list(dict.fromkeys(matches))
    if len(matches)>1:
        # Ordered aliases encode explicit preference, while recording actual mapping.
        return matches[0]
    if matches:return matches[0]
    if required:raise ValueError(f'Cannot identify field among {columns}; inspect saved actual schema')
    return None

def parse_options(value):
    if isinstance(value,np.ndarray):
        value=value.tolist()
    if isinstance(value,dict):
        return {str(k).upper().strip():str(v) if v is not None else '' for k,v in value.items()}
    if isinstance(value,list):
        if all(isinstance(x,dict) and 'label' in x and 'text' in x for x in value):
            return {str(x['label']).upper():str(x['text'] or '') for x in value}
        return {chr(65+i):str(v) if v is not None else '' for i,v in enumerate(value)}
    if isinstance(value,str):
        try:
            obj=json.loads(value)
        except json.JSONDecodeError:
            try:obj=ast.literal_eval(value)
            except (ValueError,SyntaxError):return None
        if isinstance(obj,(dict,list)):return parse_options(obj)
    return None

def mappings(frame):
    cols=list(frame.columns)
    q=choose_column(cols,['question','questiontext','questionstem','stem','prompt'],True)
    a=choose_column(cols,['correctanswer','correctoption','answerkey','answer','correctchoice','label'],True)
    o=choose_column(cols,['options','choices','answeroptions'])
    option_cols={}
    for c in cols:
        match=re.fullmatch(r'(?:option|choice)?([abcd])',keyname(c))
        if match:option_cols[match[1].upper()]=c
    if o is None and not option_cols:
        raise ValueError(f'Options not identified from actual schema {cols}; do not guess')
    return {'question':q,'answer':a,'options':o,'option_columns':option_cols,
            'question_type':choose_column(cols,['questiontype','type']),
            'difficulty':choose_column(cols,['difficulty','difficultylevel','questionlevel','level']),
            'domain_fields':[c for c in cols if keyname(c) in ['subject','domain','subjectdomain','topic','category','subsubject','subdomain','subtopic']]}

def answer_mapping(value,options):
    """No guessing of numerical index conventions; unresolved != invalid."""
    if value is None or (isinstance(value,float) and np.isnan(value)) or not str(value).strip():
        return None,'missing'
    if not options:return None,'options_missing_or_unparsed'
    text=str(value).strip()
    stripped=re.sub(r'^(?:option|answer|choice)\s*[:\-]?\s*','',text,flags=re.I).strip('()[] .:')
    if stripped.upper() in options:
        return stripped.upper(),'label'
    matches=[k for k,v in options.items() if text==v.strip()]
    if len(matches)==1:return matches[0],'exact_option_text'
    matches=[k for k,v in options.items() if normalized(text)==normalized(v)]
    if len(matches)==1:return matches[0],'normalized_option_text'
    if len(matches)>1:return None,'ambiguous_option_text'
    if re.fullmatch(r'[0-9]+',text):return None,'numeric_encoding_unresolved'
    if re.fullmatch(r'[A-Z]',stripped.upper()):return None,'invalid_label'
    return None,'answer_not_in_options'

def dist(values):
    arr=np.asarray(values,dtype=float)
    if not len(arr):return {k:None for k in ['mean','median','p10','p90','max']}
    return {'mean':float(arr.mean()),'median':float(np.median(arr)),'p10':float(np.percentile(arr,10)),
            'p90':float(np.percentile(arr,90)),'max':int(arr.max())}

def duplicates(values):
    c=Counter(values)
    return {'groups':sum(n>1 for n in c.values()),'extra_rows':sum(n-1 for n in c.values()),
            'rows_in_groups':sum(n for n in c.values() if n>1)}

def near_duplicates(questions):
    """Approximate MinHash LSH proposals; exact trigram Jaccard confirms pairs.

    64 hash permutations, 16 bands x 4, threshold .85, seed 42. This is an
    approximate candidate search, not an exhaustive guarantee of recall.
    Exact normalized copies are separately counted and not re-labelled near.
    """
    rng=np.random.default_rng(SEED)
    multipliers=rng.integers(1,2**32-1,size=64,dtype=np.uint64)|np.uint64(1)
    additions=rng.integers(0,2**32-1,size=64,dtype=np.uint64)
    buckets=defaultdict(list);grams=[];norms=[];pairs=[];candidates=set()
    for i,q in enumerate(questions):
        text=normalized(q);words=text.split();norms.append(text)
        tokens={' '.join(words[j:j+3]) for j in range(len(words)-2)} if len(words)>=3 else {text} if text else set()
        grams.append(tokens)
        if not tokens:continue
        hashed=np.array([zlib.crc32(t.encode('utf-8')) for t in tokens],dtype=np.uint64)
        signature=((hashed[:,None]*multipliers[None,:]+additions[None,:])&np.uint64(0xffffffff)).min(axis=0)
        for band in range(16):
            key=(band,signature[band*4:band*4+4].tobytes())
            for j in buckets[key]:
                if norms[j]!=text:candidates.add((j,i))
            buckets[key].append(i)
    for j,i in sorted(candidates):
        union=grams[j]|grams[i]
        score=len(grams[j]&grams[i])/len(union) if union else 0
        if score>=.85:pairs.append({'row_a':j,'row_b':i,'trigram_jaccard':score,'question_a':questions[j],'question_b':questions[i]})
    return pairs,len(candidates)


def audit_all():
    summaries=[];types=[];subjects=[];difficulty=[];quality=[];all_questions=[];domain_detail={}
    for domain in REPOS:
        meta=load(DATA/domain/'metadata.json')
        frame=pd.read_parquet(DATA/domain/(meta['split']+'.parquet'))
        mapping=mappings(frame);save(EVIDENCE/(domain+'_field_mapping.json'),mapping)
        records=frame.to_dict('records');questions=[];optionslist=[];answers=[];answerstatus=[]
        optionlens=[];optionwords=[];numeric_flags=[];native_calc=[];special=[];positions=Counter();count_options=Counter()
        issue_rows=[]
        for index,r in enumerate(records):
            value=r.get(mapping['question']);q='' if value is None or (isinstance(value,float) and np.isnan(value)) else str(value)
            options=parse_options(r.get(mapping['options'])) if mapping['options'] else {k:str(r[c] or '') for k,c in mapping['option_columns'].items()}
            a,status=answer_mapping(r.get(mapping['answer']),options)
            questions.append(q);optionslist.append(options);answers.append(a);answerstatus.append(status)
            if a:positions[a]+=1
            count_options[len(options) if options is not None else 'UNPARSED']+=1
            if options:
                optionlens.extend(len(v) for v in options.values());optionwords.extend(len(v.split()) for v in options.values())
            special.append(bool(options) and any(re.search(r'\b(?:all|none)\s+of\s+(?:the\s+)?(?:above|these|following)|\b(?:both|neither)\s+[a-d]\s+(?:and|or)\s+[a-d]\b',v,re.I) for v in options.values()))
            # Observable calculation signals, not a new gold question-type label.
            calculation_cue=bool(re.search(r'\bcalculate|\bcompute|\bsimplify|\bevaluate|\bsolve\b|\bfind the (?:value|ratio|sum|average)|\bhow much|\b(?:simple|compound) interest|\bpercentage\b',q,re.I))
            has_number=bool(re.search(r'\d',q))
            numopts=sum(bool(re.fullmatch(r'[\s\d.,%+\-−/₹$€£()]+',v)) for v in (options or {}).values())
            numeric_flags.append((calculation_cue and has_number) or numopts>=3)
            nativevalues=[str(r[c] or '').strip().casefold() for c in mapping['domain_fields']]
            native_calc.append(any(v in ['problem solving','mathematics for finance'] for v in nativevalues))
            issues=[]
            if not q.strip():issues.append('missing_question')
            if not options:issues.append('missing_or_unparsed_options')
            if options and any(not v.strip() for v in options.values()):issues.append('blank_option')
            if status not in ['label','exact_option_text','normalized_option_text']:issues.append(status)
            if issues:issue_rows.append({'row_index':index,'issues':issues,'question':q,'raw_options':r.get(mapping['options']) if mapping['options'] else options,'raw_answer':r.get(mapping['answer'])})
            all_questions.append({'domain':domain,'row_index':index,'question':q,'normalized':normalized(q)})
        qtypes=Counter(str(r.get(mapping['question_type'])) if mapping['question_type'] else 'FIELD_ABSENT' for r in records)
        for label,n in qtypes.items():types.append({'domain':domain,'question_type_field':mapping['question_type'],'question_type':label,'count':n,'percentage':n/len(frame)})
        for field in mapping['domain_fields']:
            for value,n in Counter(str(r.get(field)) for r in records).items():
                subjects.append({'domain':domain,'field':field,'value':value,'count':n,'percentage':n/len(frame)})
        if mapping['difficulty']:
            for value,n in Counter(str(r.get(mapping['difficulty'])) for r in records).items():difficulty.append({'domain':domain,'difficulty_field':mapping['difficulty'],'value':value,'count':n,'percentage':n/len(frame)})
        labelmcq=[bool(mapping['question_type']) and keyname(str(r.get(mapping['question_type']))) in ['mcq','multiplechoice','multiplechoicequestion','multiplechoicequestions'] for r in records]
        validmcq=[len(o or {})==4 and a in (o or {}) for o,a in zip(optionslist,answers)]
        broad_candidates=[is_mcq and valid for is_mcq,valid in zip(labelmcq,validmcq)]
        conservative=[candidate and not lexical and not native for candidate,lexical,native in zip(broad_candidates,numeric_flags,native_calc)]
        qnorm=[normalized(q) for q in questions]
        qa=[json.dumps([q,o,r.get(mapping['answer'])],ensure_ascii=False,sort_keys=True,default=str) for q,o,r in zip(questions,optionslist,records)]
        qanorm=[json.dumps([normalized(q),{k:normalized(v) for k,v in (o or {}).items()},normalized(r.get(mapping['answer']))],ensure_ascii=False,sort_keys=True) for q,o,r in zip(questions,optionslist,records)]
        full=[json.dumps(r,ensure_ascii=False,sort_keys=True,default=str) for r in records]
        pairs,proposals=near_duplicates(questions)
        save(EVIDENCE/(domain+'_quality_examples.json'),issue_rows)
        save(EVIDENCE/(domain+'_near_duplicates.json'),{'method':'64-permutation word-trigram MinHash LSH 16x4 + exact Jaccard >=0.85; normalized exact copies excluded',
             'candidate_pairs':proposals,'confirmed_pairs':len(pairs),'pairs':pairs})
        qs={'domain':domain,'total_count':len(frame),'missing_question':sum(not q.strip() for q in questions),
            'missing_or_unparsed_options':sum(not o for o in optionslist),'blank_option_questions':sum(bool(o) and any(not v.strip() for v in o.values()) for o in optionslist),
            'missing_answer':answerstatus.count('missing'),'answer_not_in_options':answerstatus.count('answer_not_in_options')+answerstatus.count('invalid_label'),
            'answer_encoding_unresolved':answerstatus.count('numeric_encoding_unresolved')+answerstatus.count('ambiguous_option_text'),
            'near_duplicate_pairs':len(pairs),'near_duplicate_rows':len({r[k] for r in pairs for k in ['row_a','row_b']})}
        for label,values in [('exact_full_row',full),('exact_qa',qa),('normalized_qa',qanorm),('exact_question_stem',questions),('normalized_question_stem',qnorm)]:
            for k,v in duplicates(values).items():qs[label+'_'+k]=v
            qs[label+'_extra_rate']=qs[label+'_extra_rows']/len(frame)
        quality.append(qs)
        item={'domain':domain,'repo':REPOS[domain],'split':meta['split'],'count':len(frame),
              'native_mcq_count':sum(labelmcq),'native_mcq_rate':sum(labelmcq)/len(frame),
              'four_option_valid_answer_count':sum(validmcq),'four_option_valid_answer_rate':sum(validmcq)/len(frame),
              'special_option_count':sum(special),'special_option_rate':sum(special)/len(frame),
              'native_problem_solving_or_finance_math_count':sum(native_calc),
              'lexical_calculation_signal_count':sum(numeric_flags),'lexical_calculation_signal_rate':sum(numeric_flags)/len(frame),
              'knowledge_mcq_broad_candidate_count':sum(broad_candidates),'knowledge_mcq_conservative_proxy_count':sum(conservative),
              'correct_answer_positions':dict(positions),'option_count_distribution':dict(count_options),
              'question_type_distribution':dict(qtypes),'answer_mapping_status':dict(Counter(answerstatus)),
              'field_mapping':mapping}
        for prefix,values in [('question_chars',[len(q) for q in questions]),('question_words',[len(q.split()) for q in questions]),
                              ('option_chars',optionlens),('option_words',optionwords)]:
            for k,v in dist(values).items():item[prefix+'_'+k]=v
        samples=frame.sample(n=min(200,len(frame)),random_state=SEED).copy()
        for c in samples:
            if any(isinstance(v,(dict,list,np.ndarray)) for v in samples[c]):samples[c]=samples[c].map(lambda v:json.dumps(v.tolist() if isinstance(v,np.ndarray) else v,ensure_ascii=False,default=str))
        samples.insert(0,'audit_source_row_index',samples.index)
        samples.to_csv(SAMPLES/(domain+'_200.csv'),index=False,encoding='utf-8-sig')
        domain_detail[domain]=item;summaries.append(item)
        print('AUDITED',domain,json.dumps(item,ensure_ascii=False,default=str),flush=True)
    pd.DataFrame(types).to_csv(REPORTS/'question_type_distribution.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(subjects).to_csv(REPORTS/'subject_distribution.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(difficulty).to_csv(REPORTS/'difficulty_distribution.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(quality).to_csv(REPORTS/'data_quality.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for item in summaries:
        flat={k:json.dumps(v,ensure_ascii=False,default=str) if isinstance(v,(dict,list)) else v for k,v in item.items()}
        rows.append(flat)
    pd.DataFrame(rows).to_csv(REPORTS/'domain_summary.csv',index=False,encoding='utf-8-sig')
    save(EVIDENCE/'audit_statistics.json',{'domains':domain_detail,'quality':quality,'seed':SEED})
    cross_groups=defaultdict(list)
    for row in all_questions:
        if row['normalized']:cross_groups[row['normalized']].append(row)
    save(EVIDENCE/'cross_domain_question_duplicates.json',[
         {'normalized_question':text,'rows':rows} for text,rows in cross_groups.items()
         if len(rows)>1 and len({r['domain'] for r in rows})>1])
    return domain_detail,quality,subjects,types,difficulty


def table(headers,rows):
    def text(x):return str(x).replace('|','\\|').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                      ['| '+' | '.join(map(text,row))+' |' for row in rows])

def write_report(detail,quality,subjects,types,difficulty):
    parts=['# BhashaBench V1 Original English Evaluation Data Audit',
           f'Run UTC: {datetime.now(timezone.utc).isoformat()}. Seed=42. Only the four requested original V1 repositories / English / original evaluation split are used. No Multi data, translations, filtering, deduplication or new splits. No model training.',
           '## 1. Scope, native schemas and field mappings',
           table(['domain','repo','split','actual rows','question field','answer field','options field(s)','native domain fields'],[
               [d,i['repo'],i['split'],i['count'],i['field_mapping']['question'],i['field_mapping']['answer'],i['field_mapping']['options'] or i['field_mapping']['option_columns'],i['field_mapping']['domain_fields']] for d,i in detail.items()]),
           'Full actual features and the first five rows were saved before auditing in `reports/bhashabench_evidence/*_schema.json`; mappings are saved separately. The top-level four domains are dataset-source labels, not newly classified questions. Evaluation-only data are not a pre-existing training dataset for federated fine-tuning.',
           table(['domain','actual columns / types'],[[d,', '.join(f"{c}: {v.get('dtype',v.get('_type'))}" for c,v in load(EVIDENCE/(d+'_schema.json'))['features'].items())] for d in detail]),
           '本次使用用户从指定原始 V1 仓库手工下载的四个 parquet，文件对应关系由用户明确提供。所有行 language 均为 en，四个文件大小均与已记录的 English/test 文件清单一致。输入文件 SHA256、原始字段和行顺序均保存；公开元数据 commit 只是参考 revision，不能独立证明手工下载文件所属 commit，不把它表述为已验证的下载版本。CC-BY-4.0 来自仓库元数据。未再使用 token 下载，未引入 Hindi、Multi 或翻译文件。',
           '## 2. Native question types and task format',
           table(['domain','type','count','percentage'],[[r['domain'],r['question_type'],r['count'],f"{r['percentage']:.2%}"] for r in types]),
           table(['domain','native MCQ %','4 options + valid single answer %','special option %','option counts','correct answer positions'],[
               [d,f"{i['native_mcq_rate']:.2%}",f"{i['four_option_valid_answer_rate']:.2%}",f"{i['special_option_rate']:.2%}",i['option_count_distribution'],i['correct_answer_positions']] for d,i in detail.items()]),
           'Native MCQ is counted from the actual question_type field. Structural MCQ means four options and a resolvable single option answer; fill-in-the-blank/matching/assertion/rearrangement may still use an MCQ output wrapper. These definitions do not equate factual knowledge with four choices. Special options count questions with all/none of the above/these/following, or both/neither A and/or B patterns; transparent regex only, not semantic validation.',
           '## 3. Subject/domain/topic and difficulty',
           table(['domain','actual field','original value','count','percentage'],[[r['domain'],r['field'],r['value'],r['count'],f"{r['percentage']:.2%}"] for r in subjects]),
           table(['domain','difficulty field','value','count','percentage'],[[r['domain'],r['difficulty_field'],r['value'],r['count'],f"{r['percentage']:.2%}"] for r in difficulty]),
           'Every candidate native domain column is counted separately; labels are never merged or reclassified. Missing fields remain absent rather than assumed identical across datasets.',
           '## 4. Question and option lengths',
           table(['domain','Q chars mean / median / p90 / max','Q words mean / median / p90 / max','option chars mean / median / p90 / max','option words mean / median / p90 / max'],[
               [d,*[' / '.join(str(i[p+'_'+k]) for k in ['mean','median','p90','max']) for p in ['question_chars','question_words','option_chars','option_words']]] for d,i in detail.items()]),
           'Words are whitespace-separated; chars use Python len. They are not model-token counts. Option statistics include every provided option, not just the correct option. p10 is also preserved in domain_summary.csv.',
           '## 5. Finance: calculation and problem-solving confounds']
    fin=detail['finance']
    nativefinance=[r for r in subjects if r['domain']=='finance']
    ps_count=sum(r['count'] for r in nativefinance if r['field']=='subject_domain' and r['value']=='Problem Solving')
    math_count=sum(r['count'] for r in nativefinance if r['field']=='subject_domain' and r['value']=='Mathematics for Finance')
    parts.extend([table(['actual field','native value','count','percentage'],[[r['field'],r['value'],r['count'],f"{r['percentage']:.2%}"] for r in nativefinance]),
       f"Unique finance rows with native subject/domain/topic equal to Problem Solving or Mathematics for Finance: {fin['native_problem_solving_or_finance_math_count']} / {fin['count']} = {fin['native_problem_solving_or_finance_math_count']/fin['count']:.2%}. This native-label union is a problem-solving/math-domain indicator, not a measured fraction of questions requiring arithmetic.",
       f"Transparent lexical/numeric calculation signal: {fin['lexical_calculation_signal_count']} / {fin['count']} = {fin['lexical_calculation_signal_rate']:.2%}. Signal = question contains digits AND a calculation cue (calculate/compute/simplify/evaluate/solve/find value/ratio/sum/average/how much/simple or compound interest/percentage), OR at least three options contain only numbers/currency/punctuation. It can overcount dates/numeric facts and miss verbal calculation; it is an approximate screen with unknown precision/recall, not an LLM-generated label.",
       'MCQ count alone is not evidence of knowledge-only QA. Native Problem Solving can include aptitude/logic rather than finance knowledge; other subject counts above are retained to expose source-domain heterogeneity.',
       '## 6. Duplicate and missing-value quality checks',
       table(['domain','exact full-row extra','exact QA extra','normalized QA extra','exact stem extra','normalized stem extra','near pairs','missing Q/options/A','answer not in options','unresolved encoding'],[
           [r['domain'],r['exact_full_row_extra_rows'],r['exact_qa_extra_rows'],r['normalized_qa_extra_rows'],r['exact_question_stem_extra_rows'],r['normalized_question_stem_extra_rows'],r['near_duplicate_pairs'],f"{r['missing_question']} / {r['missing_or_unparsed_options']} / {r['missing_answer']}",r['answer_not_in_options'],r['answer_encoding_unresolved']] for r in quality]),
       'Exact full-row includes all native fields; exact QA ignores IDs/metadata and uses question/options/raw answer; stem excludes separate options and target. Normalization uses lowercase, removal of Unicode punctuation, and whitespace collapse; numbers are retained. Rates in data_quality.csv use extra rows / total, and rows participating in duplicate groups are also recorded. No duplicate is deleted.',
       'Near-duplicate check uses 64-permutation word-trigram MinHash LSH (16 bands ×4, seed42) to propose pairs, then confirms exact word-trigram Jaccard ≥0.85. Normalized exact duplicates are excluded from near counts. This approximate search is not exhaustive and does not establish semantic equivalence; single numeric changes can create different correct answers. Full pair examples and unresolved/invalid answers are saved per domain. Cross-source normalized stem duplicates are saved separately for future leakage review.',
       '## 7. Knowledge-oriented MCQ candidate counts (statistics only)',
       table(['domain','native MCQ + four valid options (broad upper candidates)','same but without native math/PS or calculation signals (conservative proxy)'],[
           [d,i['knowledge_mcq_broad_candidate_count'],i['knowledge_mcq_conservative_proxy_count']] for d,i in detail.items()]),
       'These are transparent rule-based candidate counts, not correctness, factuality or knowledge-purity labels. No rows are filtered/exported as a selected dataset. Neither count is a statistical lower/upper bound on the true number of knowledge-oriented questions: broad candidates still contain calculation, and conservative proxies may discard legitimate factual questions. Context-dependent comprehension, legal reasoning and clinical application require manual inspection.',
       '## 8. Reproducibility and manual samples',
       'Original fields are preserved in data/bhashabench/{domain}/{original_split}.parquet and .jsonl. Four *_200.csv files in analysis_samples are uniform samples without replacement using random_state42, with original source row indexes; fewer than 200 rows would be reported rather than fabricated. Per-domain metadata.json records source/reference revision and license; download_manifest.json records file SHA256 and sizes. HF_TOKEN is read from environment only and never written to reports or data.',
       "```powershell\n& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/audit_bhashabench_v1.py --download-only\n& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/audit_bhashabench_v1.py --audit-existing\n```",
       '## 9. 简洁总结 A–E',
       'A. 四个领域的输出形式基本统一：所有行均有四个非空选项和有效 A/B/C/D 答案，缺失和非法答案均为 0。原生题型 MCQ 占比分别为 '+', '.join(f"{d} {i['native_mcq_rate']:.2%}" for d,i in detail.items())+'；其余填空、匹配、断言推理等也包装成四选一。因此能统一为选择题任务，但不能认定都属于相同的事实问答能力。',
       'B. 原生非 MCQ 比例分别为 '+', '.join(f"{d} {1-i['native_mcq_rate']:.2%}" for d,i in detail.items())+'。Legal 和 Finance 有 Reading Comprehension；Finance 还存在 1 条原生 Essay 标签但仍有四选项和答案，需要人工核验标签。Krishi 匹配和断言推理较多，Ayur 的原生 MCQ 比例最高。Finance 的能力混杂尤其明显。此外 Legal 有 General Academic Subjects，Krishi 有 General Knowledge & Reasoning，不能把来源领域等同于每道题的专业知识纯度。',
       f"C. Finance 的混杂严重且值得优先人工核查：Problem Solving {ps_count} 题、Mathematics for Finance {math_count} 题，合计 {fin['native_problem_solving_or_finance_math_count']} 题（{fin['native_problem_solving_or_finance_math_count']/fin['count']:.2%}）。计算词/数字规则命中 {fin['lexical_calculation_signal_count']} 题（{fin['lexical_calculation_signal_rate']:.2%}）。前者是原生学科标签，后者只是粗筛；两者不可相加，也不等于实际需要计算的题目比例。真实样例包含数列找错以及一般常识题，不能直接视为纯金融知识领域。",
       'D. 只按透明规则统计的候选量（较宽 MCQ 候选 / 排除原生数学、Problem Solving 和计算信号后的代理值）：'+', '.join(f"{d} {i['knowledge_mcq_broad_candidate_count']} / {i['knowledge_mcq_conservative_proxy_count']}" for d,i in detail.items())+'。没有人工审读前，不能给出真实 knowledge-oriented 题数或把这些代理值当作上下界。',
       'E. 下一步人工核查：Finance 的金融知识与通用 aptitude/数学、Essay 标签；Legal 的阅读理解/学术常识及地区与时效性；Krishi 的匹配/推理和一般常识；Ayur 的术语及重复。另需检查特殊 all/none 选项、近重复中的数字/答案变化、题干重复但选项不同、源文本乱码/标点异常。每域 200 条随机样本已保存；近重复证据可额外定向检查。',
       table(['domain','normalized QA extra / rate','normalized stem extra / rate','near pairs'],[[r['domain'],f"{r['normalized_qa_extra_rows']} / {r['normalized_qa_extra_rate']:.2%}",f"{r['normalized_question_stem_extra_rows']} / {r['normalized_question_stem_extra_rate']:.2%}",r['near_duplicate_pairs']] for r in quality]),
       '适用性判断：原生来源领域、统一四选一接口和约万题规模使其值得继续验证，但目前只能定位为有领域/能力混杂的英文考试选择题集合，尚不能直接作为严格受控的纯知识跨领域 benchmark。四份均是原始 test，数据本身未提供此次可用的 train；未来若讨论训练和独立评估，需要另行明确实验协议。尤其应避免把官方评测集训练后继续宣称为官方独立测试。本次完全未划分、过滤或训练。',
       f"跨领域规范化题干重复共有 {len(load(EVIDENCE/'cross_domain_question_duplicates.json'))} 组，证据保存在 cross_domain_question_duplicates.json；尚未创建 train/test，故没有虚构 train-test 泄漏率。",
       "本次离线重现命令：\n```powershell\n& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/audit_bhashabench_v1.py --import-local\n```\n也可使用 --audit-existing 直接重新审计已归档的数据。"])
    (REPORTS/'bhashabench_audit.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    manifest=[]
    for domain in REPOS:
        for p in sorted((DATA/domain).glob('*')):
            if p.is_file():manifest.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    save(EVIDENCE/'download_manifest.json',manifest)
    for path in [REPORTS/'bhashabench_audit.md',REPORTS/'domain_summary.csv',REPORTS/'question_type_distribution.csv',REPORTS/'subject_distribution.csv',REPORTS/'data_quality.csv']:
        assert path.exists() and path.stat().st_size>0
        print('OUTPUT',path.resolve(),path.stat().st_size,'bytes',flush=True)
    print('FINAL SUMMARY',flush=True)
    for d,i in detail.items():print(d,'count=',i['count'],'MCQ=',i['native_mcq_count'],'knowledge candidates=',i['knowledge_mcq_broad_candidate_count'],'conservative proxy=',i['knowledge_mcq_conservative_proxy_count'],flush=True)
    print('FINANCE native ProblemSolving/math=',fin['native_problem_solving_or_finance_math_count'],'lexical calc=',fin['lexical_calculation_signal_count'],flush=True)


def verify_outputs():
    verified=[]
    for domain in REPOS:
        metadata=load(DATA/domain/'metadata.json')
        frame=pd.read_parquet(DATA/domain/(metadata['split']+'.parquet'))
        restored=pd.read_json(DATA/domain/(metadata['split']+'.jsonl'),lines=True,dtype=False)
        pd.testing.assert_frame_equal(frame,restored,check_dtype=False)
        sample=pd.read_csv(SAMPLES/(domain+'_200.csv'),keep_default_na=False)
        expected=frame.sample(n=min(200,len(frame)),random_state=SEED)
        assert sample.audit_source_row_index.tolist()==expected.index.tolist()
        assert len(sample)==min(200,len(frame))
        for c in frame:
            assert sample[c].tolist()==expected[c].tolist(),(domain,c)
        for p in [DATA/domain/(metadata['split']+'.parquet'),DATA/domain/(metadata['split']+'.jsonl'),DATA/domain/'metadata.json',SAMPLES/(domain+'_200.csv')]:
            verified.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size})
    for p in [REPORTS/'bhashabench_audit.md',REPORTS/'domain_summary.csv',REPORTS/'question_type_distribution.csv',REPORTS/'subject_distribution.csv',REPORTS/'data_quality.csv',ROOT/'scripts/audit_bhashabench_v1.py']:
        assert p.is_file() and p.stat().st_size>0
        verified.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size})
    save(EVIDENCE/'output_verification.json',{'parquet_jsonl_identical':True,'sample_seed_and_values_verified':True,'files':verified})
    for row in verified:print('VERIFIED FILE',row['path'],row['bytes'],'bytes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--audit-existing',action='store_true')
    parser.add_argument('--import-local',action='store_true',help='Import the four user-downloaded files using their explicit filename mapping')
    args=parser.parse_args()
    if args.import_local:import_local()
    elif not args.audit_existing:download_all()
    if not args.download_only:
        write_report(*audit_all())
        verify_outputs()
