"""Reproducible source discovery and full-file label audit; never trains models."""
import argparse
import collections
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import requests
import pyarrow.parquet as pq
import tarfile
import re
from huggingface_hub import HfApi, hf_hub_download

os.environ.setdefault('HF_HUB_DISABLE_XET','1')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/dataset_search/evidence'
OUT.mkdir(parents=True,exist_ok=True)
session = requests.Session()
session.headers['User-Agent']='QA-dataset-research-audit/1.0'

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def fetch(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with session.get(url,timeout=180,stream=True) as r:
            r.raise_for_status()
            with path.open('wb') as f:
                for block in r.iter_content(1024*1024):
                    f.write(block)
    return path

def audit(name,rows,fields,source,split):
    if isinstance(rows,dict):
        rows=list(rows.values())
    result={'name':name,'source':source,'split':split,'rows':len(rows),'columns':sorted(set(k for row in rows for k in row)),
            'first_two_rows':rows[:2],'fields':{},'audit_scope':'entire downloaded file','utc':datetime.now(timezone.utc).isoformat()}
    for f in fields:
        values=[r.get(f) for r in rows]
        counter=collections.Counter(json.dumps(v,ensure_ascii=False,sort_keys=True) for v in values)
        result['fields'][f]={'unique_values_including_null':len(counter),'null_or_absent':sum(v is None for v in values),
                             'blank':sum(isinstance(v,str) and not v.strip() for v in values),
                             'counts':dict(counter.most_common())}
    save(name+'_'+split+'_audit.json',result)
    print(name,split,len(rows),result['columns'],{f:{k:v for k,v in a.items() if k!='counts'} for f,a in result['fields'].items()},flush=True)
    return result

def discover():
    api=HfApi()
    repos=['m-a-p/SuperGPQA','cais/mmlu','TIGER-Lab/MMLU-Pro','MegaScience/TextbookReasoning','MegaScience/MegaScience','openlifescienceai/medmcqa','lupantech/ScienceQA','mandarjoshi/trivia_qa','google-research-datasets/natural_questions','facebook/kilt_tasks','akariasai/PopQA','qanta','exams']
    def hf(repo):
        try:
            info=api.dataset_info(repo,files_metadata=True)
            result={'repo_id':repo,'revision':info.sha,'card':info.card_data.to_dict() if info.card_data else {},'files':[{'path':x.rfilename,'bytes':x.size} for x in info.siblings]}
        except Exception as e:
            result={'repo_id':repo,'error':str(e)}
        save('hf_'+repo.replace('/','__')+'.json',result)
        print(repo, json.dumps(result,ensure_ascii=False)[:3500],flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(hf,repos))
    for repo in ['Pinafore/qb','lupantech/ScienceQA','mhardalov/exams-qa','nlpdata/examqa','princeton-nlp/EntityQuestions','GAIR-NLP/MegaScience']:
        try:
            meta=session.get('https://api.github.com/repos/'+repo,timeout=40).json()
            branch=meta['default_branch']
            commit=session.get(f'https://api.github.com/repos/{repo}/commits/{branch}',timeout=40).json()['sha']
            tree=session.get(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1',timeout=60).json()
            save('github_'+repo.replace('/','__')+'.json',{'repo':repo,'revision':commit,'tree':tree})
            print('GITHUB',repo,commit,[(x['path'],x.get('size')) for x in tree.get('tree',[]) if x['path'].endswith(('.json','.jsonl','.csv','.parquet','.zip'))][:60],flush=True)
        except Exception as e:
            print('GITHUB ERROR',repo,str(e),flush=True)

def audit_parquet(name,path,fields,source,split):
    pf=pq.ParquetFile(path)
    counters={f:collections.Counter() for f in fields}
    nulls=collections.Counter()
    blanks=collections.Counter()
    samples=[]
    rows=0
    for batch in pf.iter_batches(batch_size=4096):
        batch_rows=batch.to_pylist()
        samples.extend(batch_rows[:max(0,2-len(samples))])
        rows+=len(batch_rows)
        for row in batch_rows:
            for f in fields:
                v=row.get(f)
                counters[f][json.dumps(v,ensure_ascii=False,sort_keys=True)]+=1
                nulls[f]+=v is None
                blanks[f]+=isinstance(v,str) and not v.strip()
    result={'name':name,'source':source,'split':split,'rows':rows,'columns':pf.schema_arrow.names,'schema':str(pf.schema_arrow),'first_two_rows':samples,
            'fields':{f:{'unique_values_including_null':len(counters[f]),'null_or_absent':nulls[f],'blank':blanks[f],'counts':dict(counters[f].most_common())} for f in fields},
            'audit_scope':'entire downloaded Parquet, all row groups','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'utc':datetime.now(timezone.utc).isoformat()}
    save(name+'_'+split+'_audit.json',result)
    print(name,split,rows,result['columns'],{f:dict(counters[f].most_common(40)) for f in fields},flush=True)

def full_audits():
    jobs=[('SuperGPQA','m-a-p/SuperGPQA','SuperGPQA-all.jsonl',['discipline','field','subfield'],'train'),
          ('MMLU','cais/mmlu','all/auxiliary_train-00000-of-00001.parquet',['subject'],'auxiliary_train'),
          ('MMLU','cais/mmlu','all/dev-00000-of-00001.parquet',['subject'],'dev'),
          ('MMLU-Pro','TIGER-Lab/MMLU-Pro','data/validation-00000-of-00001.parquet',['category'],'validation'),
          ('QANTA','qanta','mode=full,char_skip=25/guesstrain-00000-of-00001.parquet',['category','subcategory'],'guesstrain'),
          ('QANTA','qanta','mode=full,char_skip=25/buzztrain-00000-of-00001.parquet',['category','subcategory'],'buzztrain'),
          ('TextbookReasoning','MegaScience/TextbookReasoning','data/train-00000-of-00001.parquet',['subject'],'train'),
          ('MedMCQA','openlifescienceai/medmcqa','data/train-00000-of-00001.parquet',['subject_name'],'train')]
    def job(spec):
        name,repo,filename,fields,split=spec
        try:
            meta=json.loads((OUT/('hf_'+repo.replace('/','__')+'.json')).read_text(encoding='utf-8'))
            path=Path(hf_hub_download(repo,filename,repo_type='dataset',revision=meta['revision'],local_dir=str(OUT/'downloads'/name)))
            source={'repo':repo,'revision':meta['revision'],'file':filename,'url':f"https://huggingface.co/datasets/{repo}/blob/{meta['revision']}/{filename}"}
            if path.suffix=='.parquet':
                audit_parquet(name,path,fields,source,split)
            else:
                rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                result=audit(name,rows,fields,source,split)
                result['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                save(name+'_'+split+'_audit.json',result)
        except Exception as e:
            save(name+'_'+split+'_ERROR.json',{'error':str(e)})
            print('AUDIT ERROR',name,split,str(e),flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(job,jobs))
    scientific_audits()

def scientific_audits():
    gh=json.loads((OUT/'github_lupantech__ScienceQA.json').read_text(encoding='utf-8'))
    base=f"https://raw.githubusercontent.com/lupantech/ScienceQA/{gh['revision']}/data/scienceqa/"
    problems=json.loads(fetch(base+'problems.json',OUT/'downloads/ScienceQA/problems.json').read_text(encoding='utf-8'))
    splits=json.loads(fetch(base+'pid_splits.json',OUT/'downloads/ScienceQA/pid_splits.json').read_text(encoding='utf-8'))
    for split,ids in splits.items():
        if not isinstance(ids,list):
            continue
        rows=[problems[str(i)] for i in ids]
        result=audit('ScienceQA',rows,['subject','topic','category','split'],{'repo':gh['repo'],'revision':gh['revision'],'url':base+'problems.json'},split)
        result['context_counts']={'with_image':sum(bool(r.get('image')) for r in rows),'with_hint':sum(bool(r.get('hint')) for r in rows),'no_image_no_hint':sum(not r.get('image') and not r.get('hint') for r in rows)}
        result['no_image_no_hint_topics']=dict(collections.Counter(r.get('topic') for r in rows if not r.get('image') and not r.get('hint')))
        save('ScienceQA_'+split+'_audit.json',result)
    gh=json.loads((OUT/'github_mhardalov__exams-qa.json').read_text(encoding='utf-8'))
    paths=[x['path'] for x in gh['tree']['tree'] if 'multilingual' in x['path'] and 'train' in x['path'] and 'para' not in x['path']]
    for filename in paths:
        url=f"https://raw.githubusercontent.com/{gh['repo']}/{gh['revision']}/{filename}"
        path=fetch(url,OUT/'downloads/EXAMS'/Path(filename).name)
        with tarfile.open(path,'r:gz') as tar:
            rows=[]
            for member in tar.getmembers():
                if member.isfile():
                    rows.extend(json.loads(line) for line in tar.extractfile(member) if line.strip())
        projected=[{**r,'native_subject':r.get('info',{}).get('subject'),'native_language':r.get('info',{}).get('language')} for r in rows]
        audit('EXAMS',projected,['native_subject','native_language'],{'repo':gh['repo'],'revision':gh['revision'],'url':url},'train')

def qblink():
    url='https://sites.google.com/view/qanta/projects/qblink'
    html=session.get(url,timeout=60).text
    links=sorted(set(re.findall(r'https://drive.google.com/[^"<> ]+',html)))
    save('QBLink_official_download_links.json',{'source':url,'links':links})
    print('QBLINK LINKS',links,flush=True)
    for link in links:
        match=re.search(r'(?:/d/|id=)([A-Za-z0-9_-]+)',link)
        if not match:
            continue
        fid=match.group(1)
        try:
            path=fetch('https://drive.google.com/uc?export=download&id='+fid,OUT/'downloads/QBLink'/(fid+'.json'))
            rows=json.loads(path.read_text(encoding='utf-8'))
            result=audit('QBLink',rows,['category','sub-category'],{'official_page':url,'drive_url':link,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},fid)
        except Exception as e:
            print('QBLINK ERROR',fid,str(e),flush=True)

def secondary():
    api=HfApi()
    print('QBLINK HF REPOS',[x.id for x in api.list_datasets(search='qblink')],flush=True)
    jobs=[('TriviaQA','mandarjoshi/trivia_qa','rc.nocontext','train'),('NQ','google-research-datasets/natural_questions','default','train'),
          ('NQ-open','google-research-datasets/nq_open','default','train'),('zsRE-KILT','facebook/kilt_tasks','structured_zeroshot','train'),
          ('NaturalReasoning','facebook/natural_reasoning','default','train'),('MegaScience','MegaScience/MegaScience','default','train'),
          ('SciKnowEval','hicai-zju/SciKnowEval','default','train'),('PopQA','akariasai/PopQA','default','test')]
    def job(spec):
        name,repo,config,split=spec
        result={'name':name,'repo':repo,'config':config,'requested_split':split,'scope':'Hub metadata plus real dataset-server first rows, not full label audit','utc':datetime.now(timezone.utc).isoformat()}
        try:
            info=api.dataset_info(repo,files_metadata=True)
            result['revision']=info.sha
            result['files']=[{'path':s.rfilename,'bytes':s.size} for s in info.siblings]
            for endpoint in ['splits','first-rows']:
                params={'dataset':repo}
                if endpoint=='first-rows':
                    params.update({'config':config,'split':split})
                response=session.get('https://datasets-server.huggingface.co/'+endpoint,params=params,timeout=90)
                result[endpoint]={'http_status':response.status_code,'data':response.json()}
            if result['first-rows']['http_status']==200:
                result['first-rows']['data']['rows']=result['first-rows']['data'].get('rows',[])[:2]
        except Exception as e:
            result['error']=str(e)
        save(name+'_server_evidence.json',result)
        data=result.get('first-rows',{}).get('data',{})
        print('SERVER',name,result.get('revision'),result.get('splits',{}).get('data',{}).get('splits'),[(x.get('name'),x.get('type')) for x in data.get('features',[])],result.get('error'),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(job,jobs))
    gh=json.loads((OUT/'github_Pinafore__qb.json').read_text(encoding='utf-8'))
    for filename in ['qanta/ingestion/classifier.py','qanta/ingestion/normalization.py','qanta/ingestion/pipeline.py','qanta/ingestion/quizdb.py','qanta/ingestion/protobowl.py']:
        url=f"https://raw.githubusercontent.com/{gh['repo']}/{gh['revision']}/{filename}"
        fetch(url,OUT/'source_code'/Path(filename).name)
    gh=json.loads((OUT/'github_nlpdata__examqa.json').read_text(encoding='utf-8'))
    fetch(f"https://raw.githubusercontent.com/{gh['repo']}/{gh['revision']}/README.md",OUT/'source_code/ExamQA_README.md')
    gh=json.loads((OUT/'github_princeton-nlp__EntityQuestions.json').read_text(encoding='utf-8'))
    fetch(f"https://raw.githubusercontent.com/{gh['repo']}/{gh['revision']}/relation_query_templates.json",OUT/'source_code/EntityQuestions_relation_query_templates.json')

def qblink_archive():
    ids={'train':'e2b2c19c-890e-44db-8ec6-a3cb4687d40a','dev':'f8a64c59-380c-449e-8b6c-bfa277f273a8','test':'72f9cfd0-ddc8-42cf-9210-8ba00cc59cb3'}
    for split,fid in ids.items():
        url=f'https://api.drum.lib.umd.edu/server/api/core/bitstreams/{fid}/content'
        path=fetch(url,OUT/'downloads/QBLink'/('QBLink-'+split+'.json'))
        rows=json.loads(path.read_text(encoding='utf-8'))
        print('QBLINK ACTUAL KEYS',split,rows[0].keys(),flush=True)
        result=audit('QBLink',rows,['category','sub_category','sub-category'],{'official_archive':'https://drum.lib.umd.edu/items/fec38294-978e-4892-ae32-f56f08337d86','url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},split)
        qkeys=[k for k in rows[0] if re.match(r'(?:q|question)[123]$',k)]
        result['qa_keys']=qkeys
        result['qa_pairs']=sum(sum(isinstance(r.get(k),dict) for k in qkeys) for r in rows)
        save('QBLink_'+split+'_audit.json',result)

def expertqa():
    repo='chaitanyamalaviya/ExpertQA'
    meta=session.get(f'https://api.github.com/repos/{repo}',timeout=60).json()
    revision=session.get(f"https://api.github.com/repos/{repo}/commits/{meta['default_branch']}",timeout=60).json()['sha']
    tree=session.get(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1',timeout=60).json()
    save('github_ExpertQA.json',{'repo':repo,'revision':revision,'tree':tree})
    paths=[x['path'] for x in tree['tree'] if 'data/lfqa/' in x['path'] and x['type']=='blob']
    print('EXPERTQA FILES',paths,flush=True)
    paths += ['data/r2_compiled_anon.jsonl']
    for filename in paths:
        url=f'https://raw.githubusercontent.com/{repo}/{revision}/{filename}'
        path=fetch(url,OUT/'downloads/ExpertQA'/Path(filename).name)
        if path.suffix not in {'.jsonl','.json'}:
            continue
        text=path.read_text(encoding='utf-8')
        serialization='json'
        try:
            rows=json.loads(text)
        except json.JSONDecodeError:
            rows=[json.loads(line) for line in text.splitlines() if line.strip()]
            serialization='jsonl (including some files with .json extension)'
        if isinstance(rows,dict):
            print('EXPERTQA DICT',filename,list(rows)[:10],flush=True)
            save('ExpertQA_'+path.stem+'_structure.json',rows)
            continue
        projected=[{**r,'native_field':r.get('metadata',{}).get('field'),'native_specific_field':r.get('metadata',{}).get('specific_field')} for r in rows]
        audit('ExpertQA',projected,['native_field','native_specific_field'],{'repo':repo,'revision':revision,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'actual_serialization':serialization},path.stem)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--discover',action='store_true')
    parser.add_argument('--full-audits',action='store_true')
    parser.add_argument('--qblink',action='store_true')
    parser.add_argument('--secondary',action='store_true')
    parser.add_argument('--scientific-audits',action='store_true')
    parser.add_argument('--qblink-archive',action='store_true')
    parser.add_argument('--expertqa',action='store_true')
    args=parser.parse_args()
    if args.discover:
        discover()
    if args.full_audits:
        full_audits()
    if args.qblink:
        qblink()
    if args.secondary:
        secondary()
    if args.scientific_audits:
        scientific_audits()
    if args.qblink_archive:
        qblink_archive()
    if args.expertqa:
        expertqa()
