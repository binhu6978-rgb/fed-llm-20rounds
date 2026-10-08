"""Pinned full-file QA audit; seed 42; no classifiers, training or client splits."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import time
import requests
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/domain_qa_audit'
RAW=ROOT/'data/raw'
OUT.mkdir(parents=True,exist_ok=True)
SEED=42
SITES={'biology':'biology.stackexchange.com','chemistry':'chemistry.stackexchange.com',
       'cs / computer science':'cs.stackexchange.com','economics':'economics.stackexchange.com',
       'health / medical sciences':'health.stackexchange.com','history':'history.stackexchange.com',
       'law':'law.stackexchange.com','physics':'physics.stackexchange.com',
       'philosophy':'philosophy.stackexchange.com','stats / statistics':'stats.stackexchange.com'}

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

def fetch(url,path,size=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size==size):
        return path
    partial=path.with_suffix(path.suffix+'.part')
    for attempt in range(6):
        try:
            # Query only refreshes a CDN redirect; source commit remains pinned.
            address=url+('?' if '?' not in url else '&')+f'download=true&audit_refresh={time.time_ns()}'
            with requests.get(address,stream=True,timeout=(30,180)) as r:
                r.raise_for_status()
                with partial.open('wb') as f:
                    for block in r.iter_content(1024*1024):
                        f.write(block)
            if size is not None and partial.stat().st_size!=size:
                raise ValueError(f'Unexpected bytes {partial.stat().st_size} != {size}')
            partial.replace(path)
            print('DOWNLOADED',path.relative_to(ROOT),path.stat().st_size,flush=True)
            return path
        except Exception as e:
            print('RETRY',path.name,attempt+1,type(e).__name__,flush=True)
            if attempt==5:
                raise
            time.sleep(min(2**attempt,15))

def download():
    ensure_sources()
    mint=load(OUT/'mintaka_github_repository.json')
    jobs=[]
    for f in mint['tree']['tree']:
        if f['path'].startswith('data/mintaka_') and f['path'].endswith('.json'):
            url=f"https://raw.githubusercontent.com/{mint['repo']}/{mint['revision']}/{f['path']}"
            jobs.append((url,RAW/'mintaka'/Path(f['path']).name,f['size']))
    stack=load(OUT/'stackexchange_repository.json')
    for site in SITES.values():
        files=[f for f in stack['files'] if f['path'].startswith('data/'+site+'/') and f['path'].endswith('.parquet')]
        if not files:
            print('NOT FOUND',site,flush=True)
        for f in files:
            jobs.append((f"https://huggingface.co/datasets/{stack['repo']}/resolve/{stack['revision']}/{f['path']}",RAW/'stack_exchange_preferences'/f['path'],f['bytes']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for p in pool.map(lambda j:fetch(*j),jobs):
            pass
    print('DOWNLOAD COMPLETE',len(jobs),'files',flush=True)


def ensure_sources():
    """Reuse saved versions; only resolve current revisions when no manifest exists."""
    from huggingface_hub import HfApi
    for name,repo in [('mintaka','AmazonScience/mintaka'),('stackexchange','HuggingFaceH4/stack-exchange-preferences')]:
        if not (OUT/(name+'_repository.json')).exists():
            info=HfApi().dataset_info(repo,files_metadata=True)
            save(name+'_repository.json',{'repo':repo,'revision':info.sha,'card':info.card_data.to_dict() if info.card_data else {},
                                        'files':[{'path':f.rfilename,'bytes':f.size} for f in info.siblings]})
    if not (OUT/'mintaka_github_repository.json').exists():
        repo='amazon-science/mintaka'
        r=requests.get(f'https://api.github.com/repos/{repo}/commits/main',timeout=60);r.raise_for_status();revision=r.json()['sha']
        r=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1',timeout=60);r.raise_for_status()
        save('mintaka_github_repository.json',{'repo':repo,'revision':revision,'tree':r.json()})
    for name,filename in [('mintaka','mintaka.py'),('stackexchange','README.md')]:
        dest=OUT/(name+'_'+filename.replace('.','_')+'.txt')
        d=load(OUT/(name+'_repository.json'))
        fetch(f"https://huggingface.co/datasets/{d['repo']}/resolve/{d['revision']}/{filename}",dest)


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts=[]
    def handle_starttag(self,tag,attrs):
        self.parts.append(' ')
    def handle_endtag(self,tag):
        self.parts.append(' ')
    def handle_data(self,data):
        self.parts.append(data)

def plain(value):
    if value is None:
        return ''
    p=TextParser()
    p.feed(str(value))
    return ' '.join(''.join(p.parts).split())

def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):
        return {k:None for k in ['n','mean','median','p10','p90','max']}
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'p10':float(np.percentile(a,10)),'p90':float(np.percentile(a,90)),'max':int(a.max())}

def lengths(values):
    return {'chars':distribution([len(v) for v in values]),
            'words':distribution([len(v.split()) for v in values])}

def observed_schema(value):
    if isinstance(value,dict):
        return {k:observed_schema(v) for k,v in value.items()}
    if isinstance(value,list):
        return {'type':'list','first_item':observed_schema(value[0]) if value else 'empty list'}
    return type(value).__name__

def write_jsonl(name,rows):
    with (OUT/name).open('w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n')

def mintaka():
    allrows=[]; splits={}; schema={}; splitrows={}
    for split,filename in [('train','mintaka_train.json'),('validation','mintaka_dev.json'),('test','mintaka_test.json')]:
        rows=load(RAW/'mintaka'/filename)
        if not rows or not all(k in rows[0] for k in ['question','answer','category','complexityType']):
            raise ValueError('Unexpected actual Mintaka schema; do not guess a mapping')
        schema[split]={'columns':sorted(set(k for r in rows for k in r)),
                       'first_row_schema':observed_schema(rows[0]),'first_five_rows':rows[:5]}
        projected=[]
        for r in rows:
            a=r['answer']
            if not isinstance(a,dict) or 'mention' not in a:
                raise ValueError('Missing answer.mention; record schema discrepancy before proceeding')
            projected.append({'id':r['id'],'question':r['question'],'answer_text':a['mention'],
                              'category':r['category'],'complexityType':r['complexityType'],
                              'answer_type':a.get('answerType'),'split':split,'lang':'en'})
        splitrows[split]=projected;allrows.extend(projected)
        qc=collections.Counter(r['question'] for r in projected)
        translation_languages=sorted(set(k for r in rows for k in r.get('translations',{})))
        splits[split]={'rows':len(rows),'category_counts':dict(collections.Counter(r['category'] for r in rows)),
                       'complexity_counts':dict(collections.Counter(r['complexityType'] for r in rows)),
                       'answer_types':dict(collections.Counter(r['answer']['answerType'] for r in rows)),
                       'empty_question':sum(not str(r['question'] or '').strip() for r in rows),
                       'empty_answer':sum(not str(r['answer']['mention'] or '').strip() for r in rows),
                       'duplicate_question_excess':sum(n-1 for q,n in qc.items() if q),
                       'duplicate_question_rows':sum(n for q,n in qc.items() if q and n>1),
                       'question_lengths':lengths([r['question'] or '' for r in projected]),
                       'answer_lengths':lengths([r['answer_text'] or '' for r in projected]),
                       'translation_languages':translation_languages,
                       'translation_nonempty_counts':{k:sum(bool(r.get('translations',{}).get(k)) for r in rows) for k in translation_languages}}
    df=pd.DataFrame(allrows)
    path=ROOT/'data/processed/mintaka_en.parquet';path.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(path,index=False)
    cats=sorted(df.category.unique());comps=sorted(df.complexityType.unique());out=[];samples=[]
    rng=random.Random(SEED)
    for c in cats:
        sub=df[df.category==c];tr=sub[sub.split=='train'];totalq=collections.Counter(sub.question)
        item={'category':c,'train_count':len(tr),'dev_count':int((sub.split=='validation').sum()),
              'test_count':int((sub.split=='test').sum()),'total_count':len(sub),
              'null_or_blank_category_count':int(sub.category.isna().sum())+int((sub.category=='').sum()),
              'train_duplicate_question_excess':len(tr)-tr.question.nunique(),
              'total_duplicate_question_excess':len(sub)-sub.question.nunique()}
        for prefix,values in [('train_question',tr.question.fillna('')),('train_answer',tr.answer_text.fillna('')),
                              ('total_question',sub.question.fillna('')),('total_answer',sub.answer_text.fillna(''))]:
            for unit,stats in lengths(list(values)).items():
                for k,v in stats.items():
                    item[prefix+'_'+unit+'_'+k]=v
        aw=tr.answer_text.fillna('').map(lambda x:len(x.split()))
        item.update({'train_answer_le_1_word_rate':float((aw<=1).mean()),'train_answer_le_5_words_rate':float((aw<=5).mean()),
                     'train_answer_le_10_words_rate':float((aw<=10).mean()),'train_answer_gt_20_words_count':int((aw>20).sum())})
        out.append(item)
        choices=rng.sample(tr.to_dict('records'),min(20,len(tr)))
        samples.extend({**r,'sample_seed':SEED,'sampling_pool':'English train within category'} for r in choices)
    pd.DataFrame(out).to_csv(OUT/'mintaka_category_stats.csv',index=False,encoding='utf-8-sig')
    cross=df.groupby(['split','category','complexityType']).size().rename('count').reset_index()
    # Explicit zero cells, so absence of a complexity type cannot be hidden.
    idx=pd.MultiIndex.from_product([list(splits),cats,comps],names=['split','category','complexityType'])
    cross=cross.set_index(['split','category','complexityType']).reindex(idx,fill_value=0).reset_index()
    cross['category_split_count']=cross.apply(lambda r:splits[r['split']]['category_counts'].get(r['category'],0),axis=1)
    cross['percentage_within_category_split']=cross['count']/cross.category_split_count
    cross.to_csv(OUT/'mintaka_category_complexity.csv',index=False,encoding='utf-8-sig')
    df.groupby(['split','complexityType']).size().rename('count').reset_index().to_csv(OUT/'mintaka_complexity_distribution.csv',index=False,encoding='utf-8-sig')
    write_jsonl('mintaka_samples.jsonl',samples)
    qc=collections.Counter(df.question)
    overlap=set(r['question'] for r in splitrows['train'])&set(r['question'] for r in splitrows['test'])
    duplicates=[{'question':q,'count':n,'rows':[{'id':r['id'],'split':r['split'],'category':r['category']} for r in allrows if r['question']==q]} for q,n in qc.items() if n>1]
    save('mintaka_duplicate_questions.json',duplicates)
    save('mintaka_train_test_overlap.json',sorted(overlap))
    result={'splits':splits,'categories':cats,'complexityTypes':comps,'length_scope':'English question and original English answer.mention',
            'total_rows':len(df),'all_splits_duplicate_question_excess':len(df)-df.question.nunique(),
            'train_test_exact_unique_question_overlap':len(overlap),
            'train_test_overlap_train_rows':sum(r['question'] in overlap for r in splitrows['train']),
            'train_test_overlap_test_rows':sum(r['question'] in overlap for r in splitrows['test']),
            'category_stats':out,'answer_lengths':lengths(list(df.answer_text.fillna('')))}
    save('mintaka_schema.json',schema);save('mintaka_statistics.json',result)
    print('MINTAKA COUNTS',flush=True)
    for r in out:
        print(r['category'],r['train_count'],r['dev_count'],r['test_count'],flush=True)
    return result


CODE_BLOCK=re.compile(r'<pre\b|```|~~~',re.I)
ANY_CODE=re.compile(r'<(?:pre|code)\b|```|~~~',re.I)
URL=re.compile(r'https?://|www\.',re.I)
# Surface-string indicators only; they are not question-type or domain labels.
DEBUG_LITERAL=re.compile(r'\b(?:debug(?:ging)?|traceback|exception|compiler|compile|segmentation fault)\b',re.I)
ADVICE_LITERAL=re.compile(r'\b(?:should I|what should|recommend(?:ation|ations)?|advice|best way|in your opinion)\b',re.I)

def stackexchange():
    schema={};sites=[];samples=[];scores=[];quality=[]
    dbpath=ROOT/'data/processed/stackexchange_question_audit.sqlite'
    con=sqlite3.connect(dbpath)
    con.execute('DROP TABLE IF EXISTS questions')
    con.execute('CREATE TABLE questions (site TEXT,qid INTEGER,question_html TEXT,question_text TEXT)')
    con.execute('PRAGMA journal_mode=OFF');con.execute('PRAGMA synchronous=OFF')
    for requested,site in SITES.items():
        repository=load(OUT/'stackexchange_repository.json')
        expected=[f for f in repository['files'] if f['path'].startswith('data/'+site+'/') and f['path'].endswith('.parquet')]
        paths=sorted((RAW/'stack_exchange_preferences/data'/site).glob('*.parquet'))
        if not expected:
            sites.append({'requested_domain':requested,'site':site,'status':'NOT FOUND','question_count':0})
            continue
        for f in expected:
            p=RAW/'stack_exchange_preferences'/f['path']
            if not p.exists() or p.stat().st_size!=f['bytes']:
                raise ValueError(f'Incomplete download: {p}; do not report partial site counts or NOT FOUND')
        if len(paths)!=len(expected):
            raise ValueError(f'Unexpected number of local shards for {site}')
        vectors=collections.defaultdict(list);flags=collections.Counter();scorecounts=collections.Counter();answercounts=collections.Counter()
        rng=random.Random(SEED);reservoir=[];n=0;first=[];source_urls=collections.Counter();root_keys=set();answer_keys=set()
        for p in paths:
            pf=pq.ParquetFile(p)
            if not {'qid','question','answers','metadata'}.issubset(pf.schema_arrow.names):
                raise ValueError(f'Unexpected actual schema in {p}; do not silently map missing fields')
            schema[p.relative_to(ROOT).as_posix()]={'rows':pf.metadata.num_rows,'schema':str(pf.schema_arrow)}
            for batch in pf.iter_batches(batch_size=512):
                records=[]
                for r in batch.to_pylist():
                    n+=1;root_keys.update(r);first.extend([r] if len(first)<2 else [])
                    qhtml=r['question'] or '';qtext=plain(qhtml);answers=r['answers'] or []
                    if not all({'text','pm_score','selected'}.issubset(a) for a in answers):
                        raise ValueError(f'Unexpected actual answer schema in {site}')
                    accepted=[a for a in answers if a['selected'] is True]
                    top=max(answers,key=lambda a:(a['pm_score'] if a['pm_score'] is not None else -float('inf'),-(a.get('answer_id') or 0))) if answers else None
                    top_score_ties=sum(a['pm_score']==top['pm_score'] for a in answers) if top else 0
                    vectors['question_words'].append(len(qtext.split()));vectors['question_chars'].append(len(qtext));vectors['question_html_chars'].append(len(qhtml))
                    answercounts[len(answers)]+=1
                    flags['at_least_one_answer']+=len(answers)>=1;flags['at_least_two_answers']+=len(answers)>=2
                    flags['with_accepted_answer']+=bool(accepted);flags['multiple_selected_answers']+=len(accepted)>1
                    flags['empty_question']+=not bool(qtext);flags['question_with_code_block']+=bool(CODE_BLOCK.search(qhtml))
                    flags['question_with_any_code_tag']+=bool(ANY_CODE.search(qhtml));flags['question_with_url']+=bool(URL.search(qhtml))
                    flags['question_with_image_tag']+=bool(re.search(r'<img\b',qhtml,re.I))
                    flags['question_gt_200_words']+=len(qtext.split())>200;flags['question_gt_512_words']+=len(qtext.split())>512
                    flags['question_debug_literal']+=bool(DEBUG_LITERAL.search(qtext));flags['question_advice_literal']+=bool(ADVICE_LITERAL.search(qtext))
                    flags['with_any_answer_gt_512_words']+=any(len(plain(a['text']).split())>512 for a in answers)
                    flags['with_any_answer_gt_1024_words']+=any(len(plain(a['text']).split())>1024 for a in answers)
                    flags['selected_equals_top_pm_score']+=bool(accepted) and any(a['pm_score']==top['pm_score'] for a in accepted)
                    flags['top_pm_score_tie_questions']+=top_score_ties>1
                    flags['tags_field_present']+='tags' in r;flags['title_field_present']+='title' in r
                    for a in answers:
                        answer_keys.update(a);atext=plain(a['text']);aw=len(atext.split())
                        vectors['answer_words'].append(aw);vectors['answer_chars'].append(len(atext));vectors['answer_html_chars'].append(len(a['text'] or ''))
                        flags['empty_answer']+=not bool(atext);flags['answer_gt_512_words']+=aw>512;flags['answer_gt_1024_words']+=aw>1024
                        flags['answer_with_code_block']+=bool(CODE_BLOCK.search(a['text'] or ''));flags['answer_with_url']+=bool(URL.search(a['text'] or ''))
                        scorecounts[a['pm_score']]+=1
                        flags['answer_selected_null']+=a['selected'] is None;flags['answer_pm_score_null']+=a['pm_score'] is None
                    for a in accepted:
                        vectors['accepted_answer_words'].append(len(plain(a['text']).split()));vectors['accepted_answer_chars'].append(len(plain(a['text'])))
                    if top:
                        vectors['top_pm_score_answer_words'].append(len(plain(top['text']).split()))
                    metadata=r.get('metadata') or []
                    urls=[u for u in metadata if isinstance(u,str) and '/questions/' in u]
                    flags['source_url_missing']+=not urls
                    for u in urls:
                        host=u.split('/')[2];source_urls[host]+=1
                        flags['source_site_mismatch']+=host!=site
                    sample={'site':site,'source':urls[0] if urls else None,'qid':r['qid'],
                            'question':qtext,'question_html':qhtml,'title':r.get('title'),
                            'accepted_answer':accepted[0] if len(accepted)==1 else (accepted if accepted else None),
                            'accepted_answer_text':plain(accepted[0]['text']) if len(accepted)==1 else None,
                            'top_score_answer':top,'top_score_answer_text':plain(top['text']) if top else None,
                            'top_score_definition':'maximum published pm_score; minimum answer_id breaks ties',
                            'top_score_tie_count':top_score_ties,'tags':r.get('tags'),
                            'metadata':metadata,'date':r.get('date'),'all_answers':answers,
                            'sample_seed':SEED,'sampling_pool':'all downloaded rows for this native site'}
                    if len(reservoir)<30:
                        reservoir.append(sample)
                    else:
                        j=rng.randrange(n)
                        if j<30:reservoir[j]=sample
                    records.append((site,r['qid'],qhtml,qtext))
                con.executemany('INSERT INTO questions VALUES (?,?,?,?)',records)
            con.commit()
        schema[site]={'actual_columns':sorted(root_keys),'actual_answer_columns':sorted(answer_keys),'first_two_rows':first}
        item={'requested_domain':requested,'site':site,'status':'FOUND','question_count':n,'answer_count':len(vectors['answer_words']),
              'accepted_answer_count':flags['with_accepted_answer'],'accepted_answer_object_count':len(vectors['accepted_answer_words']),
              'accepted_answer_rate':flags['with_accepted_answer']/n,'native_domain_source':'repository directory + original question URL',
              'title_available':flags['title_field_present']>0,'tags_available':flags['tags_field_present']>0,
              'title_chars_median':None,'title_words_median':None,**flags}
        for k,v in list(flags.items()):
            denom=len(vectors['answer_words']) if k.startswith('answer_') or k=='empty_answer' else n
            item[k+'_rate']=v/denom if denom else None
        for k,values in vectors.items():
            for metric,value in distribution(values).items():
                item[k+'_'+metric]=value
        for column,label in [('question_html','exact_html'),('question_text','exact_plain_text')]:
            groups=con.execute(f'SELECT COUNT(*),SUM(c),SUM(c-1) FROM (SELECT COUNT(*) c FROM questions WHERE site=? AND {column}!=? GROUP BY {column} HAVING COUNT(*)>1)',(site,'')).fetchone()
            item[label+'_duplicate_groups']=groups[0];item[label+'_duplicate_rows']=groups[1] or 0;item[label+'_duplicate_excess']=groups[2] or 0
            item[label+'_duplicate_rate']=(groups[2] or 0)/n
        distinct_ids=con.execute('SELECT COUNT(DISTINCT qid) FROM questions WHERE site=?',(site,)).fetchone()[0]
        item['duplicate_qid_excess']=n-distinct_ids
        item['answer_count_distribution']=dict(answercounts);item['source_host_counts']=dict(source_urls)
        sites.append(item);samples.extend(reservoir)
        scores.extend({'site':site,'pm_score':k,'answer_count':v,'percentage_of_answers':v/item['answer_count']} for k,v in sorted(scorecounts.items(),key=lambda x:(x[0] is None,x[0] or 0)))
        quality.append({'dataset':'StackExchange','domain':site,'count':n,'duplicate_question_excess':item['exact_html_duplicate_excess'],
                        'duplicate_rate':item['exact_html_duplicate_rate'],'duplicate_plain_text_excess':item['exact_plain_text_duplicate_excess'],
                        'empty_question':flags['empty_question'],'empty_answer':flags['empty_answer'],
                        'accepted_answer_count':flags['with_accepted_answer'],'multiple_selected_answers':flags['multiple_selected_answers']})
        print('SITE',site,'questions',n,'accepted',item['accepted_answer_count'],'answers',item['answer_count'],flush=True)
    pairs=[];cross={}
    for column,label in [('question_html','exact_html'),('question_text','exact_plain_text')]:
        groups=list(con.execute(f'SELECT {column},COUNT(*),COUNT(DISTINCT site) FROM questions WHERE {column}!=? GROUP BY {column} HAVING COUNT(DISTINCT site)>1',('',)))
        cross[label]={'unique_questions_shared_across_sites':len(groups),'rows_in_shared_groups':sum(r[1] for r in groups)}
        for q,count,numsites in groups:
            matches=list(con.execute(f'SELECT site,qid FROM questions WHERE {column}=?',(q,)))
            pairs.append({'match_type':label,'question':q,'total_rows':count,'site_count':numsites,'matches':[{'site':s,'qid':i} for s,i in matches]})
    con.close()
    frame=pd.DataFrame(sites)
    # JSON structured fields avoid Python repr in CSV.
    for c in ['answer_count_distribution','source_host_counts']:
        if c in frame:frame[c]=frame[c].map(lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else '')
    frame.to_csv(OUT/'stackexchange_site_stats.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(scores).to_csv(OUT/'stackexchange_answer_scores.csv',index=False,encoding='utf-8-sig')
    write_jsonl('stackexchange_samples.jsonl',samples)
    write_jsonl('stackexchange_cross_site_duplicates.jsonl',pairs)
    save('stackexchange_schema.json',schema)
    result={'sites':sites,'cross_site_duplicates':cross,'scope':'all shards of ten requested main-site directories; excludes meta',
            'accepted_mapping':'answers[].selected == true; frozen publisher acceptance flag',
            'score_mapping':'answers[].pm_score; transformed preference score, not original net votes',
            'title_and_tags':'not available when absent; never inferred from body or question URL'}
    save('stackexchange_statistics.json',result)
    return result,quality


def md_table(headers,rows):
    def fmt(x):
        if x is None or (isinstance(x,float) and np.isnan(x)):
            return 'N/A'
        if isinstance(x,float):
            return f'{x:.4f}'.rstrip('0').rstrip('.')
        return str(x).replace('|','\\|').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(map(fmt,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
                      ['| '+' | '.join(map(fmt,r))+' |' for r in rows])


def diagnose_empty_answers():
    """Distinguish genuinely blank HTML from image-only empty visible text."""
    import pyarrow.compute as pc
    summary={};examples=[]
    for site in SITES.values():
        raw_empty=0;image_only=0
        for file in sorted((RAW/'stack_exchange_preferences/data'/site).glob('*.parquet')):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=1024,columns=['qid','answers']):
                flat=pc.list_flatten(batch.column('answers'))
                texts=pc.struct_field(flat,'text')
                empty=pc.or_(pc.is_null(texts),pc.fill_null(pc.equal(pc.utf8_trim_whitespace(texts),''),False))
                raw_empty+=pc.sum(pc.cast(empty,'int64')).as_py() or 0
                stripped=pc.replace_substring_regex(texts,pattern='<[^>]*>',replacement='')
                if pc.any(pc.match_substring_regex(stripped,pattern=r'^\s*$')).as_py():
                    for r in batch.to_pylist():
                        for answer in r['answers']:
                            if not plain(answer['text']):
                                has_image=bool(re.search(r'<img\b',answer['text'] or '',re.I))
                                image_only+=has_image
                                examples.append({'site':site,'qid':r['qid'],'image_only':has_image,'answer':answer})
        summary[site]={'empty_raw_answer':raw_empty,'image_only_answer_count':image_only}
    save('empty_visible_answer_examples.json',examples)
    save('raw_answer_empty_diagnostics.json',summary)
    return summary

def report(m,s,quality):
    empty_diagnostics=diagnose_empty_answers()
    for r in s['sites']:
        r.update(empty_diagnostics.get(r['site'],{}))
        r['empty_visible_text_answer']=r.get('empty_answer')
    save('stackexchange_statistics.json',s)
    frame=pd.DataFrame(s['sites'])
    for c in ['answer_count_distribution','source_host_counts']:
        frame[c]=frame[c].map(lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else '')
    frame.to_csv(OUT/'stackexchange_site_stats.csv',index=False,encoding='utf-8-sig')
    for r in quality:
        if r.get('dataset')=='StackExchange':
            r.update(empty_diagnostics.get(r.get('domain'),{}))
            r['empty_visible_text_answer']=r.get('empty_answer')
    compare=[]
    for r in m['category_stats']:
        compare.append({'dataset/domain':'Mintaka/'+r['category'],'train_or_total_count':r['train_count'],
                        'accepted_answer_count':None,'median_question_words':r['train_question_words_median'],
                        'median_answer_words':r['train_answer_words_median'],'p90_answer_words':r['train_answer_words_p90'],
                        'duplicate_rate':r['train_duplicate_question_excess']/r['train_count'],
                        'native_domain_source':'raw category in author dataset',
                        'notes':'English train; answer.mention; all-split total '+str(r['total_count'])+'; accepted concept not applicable'})
    for r in s['sites']:
        compare.append({'dataset/domain':'StackExchange/'+r['site'],'train_or_total_count':r['question_count'],
                        'accepted_answer_count':r.get('accepted_answer_count'),'median_question_words':r.get('question_words_median'),
                        'median_answer_words':r.get('answer_words_median'),'p90_answer_words':r.get('answer_words_p90'),
                        'duplicate_rate':r.get('exact_html_duplicate_rate'),'native_domain_source':r.get('native_domain_source'),
                        'notes':r['status']+'; all published questions; length over all answers, not one answer/question; pm_score transformed; no title/tags'})
    pd.DataFrame(compare).to_csv(OUT/'dataset_domain_comparison.csv',index=False,encoding='utf-8-sig')
    for split,r in m['splits'].items():
        quality.append({'dataset':'Mintaka','domain':split,'count':r['rows'],'duplicate_question_excess':r['duplicate_question_excess'],
                        'duplicate_rate':r['duplicate_question_excess']/r['rows'],'empty_question':r['empty_question'],'empty_answer':r['empty_answer'],
                        'train_test_unique_question_overlap':m['train_test_exact_unique_question_overlap']})
    quality.append({'dataset':'StackExchange','domain':'cross-site','count':sum(r['question_count'] for r in s['sites']),
                    'cross_site_exact_html_unique_questions':s['cross_site_duplicates']['exact_html']['unique_questions_shared_across_sites'],
                    'cross_site_exact_plain_unique_questions':s['cross_site_duplicates']['exact_plain_text']['unique_questions_shared_across_sites']})
    pd.DataFrame(quality).to_csv(OUT/'data_quality.csv',index=False,encoding='utf-8-sig')
    mint_source=load(OUT/'mintaka_github_repository.json');mhf=load(OUT/'mintaka_repository.json');shf=load(OUT/'stackexchange_repository.json')
    found=[r for r in s['sites'] if r['status']=='FOUND']
    parts=[]
    def add(text):parts.append(text.strip())
    add(f'''# Mintaka 与 StackExchange 专业领域 QA 数据审计

运行时间（UTC）：{datetime.now(timezone.utc).isoformat()}。使用指定 `fd` Python 环境；全部计数来自固定版本实际文件。仅下载、结构检查、统计与抽样，没有训练模型、创建客户端或自动分类问题。

## 1. 数据范围、原生字段与统计定义

Mintaka：Hugging Face `AmazonScience/mintaka` 当前固定 commit `{mhf['revision']}` 只有 loader/README 等文件，没有实际 parquet。实际阅读 loader 后，按其官方地址下载作者 GitHub `{mint_source['revision']}` 的完整 `mintaka_train.json`、`mintaka_dev.json`、`mintaka_test.json`，合计 {m['total_rows']:,} 个原始问题；每条同时包含英文及八种翻译，未漏掉原始译文。dev 在报告统一标为 validation。此报告核心英文统计每个原始问题只数一次，**没有把九种语言版本当成九个独立知识样本**。

StackExchange：固定 Hugging Face commit `{shf['revision']}`，下载 10 个指定主站目录下的全部 parquet shards；没有把 `*.meta.stackexchange.com` 混入学科。原始文件按站点分目录而不是单表必有 subject 列；领域来源是发布目录及保留的原问题 URL。`health.stackexchange.com` 是该版本 health / medical sciences 请求对应的真实目录名，未改成不存在的 medicalsciences 目录，也没有拿其他站点补缺。

所有保存文件大小和 SHA256 在 [dataset_metadata.json](dataset_metadata.json)。原始文件分别保存于 `data/raw/mintaka/` 与 `data/raw/stack_exchange_preferences/`；后者只下载请求的 10 个主站，不宣称下载全网络 10M 级数据。

长度定义：英文 whitespace words，字符数使用 Python `len`；Mintaka 对 question 与原始 `answer.mention` 直接计算，StackExchange 对 HTML 解码、标签去除并合并空白后的可见文本计算。HTML 原始字符长度另存 CSV。不使用模型 tokenizer；公式、代码和 URL 可能使 whitespace words 低估真实模型 token 数。

重复率定义为**额外重复行数 `(N - unique questions) / N`**。Mintaka 按原始英文字符串 exact match；StackExchange 首要重复按原始 question HTML 完全相同，另给 HTML 解析及空白标准化后的 exact text 重复，不把后者称为原始字符串 exact。空问题另报，不把空串当成知识重题。cross-site 重复给共享 unique question 数与涉及行数，不将不同站点同一个 qid 当作重题。

“特别长”只是透明阈值：question >200 words，补充 >512；answer >512 words，补充 >1024。代码块按 `<pre>` 或 Markdown fences；`<code>` 含内联代码另报。URL 按 `http(s)://` 或 `www.`。这些是表面特征，不是 factual/opinion/advice 或领域纯度的自动标签。''')
    add('''## 2. 实际 Schema

| 数据 | 真实字段 | QA 与领域映射 | 不可获得或需注意 |
| --- | --- | --- | --- |
| Mintaka raw JSON | id, question, translations, questionEntity, answer, category, complexityType | Q=question；A=answer.mention；领域=category；复杂度=complexityType | answer 是嵌套对象，不直接当字符串；answer.answer 是实体/数值等结构 |
| Mintaka loader 输出 | id, lang, question, answerText, category, complexityType, questionEntity, answerEntity | answerText 由 raw answer.mention 映射；en 是英文配置 | 原始版本没有 lang 列；processed/en 是从 source 明确抽取而非改标签 |
| StackExchange parquet | qid:int64, question:string, answers:list(struct), date:string, metadata:list(string) | Q=question HTML；A=answers[].text；领域=原站点目录和 metadata URL | 无独立 title、无 tags 字段、无原始 vote score、无正式 val/test |
| StackExchange answer struct | answer_id:int64, author:string, author_id:int64, author_profile:string, pm_score:int64, selected:bool, text:string | accepted 标记=selected true；最高发布分数=pm_score | accepted 不是正确性证明；pm_score 不是原始 upvote/net score |

实读每个 split/shard 的 schema、真实前几条与 dtype 已保存到 [mintaka_schema.json](mintaka_schema.json)、[stackexchange_schema.json](stackexchange_schema.json)。脚本会在缺少关键字段或 shard 未下载完整时中止，而不会猜映射或把下载未完成写成 NOT FOUND。

**Title/body 限制：** 本版本只能统计发布的 question HTML 及其可见文本。独立 title 字段已丢失，不能可靠还原原站点 title 与 body 边界；所有 title 长度为 N/A，不以首句/第一段冒充 title。tags 也记 null/N/A，未从问题或 URL 推断。`metadata` 中保留 question URL/site/profile 信息，样本保存全部原始 metadata 和答案作者出处。

**Score 限制：** 发布说明将 pm_score 定义为 rounded log2(1+upvotes)，接受答案额外加 1，负分赋 -1。此次实际只持有 pm_score，无法重建原始票数；`top_score_answer` 指最高 pm_score（并列时最小 answer_id），不声称它一定是原站点最高票答案。全部分数频次保存于 [stackexchange_answer_scores.csv](stackexchange_answer_scores.csv)。[固定发布说明](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences)''')
    add('## 3. Mintaka Splits、Category 与复杂度\n\n'+md_table(['category','train','dev / validation','test','total'],[
        [r['category'],r['train_count'],r['dev_count'],r['test_count'],r['total_count']] for r in m['category_stats']]))
    add(f'''实际 train/validation/test 为 {m['splits']['train']['rows']:,} / {m['splits']['validation']['rows']:,} / {m['splits']['test']['rows']:,}。8 类不是假设：从三份文件读出的完整名称为 `{', '.join(m['categories'])}`。**每类 train 1,750 / dev 250 / test 500，数量完全平衡**，最大/最小训练类比例为 1。需要多少本地训练样本依赖后续设计；这些数字能支持小规模 controlled study，并不能保证任何模型/20 客户端训练的统计功效。

complexityType 真实共 {len(m['complexityTypes'])} 类：`{', '.join(m['complexityTypes'])}`；generic 加八种复杂类型，不误写成总共八类。''')
    add(md_table(['complexityType','train','validation','test'],[[k,*[m['splits'][sp]['complexity_counts'].get(k,0) for sp in ['train','validation','test']]] for k in m['complexityTypes']]))
    comp=pd.read_csv(OUT/'mintaka_category_complexity.csv')
    for split in ['train','validation','test']:
        pivot=comp[comp.split==split].pivot(index='category',columns='complexityType',values='count')
        add(f'### {split}：category × complexityType\n\n'+md_table(['category']+list(pivot.columns),[[c]+[int(x) for x in row] for c,row in pivot.iterrows()]))
    identical=all(len(set(comp[(comp.split==sp)&(comp.complexityType==ct)]['count']))==1 for sp in m['splits'] for ct in m['complexityTypes'])
    add(f'''实际交叉表中各领域复杂类型分布是否完全一致：**{identical}**。因此当前文件不仅类别数量平衡，也直接控制了原生 complexityType 分布，减少按域同时切换任务复杂度的混淆。但同一短答案任务仍包含数值、布尔、实体以及组合推理，不代表八域都在测完全相同的认知能力；话题间也可能共享实体与知识。

完整计数和类别内比例：[mintaka_category_complexity.csv](mintaka_category_complexity.csv)；split 总分布：[mintaka_complexity_distribution.csv](mintaka_complexity_distribution.csv)。''')
    add('## 4. Mintaka QA 长度与质量\n\n'+md_table(['category（train）','Q median words','Q p90 words','A mean words','A median words','A p90 words','A max words','A ≤5 words'],[
       [r['category'],r['train_question_words_median'],r['train_question_words_p90'],r['train_answer_words_mean'],r['train_answer_words_median'],r['train_answer_words_p90'],r['train_answer_words_max'],f"{r['train_answer_le_5_words_rate']:.2%}"] for r in m['category_stats']]))
    add(md_table(['split','Q mean / median / p90 / max chars','Q mean / median / p90 / max words','A mean / median / p90 / max chars','A mean / median / p90 / max words'],[
        [sp,*[' / '.join(f"{v[metric]:.2f}" for metric in ['mean','median','p90','max']) for v in [r['question_lengths']['chars'],r['question_lengths']['words'],r['answer_lengths']['chars'],r['answer_lengths']['words']]]] for sp,r in m['splits'].items()]))
    a=m['answer_lengths'];train=m['splits']['train']
    add(f'''全部英文 20,000 答案的 word mean={a['words']['mean']:.4f}，median={a['words']['median']:g}，p90={a['words']['p90']:g}，max={a['words']['max']:g}；字符 median={a['chars']['median']:g}、p90={a['chars']['p90']:g}、max={a['chars']['max']:g}。**主要确实是极短答案，但并非全部一词或全部实体**。逐类 train ≤1/≤5/≤10 words 比例和 >20 words 数量、mean/median/p10/p90/max 全存 CSV，避免只看总体。

类别内原生 answerType 不是全部 entity。train 类型计数：`{json.dumps(train['answer_types'],ensure_ascii=False)}`。raw answer 对象保留实体/支持实体/日期等信息，未把 answerEntity 列表拼接成答案以改变任务。''')
    add(md_table(['split','empty Q','empty A','duplicate extra rows','duplicate rows membership'],[[sp,r['empty_question'],r['empty_answer'],r['duplicate_question_excess'],r['duplicate_question_rows']] for sp,r in m['splits'].items()]))
    add(f'''跨三个 split 的 extra duplicate questions：{m['all_splits_duplicate_question_excess']}。train–test 原始英文 exact question overlap：{m['train_test_exact_unique_question_overlap']} unique questions，涉及 train {m['train_test_overlap_train_rows']} 行 / test {m['train_test_overlap_test_rows']} 行。**exact overlap 为零不证明无语义近似题、实体共享或预训练污染**；此次不使用模型做语义分类/去重。重复与 overlap 实物列表亦已保存。

每类从**英文 train**均匀抽 20 条，共 {sum(1 for _ in (OUT/'mintaka_samples.jsonl').open(encoding='utf-8'))} 条；seed=42，按排序后的 category 顺序抽样。抽样不会改变训练标签或创建新 train/test 分割。[人工检查样本](mintaka_samples.jsonl)''')
    add('## 5. StackExchange 10 站存在性、QA 与 accepted 覆盖\n\n'+md_table(['请求领域','真实 site','状态','问题数','至少1答案','≥2答案','答案对象总数','有accepted问题数','accepted覆盖率'],[
       [r['requested_domain'],r['site'],r['status'],r['question_count'],r.get('at_least_one_answer'),r.get('at_least_two_answers'),r.get('answer_count'),r.get('accepted_answer_count'),f"{r['accepted_answer_rate']:.2%}" if r.get('accepted_answer_rate') is not None else 'N/A'] for r in s['sites']]))
    add(f'''统计的是**这个固定偏好语料已收录的 QA 问题**，不是 2026 年整个站点所有问题。一个问题有多答案，question_count 不能与 answer_count 混用。全量读取合计问题 {sum(r['question_count'] for r in found):,}，答案 {sum(r['answer_count'] for r in found):,}。accepted 分母为本语料所有问题，含没有 selected true 的问题；多 selected 的异常也单列。

发布 README 声称过滤 ≥2 answers；此次以实读数量分布核验，未依赖说明做强制过滤。答案数 0/1/2/更多的频次在 site CSV `answer_count_distribution`，若存在不符，会原样显示而非删掉。`selected` 是历史快照保留状态，不是当前站点实时接受率，也不是专家判定正确率。

来源 URL 的 host 不匹配行数合计 {sum(r.get('source_site_mismatch',0) for r in found)}，缺 question URL 行数合计 {sum(r.get('source_url_missing',0) for r in found)}；multiple selected 问题 {sum(r.get('multiple_selected_answers',0) for r in found)}。''')
    add('## 6. StackExchange 长度、分数与可直接计算的结构特征\n\n'+md_table(['site','Q median / p90 / max words','全答案 median / p90 / max words','accepted median / p90 words','top pm_score median / p90 words','答案 >512比例','答案 >1024比例'],[
        [r['site'],' / '.join(str(r['question_words_'+k]) for k in ['median','p90','max']),
         ' / '.join(str(r['answer_words_'+k]) for k in ['median','p90','max']),
         ' / '.join(str(r['accepted_answer_words_'+k]) for k in ['median','p90']),
         ' / '.join(str(r['top_pm_score_answer_words_'+k]) for k in ['median','p90']),
         f"{r.get('answer_gt_512_words_rate',0):.2%}",f"{r.get('answer_gt_1024_words_rate',0):.2%}"] for r in found]))
    add('''所有字符/word mean、median、p10、p90、max 保存在 [stackexchange_site_stats.csv](stackexchange_site_stats.csv)。accepted 长度只对 selected=true 答案对象计数；全答案分布包含所有候选，回答多的问题权重更大；top 分布是一问题一答案的额外对照。本报告没有决定最终使用 accepted 还是 top 作为训练目标。

文章长不长与是否适合微调是两件事：此表直接给长度和尾部比例，但没有指定模型 tokenizer、上下文上限或学习目标，不能依据字符/whitespace words 宣称全部不适合。相对于 Mintaka 一两词，社区答案通常是解释性回复；应由长度预算及知识 QA 目标决定下一步是否值得筛选，而不是现在静默截断。''')
    add(md_table(['site','Q代码块','Q任何code标记','Q URL','Q图像标签','Q>200 words','问题有>512 word答案','debug字面词','advice字面短语'],[
        [r['site'],*[f"{r.get(k+'_rate',0):.2%}" for k in ['question_with_code_block','question_with_any_code_tag','question_with_url','question_with_image_tag','question_gt_200_words','with_any_answer_gt_512_words','question_debug_literal','question_advice_literal']]] for r in found]))
    add('''debug 字面匹配规则：`debug(ging), traceback, exception, compiler, compile, segmentation fault`；advice 字面短语：`should I, what should, recommend(ation/ations), advice, best way, in your opinion`。比例仅指问题中存在这些字符串，不是编程/意见/建议题比例；有 advice 短语可仍是事实性实验方法题，有代码可仍是算法理论题，无代码也可能是调试题。**没有 LLM 分类，亦未人为给样本新领域/任务标签**。

每站对全部问题做单次 reservoir sampling，seed=42、按固定 shard/row 顺序，30 条/站。样本保存 question 原文/可见文本、所有答案、accepted（没有时 null）、top pm_score、metadata、作者与出处、tags=null、title=null；并列最高分数量也记录。[人工检查样本](stackexchange_samples.jsonl)''')
    scoreframe=pd.read_csv(OUT/'stackexchange_answer_scores.csv')
    add('### 发布 pm_score 的完整频次\n\n'+md_table(['site','pm_score → answer_count'],[
        [site,', '.join(f"{r.pm_score:g}: {int(r.answer_count)}" for r in scoreframe[scoreframe.site==site].itertuples())] for site in SITES.values() if site in set(scoreframe.site)]))
    add('## 7. 重复、跨站一致题与质量\n\n'+md_table(['site','HTML exact extra重复','HTML重复率','解析文本 extra重复','解析文本重复率','重复qid extra','空Q可见文本','空A可见文本'],[
        [r['site'],r['exact_html_duplicate_excess'],f"{r['exact_html_duplicate_rate']:.4%}",r['exact_plain_text_duplicate_excess'],f"{r['exact_plain_text_duplicate_rate']:.4%}",r['duplicate_qid_excess'],r.get('empty_question',0),r.get('empty_answer',0)] for r in found]))
    add(md_table(['跨站匹配定义','shared unique questions','涉及行数'],[[k,v['unique_questions_shared_across_sites'],v['rows_in_shared_groups']] for k,v in s['cross_site_duplicates'].items()]))
    add(f'''可见文本为空的答案共有 {sum(r.get('empty_answer',0) for r in found)} 个，额外检查原始 HTML 后，原始 null/空白答案为 {sum(r.get('empty_raw_answer',0) for r in found)}，仅有图像而无可见文字的答案为 {sum(r.get('image_only_answer_count',0) for r in found)}。这些不能直接称为缺失答案；`empty_answer` / `empty_visible_text_answer` 指解析后的文本为空，`empty_raw_answer` 单独保存。实例在 [empty_visible_answer_examples.json](empty_visible_answer_examples.json)，计数在 [raw_answer_empty_diagnostics.json](raw_answer_empty_diagnostics.json)。这也说明纯文本转换可能丢失答案信息，未下载图像或替图像生成答案。

实际跨站共享题/站点/qid 保存于 [stackexchange_cross_site_duplicates.jsonl](stackexchange_cross_site_duplicates.jsonl)；空文件表示计数为零。HTML exact 重复只覆盖逐字相同的题；文本解析合并空白的第二套统计是额外更宽的标准，不证明语义不重叠。跨站 repost、迁移或近似问题可仍未检出。全部质量指标见 [data_quality.csv](data_quality.csv)。''')
    add('## 8. 特别比较：原生领域展开\n\n'+md_table(list(compare[0]),[[r[k] for k in compare[0]] for r in compare]))
    add('''Mintaka 行使用 **train** 长度/数量；StackExchange 行使用指定站点发布语料的 **total**，Hub 将其容器称为 train，没有现成独立 val/test。Mintaka accepted 为 N/A，因为它没有社区接受机制；不能写为零使人误解质量。StackExchange answer 长度为所有答案，top/accepted 的单独统计在上一节。这个比较不表示两套来源或任务已被控制为完全一致。[CSV](dataset_domain_comparison.csv)

## 9. 最后回答研究问题
''')
    qshort=m['answer_lengths']['words'];mx=max(found,key=lambda r:r['question_with_code_block_rate']);mn=min(found,key=lambda r:r['question_with_code_block_rate'])
    add(f'''1. **Mintaka 8 个 category 是否真实存在于 train？** 是，逐行确认的八个名称都在 train；没有用 test 标签替代 train 标签。

2. **每域是否足够、相对平衡？** 每域 train 1,750 / dev 250 / test 500，类别和 complexityType 交叉数量都平衡；适合验证性 controlled study。是否“足够”取决于后续模型、功效、客户端数与训练设计，目前未定义绝对足量标准，不能由平衡性推出训练规模充分。

3. **Mintaka 答案有多短？** 全集平均 {qshort['mean']:.4f} words，中位 {qshort['median']:g}，p90 {qshort['p90']:g}，最大 {qshort['max']:g}；答案字符中位 {m['answer_lengths']['chars']['median']:g}。实体之外包含 boolean/numerical/date/string；它是短答案知识与组合推理 QA，不是纯单跳记忆题。

4. **StackExchange 哪些站真实存在？** 本次固定仓库清单中 {len(found)}/10 主站 FOUND。真实 host 名列于第 5 节，包括 `cs.stackexchange.com`、`health.stackexchange.com`、`stats.stackexchange.com`；不是擅自替换为 StackOverflow 或 meta 站。

5. **每站多少 QA？** 完整 shard 数出的 question_count / answer_count 在第 5 节和 CSV，合计 {sum(r['question_count'] for r in found):,} 个问题。它们只是这个偏好过滤快照的规模，不是站点总用户问题数。

6. **accepted 覆盖如何？** 以 `selected=true` 直接计数，所有站具体 accepted 数/分母/比例列于第 5 节。没有 accepted 的问题不凭 pm_score 猜 accepted；最高偏好分答案也不能视为被作者接受或事实正确。

7. **StackExchange 答案长到不适合当前微调吗？** 其答案长度明显不同于 Mintaka；具体 median/p90、512/1024 尾部比例见第 6 节。没有给定目标模型 token 预算，不能客观宣布某站整体不适合；更适合检验解释型专业 QA 的可能性，而非当成同样的一词事实任务。含 HTML/公式/代码时尤其需要下一步 tokenizer 核验，当前不截断。

8. **哪些站领域最纯？** 当前可证实的是原生 host 一致性和结构特征，不能由站点名、低代码率或 accepted 率推出语义领域纯度。表面代码块率最低的 `{mn['site']}` 为 {mn['question_with_code_block_rate']:.2%}，这个排名仅是该特征，**不是领域纯度排名**。建议用已保存的 30 条/站做下一阶段人工核验；此次不人为标领域、不生成 purity 分数。

9. **哪些站明显大量编程/debug/opinion/advice？** 可精确指出代码块、debug 字面词及 advice 短语的率；代码块率最高的 `{mx['site']}` 为 {mx['question_with_code_block_rate']:.2%}。这些只能作为任务混杂信号，不能把代码/句式直接等同于完整任务类型；特别是 stats 的分析代码与 cs 的算法表达不必然是调试。没有自动分类就不能严谨宣称各站 opinion/advice 的真实比例，数据中也无原生 question_type/tags 支撑这种结论。

10. **纯数据角度的可行性判断：** Mintaka 的同一来源、统一短答案 QA、train 原生八域、完全平衡的 category×complexity 以及已测 exact overlap，使它**值得作为 controlled topic-domain benchmark 进一步验证**；但它的类别是 books/movies/music/sports 等知识话题，**不是 medicine/law/chemistry/CS 等专业学科**，且八域间不保证知识互斥。StackExchange 十个专业主站真实可得、有 native host 与多答案/接受标记，**值得继续核验 professional-domain benchmark 的构建可行性**；目前仍有来源社区差异、偏好过滤选择偏差、目标答案选择、较长解释、题目类型与上下文依赖等变量，不能作为已受控知识整合 benchmark 直接使用。此次不决定最终数据集、不筛选新训练集、不划客户端。

## 10. 可复现文件与局限

主脚本：[audit_mintaka_stackexchange.py](../../scripts/audit_mintaka_stackexchange.py)。统计/采样固定 seed=42。下载复用已保存 commit，不随每次运行漂移；从空目录首次运行才解析当前 revision 并保存。完整样本保留原有类别与源站字段，不另造标签。

```powershell
& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/audit_mintaka_stackexchange.py --download-only
& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/audit_mintaka_stackexchange.py
```

官方来源只用于确认发布 provenance / score 定义，所有数字与存在性结论来自实物：[Mintaka 固定源](https://github.com/amazon-science/mintaka/tree/{mint_source['revision']})、[Mintaka COLING 2022 论文](https://aclanthology.org/2022.coling-1.138/)、[StackExchange 固定仓库](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences/tree/{shf['revision']})。

原始文件与统计不证明 factual correctness、领域纯度或模型未见过题。Mintaka 的实体标注/复杂度是原生数据，StackExchange 接受标记与评分是社区反馈；都不是新的事实核验。因 schema 缺 title/tags，相关统计 N/A；原始 vote score 不可重建；没有把缺失指标编造为已知。已下载翻译留存，但长度/重复主分析仅英语；跨语言版本共享原始 ID 与答案，若日后混用不得误当独立题目或跨 split 新样本。

许可声明随 pinned card 留存：Mintaka CC-BY-4.0，H4 StackExchange CC-BY-SA-4.0；样本保存原出处/作者信息。此处不做新的发布或许可解释。没有改动系统包；仅复用 fd 中 requests/numpy/pandas/pyarrow 等现有依赖。''')
    (OUT/'REPORT.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    metadata={'utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'python':__import__('sys').executable,
              'mintaka_hf_revision':mhf['revision'],'mintaka_source_revision':mint_source['revision'],
              'stackexchange_revision':shf['revision'],'repositories':[mhf['repo'],shf['repo']],
              'licenses':{'mintaka':mhf['card'].get('license'),'stackexchange':shf['card'].get('license')},
              'field_mappings':{'Mintaka':{'question':'raw.question','answer':'raw.answer.mention','domain':'raw.category'},
                                'StackExchange':{'question':'question','answers':'answers[].text','accepted':'answers[].selected','score':'answers[].pm_score','domain':'source directory and metadata host'}},
              'dependencies':{'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':__import__('pyarrow').__version__,'requests':requests.__version__},
              'files':[]}
    for folder in [RAW/'mintaka',RAW/'stack_exchange_preferences']:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix!='.part':
                source=(f"https://raw.githubusercontent.com/{mint_source['repo']}/{mint_source['revision']}/data/{p.name}" if folder.name=='mintaka' else
                        f"https://huggingface.co/datasets/{shf['repo']}/resolve/{shf['revision']}/{p.relative_to(folder).as_posix()}")
                metadata['files'].append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pinned_source_url':source})
    save('dataset_metadata.json',metadata)
    required=['REPORT.md','mintaka_category_stats.csv','mintaka_samples.jsonl','stackexchange_site_stats.csv','stackexchange_samples.jsonl','data_quality.csv']
    manifest=[]
    for name in required:
        p=OUT/name
        assert p.exists() and p.stat().st_size>0
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!='output_manifest.json':
            manifest.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size})
            print('OUTPUT',p.resolve(),f'{p.stat().st_size:,} bytes',flush=True)
    save('output_manifest.json',manifest)
    print('REPORT:',(OUT/'REPORT.md').resolve(),flush=True)
    print('RAW DOWNLOADS',len(metadata['files']),sum(f['bytes'] for f in metadata['files']),'bytes',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--mintaka-only',action='store_true')
    args=parser.parse_args()
    if args.download_only:
        download()
    elif args.mintaka_only:
        mintaka()
    else:
        m=mintaka()
        s,quality=stackexchange()
        report(m,s,quality)
