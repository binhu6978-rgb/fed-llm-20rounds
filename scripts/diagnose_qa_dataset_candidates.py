"""Additional deterministic audits. Does not normalize labels or allocate clients."""
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import argparse
from audit_qa_dataset_candidates import OUT, fetch, save, audit_parquet, session


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def local():
    result = {}
    base = OUT/'downloads/ExpertQA'
    for prefix in ['rand', 'domain']:
        for split in ['train', 'val', 'test']:
            raw = lines(base/f'{prefix}_{split}.jsonl')
            conv = lines(base/f'{prefix}_lfqa_{split}.json')
            by_question = defaultdict(list)
            for row in raw:
                by_question[row['question']].append(row)
            revised = [a.get('revised_answer_string') for r in raw for a in r['answers'].values()]
            item = {'raw_rows':len(raw), 'converted_rows':len(conv),
                    'raw_duplicate_question_excess':len(raw)-len(by_question),
                    'converted_question_exact_match':sum(r['question'] in by_question for r in conv),
                    'converted_unique_question_match':sum(len(by_question[r['question']])==1 for r in conv),
                    'converted_context_nonempty':sum(bool(r.get('context')) for r in conv),
                    'converted_answer_empty':sum(not str(r.get('answer') or '').strip() for r in conv),
                    'raw_revised_answer_nonempty':sum(bool(a and a.strip()) for a in revised),
                    'raw_revised_answer_slots':len(revised),
                    'question_types':dict(Counter(json.dumps(r['metadata'].get('question_type'),ensure_ascii=False) for r in raw))}
            fields = Counter()
            for r in conv:
                matches=by_question[r['question']]
                if len(matches)==1:
                    fields[matches[0]['metadata']['field']]+=1
            item['converted_fields_via_unique_exact_question_join']=dict(fields)
            result[prefix+'_'+split]=item
    qb=[]
    for split in ['train','dev','test']:
        rows=json.loads((OUT/f'downloads/QBLink/QBLink-{split}.json').read_text(encoding='utf-8'))
        pairs=[r[k] for r in rows for k in ['q1','q2','q3']]
        qb.append({'split':split,'sequences':len(rows),'qa_pairs':len(pairs),
                   'blank_lead_in':sum(not str(r.get('lead_in') or '').strip() for r in rows),
                   'empty_question':sum(not str(q.get('quetsion_text') or '').strip() for q in pairs),
                   'empty_raw_answer':sum(not str(q.get('raw_answer') or '').strip() for q in pairs),
                   'empty_wiki_page':sum(not str(q.get('wiki_page') or '').strip() for q in pairs),
                   'nested_columns':sorted(set(k for q in pairs for k in q))})
    save('additional_diagnostics.json',{'ExpertQA':result,'QBLink':qb})
    print(json.dumps({'ExpertQA':result,'QBLink':qb},ensure_ascii=False,indent=2))


def network():
    revision='5dd9790a83002ad084ddeb7c420dc716852c6f28'
    file='nq_open/train-00000-of-00001.parquet'
    url=f'https://huggingface.co/datasets/google-research-datasets/nq_open/resolve/{revision}/{file}'
    path=fetch(url,OUT/'downloads/NQ-open/train.parquet')
    audit_parquet('NQ-open',path,[],{'repo':'google-research-datasets/nq_open','revision':revision,'url':url,
                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'train')
    resp=session.get('https://datasets-server.huggingface.co/splits',params={'dataset':'m-a-p/SuperGPQA'},timeout=90)
    save('SuperGPQA_server_splits.json',{'status_code':resp.status_code,'data':resp.json(),
                                      'scope':'dataset-server split inventory; raw file separately audited at pinned revision'})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--network',action='store_true')
    args=parser.parse_args()
    local()
    if args.network:
        network()
