"""Full deterministic local analysis. No training, relabeling, or client allocation."""
import argparse
import hashlib
import importlib.metadata
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/multi_subject_rlvr'
OUT = ROOT / 'analysis/multi_subject_rlvr'
PROCESSED = ROOT / 'data/processed'
MISSING = '__MISSING_NULL_BUCKET__'
UNKNOWN = {'unknown', 'unclassified', 'unlabeled', 'unlabelled', 'none', 'null', 'n/a', 'na', 'unspecified', 'others'}

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def csv(name, frame):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def table(frame):
    def cell(v):
        return str(v).replace('|', '\\|').replace('\n', '<br>')
    return '| ' + ' | '.join(frame.columns) + ' |\n|' + '|'.join(['---'] * len(frame.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None))

def truncate(v):
    if isinstance(v, str):
        return v[:700] + (' ...[truncated]' if len(v) > 700 else '')
    if isinstance(v, dict):
        return {k: truncate(x) for k, x in v.items()}
    if isinstance(v, list):
        return [truncate(x) for x in v[:10]]
    return v

def scalar(v):
    return isinstance(v, str) or v is None

def extract_question(v):
    if v is None or isinstance(v, str):
        return v, None
    if isinstance(v, list) and all(isinstance(x, dict) for x in v):
        users = [x.get('content') for x in v if x.get('role') == 'user']
        if users and all(isinstance(x, str) for x in users):
            return '\n'.join(users), None if len(users) == 1 else 'multiple_user_messages'
        return None, 'no_string_user_message'
    return None, 'unsupported_question_structure'

def counts(frame, field, splits):
    series = frame[field]
    # This is an analysis bucket only. Source labels are never overwritten.
    keys = series.map(lambda x: MISSING if x is None else x)
    result = pd.crosstab(keys, frame['split'], dropna=False).reindex(columns=splits, fill_value=0)
    result.index.name = field
    result = result.reset_index()
    for s in sorted(set(splits) | {'train', 'test'}):
        if s not in result:
            result[s] = 0
        result.rename(columns={s: s + '_count'}, inplace=True)
        denom = int((frame['split'] == s).sum())
        result[s + '_percentage'] = result[s + '_count'] * 100 / denom if denom else 0.0
    result['total_count'] = sum(result[s + '_count'] for s in splits)
    result['total_percentage'] = result.total_count * 100 / len(frame)
    result['is_missing'] = result[field] == MISSING
    result.loc[result.is_missing, field] = None
    result['is_unknown_or_unclassified'] = result[field].map(lambda x: isinstance(x, str) and (x.strip().lower() in UNKNOWN or not x.strip()))
    cols = [field, 'train_count', 'test_count', 'total_count', 'train_percentage', 'test_percentage']
    cols += [x for x in result.columns if x not in cols]
    return result[cols].sort_values(['train_count', 'total_count'], ascending=False, kind='stable').reset_index(drop=True)

def duplicates(frame, columns):
    valid = frame.loc[frame[columns].notna().all(axis=1) & frame[columns].map(lambda x: isinstance(x, str) and bool(x.strip())).all(axis=1)]
    return {'eligible_rows': len(valid), 'excluded_empty_or_null_rows': len(frame)-len(valid),
            'duplicate_excess_rows': int(valid.duplicated(columns).sum()),
            'duplicate_excess_percentage_of_all_rows': float(valid.duplicated(columns).sum()*100/len(frame)),
            'rows_in_duplicate_groups': int(valid.duplicated(columns, keep=False).sum())}

def main():
    random.seed(42)
    np.random.seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((OUT / 'download_metadata.json').read_text(encoding='utf-8'))
    for item in metadata['files']:
        path = RAW / item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError(f'Raw file checksum mismatch: {path}')
    readme = (RAW / 'README.md').read_text(encoding='utf-8')
    front = yaml.safe_load(readme.split('---')[1]) if readme.startswith('---') else {}
    files = sorted(RAW.rglob('*.parquet'))
    if not files:
        raise RuntimeError('No Parquet files; do not silently substitute another dataset.')
    schema_rows, previews, split_files = [], {}, {}
    schemas = {}
    for path in files:
        match = re.match(r'(.+?)-\d+-of-\d+\.parquet$', path.name)
        if not match:
            raise RuntimeError(f'Cannot infer split from filename: {path}; explicit mapping required.')
        split = match.group(1)
        split_files.setdefault(split, []).append(path)
        pf = pq.ParquetFile(path)
        schemas[split] = str(pf.schema_arrow)
        for f in pf.schema_arrow:
            schema_rows.append({'split': split, 'column': f.name, 'arrow_type': str(f.type), 'nullable': f.nullable})
        sample = next(pf.iter_batches(batch_size=5)).to_pylist()
        previews.setdefault(split, [truncate(x) for x in sample])
    schema_frame = pd.DataFrame(schema_rows).drop_duplicates()
    csv('schema.csv', schema_frame)
    dump('schema.json', {'arrow': schemas, 'huggingface_features': {s: json.loads((pq.ParquetFile(ps[0]).schema_arrow.metadata or {}).get(b'huggingface', b'{}')).get('info',{}).get('features',{}) for s, ps in split_files.items()}})
    dump('sample_rows.json', previews)
    print('ACTUAL SCHEMA\n' + table(schema_frame), flush=True)
    print('FIRST 5 ROWS PER SPLIT\n' + json.dumps(previews, ensure_ascii=False, indent=2), flush=True)
    columns = set(schema_frame.column)
    qnames = [x for x in ['question', 'query', 'prompt', 'input', 'problem', 'messages'] if x in columns]
    anames = [x for x in ['answer', 'label', 'response', 'output', 'solution', 'target', 'completion'] if x in columns]
    if len(qnames) != 1 or len(anames) != 1:
        raise RuntimeError(f'Ambiguous QA mapping; inspect schema: {qnames}, {anames}')
    qcol, acol = qnames[0], anames[0]
    domain_fields = [c for c in sorted(columns) if re.search(r'subject|domain|category|field|discipline|topic|area|subset|specialty|speciality', c, re.I)]
    rows, extraction_issues = [], []
    for split, paths in split_files.items():
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
                for raw in batch.to_pylist():
                    question, issue = extract_question(raw[qcol])
                    answer = raw[acol]
                    if not scalar(answer):
                        raise RuntimeError(f'Unsupported non-string answer in {split}; no silent coercion.')
                    row = {'split': split, 'question': question, 'answer': answer, **{f: raw.get(f) for f in domain_fields}}
                    if issue:
                        extraction_issues.append({'split': split, 'row_index': len(rows), 'issue': issue})
                    rows.append(row)
                print(f'Loaded {split}: {len(rows):,} total rows', flush=True) if len(rows) % 131072 == 0 else None
    frame = pd.DataFrame(rows)
    del rows
    splits = list(split_files)
    split_counts = frame.groupby('split', sort=False).size().to_dict()
    domain_summaries, domain_tables = [], {}
    for f in domain_fields:
        if not frame[f].map(scalar).all():
            raise RuntimeError(f'Non-scalar domain field {f}; explicit analysis needed.')
        ct = counts(frame, f, splits)
        domain_tables[f] = ct
        csv(f'{f}_counts.csv', ct)
        s = frame[f]
        masks = {'null': s.isna(), 'blank': s.map(lambda x: isinstance(x,str) and not x.strip()),
                 'none_string': s.map(lambda x: isinstance(x,str) and x.strip().lower() == 'none'),
                 'unclassified': s.map(lambda x: isinstance(x,str) and x.strip().lower() == 'unclassified'),
                 'unknown_or_unclassified': s.map(lambda x: isinstance(x,str) and x.strip().lower() in UNKNOWN)}
        for split in ['ALL'] + splits:
            eligible = pd.Series(True,index=frame.index) if split == 'ALL' else frame.split.eq(split)
            n = int(eligible.sum())
            domain_summaries.append({'field': f, 'split': split, 'rows': n, 'unique_non_null_values': int(s[eligible].nunique(dropna=True)),
                                     **{k+'_count':int((mask & eligible).sum()) for k,mask in masks.items()},
                                     **{k+'_percentage': float((mask & eligible).sum()*100/n) for k,mask in masks.items()}})
    domain_summary = pd.DataFrame(domain_summaries)
    csv('domain_field_summary.csv', domain_summary)
    subject = 'subject' if 'subject' in domain_fields else next((f for f in domain_fields if 'subject' in f.lower()), None)
    for col in ['question', 'answer']:
        frame[col+'_chars'] = frame[col].map(lambda x: len(x) if isinstance(x,str) else np.nan)
        frame[col+'_words'] = frame[col].map(lambda x: len(x.split()) if isinstance(x,str) else np.nan)
    length_rows = []
    groups = [('ALL', frame)] + list(frame.groupby('split', sort=False))
    if subject:
        groups += [(str(s),g) for s,g in frame.groupby(subject,dropna=False,sort=False)]
    for idx,(name,g) in enumerate(groups):
        scope = 'overall' if idx == 0 else 'split' if idx <= len(splits) else 'subject'
        for metric in ['question_chars','question_words','answer_chars','answer_words']:
            v = g[metric].dropna()
            length_rows.append({'scope':scope,'group':name,'metric':metric,'row_count':len(g),'valid_count':len(v),
                                'mean':v.mean(),'median':v.median(),'p10':v.quantile(.1),'p90':v.quantile(.9),'max':v.max()})
    lengths = pd.DataFrame(length_rows)
    if subject:
        split_length_rows = []
        for (sp,s),g in frame.groupby(['split',subject],dropna=False,sort=False):
            for metric in ['question_chars','question_words','answer_chars','answer_words']:
                v = g[metric].dropna()
                split_length_rows.append({'scope':'subject_split','group':str(s),'split':sp,'metric':metric,'row_count':len(g),'valid_count':len(v),'mean':v.mean(),'median':v.median(),'p10':v.quantile(.1),'p90':v.quantile(.9),'max':v.max()})
        lengths = pd.concat([lengths,pd.DataFrame(split_length_rows)],ignore_index=True)
    csv('qa_length_statistics.csv', lengths)
    stats = []
    if subject:
        sc = domain_tables[subject].rename(columns={subject:'subject'})
        csv('subject_counts.csv',sc)
        for s,g in frame.groupby(subject,dropna=False,sort=False):
            rec = {'subject': None if pd.isna(s) else s, 'is_missing':pd.isna(s), 'total_count':len(g)}
            for metric in ['question_chars','question_words','answer_chars','answer_words']:
                v = g[metric].dropna()
                for stat,value in [('mean',v.mean()),('median',v.median()),('p10',v.quantile(.1)),('p90',v.quantile(.9)),('max',v.max())]:
                    rec[metric+'_'+stat] = value
            stats.append(rec)
        statistics = pd.DataFrame(stats)
        csv('subject_statistics.csv',statistics)
        valid = sc.loc[~sc.is_missing & ~sc.is_unknown_or_unclassified].copy()
        top = valid.head(20).merge(statistics.drop(columns=['total_count','is_missing']),on='subject',how='left')
        top.insert(0,'rank',range(1,len(top)+1))
        top = top.rename(columns={'train_percentage':'percentage_of_train','question_words_mean':'avg_question_words','question_words_median':'median_question_words','answer_words_mean':'avg_answer_words','answer_words_median':'median_answer_words'})
        top = top[['rank','subject','train_count','test_count','total_count','percentage_of_train','avg_question_words','median_question_words','avg_answer_words','median_answer_words']]
        candidates = valid[['subject','train_count','test_count','total_count']].copy()
        for k,mask in {'A_train_ge_10000':valid.train_count.ge(10000),'B_train_ge_5000':valid.train_count.ge(5000),'C_train_ge_2000':valid.train_count.ge(2000),'D_test_ge_100':valid.test_count.ge(100),'E_train_ge_5000_test_ge_100':valid.train_count.ge(5000)&valid.test_count.ge(100)}.items():
            candidates[k] = mask
    else:
        sc = pd.DataFrame(columns=['subject','train_count','test_count','total_count','train_percentage','test_percentage'])
        statistics = pd.DataFrame(columns=['subject'])
        top = pd.DataFrame(columns=['rank','subject','train_count','test_count','total_count','percentage_of_train','avg_question_words','median_question_words','avg_answer_words','median_answer_words'])
        candidates = pd.DataFrame(columns=['subject','train_count','test_count','total_count','A_train_ge_10000','B_train_ge_5000','C_train_ge_2000','D_test_ge_100','E_train_ge_5000_test_ge_100'])
        csv('subject_counts.csv',sc)
        csv('subject_statistics.csv',statistics)
    csv('top20_subjects.csv',top)
    csv('candidate_subjects.csv',candidates)
    if subject:
        csv('top20_subject_values_including_placeholders.csv',sc.head(20))
    answer_flags = []
    for pattern_name,pattern in {'empty_or_whitespace':r'^\s*$','placeholder_only':r'(?i)^\s*(none|null|n/?a|unknown|undefined)\s*$','replacement_character':'\ufffd','control_character':r'[\x00-\x08\x0b\x0c\x0e-\x1f]','html_tag':r'<[A-Za-z][^>]*>','markdown_code_fence':r'```','json_like_start':r'^\s*[\[{]'}.items():
        flagged = frame.loc[frame.answer.map(lambda x:isinstance(x,str) and bool(re.search(pattern,x)))]
        for index,row in flagged.head(5).iterrows():
            answer_flags.append({'flag':pattern_name,'row_index':index,'split':row.split,'question':truncate(row.question),'answer':truncate(row.answer),'subject':row.get(subject) if subject else None})
    dump('answer_format_examples.json',answer_flags)
    quality = {}
    for split,g in [('ALL',frame)] + list(frame.groupby('split',sort=False)):
        a = g.answer
        quality[split] = {'rows':len(g),'duplicate_question':duplicates(g,['question']),'duplicate_qa':duplicates(g,['question','answer']),
                          'null_question':int(g.question.isna().sum()),'empty_question_including_null':int(g.question.map(lambda x:x is None or not x.strip()).sum()),
                          'null_answer':int(a.isna().sum()),'empty_answer_including_null':int(a.map(lambda x:x is None or not x.strip()).sum()),
                          'answer_format_flags':{key:int(a.map(lambda x:isinstance(x,str) and bool(re.search(pattern,x))).sum()) for key,pattern in {
                              'replacement_character':'\ufffd','control_character':r'[\x00-\x08\x0b\x0c\x0e-\x1f]',
                              'html_tag':r'<[A-Za-z][^>]*>','markdown_code_fence':r'```','json_like_start':r'^\s*[\[{]',
                              'placeholder_only':r'(?i)^\s*(none|null|n/?a|unknown|undefined)\s*$'}.items()}}
    leak = {}
    if 'train' in splits and 'test' in splits:
        train,test = frame.loc[frame.split.eq('train')],frame.loc[frame.split.eq('test')]
        for mode in ['exact','whitespace_normalized']:
            t,r = train[['question','answer']].copy(),test[['question','answer']].copy()
            if mode != 'exact':
                for c in t:
                    t[c] = t[c].map(lambda x:' '.join(x.split()) if isinstance(x,str) else x)
                    r[c] = r[c].map(lambda x:' '.join(x.split()) if isinstance(x,str) else x)
            qs = set(t.question.dropna()) - {''}
            qas = set(t.loc[t.notna().all(axis=1)&t.question.ne('')&t.answer.ne('')].itertuples(index=False,name=None))
            rq = set(r.question.dropna()) - {''}
            rqa = set(r.loc[r.notna().all(axis=1)&r.question.ne('')&r.answer.ne('')].itertuples(index=False,name=None))
            leak[mode] = {'unique_question_overlap':len(qs & rq),'test_rows_with_question_in_train':int(r.question.isin(qs).sum()),
                          'percentage_of_test_rows_question_overlap':float(r.question.isin(qs).mean()*100),
                          'unique_qa_overlap':len(qas & rqa),'test_rows_with_exact_qa_in_train':sum(x in qas for x in r.itertuples(index=False,name=None))}
    label_quality = domain_summary.to_dict('records')
    imbalance = {'valid_subject_count':len(valid) if subject else 0}
    if subject and len(valid):
        nonzero = valid.loc[valid.train_count.gt(0)].train_count
        imbalance.update({'min_train_count':int(nonzero.min()) if len(nonzero) else None,'max_train_count':int(nonzero.max()) if len(nonzero) else None,'max_min_ratio':float(nonzero.max()/nonzero.min()) if len(nonzero) else None,'train_subjects_below_2000':int(valid.train_count.lt(2000).sum()),'test_max_min_ratio':float(valid.test_count.max()/valid.loc[valid.test_count.gt(0)].test_count.min())})
        csv('subject_imbalance.csv', valid[['subject','train_count','test_count','train_percentage','test_percentage']].assign(train_below_2000=valid.train_count.lt(2000),test_below_100=valid.test_count.lt(100)))
    dump('data_quality.json',{'by_split':quality,'train_test_leakage':leak,'domain_labels':label_quality,'imbalance':imbalance,'extraction_issues':extraction_issues})
    frame.to_parquet(PROCESSED / 'multi_subject_rlvr_qa_features.parquet',index=False)
    observed = {'splits':split_counts,'columns':sorted(columns)}
    discrepancies = []
    # Explicit card claims; values below are comparisons, never substituted statistics.
    if '638k' in readme and len(frame) != 638000:
        discrepancies.append({'kind':'prose_total_count','readme_claim':638000,'actual_downloaded_total':len(frame),'note':'Card may refer to original corpus; published snapshot differs.'})
    if subject and frame.loc[frame.split.eq('train'),subject].map(lambda x: x is None or (isinstance(x,str) and x.strip().lower() in UNKNOWN)).all():
        discrepancies.append({'kind':'subject_label_coverage','note':'README describes subject classification, but every published training subject is missing or an unknown placeholder. Exact raw value distribution is in subject_counts.csv; string None is distinct from null. Test labels must not be extrapolated to train.'})
    expected_info = front.get('dataset_info',{})
    if isinstance(expected_info,dict):
        for spec in expected_info.get('splits',[]):
            if split_counts.get(spec['name']) != spec.get('num_examples'):
                discrepancies.append({'kind':'split_count','readme':spec,'actual':split_counts.get(spec['name'])})
        expected_columns = [f['name'] for f in expected_info.get('features',[])]
        if expected_columns and set(expected_columns) != columns:
            discrepancies.append({'kind':'columns','readme':expected_columns,'actual':sorted(columns)})
    metadata.update({'analysis_date_utc':datetime.now(timezone.utc).isoformat(),'python_executable':sys.executable,'python_version':sys.version,
                     'package_versions':{p:importlib.metadata.version(p) for p in ['datasets','huggingface_hub','pandas','pyarrow','numpy','matplotlib','pyyaml']},
                     'random_seed':42,'observed':observed,'qa_mapping':{'question':qcol,'answer':acol,'question_extraction':'string as-is or newline join of user-role message contents; excludes system instructions'},
                     'domain_fields':domain_fields,'readme_discrepancies':discrepancies,'download_verified':True})
    dump('dataset_metadata.json',metadata)
    (OUT/'requirements-analysis.txt').write_text('\n'.join(f'{k}=={v}' for k,v in metadata['package_versions'].items())+'\n',encoding='utf-8')
    overview = f"Repository: `{metadata['repo_id']}`\n\nRevision: `{metadata['revision']}`\n\nLicense: `{metadata['license']}`\n\nDownloaded UTC: {metadata['download_date_utc']}\n\nFull repository downloaded and checksums verified.\n\n" + table(pd.DataFrame([{'split':s,'rows':n} for s,n in split_counts.items()]))
    schema_text = table(schema_frame) + '\n\nQA mapping: ' + json.dumps(metadata['qa_mapping'],ensure_ascii=False) + '\n\nFirst five raw samples per split (long strings truncated):\n\n```json\n' + json.dumps(previews,ensure_ascii=False,indent=2) + '\n```'
    (OUT/'dataset_overview.md').write_text('# Dataset Overview\n\n'+overview+'\n\n'+schema_text,encoding='utf-8')
    quality_text = 'Exact duplication compares original extracted strings, without trimming or case folding. Duplicate rate counts excess occurrences (N minus distinct), divided by all rows; null/blank QA excluded and reported separately. Format flags are structural screening, not proof of incorrect answers. Deterministic examples are saved in `answer_format_examples.json`. String None and Unclassified are counted separately from true null.\n\n```json\n'+json.dumps(quality,ensure_ascii=False,indent=2)+'\n```\n\nLabel coverage:\n\n'+table(domain_summary)+'\n\nSubject imbalance:\n\n```json\n'+json.dumps(imbalance,indent=2)+'\n```\n\nPer-subject train/test shares and train <2000/test <100 flags are in `subject_imbalance.csv`. A max/min ratio describes imbalance, not an automatic suitability decision. If all training labels are placeholders, training imbalance by named subject cannot be evaluated.'
    leakage_text = 'Exact and whitespace-normalized checks shown separately. Counts include unique overlaps and affected test rows. No rows were removed.\n\n```json\n'+json.dumps(leak,indent=2)+'\n```'
    (OUT/'data_quality_report.md').write_text('# Data Quality\n\n'+quality_text+'\n\n# Train-Test Leakage\n\n'+leakage_text,encoding='utf-8')
    candidate_text = 'Objective thresholds only; missing/blank/unknown labels cannot serve as knowledge domains. No labels inferred, no clients created, and no final domains selected. Target: 20 clients, 8–10 broad domains, roughly 2–3 clients per domain. Sample volume alone cannot establish broad domain suitability.\n\n'+table(candidates)
    for col in [c for c in candidates if c.startswith(tuple('ABCDE'))]:
        candidate_text += '\n\n### '+col+'\n\n'+table(candidates.loc[candidates[col],['subject','train_count','test_count','total_count']])
    broad = [f for f in domain_fields if re.search(r'domain|field|discipline',f,re.I)]
    if 'subset' in domain_fields:
        subset_values = sorted(frame['subset'].dropna().unique().tolist())
        dump('original_subject_subset_mapping.json', frame.loc[frame[subject].notna(),[subject,'subset']].drop_duplicates().to_dict('records') if subject else [])
        if {x for x in subset_values if x.strip().lower() not in UNKNOWN} <= {'STEM','Social Sciences','Humanities','Applied Sciences'} and len(subset_values):
            broad.append('subset')
            csv('broad_field_counts.csv',domain_tables['subset'].rename(columns={'subset':'broad_field'}))
    metadata['broad_field_columns'] = broad
    dump('dataset_metadata.json',metadata)
    caveats = ['Only original repository labels are reported. Null is not a subject or a domain; the missing CSV row is an analysis bucket.',
               'Whitespace word counts are not model tokenizer counts; answers are reference labels, not assumed reasoning traces.',
               'All statistics use every downloaded row; no sampling, model training, or client partitioning.',
               'No broad field/discipline column found.' if not broad else 'Broad field/discipline columns: '+', '.join(broad),
               'The actual subset values and original subject/subset pairs are recorded. If subset matches the card broad fields, it is reported as the provided broad field; no new classification is assigned.',
               'README comparison covers machine-readable split counts and feature names. Prose and paper descriptions do not establish label completeness.',
               'README discrepancies: '+json.dumps(discrepancies,ensure_ascii=False),
               'Semantic correctness, factual correctness and domain purity cannot be established by these structural checks.']
    if subject and not len(valid):
        caveats.insert(0,'CRITICAL: no usable subject labels are present. Per-subject named-domain statistics and top-20 named subjects are unavailable; candidate sets A–E are empty. This dataset cannot support label-based domain selection in its downloaded form. No inferred relabeling was attempted.')
    elif subject and not valid.train_count.gt(0).any():
        caveats.insert(0,'CRITICAL: all training subject labels are missing/unknown placeholders. The literal string None is not a genuine null and is preserved in raw value tables. Named subjects occur only in test. Candidate sets A/B/C/E are empty; D lists only test-supported subjects. A top-20 named-subject table sorted by train_count has all-zero ties (secondary total_count descending) and cannot identify the largest training domains. Per-subject named QA lengths come only from test. The snapshot cannot support label-based training-domain selection without additional authentic annotations.')
    sections = [('1. Dataset Overview',overview),('2. Schema',schema_text),('3. Splits',table(pd.DataFrame([{'split':s,'rows':n} for s,n in split_counts.items()]))),
                ('4. Subject Distribution',table(sc)+'\n\nAll potential domain fields:\n\n'+table(domain_summary)+'\n\n'+'\n\n'.join('### '+f+'\n\n'+table(t) for f,t in domain_tables.items() if f != subject)),
                ('5. Top 20 Subjects',table(top)+'\n\nOnly non-missing, non-unknown original labels qualify. All named train counts are tied at zero when train contains only placeholders; test counts determine the secondary order.\n\nRaw top-20 values including placeholders:\n\n'+(table(sc.head(20)) if subject else 'Unavailable.')),('6. QA Length Statistics',table(lengths.fillna(''))+'\n\nPer-subject wide statistics: `subject_statistics.csv`; original placeholder groups are preserved. Named-subject statistics are test-only when training lacks subject annotations.'),
                ('7. Data Quality',quality_text),('8. Train-Test Leakage Check',leakage_text),('9. Candidate Domains for 20-client Federated Setting',candidate_text),('10. Important Caveats','\n'.join('- '+x for x in caveats))]
    (OUT/'REPORT.md').write_text('# Multi-subject-RLVR Dataset Analysis\n\n'+'\n\n'.join('## '+title+'\n\n'+body for title,body in sections)+'\n',encoding='utf-8')
    summary = 'Splits: '+json.dumps(split_counts)+'\nSubject values (non-null, including placeholders): '+str(frame[subject].nunique() if subject else 0)+'\nNamed usable subjects: '+str(len(valid) if subject else 0)+'\n'+table(domain_summary)+'\nTop 20 named subjects (all train=0; test count breaks ties):\n'+table(top)+'\nSubjects satisfying train >=5000 and test >=100:\n'+table(candidates.loc[candidates.get('E_train_ge_5000_test_ge_100',pd.Series(False,index=candidates.index)).astype(bool)])+'\nDuplicates:\n'+json.dumps(quality['ALL'],indent=2)+'\nOverlap:\n'+json.dumps(leak,indent=2)+'\nREPORT: '+str(OUT/'REPORT.md')
    (OUT/'terminal_summary.txt').write_text(summary+'\n',encoding='utf-8')
    print(summary,flush=True)
    required = ['dataset_overview.md','subject_counts.csv','subject_statistics.csv','top20_subjects.csv','candidate_subjects.csv','data_quality_report.md','dataset_metadata.json','REPORT.md']
    for name in required:
        assert (OUT/name).is_file() and (OUT/name).stat().st_size > 0,name
    inventory = [{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size} for base in [OUT,RAW,PROCESSED] for p in sorted(base.rglob('*')) if p.is_file() and '.cache' not in p.parts]
    csv('output_file_inventory.csv',pd.DataFrame(inventory))
    print('VERIFIED OUTPUT FILES\n'+table(pd.DataFrame(inventory)),flush=True)

if __name__ == '__main__':
    main()
