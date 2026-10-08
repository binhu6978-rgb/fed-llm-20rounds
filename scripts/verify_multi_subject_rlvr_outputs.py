"""Verify saved analysis totals against downloaded Parquet metadata and derived rows."""
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

root = Path(__file__).resolve().parents[1]
out = root / 'analysis/multi_subject_rlvr'
metadata = json.loads((out/'dataset_metadata.json').read_text(encoding='utf-8'))
required = ['dataset_overview.md','subject_counts.csv','subject_statistics.csv','top20_subjects.csv','candidate_subjects.csv','data_quality_report.md','dataset_metadata.json','REPORT.md']
for name in required:
    assert (out/name).is_file() and (out/name).stat().st_size > 0, name
counts = pd.read_csv(out/'subject_counts.csv',keep_default_na=False)
broad = pd.read_csv(out/'broad_field_counts.csv',keep_default_na=False)
for split,count in metadata['observed']['splits'].items():
    raw_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in (root/'data/raw/multi_subject_rlvr').rglob(f'{split}-*.parquet'))
    assert count == raw_rows == int(counts[split+'_count'].sum()) == int(broad[split+'_count'].sum())
derived = pd.read_parquet(root/'data/processed/multi_subject_rlvr_qa_features.parquet')
assert len(derived) == sum(metadata['observed']['splits'].values())
assert derived.loc[derived.split.eq('train'),'subject'].eq('None').all()
assert derived.loc[derived.split.eq('train'),'subset'].eq('None').all()
quality = json.loads((out/'data_quality.json').read_text(encoding='utf-8'))
assert int(derived.duplicated(['question','answer']).sum()) == quality['by_split']['ALL']['duplicate_qa']['duplicate_excess_rows']
assert int(derived.duplicated('question').sum()) == quality['by_split']['ALL']['duplicate_question']['duplicate_excess_rows']
assert not quality['extraction_issues']
candidates = pd.read_csv(out/'candidate_subjects.csv',keep_default_na=False)
print('VERIFIED: raw split counts, CSV totals, processed rows, complete QA extraction, duplicates, placeholder coverage.')
print('train/test:',metadata['observed']['splits'])
print('subject raw values:',len(counts),'named subjects:',len(candidates))
uc = counts.loc[counts.subject.eq('Unclassified')].iloc[0]
print(f"Unclassified: {uc.total_count} ({uc.test_percentage:.4f}% of test, {uc.total_percentage:.4f}% of all rows)")
print('TOP20 named subjects (train ties at zero; ordered secondarily by test):')
print(pd.read_csv(out/'top20_subjects.csv')[['rank','subject','train_count','test_count']].to_string(index=False))
print('Candidate set sizes:',{c:int(candidates[c].sum()) for c in candidates if c.startswith(tuple('ABCDE'))})
print('train>=5000 AND test>=100:',candidates.loc[candidates.E_train_ge_5000_test_ge_100,'subject'].tolist())
print('Duplicate question/QA excess rows and %:',quality['by_split']['ALL']['duplicate_question']['duplicate_excess_rows'],quality['by_split']['ALL']['duplicate_question']['duplicate_excess_percentage_of_all_rows'],quality['by_split']['ALL']['duplicate_qa']['duplicate_excess_rows'],quality['by_split']['ALL']['duplicate_qa']['duplicate_excess_percentage_of_all_rows'])
print('Train/test overlap:',quality['train_test_leakage'])
print('REPORT:',out/'REPORT.md')
inventory = []
for base in [out,root/'data/raw/multi_subject_rlvr',root/'data/processed',root/'scripts']:
    for path in sorted(base.rglob('*')):
        if path.is_file() and '.cache' not in path.parts and '__pycache__' not in path.parts and path.name != 'output_file_inventory.csv':
            inventory.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size})
pd.DataFrame(inventory).to_csv(out/'output_file_inventory.csv',index=False,encoding='utf-8-sig')
inventory.append({'path':str((out/'output_file_inventory.csv').relative_to(root)),'bytes':(out/'output_file_inventory.csv').stat().st_size})
print('VERIFIED FILE SIZES (bytes):\n'+pd.DataFrame(inventory).to_string(index=False))
