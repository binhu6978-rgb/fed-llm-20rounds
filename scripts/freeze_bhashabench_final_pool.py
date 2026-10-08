"""Freeze v5 minus explicitly recorded audit issues, by (domain, native id).

No classifier, new regex, balancing, downsampling, split, or training.
Existing normalization/near-duplicate definitions are used only for verification.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import hashlib, json
import pandas as pd
import build_bhashabench_candidates as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/bhashabench_final_pool'
REPORT=ROOT/'reports/final_pool_summary.md'
DOMAINS=['ayur','legal','krishi']
OPTIONS=['option_a','option_b','option_c','option_d']
NOTE='No train/dev/test split or client partition has been performed.'
CONFIRM='Final usable pools are frozen. No split, client allocation, downsampling, or training has been performed.'
BAD_TAXONOMY={'reasoning_application','calculation_aptitude','general_offdomain'}
YES={'yes','true','1'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(path):return path.relative_to(ROOT).as_posix()
def md(frame):
    columns=frame.columns.tolist()
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in row)+' |')
    return '\n'.join(lines)

def evidence_reasons(row):
    """Use literal prior annotation values only; do not classify question text."""
    reasons=[]
    value=str(row.get('taxonomy','')).strip().lower()
    if value in BAD_TAXONOMY:reasons.append(value)
    if str(row.get('self_contained','')).strip().lower()=='no':reasons.append('not_self_contained')
    if str(row.get('quality_issue','')).strip().lower() in YES:reasons.append('quality_issue')
    if str(row.get('semantic_duplicate','')).strip().lower() in YES:reasons.append('semantic_duplicate')
    flag=str(row.get('additional_quality_flag','')).strip()
    if flag:reasons.append('prior_explicit_quality_flag:'+flag)
    # Support explicit boolean annotations if older/newer ledgers use these fields.
    for column in ['corrupted','ambiguous','special_option','special-option']:
        if str(row.get(column,'')).strip().lower() in YES:reasons.append(column)
    return sorted(set(reasons))

def collect_audits():
    sources=[];registry=defaultdict(list)
    markers={'taxonomy','self_contained','quality_issue','semantic_duplicate','additional_quality_flag','corrupted','ambiguous','special_option','special-option'}
    paths=[]
    for path in sorted((ROOT/'reports').rglob('*.csv')):
        if path==REPORT:continue
        head=pd.read_csv(path,nrows=0)
        if {'id','domain'}.issubset(head.columns) and markers&set(head.columns):paths.append(path)
    paths += [ROOT/'reports/bhashabench_evidence/final_review_annotations.json',
              ROOT/'reports/bhashabench_evidence/heldout_explicit_review_ledger.json']
    for path in paths:
        if path.suffix=='.csv':frame=pd.read_csv(path,dtype=str).fillna('')
        else:frame=pd.DataFrame(json.loads(path.read_text(encoding='utf8'))).fillna('')
        recorded=0
        for _,row in frame.iterrows():
            domain=str(row.get('domain','')).strip().lower()
            if domain not in DOMAINS:continue
            native_id=str(row.get('id','')).strip()
            assert native_id, f'Missing native id in annotation {path}'
            reasons=evidence_reasons(row)
            if not reasons:continue
            registry[(domain,native_id)].append({'audit_file':relative(path),'reasons':reasons,
                'short_reason':str(row.get('short_reason','')),
                'quality_issue_type':str(row.get('quality_issue_type','')),
                'source_row_index':str(row.get('source_row_index',row.get('audit_source_row_index',''))),
                'audit_index':str(row.get('heldout_audit_index',row.get('final_audit_index',row.get('sample_index',''))))})
            recorded+=1
        sources.append({'path':relative(path),'sha256':sha(path),'rows':len(frame),'problem_annotation_rows':recorded})
    assert sources and registry
    return sources,registry

def main():
    if (OUT/'manifest.json').exists():
        existing=json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
        for record in existing['input_candidate_files']+existing['audit_files']+existing['output_files']:
            assert sha(ROOT/record['path'])==record['sha256'], f'Frozen input/output changed: {record["path"]}'
        for d in DOMAINS:print(f'{d.capitalize()} final pool: {existing["domain_counts"][d]}')
        print(CONFIRM);return
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[]
    run=json.loads((ROOT/'reports/bhashabench_evidence/targeted_cleanup_run.json').read_text(encoding='utf8'))
    frozen=ROOT/'reports/bhashabench_evidence/targeted_frozen_rules.json'
    assert sha(frozen)==run['frozen_rules_sha256']
    frozen_sha=sha(frozen)
    prior=json.loads((ROOT/'reports/bhashabench_evidence/cleaning_results.json').read_text(encoding='utf8'))
    for source in prior['source_files']:assert sha(ROOT/source['path'])==source['sha256']
    audit_files,known=collect_audits()
    removed=[];counts=[];checks=[];frames=[]
    for d in DOMAINS:
        path=ROOT/f'data/bhashabench_final/{d}.parquet'
        frame=pd.read_parquet(path)
        expected=next(x for x in run['heldout_manifest'] if x['domain']==d)
        assert sha(path)==expected['candidate_sha256'], f'Not the frozen v5 input: {path}'
        assert frame.final_cleaning_version.eq('targeted-v5-frozen').all()
        assert frame.domain.eq(d).all() and frame.id.is_unique
        inputs.append({'domain':d,'path':relative(path),'sha256':sha(path),'rows':len(frame)})
        mask=frame.id.map(lambda native_id:(d,str(native_id).strip()) in known)
        for _,row in frame[mask].iterrows():
            evidence=known[(d,str(row.id).strip())]
            removed.append({**row.to_dict(),'confirmed_issue_reasons':json.dumps(sorted(set(r for e in evidence for r in e['reasons'])),ensure_ascii=False),
                'audit_source_files':json.dumps(sorted(set(e['audit_file'] for e in evidence)),ensure_ascii=False),
                'audit_evidence':json.dumps(evidence,ensure_ascii=False)})
        pool=frame[~mask].copy()
        assert all((d,str(native_id).strip()) not in known for native_id in pool.id)
        columns=['id','domain','question',*OPTIONS,'correct_answer','subject_domain']
        if 'topic' in frame.columns:columns.append('topic')
        columns += ['question_level','source_row_index']
        pool=pool[columns].copy()
        for col in ['id','question',*OPTIONS,'correct_answer']:
            assert pool[col].notna().all(), (d,col,'null')
            assert pool[col].map(lambda x:isinstance(x,str) and bool(x.strip())).all(), (d,col,'empty or wrong type')
        assert pool.correct_answer.isin(list('ABCD')).all()
        assert pool.apply(lambda r:len(set(r[c] for c in OPTIONS))==4,axis=1).all()
        assert pool.apply(lambda r:len(set(base.norm(r[c]) for c in OPTIONS))==4,axis=1).all()
        pool.insert(pool.columns.get_loc('correct_answer')+1,'answer_text',pool.apply(lambda r:r[OPTIONS['ABCD'.index(r.correct_answer)]],axis=1))
        assert pool.answer_text.map(lambda x:isinstance(x,str) and bool(x.strip())).all()
        assert pool.source_row_index.notna().all()
        original=pd.read_parquet(ROOT/f'data/bhashabench/{d}/test.parquet')
        for c in columns:
            if c in original.columns:assert pool[c].tolist()==original.loc[pool.source_row_index,c].tolist(),(d,c,'original field changed')
        pool=pool.reset_index(drop=True)
        pool.to_parquet(OUT/f'{d}.parquet',index=False)
        pool.to_json(OUT/f'{d}.jsonl',orient='records',lines=True,force_ascii=False)
        pd.testing.assert_frame_equal(pool,pd.read_parquet(OUT/f'{d}.parquet'))
        pd.testing.assert_frame_equal(pool,pd.read_json(OUT/f'{d}.jsonl',lines=True,dtype=False),check_dtype=False)
        counts.append({'domain':d,'before_freeze':len(frame),'removed_confirmed_issues':int(mask.sum()),'final_pool_count':len(pool)})
        checks.append({'domain':d,'empty_question':0,'empty_options':0,'duplicate_normalized_options':0,
            'invalid_answer_label':0,'answer_text_parse_failures':0,'missing_native_id':0,'source_alignment_failures':0})
        assert len(frame)==len(pool)+int(mask.sum())
        frames.append(pool)
    allpool=pd.concat(frames,ignore_index=True)
    stems=allpool.question.map(base.norm)
    qa=allpool.apply(lambda r:json.dumps([base.norm(r.question),*[base.norm(r[c]) for c in OPTIONS],r.correct_answer]),axis=1)
    assert not stems.duplicated().any() and not qa.duplicated().any()
    near,pair_proposals=base.near_edges(allpool.question.tolist())
    assert not near
    duplicates={'normalized_exact_qa_duplicate_excess_rows':0,'normalized_stem_duplicate_excess_rows':0,
        'near_duplicate_pairs_word_trigram_jaccard_gte_085':0,'cross_domain_normalized_qa_pairs':0,
        'cross_domain_normalized_stem_pairs':0,'cross_domain_near_pairs_gte_085':0,
        'near_definition':'Exact set Jaccard over normalized word trigrams >= 0.85, same implementation as v4/v5. No semantic classifier or threshold change.'}
    pd.DataFrame(removed).to_csv(OUT/'removed_known_issues.csv',index=False,encoding='utf-8-sig')
    summary=pd.DataFrame(counts)
    files=[OUT/f'{d}{suffix}' for d in DOMAINS for suffix in ['.parquet','.jsonl']]+[OUT/'removed_known_issues.csv',OUT/'manifest.json']
    report='''# BhashaBench Final Usable Pool

仅从 targeted-v5-frozen 三个输入候选集按 domain + native id 删除已有审计问题记录；没有修改自动规则、扩展 regex、按 subject 推断删除或平衡领域数量。quality_issue=yes 和旧审计显式 additional_quality_flag 按本次要求进入删除清单；记录保留原始理由及“疑似/待专家检查”措辞，不把它们重新宣称为专家认证结论。

## 数量

'''+md(summary)+'''

同一问题 ID 在多个审计文件出现只删除一次；已在 v5 删除或不在本次输入中的 ID 不再次计数。问题清单可包含此前人工审计中的 taxonomy 非 recall、self_contained=no、quality_issue/semantic_duplicate 显式标记及旧 quality flag。原始 BhashaBench、v5 候选集、历史审计及清洗规则 hash 均校验未变。

## 完整性检查（失败数）

'''+md(pd.DataFrame(checks))+'''

question/四个 options 均非空；选项按原文及现有 normalization 均互异；correct_answer 严格为 A/B/C/D，answer_text 直接解析到对应原始选项，未修改题干、选项或答案。原始 id/source_row_index 保留且逐字段核对源行。Ayur/Legal 保留原生 topic；Krishi 原始数据没有 topic，因此没有编造该字段。

## 全局重复检查

'''+md(pd.DataFrame([{'check':k,'result':v} for k,v in duplicates.items() if k!='near_definition']))+'''

near-duplicate 使用现有 normalized word-trigram 集合 Jaccard ≥0.85 定义；没有改阈值或新增语义规则。检查覆盖三个数据池的合并全集，故也覆盖域内和跨域。此定义不保证消除所有语义改写；已人工标记的语义重复样本按 ID 删除。

## 冻结文件

'''+md(pd.DataFrame([{'file':relative(p),'absolute_path':str(p)} for p in files]))+'''

manifest.json 记录输入/审计/输出 SHA256、各域数量、唯一问题 ID 数、创建时间和执行脚本 hash。manifest 本身不做递归自哈希，其余输出包含文件 SHA256。removed_known_issues.csv 包含本次实际删除的每个原始 ID、原字段、去除理由及所有审计来源；没有额外删除未记录问题的样本。

本次冻结仅表示已知问题精确移除后的版本固定，不重新估计全量 purity；此前 held-out 题已用于本轮问题清单，不能继续作为本池的独立质量验证。

'''+NOTE+'\n\n'+CONFIRM+'\n'
    REPORT.write_text(report,encoding='utf8')
    output_paths=[p for p in files if p.name!='manifest.json']+[REPORT]
    outputs=[{'path':relative(p),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in output_paths]
    manifest={'version':'targeted-v5-known-issues-removed-final-pool','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_candidate_version':'targeted-v5-frozen','domain_counts':{r['domain']:r['final_pool_count'] for r in counts},
        'counts':counts,'input_candidate_files':inputs,'output_files':outputs,
        'removed_confirmed_issue_count':len(removed),'removed_confirmed_issues_by_domain':{r['domain']:r['removed_confirmed_issues'] for r in counts},
        'audit_files':audit_files,'known_problem_unique_ids_all_audits':{d:sum(k[0]==d for k in known) for d in DOMAINS},
        'integrity_checks':checks,'duplicate_checks':duplicates,
        'frozen_rule_file':relative(frozen),'frozen_rule_sha256':frozen_sha,
        'builder_script':relative(Path(__file__)),'builder_script_sha256':sha(Path(__file__)),
        'original_files':prior['source_files'],'output_schema_by_domain':{d:list(f.columns) for d,f in zip(DOMAINS,frames)},
        'operations_performed':['exact removal of prior explicitly flagged domain/native-id pairs','integrity and existing duplicate checks','serialization and SHA256 freeze'],
        'note':NOTE,'confirmation':CONFIRM,
        'quality_scope':'Removal list follows existing annotations, including recorded quality concerns. No new classification or renewed purity certification. Prior heldout samples now used as known-issue development evidence.',
        'manifest_hash_note':'Output SHA256 excludes manifest.json itself to avoid recursive self-hashing.'}
    for item in inputs+audit_files+outputs:assert sha(ROOT/item['path'])==item['sha256']
    assert sha(frozen)==frozen_sha
    for item in prior['source_files']:assert sha(ROOT/item['path'])==item['sha256']
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    for r in counts:print(f'{r["domain"].capitalize()} final pool: {r["final_pool_count"]}')
    print(CONFIRM)
    print(summary.to_string(index=False))
    for p in [*files,REPORT]:
        assert p.exists() and p.stat().st_size>0
        print(f'OUTPUT {p} {p.stat().st_size:,} bytes')

if __name__=='__main__':main()
