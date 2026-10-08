"""Frozen targeted delta over conservative-content-v4. No experiment splitting.

All amendments are based solely on the previous final_audit_issues.csv.
Never revise these rules using the held-out validation samples.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, inspect, json, re
import pandas as pd
import build_bhashabench_candidates as base

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'reports/bhashabench_evidence'
OUT=ROOT/'data/bhashabench_final'
SEED=2026
VERSION='targeted-v5-frozen'

RULES={
 'T01_special_option':r'\beach\s+of\s+(?:these|them|the\s+above)\b|\b(?:more\s+than\s+one|multiple)\s+(?:answer|option)s?\s+(?:is|are)\s+correct\b|(?<!\w)\(?[a-d1-4]\)?\s*\+\s*\(?[a-d1-4]\)?(?!\w)',
 'T02_ayur_patient':r'\b\d+\s*[- ]?year(?:s)?\s*[- ]?old\b|\b(?:a\s+)?(?:person|patient|child|worker)\b.{0,80}\b(?:presented|presents|comes|complains|suffered|suffering|admitted)\b',
 'T03_legal_letter_coding':r'\b(?:a certain code|is coded|find the code|coded language)\b|\bhow is\b.{0,80}\bwritten\b.{0,30}\bcode\b',
 'T04_legal_unanchored_office_holder':r'\bwho\b.{0,45}\b(?:is|current|present)\b.{0,60}\b(?:chief justice|chief minister|president|governor)\b',
 'T05_legal_named_event':r'\b[A-Z][a-z]{2,}\s+(?:is\s+charged|agreed|assures|signed|signs|gave|gives|borrowed|sold|sells|killed|murders|transferred)\b',
 'T06_legal_narrative_application':r'\b(?:landlord|tenant|payee|holder|plaintiff|defendant|appellant|child)\b.{0,90}\b(?:files a suit|recovered possession|withdraws the appeal|crossing the age|fills up|dies before|survives|declared adult|should be tried as an adult)\b|\bin such a (?:situation|case)\b|\bif he survives\b',
 'T07_krishi_genetics_inference':r'\bF[12]\b.{0,100}\b(?:segregat\w*|ratio|susceptible|resistant)\b.{0,180}\b(?:test cross|testcross|will segregate|expected)\b',
 'T08_krishi_calculation':r'\b(?:gypsum requirement|amount of gypsum)\b.{0,160}\b(?:CEC|ESP|ha\b|hectares?|meq)\b|\b(?:calculate|compute)\b.{0,100}\b(?:requirement|efficiency|yield|fertiliz\w*)\b',
 'T09_krishi_missing_vaccine':r'^\s*vaccine is given to sheep\b',
}
RX={key:re.compile(value,re.I|re.S) for key,value in RULES.items()}
NAMED_EVENT=re.compile(RULES['T05_legal_named_event'])

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def targeted_rules(domain,row):
    q=str(row.question);opts=[str(row[c]) for c in base.OPTIONS];hits=[]
    def add(reason,rule):hits.append((reason,rule))
    if any(RX['T01_special_option'].search(o) for o in opts):add('special_option','T01_special_option')
    # A list of the three other literal option texts is a composite answer.
    for i,o in enumerate(opts):
        others=[base.norm(x) for j,x in enumerate(opts) if i!=j]
        normo=base.norm(o)
        if ',' in o and all(len(x)>=3 and x in normo for x in others):add('special_option','T01b_literal_union_of_other_options')
    if domain=='ayur':
        if RX['T02_ayur_patient'].search(q):add('reasoning_application','T02_ayur_patient')
        fixed=base.DOMAIN_PATTERNS['ayur'].replace('must[a]?','musta')
        if not re.search(fixed,q,re.I):add('off_domain','T02b_fix_must_vs_musta_false_domain_cue')
    if domain=='legal':
        if RX['T03_legal_letter_coding'].search(q):add('calculation_aptitude','T03_legal_letter_coding')
        if RX['T04_legal_unanchored_office_holder'].search(q) and not re.search(r'\b(?:18|19|20)\d{2}\b',q):add('off_domain','T04_legal_unanchored_office_holder')
        if NAMED_EVENT.search(q):add('reasoning_application','T05_legal_named_event')
        if RX['T06_legal_narrative_application'].search(q):add('reasoning_application','T06_legal_narrative_application')
    if domain=='krishi':
        no=[base.norm(o) for o in opts]
        if set(no)=={'1','2','3','4'} and not re.search(r'\b(?:how many|number of|total number|types of|pairs of)\b',q,re.I):add('not_self_contained','T07a_bare_1234_options_without_explicit_count_query')
        if any(o in {'1','2','3','4'} for o in no) and sum(len(o.split())>=2 for o in no)>=2:add('corrupted','T07b_lone_digit_placeholder_among_textual_options')
        if RX['T07_krishi_genetics_inference'].search(q):add('reasoning_application','T07_krishi_genetics_inference')
        if RX['T08_krishi_calculation'].search(q):add('calculation_aptitude','T08_krishi_calculation')
        if RX['T09_krishi_missing_vaccine'].search(q):add('not_self_contained','T09_krishi_missing_vaccine')
    return hits

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    issuepath=ROOT/'reports/final_audit_issues.csv'
    issues=pd.read_csv(issuepath,dtype=str).fillna('')
    snapshots={};history=[];excluded_ids={d:set() for d in base.DOMAINS}
    for d in base.DOMAINS:
        paths=[ROOT/f'analysis_samples/{d}_200.csv',EV/f'pilot_v1_{d}_300.csv',ROOT/f'reports/final_audit_{d}_300.csv']
        for path in paths:
            assert path.exists(),path
            data=pd.read_csv(path,dtype=str).fillna('')
            assert 'id' in data
            excluded_ids[d].update(data.id)
            history.append({'domain':d,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'rows':len(data),'ids':data.id.tolist()})
    # Exclude the consolidated previous manual ledger as an extra safeguard.
    manualpath=ROOT/'reports/manual_taxonomy_audit.csv'
    if manualpath.exists():
        manual=pd.read_csv(manualpath,dtype=str).fillna('')
        for d in base.DOMAINS:excluded_ids[d].update(manual.loc[manual.domain.eq(d),'id'])
        history.append({'domain':'all','path':manualpath.relative_to(ROOT).as_posix(),'sha256':sha(manualpath),'rows':len(manual)})
    frozen={'rule_version':VERSION,'seed':SEED,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'targeted_script_sha256':sha(Path(__file__)),'base_script_sha256':sha(ROOT/'scripts/build_bhashabench_candidates.py'),
        'issues_sha256':sha(issuepath),'patterns':RULES,'function_source':inspect.getsource(targeted_rules),
        'quarantine_source':'All 52 individually reviewed problem rows in final_audit_issues.csv, by domain and native ID; no new validation IDs used.',
        'historical_audits':history,'prior_audited_unique_ids':{d:len(ids) for d,ids in excluded_ids.items()},
        'heldout_must_never_tune_rules':True}
    # Persist rules before drawing any new validation samples.
    (EV/'targeted_frozen_rules.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf8')
    frames=[];removed=[];stats=[];manifest=[]
    for d in base.DOMAINS:
        path=base.OUT/f'{d}_candidates.parquet'
        data=pd.read_parquet(path).fillna('');snapshots[d]=sha(path)
        bad={r.id:r for _,r in issues[issues.domain.eq(d)].iterrows()}
        keep=[]
        for i,r in data.iterrows():
            hits=targeted_rules(d,r)
            if r.id in bad:
                issue=bad[r.id]
                reason={'reasoning_application':'reasoning_application','calculation_aptitude':'calculation_aptitude','general_offdomain':'off_domain'}.get(issue.taxonomy,'not_self_contained' if issue.self_contained=='no' else 'corrupted')
                if 'special_option' in issue.additional_quality_flag:reason='special_option'
                hits.append((reason,'T00_reviewed_issue_quarantine'))
            if hits:
                removed.append({**r.to_dict(),'targeted_removal_reason':hits[0][0],
                    'targeted_rule_ids':json.dumps(sorted(set(x[1] for x in hits))),
                    'all_targeted_reasons':json.dumps(sorted(set(x[0] for x in hits))),
                    'previous_review_reason':bad[r.id].short_reason if r.id in bad else '',
                    'previous_quality_flag':bad[r.id].additional_quality_flag if r.id in bad else ''})
            else:keep.append(i)
        final=data.loc[keep].copy().reset_index(drop=True)
        final['final_cleaning_version']=VERSION
        final.to_parquet(OUT/f'{d}.parquet',index=False)
        pd.testing.assert_frame_equal(final,pd.read_parquet(OUT/f'{d}.parquet'))
        frames.append(final)
        pool=final[~final.id.isin(excluded_ids[d])]
        assert len(pool)>=300
        sample=pool.sample(n=300,random_state=SEED).reset_index(drop=True)
        assert not set(sample.id)&excluded_ids[d]
        sample.insert(0,'heldout_audit_index',range(300))
        for c in ['taxonomy','self_contained','quality_issue','short_reason']:sample[c]=''
        sample.to_csv(ROOT/f'reports/heldout_validation_{d}_300.csv',index=False,encoding='utf-8-sig')
        # Compact reading artifact hides native subject/answer to avoid shortcuts.
        (EV/f'heldout_reading_{d}.txt').write_text('\n'.join(f'{i} {r.question} | '+ ' / '.join(str(r[c]) for c in base.OPTIONS) for i,r in sample.iterrows()),encoding='utf8')
        identity=sample[['id','source_row_index','question',*base.OPTIONS]].astype(str).values.tolist()
        manifest.append({'domain':d,'candidate_sha256':sha(OUT/f'{d}.parquet'),
            'sample_content_sha256':hashlib.sha256(json.dumps(identity,ensure_ascii=False).encode()).hexdigest(),
            'new_seed':SEED,'prior_audited_overlap_count':0,'pool_n':len(pool),'review_n':300})
        rows=[r for r in removed if r['domain']==d]
        stats.append({'domain':d,'original':len(pd.read_parquet(ROOT/f'data/bhashabench/{d}/test.parquet')),
            'previous_candidates':len(data),'targeted_removed':len(rows),'final_count':len(final),
            'prior_audited_unique_ids':len(excluded_ids[d]),'heldout_eligible_pool':len(pool),
            'quarantined_reviewed_issues':sum('T00_reviewed_issue_quarantine' in r['targeted_rule_ids'] for r in rows)})
    allfinal=pd.concat(frames,ignore_index=True).fillna('')
    assert allfinal.cluster_id.is_unique
    assert not allfinal.question.map(base.norm).duplicated().any()
    qa=allfinal.apply(lambda r:json.dumps([base.norm(r.question),*[base.norm(r[c]) for c in base.OPTIONS],base.norm(r.correct_answer)]),axis=1)
    assert not qa.duplicated().any()
    near,proposals=base.near_edges(allfinal.question.tolist());assert not near
    for d in base.DOMAINS:assert snapshots[d]==sha(base.OUT/f'{d}_candidates.parquet')
    pd.DataFrame(removed).to_csv(ROOT/'reports/targeted_removed_examples.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(stats).to_csv(ROOT/'reports/final_cleanup_counts.csv',index=False,encoding='utf-8-sig')
    summary={'version':VERSION,'counts':stats,'frozen_rules_sha256':sha(EV/'targeted_frozen_rules.json'),
        'candidate_sources_sha256':snapshots,'heldout_manifest':manifest,
        'final_normalized_qa_duplicates':0,'final_normalized_stem_duplicates':0,
        'final_duplicate_cluster_ids':0,'final_near_pairs_085':0,'final_cross_domain_duplicate_pairs':0,
        'heldout_review_status':'PENDING_CONTENT_REVIEW_RULES_FROZEN'}
    (EV/'targeted_cleanup_run.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(pd.DataFrame(stats).to_string(index=False))
    print('RULES FROZEN; QA/stem/cluster/near/cross-domain duplicates = 0; new audit overlap = 0.')

if __name__=='__main__':main()
