"""Materialize the assistant's completed question-and-option review and report.

The sparse exception ledger below records judgments AFTER reading all 900 rows.
It is not a keyword classifier: unlisted reviewed rows were individually judged
knowledge recall. No native subject, topic, answer key or previous audit label
was used to decide taxonomy. Candidate data remain frozen during final review.
"""
from pathlib import Path
import hashlib, inspect, json, math
import pandas as pd
import build_bhashabench_candidates as build

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
EV=REPORTS/'bhashabench_evidence'
TYPES=['knowledge_recall','reasoning_application','calculation_aptitude','general_offdomain']
FROZEN_CONTENT_SHA256={
 'ayur':'c686ae72ce5a61ec8ba02d2d812373806bba8f6d4cdfd026c87a3a8a7e3e84a6',
 'legal':'7ec10b3933b8ee423b808e486406bb041b9e7fbf04de5234bb43af8c583aeb7c',
 'krishi':'5b018c7a163d6bb5461b779bac963353b85dfbf84afee20c0761ee2519414824',
}

# index: (taxonomy, self_contained, reason); index is in frozen v4 audit CSV.
EXCEPTIONS={
 'ayur':{
 6:('knowledge_recall','no','陈述句与四个疾病选项没有明确的提问关系，无法确定应回答什么。'),
 16:('knowledge_recall','no','Cpinda 表述疑似缺字，所问的制剂关系不完整。'),
 18:('knowledge_recall','no','本题考查手术指征，但第四选项的压力比较符号缺损，影响选择。'),
 82:('knowledge_recall','no','只说孕妇用蜂蜜治疗，没有给出足以对应疾病选项的完整治疗关系。'),
 166:('reasoning_application','yes','需要根据儿童无 BCG 瘢痕这一临床情境决定免疫处理。'),
 177:('calculation_aptitude','yes','考查计算中位数前的数据排序，属于通用统计操作而非医学知识。'),
 199:('knowledge_recall','no','胎儿性别与双胎的记忆题含无法解释的 228 和残缺表述。'),
 257:('reasoning_application','yes','需要把压缩空气作业者的症状与临床诊断相对应。'),
 293:('knowledge_recall','no','题意为经典症状记忆，但四个选项含难以辨认的转码文字。'),
 },
 'legal':{
 18:('reasoning_application','yes','要求说明在线诽谤中表达自由与名誉权的权衡方式，而非直接回忆法条编号。'),
 72:('reasoning_application','yes','需要对房东收回房屋后再出租的事实适用限制规则。'),
 79:('general_offdomain','yes','问现任高院首席法官姓名，主要是没有时间锚点的人员时事。'),
 147:('reasoning_application','yes','需要对 Ali 无票乘车这一具体情境适用举证责任规则。'),
 163:('calculation_aptitude','yes','需识别字母编码并将变换应用到新词，属于 aptitude。'),
 178:('reasoning_application','yes','需要判断撤回缺席判决上诉后另一申请的法律后果。'),
 189:('reasoning_application','yes','需对少年司法委员会的具体审查情境选择后续程序。'),
 221:('reasoning_application','yes','需综合诉请缺漏、违约与特定履行裁量判断法院可否赔偿。'),
 231:('knowledge_recall','no','退休雇员定义题的 a 后缺少对象，且条款标号残缺。'),
 232:('reasoning_application','yes','需要把录制临终陈述后存活的情境应用到证据规则。'),
 264:('reasoning_application','yes','需根据 Kaushik 与 Vinayak 的交易安排识别抵押类型。'),
 277:('calculation_aptitude','yes','需要推导 DURATION 的编码规则并编码 FORECAST。'),
 285:('reasoning_application','yes','需要根据少年在审判中成年这一情境判断法院处理。'),
 292:('reasoning_application','yes','需要判断空白支票被填写并遭拒付后的法律救济。'),
 },
 'krishi':{
 17:('knowledge_recall','no','考查羧化酶辅因子，但维生素编号被写为 2/Be 等，选项含缺损。'),
 55:('knowledge_recall','no','粉粒粒径题的选项只有 1/2/3/4，占位数字未给出粒径或单位。'),
 81:('reasoning_application','yes','需根据 F1/F2 抗性分离信息推断测交比例。'),
 86:('knowledge_recall','no','扩散率与孔隙度关系题的公式选项发生缺字和转码，无法可靠辨认。'),
 92:('knowledge_recall','no','钾活度关系所指性质题仅提供数字 1/2/3/4，缺少实际选项。'),
 106:('knowledge_recall','no','农业用水比例题只有 1/2/3/4，未提供可解释的百分比选项。'),
 108:('reasoning_application','yes','题干直接提供肥料类别，需要利用给定信息识别所属类别。'),
 118:('knowledge_recall','no','cultivated and percentage 表述缺少被询问的面积或土地对象。'),
 191:('knowledge_recall','no','杀虫剂分组题的第二选项只有 2，源题选项内容缺失。'),
 210:('knowledge_recall','no','堆肥阶段题仅有 1/2/3/4，没有阶段名称或编号定义。'),
 223:('knowledge_recall','no','防风林方位取决于主导风向，题目没有提供该条件。'),
 253:('general_offdomain','yes','询问所有经济部门的温室气体排放排名，缺少农业知识限定。'),
 266:('general_offdomain','yes','询问最先测序的蛋白质，属于通用生物化学史且没有农业限定。'),
 287:('knowledge_recall','no','只说 sheep vaccine 接种间隔，没有指定疫苗或疾病。'),
 291:('calculation_aptitude','yes','需要利用面积、土深、CEC 和 ESP 计算石膏用量。'),
 },
}
FLAGS={
 'ayur':{10:'special_option_residual: Each of these',25:'special_option_residual: a+d',237:'answer_option_correctness_requires_expert_check'},
 'legal':{47:'potential_multiple_correct_options',75:'potential_multiple_correct_options',220:'potential_multiple_correct_options',79:'time_sensitive_unanchored'},
 'krishi':{87:'multiple_correct_options_and_answer_hints',120:'answer_option_correctness_requires_expert_check',166:'answer_option_correctness_requires_expert_check',169:'potential_multiple_correct_options',206:'ambiguous_NOT_incorrect_and_multiple_correct_options',272:'overlapping_numeric_options',285:'special_option_residual: More than one answer are correct',299:'special_option_residual: listed_union_of_options'},
}

def md(frame):
    cols=list(frame.columns)
    rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in row)+' |')
    return '\n'.join(rows)

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [c-h,c+h]

def main():
    result=json.loads((EV/'cleaning_results.json').read_text(encoding='utf8'))
    stats=pd.DataFrame(result['stats']); reviews=[]; purity=[]; manifest=[]
    for domain in build.DOMAINS:
        path=REPORTS/f'final_audit_{domain}_300.csv'
        sample=pd.read_csv(path,dtype=str).fillna('')
        content=sample[['id','source_row_index','question','option_a','option_b','option_c','option_d']].values.tolist()
        digest=hashlib.sha256(json.dumps(content,ensure_ascii=False).encode()).hexdigest()
        assert digest==FROZEN_CONTENT_SHA256[domain], 'Audit content changed: fresh question-by-question review required.'
        candidate=pd.read_parquet(build.OUT/f'{domain}_candidates.parquet')
        expected=candidate.sample(n=300,random_state=42).reset_index(drop=True)
        assert len(sample)==300 and sample.id.tolist()==expected.id.tolist()
        annotations=[]
        for _,r in sample.iterrows():
            idx=int(r.final_audit_index)
            judgment=EXCEPTIONS[domain].get(idx)
            if judgment is None:
                quote=' '.join(r.question.split())[:105]
                judgment=('knowledge_recall','yes',f'“{quote}”直接考查该领域的术语、定义或固定事实，未要求情境推导或计算。')
            category,selfcontained,reason=judgment
            annotations.append({'domain':domain,'source_row_index':int(r.source_row_index),'id':r.id,'final_audit_index':idx,
                'taxonomy':category,'self_contained':selfcontained,'short_reason':reason,
                'additional_quality_flag':FLAGS[domain].get(idx,''),
                'reviewer':'assistant_content_review_not_domain_expert','review_rule_version':result['version']})
        for col in ['taxonomy','self_contained','short_reason','additional_quality_flag','reviewer','review_rule_version']:
            sample[col]=[a[col] for a in annotations]
        sample.to_csv(path,index=False,encoding='utf-8-sig')
        reviews.extend(annotations)
        nk=sum(a['taxonomy']=='knowledge_recall' for a in annotations)
        ns=sum(a['self_contained']=='yes' for a in annotations)
        nj=sum(a['taxonomy']=='knowledge_recall' and a['self_contained']=='yes' for a in annotations)
        nf=sum(bool(a['additional_quality_flag']) for a in annotations)
        strict=sum(a['taxonomy']=='knowledge_recall' and a['self_contained']=='yes' and not a['additional_quality_flag'] for a in annotations)
        row={'domain':domain,'review_n':300,'knowledge_recall_count':nk,'self_contained_count':ns,
             'joint_knowledge_self_contained_count':nj,'additional_quality_flag_count':nf,'strict_proxy_count':strict,
             'estimated_knowledge_purity':nk/300,'estimated_self_contained_purity':ns/300,'estimated_joint_purity':nj/300,
             'estimated_strict_proxy_purity':strict/300}
        for t in TYPES:row[t]=sum(a['taxonomy']==t for a in annotations)
        for metric,count in [('knowledge',nk),('self_contained',ns),('joint',nj),('strict_proxy',strict)]:
            lo,hi=wilson(count,300);row[metric+'_wilson95_low']=lo;row[metric+'_wilson95_high']=hi
        purity.append(row)
        overlaps={}
        for name,p in [('previous_200',ROOT/f'analysis_samples/{domain}_200.csv'),('pilot_v1',EV/f'pilot_v1_{domain}_300.csv')]:
            if p.exists():
                old=pd.read_csv(p,dtype=str).fillna('')
                overlaps[name]=len(set(sample.id)&set(old.id)) if 'id' in old else None
        manifest.append({'domain':domain,'annotated_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'overlap_by_source_id':overlaps})
    (EV/'final_review_annotations.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame(reviews).to_csv(REPORTS/'final_audit_annotations.csv',index=False,encoding='utf-8-sig')
    purity=pd.DataFrame(purity)
    purity.to_csv(REPORTS/'final_audit_purity.csv',index=False,encoding='utf-8-sig')
    merged=stats.merge(purity,on='domain')
    merged.to_csv(REPORTS/'cleaning_stats.csv',index=False,encoding='utf-8-sig')

    allcand=pd.concat([pd.read_parquet(build.OUT/f'{d}_candidates.parquet') for d in build.DOMAINS],ignore_index=True).fillna('')
    assert allcand.cluster_id.is_unique and not allcand.question.map(build.norm).duplicated().any()
    finalnear,proposals=build.near_edges(allcand.question.tolist())
    assert not finalnear
    for snapshot in result['source_files']:
        assert hashlib.sha256((ROOT/snapshot['path']).read_bytes()).hexdigest()==snapshot['sha256']
    result.update({'purity_review_status':'COMPLETED_ASSISTANT_CONTENT_REVIEW_NOT_EXPERT_GOLD',
        'purity_review':purity.to_dict(orient='records'),'final_review_manifest':manifest,
        'final_candidate_normalized_stem_duplicates':0,'final_candidate_cross_domain_duplicates':0,
        'final_candidate_near_pairs_at_trigram_jaccard_085':len(finalnear),
        'federated_split_ready':False,'readiness_reason':'Residual non-recall, incomplete/options problems and special options in frozen final sample; no gold answer correctness validation.'})
    (EV/'cleaning_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')

    stages=merged[['domain','original','native_mcq','content_and_format_pass_before_dedup','after_normalized_qa_dedup','after_normalized_stem_dedup','after_cleaning','retention_rate']].copy()
    stages.retention_rate=stages.retention_rate.map(lambda x:f'{100*x:.2f}%')
    reasons=merged[['domain']+['removed_'+r for r in build.PRIORITY]]
    distribution=purity[['domain','review_n']+TYPES].copy()
    for t in TYPES:distribution[t]=distribution[t].map(lambda k:f'{k} ({k/3:.2f}%)')
    display=merged[['domain','original','after_cleaning','estimated_knowledge_purity','estimated_self_contained_purity','estimated_joint_purity','estimated_strict_proxy_purity']].copy()
    for c in display.columns[3:]:display[c]=display[c].map(lambda p:f'{100*p:.2f}%')
    intervals=[]
    for _,r in purity.iterrows():
        for key in ['knowledge','self_contained','joint','strict_proxy']:
            intervals.append({'domain':r.domain,'metric':key,'Wilson_95_interval':f'{100*r[key+"_wilson95_low"]:.2f}%–{100*r[key+"_wilson95_high"]:.2f}%'})
    lengths=[]
    for d,metrics in result['question_lengths'].items():
        for metric,v in metrics.items():lengths.append({'domain':d,'metric':metric,**{k:round(x,2) for k,x in v.items()}})
    difficulty=pd.DataFrame(result['difficulty']);difficulty['rate']=difficulty.rate.map(lambda p:f'{p*100:.2f}%')
    flagged=pd.DataFrame(reviews)
    flagged=flagged[(flagged.taxonomy!='knowledge_recall')|flagged.self_contained.eq('no')|flagged.additional_quality_flag.ne('')]
    flagged.to_csv(REPORTS/'final_audit_issues.csv',index=False,encoding='utf-8-sig')
    native=pd.crosstab(allcand.domain,allcand.subject_domain)
    native.reset_index().to_csv(REPORTS/'cleaning_retained_native_subjects.csv',index=False,encoding='utf-8-sig')
    rules=json.loads((EV/'cleaning_rules.json').read_text(encoding='utf8'))
    text='''# BhashaBench V1 knowledge-oriented MCQ 候选集清洗

## 结论与范围

仅处理原始英文 V1 的 Ayur、Legal、Krishi；Finance 未参与。输出是严格保守自动筛选的**候选集**，并非逐题专家认证的数据集。没有修改源文件，没有分 train/dev/test、分 client、下采样或训练。

**目前还不能判定已足够干净，建议暂缓正式 federated split。** 最终随机复核发现残留非 recall、残缺选项、特殊选项和答案歧义；高 taxonomy purity 不等于完整、唯一正确的 MCQ purity。保留这些复核发现，不用本次审计样本反向调规则或删除失败项，以避免把修正后的样本结果伪装为独立验证。

## 1. 数量与保留率

'''+md(stages)+'''

去重列表示在通过内容和格式规则的样本上依次进行 QA、stem、全局近重复簇约束。原始非 MCQ 与其他无效题也参加重复关系建图，但不作为可选代表。

## 2. 删除原因（互斥主原因）

'''+md(reasons)+'''

同一行可能命中多个规则，主原因依次按 non_mcq → corrupted → not_self_contained → non_english → special_option → calculation_aptitude → reasoning_application → off_domain → duplicate 记账。上表主原因总和加最终保留数等于原始数。removed_examples.csv 保存所有删除行、原始文字、cluster_id、主原因、全部原因和规则 ID；不是只摘取少数示例。完整日志在 cleaning_decisions.parquet。

**原因是筛选标记，不是逐题事实判定。** 尤其 off_domain 包含“题干没有命中正向领域词表”的保守排除，不能把其数量解释成真实 off-domain 数；长于 60 个 whitespace words 的题统一标为 reasoning_application，也不证明这些题确实需要推理。非英语有独立 non_english 原因。

## 3. 透明自动规则

不读取 subject_domain/topic 决定去留，亦不使用之前 800 条内容标签作为全量分类器。只在 question/options 的原始文字上执行规则。要求原生 question_type 精确为 MCQ、四个选项非空互异、答案标签 A–D；识别 OCR/控制字符、明显缺句、缺失关系、组合/匹配选项。匹配领域正向词表；排除英语、编程、时事词、计算、案例、因果推理、上下文材料包装及超过 60 words 的题。即使材料实际完整，材料包装仍被保守排除，不能称其必然缺材料。

词表是人工列出的可复现代理，存在误删和漏判。特别选项规则把任何 whole-word all/none/both/neither 都排除，普通合法量词也可能误删。原始 Sanskrit 罗马转写不因生僻自动判乱码。原始英语文件中的 Devanagari 单独记录；同 stem 不同正确答案文字的组全排除，可能误删同义措辞。

下面附运行的全部筛选函数及完整 regex/领域词表，避免隐藏内联规则：

```python
'''+inspect.getsource(build.assess)+'''
```

```json
'''+json.dumps(rules,ensure_ascii=False,indent=2)+'''
```

已知残余风险：Ayur 词表中的 must[a]? 会错误匹配普通英文 must；Legal 的 code/contract 等词可能匹配 aptitude；数字 1/2/3/4 既可能是合法计数答案也可能是遗失选项的占位，无法凭格式一刀切。这些问题在最终冻结复核中如实记录，尚未利用复核结果再调参。

## 4. 全局去重与跨域检查

Unicode 标点置空格、lowercase、折叠空白，保留数字；归一化仅用于比较，不改原字段。QA key 为 stem、四个原顺序选项、答案标签。stem 完全重复合簇；正确选项按 A–D 解析到文字，若同 stem 的文字目标不一致则保守排除整组。

近重复为 normalized word-trigram **集合 Jaccard ≥ 0.85** 的精确 prefix-filter join，不是 embedding、语义模型或只靠 MinHash。短于三词无法形成 trigram；改写、翻译、同义词、低于阈值的相似题未保证消除。数字不同也可能达阈值而合簇，优先保守去重。所有关系做 union-find 连通分量，同一传递簇仅保留一个通过规则的样本；按 domain|native id 的 SHA256 最小值选代表，不偏好难度或答案位置。cluster_id 基于簇源 ID，删除日志仍保留其余成员。

'''
    text+=f'全量确认近重复边 **{result["confirmed_near_pairs"]}**；全量跨域重复/近重复簇 **{result["cross_domain_clusters"]}**；同 stem 答案文字冲突组 **{result["conflicting_exact_stem_groups"]}**。最终合并三个候选文件重新检查：重复 cluster_id、normalized stem、跨域重复及 Jaccard≥0.85 近重复对均为 **0**。这不代表无语义重复。\n\n'
    text+='''## 5. 题长与原生难度

'''+md(pd.DataFrame(lengths))+'\n\n'+md(difficulty)+'''

难度仍为源标签，未重新评级。各域难度比例不同，后续不能只平衡条数而忽略难度与题长差异。retained_native_subjects 的独立表可检查保留题的原生 subject，不把该列当作筛选结果标签。

## 6. 冻结候选集的逐题内容复核

每域 seed=42 无放回抽 300，逐题读取 question 与四个 options，四选一 taxonomy；原生 subject/topic 未直接决定 taxonomy。self_contained 表示题意和必要材料/选项可理解且完整，不意味着答案 key 已通过专家认证。歧义、答案疑误、选项技巧另记 additional_quality_flag。统计代码只汇总已完成的内容判断，不将“规则通过”自动转成 recall。

这是同一助手的独立内容检查，**不是另一位盲审人员或领域专家复核**，也没有测评标注者一致性。此前部分 pilot 样本用于调整规则，本轮仍可能与 pilot/前次 200 条重叠；因此属于冻结规则后的内部质量估计，不是严格独立的留出验证。逐域重叠数和文件 hash 保存于 cleaning_results.json。泛医学知识视为 Ayur 来源的 medicine 范围，农业支撑生物学算 Krishi 范围；故不是“只含传统医学”或“各域无任何知识交叉”的保证。

'''+md(distribution)+'\n\n'+md(display)+'\n\n'+md(pd.DataFrame(intervals))+'''

joint 为 recall 且 self-contained；strict_proxy 还排除了额外质量 flag。Wilson 95% 区间只描述抽样不确定性，未覆盖标注错误、规则过拟合和医学/法律答案正确性。final_audit_issues.csv 列出全部复核异常；没有将异常样本事后删除以提高本轮 purity。

## 7. 规模建议与下一步

Ayur 是最小域，当前 **2,476** 条，Legal **4,075**，Krishi **4,339**。若当前仅比较候选规模，可把 **2,400 条/domain** 当作未来的临时上限建议；现在没有下采样，且不能保证补充内容/选项清洗后仍够 2,400。应先处理复核发现、冻结更新规则，再用未参与调规则的新样本验证，然后以更新后的最小域决定共同规模，不能提前承诺 8,000 或固定数量。

优先处理：Ayur 临床情境、Each of these/a+d 特殊选项、Sanskrit 编码残损；Legal 编码 aptitude、具体法律事实适用题、多人选项歧义及没有时间锚点的人名时事；Krishi 遗失为数字占位的选项、缺少疫苗/风向条件、遗传推断和石膏计算。专家还需检查唯一正确选项与 source answer，尤其法律时效和已变化的法条。

**最终判断：三域可继续作为主方案候选，但目前不应直接进入正式 federated split；仍需一轮针对残留问题的清洗及未参与规则调整的复核。**

## 8. 复现与文件

使用 fd 环境，先运行 scripts/build_bhashabench_candidates.py，再运行 scripts/final_bhashabench_content_review.py；后者重放固定的逐题审计判断并校验 seed=42 样本 ID。若改变清洗脚本或候选集，必须重新逐题审核，而不能把旧 index 标签复用。原始文件 SHA256 已在运行前后校验一致，见下表。源 split=test 是数据来源元信息，不是新创建的实验 split；后续用原始 evaluation 数据训练时必须声明为 derived benchmark，不能仍声称采用官方未见测试集。

'''+md(pd.DataFrame(result['source_files']))+'\n'
    (REPORTS/'cleaning_summary.md').write_text(text,encoding='utf8')
    print(display[['domain','original','after_cleaning','estimated_knowledge_purity','estimated_self_contained_purity']].to_string(index=False))
    print('READINESS: NO — residual content/option defects; no split or clients created.')
    paths=[*build.OUT.glob('*_candidates.*'),REPORTS/'cleaning_summary.md',REPORTS/'cleaning_stats.csv',REPORTS/'removed_examples.csv',*REPORTS.glob('final_audit_*_300.csv'),REPORTS/'final_audit_purity.csv',REPORTS/'final_audit_issues.csv',Path(__file__),ROOT/'scripts/build_bhashabench_candidates.py']
    for path in paths:
        assert path.exists() and path.stat().st_size>0
        print(f'OUTPUT {path} {path.stat().st_size:,} bytes')

if __name__=='__main__':main()
