"""Reproduce frozen held-out content judgments and their statistics.

All 900 question/options were read individually before creating this ledger.
Exceptions below are an authored annotation ledger, NOT an automatic classifier.
Unlisted rows were reviewed and judged recall, complete, without observed issues.
Does not edit frozen rules or candidate data; never tunes on validation findings.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import pandas as pd
from final_bhashabench_content_review import md, wilson

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'reports/bhashabench_evidence'
R=ROOT/'reports'
DOMAINS=['ayur','legal','krishi']
TAXONOMY={'K':'knowledge_recall','R':'reasoning_application','C':'calculation_aptitude','G':'general_offdomain'}
REVIEWED_CONTENT_SHA256={
 'ayur':'8e4233871ddd82b6ee06015f61c991ffb60c4b66dd3c2a0ba13da9dcabea84ad',
 'legal':'8ba3875b15b5494352be524292347f155b0963a29ff614043115ae5b0f82cdac',
 'krishi':'4c736d7f5c1e7e94098d9049816928d1248b2fd5e83fed7fcbb79921da7acd40',
}

# Explicit content-based classifications for reviewed non-recall questions.
NONRECALL={
 'ayur':{
 1:('R','需将孕妇的症状及血红蛋白检测结果应用到缺乏症判断。'),
 82:('R','需要把 Prakriti 这一变量映射到统计数据类型，属于概念应用。'),
 86:('R','需根据产妇及丈夫的 Rh 血型情境选择预防措施。'),
 104:('R','需要对学生类比长骨的具体认知例子识别 Praman 类型。'),
 257:('R','需要将学生借助血液与其他对象的比较识别为特定认知方法。'),
 288:('R','需要把冬季积累春季发病的具体因果例子归入 Hetu 类型。'),
 },
 'legal':{
 8:('R','四个选项给出不同协议情境，需要应用合同有效性规则。'),
 27:('R','需要对房东以转让取得房屋后要求驱逐的情境适用等待期。'),
 50:('R','需要将教师与校长的笔迹鉴定情境对应到证据法规定。'),
 58:('C','需要由 rule 的数字编码推导 evidence 的编码，属于 aptitude。'),
 109:('R','需要对雇主扣取公积金但未入账的情境识别罪名。'),
 111:('R','具体合同情境写在选项中，需要判断哪项可获特定履行。'),
 132:('R','需要比较教育法和童工条款的冲突并应用国际公约。'),
 186:('R','四个选项均是具体行为情境，需要适用犯罪未遂规则。'),
 197:('R','需要对当事人与调解员签署商业争议和解的情境判定法律效力。'),
 250:('G','询问目前法律委员会主席，主要是没有日期锚点的人事时事。'),
 267:('G','询问某退休法官目前任职，主要是没有日期锚点的人事时事。'),
 },
 'krishi':{
 14:('C','需要依据株行距计算种植数量，且缺少总面积这一条件。'),
 25:('G','询问定义货币的信用理论，属于一般金融经济学而非农业知识。'),
 30:('R','需要对农业增长未来受限的原因作预测性判断，时间与依据不明确。'),
 48:('C','需要按多个杂合位点计算 F2 基因型总数。'),
 58:('R','需要对给出的 DNA 序列应用转录配对规则。'),
 91:('G','询问关于金钱的名言作者，没有农业知识限定。'),
 140:('C','需要将给定摄氏温度换算为华氏温度。'),
 149:('R','需要把给定砂粉黏粒比例应用到 USDA 质地分类。'),
 150:('R','需要根据给出的前体 RNA 与成熟 mRNA 长度差推断加工机制。'),
 161:('R','需根据耕作和氮水平的实验安排选择统计试验设计。'),
 175:('G','比较食盐和水果的需求价格弹性，主要是通用微观经济学。'),
 217:('G','询问银行的联合贷款业务发布，没有明确农业对象。'),
 230:('R','需要根据研究者的温湿度实验方案选择相应饱和盐溶液。'),
 274:('C','需要用面积、流量、时间及水深计算灌溉效率。'),
 280:('G','询问镰刀型贫血健康项目，属于公共卫生而非农业领域。'),
 },
}

# Missing premises/material/essential wording/options. Minor interpretable typos
# stay self_contained=yes and can still receive a separate quality flag.
INCOMPLETE={
 'ayur':{
 160:'题干称从某 area 放血，但选项是容量，关键句结构与单位关系残缺。',
 205:'计数题的第四选项只有 T，无法确定原本的数字内容。',
 213:'没有说明何地何项政策强制增加 HCV 检测，只有四个具体日期。',
 235:'as per drug policy 没有给出地区或政策时间，无法唯一确定首选药。',
 295:'题干询问数量却把 Charaka 写在空白之后，四个选项全是作者。',
 296:'只有 Anuvasan basti is advised in 和长空白，选项序数缺少所问阶段或单位。',
 },
 'legal':{
 42:'biggest High Court 未说明按法官数、面积、辖区或其他指标比较。',
 115:'mens rea 与刑事责任题的数个选项语义损坏，无法稳定解释判断条件。',
 117:'句子结束于 he may within the period of，未给出这段期限内可以采取的行为。',
 168:'询问一般合同的法律承认，却没有电子合同这一必要限定，选项以 IT 法条为主。',
 179:'题干双重否定与 exception does not apply 连在一起，无法辨明要求的规则关系。',
 200:'只给条款数字并问 summons，未明确是哪一部法典。',
 241:'只称 SC/ST Amendments Act，未明确修法年份或适用时点。',
 263:'未限定警察羁押、司法羁押或相应条款，maximum detention 的含义不明确。',
 },
 'krishi':{
 14:'给出了株行距但未给土地面积，不能唯一计算种植数量。',
 36:'询问全国粮食种植面积，却没有统计年份。',
 60:'两个不相干问题片段合并进同一题干，源题结构损坏。',
 101:'农药禁用状态没有给出国家或适用时间。',
 111:'液氮温度题的所有摄氏温度选项均为正数，缺少必要负号。',
 142:'贷款额度题没有说明贷款计划、年份、机构或适用条件。',
 148:'Agri Export Zones 数量没有给出国家和统计时点。',
 201:'题干以 bacterial disease in 结束，缺少宿主，且选项含严重缺字。',
 214:'询问施肥消费领先的州，未说明总量/单位面积或统计年份。',
 223:'种子繁殖比被写成 1:02:00 等时间格式，原始比值无法可靠确定。',
 225:'农业 CRISPR 审批未限定编辑类型和政策时点，无法唯一选择程序。',
 245:'第一选项只有 deficiency，缺少所指营养元素。',
 251:'pregnancy 与 in milk 混合，所问计时基准不清且选项重复数字。',
 263:'细胞壁组成题的第四选项 cottatowe 无法辨认为完整物质名称。',
 266:'大陆保育农业面积排名没有给出统计年份，America 的地域口径也不清。',
 288:'题干把太阳光与粒子尺寸混在一起，所有纳米选项不能清楚对应所问概念。',
 },
}

# Observed or conservatively suspected quality problems, distinct from taxonomy.
QUALITY={
 'ayur':{
 4:('damaged_prefix','题干开头有竖线和残损字符，虽然容器用途可辨认，仍有结构噪声。'),
 25:('synonymous_options','Madhura-Tikta 与 Swaadu-Tikta 可能是同义选项，需要确认唯一答案。'),
 48:('overlapping_numeric_options','>2 weeks 与 >1 week 区间重叠，阈值表述可能导致不唯一。'),
 88:('OCR_option','一个剂量选项写成 l ml，存在字母 l/数字 1 的 OCR 混淆。'),
 117:('damaged_blank_marker','可由四个疾病选项理解需填主语，但句首缺少明确空缺标记，结构有残损。'),
 141:('missing_unit_word','由孕期和序数选项可推知月份，但题干仍缺少 month 这一单位词。'),
 144:('synonymous_options','Vata-Pittaj 与 Pitta-Vataj 仅词序不同，存在语义重复选项。'),
 171:('damaged_suffix','题干末尾 of M 多出无法解释的字母，虽然症状识别意图可辨认。'),
 194:('OCR_option','年龄选项 l year 含数字与字母的 OCR 混淆。'),
 224:('damaged_blank_marker','可由 Down syndrome 和疾病选项理解关联问题，但句首空缺标记与 is 重复有残损。'),
 242:('question_option_category_mismatch','疾病题的四个选项却全是治疗程序，语义完整可读但不能给出可靠单一答案。'),
 260:('temporal_scope_ambiguity','仅给出 MTP Act 1971 的名称，未限定原始条文还是后续修订时点。'),
 },
 'legal':{
 3:('nested_answer_options','legitimate 与 legitimate but having... 属于包含关系，需明确单一最佳答案准则。'),
 17:('potential_multiple_correct','Article 120 和 Article 121 两个配对都具有正确含义，需专家确认唯一答案。'),
 33:('semantic_duplicate','与本次 held-out 的 #262 是相同私人防卫规则题，仅措辞和标点略变。'),
 46:('potential_multiple_correct','吊扇与领取赡养费权利的动产判断可能都符合所问，需专家确认。'),
 53:('semantic_duplicate','与本次 held-out 的 #63 询问同一证人三分类判例，引用格式不同。'),
 60:('damaged_legal_term','Procreation of minor girl 疑将 procuration 误写，可能改变罪名含义。'),
 63:('semantic_duplicate','与本次 held-out 的 #53 为同一证人三分类判例题。'),
 75:('OCR_case_names','多个判例名明显拼写变形，包括 Hedley/Baxendal，需要核对原文。'),
 95:('missing_section_reference','U/S 后缺少实际条款号，保留了残缺法律引用。'),
 99:('potential_multiple_correct','War 与 External aggression 为分别可成立的情形，没有组合答案。'),
 106:('special_option','存在 To opt either (A) or (B) 这一组合选项，仍依赖其他选项。'),
 119:('nested_answer_options','fundamental right 和 human right 不互斥，单选解释可能不唯一。'),
 124:('damaged_act_year','题干称 Hindu Succession Act 1958，法律名称的年份疑误，须核对。'),
 125:('special_option','存在 Either (a) or (b) 及 Only (b) and not (a) 组合选项。'),
 130:('answer_options_need_expert_check','alimony 没有临时/永久限定，且未提供常见的永久赡养费条款选项。'),
 150:('damaged_section_range','Containing Sections 38A to 38 的范围缺失尾部字母。'),
 151:('potential_multiple_correct','普通程序与 Order 37 简易程序都表述为 can be filed，需确认单一答案。'),
 165:('nested_answer_options','express or implied 与单独 express/implied 选项有包含关系。'),
 180:('special_option','Either (a) or (b) 是组合选项，冻结规则未覆盖 or 变体。'),
 250:('unanchored_current_affairs','present chairman 没有时间锚点，答案会随人事变动。'),
 255:('duplicate_option_meaning','Thirty days 与 One month 的期限选项近义，缺少清楚区别。'),
 262:('semantic_duplicate','与本次 held-out 的 #33 是同一私人防卫规则题。'),
 267:('unanchored_current_affairs','presently 任职题未提供日期，无法固定人事状态。'),
 268:('answer_options_need_expert_check','family and locality 的组合措辞可能与实际见证要求不一致，需核对。'),
 271:('potential_multiple_correct','四个选项均像法条禁止情形，却要求选择 incorrect，需确认有效唯一答案。'),
 273:('ambiguous_criterion','judicial remedy 没有限定区分准则，多种 writ 均为司法救济。'),
 295:('potential_multiple_correct','议会职务和高院法官可能同时符合出庭豁免，需确认唯一答案。'),
 },
 'krishi':{
 0:('answer_hints_in_options','多个干扰项附其真实害虫名称，而正确候选较短，构成额外答题提示。'),
 7:('answer_hints_in_options','选项直接标注 Complete/Incomplete metamorphosis，并可能有错误注释。'),
 11:('overlapping_numeric_options','20–70 与其他粒重范围重叠，可能不能给出唯一范围选择。'),
 16:('potential_multiple_correct','多个叶片功能组合均可成立，题目没有明确区分依据。'),
 30:('unanchored_forecast','very soon likely 的预测没有时间、来源或可验证依据。'),
 56:('semantic_duplicate','与本次 held-out 的 #155 是相同除草剂施用方法题，另一题将选项写入题干。'),
 59:('potential_multiple_correct','四项均是经典光合研究实验，题干没有选出唯一项的限定。'),
 66:('answer_hints_in_options','干扰项附不同口器类型的解释，提供超出统一选项的答题提示。'),
 80:('answer_hints_in_options','一个干扰项附其传播的病毒类别，影响选项格式一致性并泄露提示。'),
 89:('answer_hints_in_options','多个干扰项附已对应的寄生蜂例子，存在答案解释型提示。'),
 90:('potential_multiple_correct','浅根、水平叶及高需水量可能都不符合旱地理想型，需确认唯一答案。'),
 102:('question_option_category_mismatch','题干称 enzyme，四个选项实际是植物激素，语义可读但存在术语错配。'),
 108:('species_scope_ambiguity','Striga 没有限定物种，不同物种寄生不同选项作物，需专家确认。'),
 128:('semantic_duplicate','与本次 held-out 的 #226 为同一 triose phosphate 题，后者把选项写入题干。'),
 143:('OCR_missing_subscript','C plants 丢失 C4 的数字或下标，虽然前文四碳信息可帮助理解。'),
 155:('semantic_duplicate','与本次 held-out 的 #56 是同一除草剂施用方法题。'),
 194:('OCR_option','185 疑为 18S 的 OCR 错读，需要核对原始核糖体选项。'),
 215:('answer_hints_in_options','多个干扰项明确标注其口器类型，而正确项没有注释。'),
 226:('semantic_duplicate','与本次 held-out 的 #128 为同一 triose phosphate 题。'),
 231:('synonymous_options','critical velocity 与 threshold velocity 等词可能同义，需明确所用术语体系。'),
 233:('potential_multiple_correct','多种候选鱼可进行空气呼吸，题目没有进一步限定，需核对。'),
 265:('potential_multiple_correct','蚜虫的孤雌生殖与胎生均可能符合所问，需确认唯一正确选项。'),
 283:('damaged_term','题干问 inhibitors 却给出 Trypsin，可能省略 inhibitor 这一关键术语。'),
 290:('answer_hints_in_options','多个站址选项附其他研究机构的解释，提供额外排除提示。'),
 294:('answer_options_need_expert_check','题干问种皮来源，但没有 integument 选项，Testa 是种皮名称，需核对题意与 key。'),
 },
}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    runpath=EV/'targeted_cleanup_run.json'
    run=json.loads(runpath.read_text(encoding='utf8'))
    frozenpath=EV/'targeted_frozen_rules.json'
    frozen=json.loads(frozenpath.read_text(encoding='utf8'))
    assert run['frozen_rules_sha256']==sha(frozenpath)
    assert frozen['targeted_script_sha256']==sha(ROOT/'scripts/targeted_bhashabench_cleanup.py')
    previous_run=json.loads((EV/'cleaning_results.json').read_text(encoding='utf8'))
    for source in previous_run['source_files']:
        assert sha(ROOT/source['path'])==source['sha256'], 'Original raw data changed.'
    all_annotations=[];stats=[];issues=[];checks=[]
    counts={r['domain']:r for r in run['counts']}
    for d in DOMAINS:
        manifest=next(r for r in run['heldout_manifest'] if r['domain']==d)
        assert manifest['sample_content_sha256']==REVIEWED_CONTENT_SHA256[d], 'Different sample: fresh individual review required.'
        assert sha(ROOT/f'data/bhashabench_final/{d}.parquet')==manifest['candidate_sha256']
        path=R/f'heldout_validation_{d}_300.csv'
        sample=pd.read_csv(path,dtype=str).fillna('')
        content=sample[['id','source_row_index','question','option_a','option_b','option_c','option_d']].values.tolist()
        assert hashlib.sha256(json.dumps(content,ensure_ascii=False).encode()).hexdigest()==manifest['sample_content_sha256']
        previous=set()
        for record in frozen['historical_audits']:
            p=ROOT/record['path'];assert sha(p)==record['sha256'],f'Historical audit changed: {p}'
            frame=pd.read_csv(p,dtype=str).fillna('')
            if record['domain']==d:previous.update(frame.id)
            elif record['domain']=='all':previous.update(frame.loc[frame.domain.eq(d),'id'])
        assert not set(sample.id)&previous
        candidates=pd.read_parquet(ROOT/f'data/bhashabench_final/{d}.parquet')
        redrawn=candidates[~candidates.id.isin(previous)].sample(n=300,random_state=2026)
        assert sample.id.tolist()==redrawn.id.tolist(), 'Held-out sample does not match seed=2026.'
        annotations=[]
        for _,row in sample.iterrows():
            i=int(row.heldout_audit_index)
            code,reason=NONRECALL[d].get(i,('K',None))
            if reason is None:
                quote=' '.join(row.question.split())[:90]
                reason=f'“{quote}”主要询问该领域的固定事实、术语或定义，未见明显计算或情境推导。'
            selfcontained='no' if i in INCOMPLETE[d] else 'yes'
            quality='yes' if i in QUALITY[d] or i in INCOMPLETE[d] else 'no'
            kind,qreason=QUALITY[d].get(i,('incomplete_or_damaged_structure','')) if quality=='yes' else ('','')
            if i in INCOMPLETE[d]:
                reason=INCOMPLETE[d][i];kind='incomplete_or_damaged_structure'
            elif qreason:reason=qreason
            if code!='K' and (i in QUALITY[d] or i in INCOMPLETE[d]):reason=NONRECALL[d][i][1]+' '+reason
            annotations.append({'domain':d,'id':row.id,'source_row_index':int(row.source_row_index),
                'heldout_audit_index':i,'taxonomy':TAXONOMY[code],'self_contained':selfcontained,
                'quality_issue':quality,'quality_issue_type':kind,'short_reason':reason,
                'reviewer':'assistant_individual_content_review_not_expert_gold',
                'review_protocol':'frozen_v5_new_seed_2026_prior_ids_excluded_no_retuning'})
        for col in ['taxonomy','self_contained','quality_issue','quality_issue_type','short_reason','reviewer','review_protocol']:
            sample[col]=[a[col] for a in annotations]
        sample.to_csv(path,index=False,encoding='utf-8-sig')
        all_annotations.extend(annotations)
        issues.extend(a for a in annotations if a['taxonomy']!='knowledge_recall' or a['self_contained']=='no' or a['quality_issue']=='yes')
        k=sum(a['taxonomy']=='knowledge_recall' for a in annotations)
        j=sum(a['taxonomy']=='knowledge_recall' and a['self_contained']=='yes' for a in annotations)
        s=sum(a['taxonomy']=='knowledge_recall' and a['self_contained']=='yes' and a['quality_issue']=='no' for a in annotations)
        n=300;row={'domain':d,**counts[d],'review_n':n,'knowledge_count':k,'joint_count':j,'strict_count':s,
            'self_contained_no_count':sum(a['self_contained']=='no' for a in annotations),
            'quality_issue_count':sum(a['quality_issue']=='yes' for a in annotations)}
        for name,count in [('knowledge',k),('joint',j),('strict',s)]:
            lo,hi=wilson(count,n)
            row[name+'_purity']=count/n;row[name+'_wilson95_low']=lo;row[name+'_wilson95_high']=hi
        for t in TAXONOMY.values():row[t+'_count']=sum(a['taxonomy']==t for a in annotations)
        row['knowledge_point_pass']=k/n>=.95
        row['joint_point_pass']=j/n>=.95
        row['strict_point_pass']=s/n>=.95
        # Observed recurring clinical/coding/option-wrapper/OCR/calculation defects.
        row['no_systematic_leakage_pass']=False
        row['overall_pass']=False
        stats.append(row)
        checks.append({'domain':d,'sample_n':300,'unique_ids':len(set(sample.id)),
            'previous_id_overlap':0,'sample_content_unchanged':True,'candidate_sha256_unchanged':True})
    pd.DataFrame(all_annotations).to_csv(R/'heldout_validation_annotations.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(issues).to_csv(R/'heldout_validation_issues.csv',index=False,encoding='utf-8-sig')
    stats=pd.DataFrame(stats)
    stats.to_csv(R/'heldout_validation_statistics.csv',index=False,encoding='utf-8-sig')
    (EV/'heldout_explicit_review_ledger.json').write_text(json.dumps(all_annotations,ensure_ascii=False,indent=2),encoding='utf8')
    run.update({'heldout_review_status':'COMPLETED_NO_RETUNING_FAILED_READINESS',
        'review_completed_at_utc':datetime.now(timezone.utc).isoformat(),'validation_statistics':stats.to_dict(orient='records'),
        'validation_checks':checks,'federated_split_ready':False,
        'rules_and_candidates_unchanged_after_validation':True})
    runpath.write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf8')
    assert sha(frozenpath)==run['frozen_rules_sha256']
    table=stats[['domain','final_count','knowledge_purity','joint_purity','strict_purity']].copy()
    for name in ['knowledge','joint','strict']:
        table[name+'_purity']=[f'{r[name+"_purity"]*100:.2f}% [{r[name+"_wilson95_low"]*100:.2f}, {r[name+"_wilson95_high"]*100:.2f}]' for _,r in stats.iterrows()]
    tax=stats[['domain','knowledge_recall_count','reasoning_application_count','calculation_aptitude_count','general_offdomain_count','self_contained_no_count','quality_issue_count']]
    removed=pd.read_csv(R/'targeted_removed_examples.csv').fillna('')
    reasons=pd.crosstab(removed.domain,removed.targeted_removal_reason).reset_index()
    rule_hits=[]
    for _,r in removed.iterrows():
        for rule in json.loads(r.targeted_rule_ids):rule_hits.append({'domain':r.domain,'rule_id':rule})
    hitframe=pd.DataFrame(rule_hits)
    hits=pd.crosstab(hitframe.rule_id,hitframe.domain).reset_index()
    hits.to_csv(R/'targeted_rule_hit_counts.csv',index=False,encoding='utf-8-sig')
    report='''# BhashaBench 三域 targeted cleanup 与新 held-out 验证

## 最终判断

**三域本轮均未通过完整准入要求；目前不能进入正式 federated split，也不推荐冻结为训练/评测版本。** 数据和规则已作为可追溯的冻结候选版本保留。knowledge 比例达标不等于 joint 或 strict 达标；本轮还观察到重复发生的结构/OCR、组合选项、案例/计算与域外漏判。

没有使用新验证题修改清洗规则，也没有删除验证失败题来重算 purity。未改原始文件、未划 train/dev/test、未分 client、未训练或下采样。

## 1. 最终规模与 targeted delta

'''+md(pd.DataFrame(run['counts']))+'\n\n'+md(reasons)+'''

52 条前轮已确认的问题样本逐条按 domain + native id 隔离，并记录原复核理由；这是明确可追溯的修正，不把人工结果推断到整个 subject 类别。另对相同残留形式增加狭窄规则：Each of these、A+D、literal option union；Ayur year-old/patient 症状与 must/musta 拼写匹配；Legal 少数编码表达、未锚定职位人物、具名事件/具体程序叙述；Krishi 无计数询问的裸 1/2/3/4、混合文字的单数字占位、F1/F2 推断、CEC/ESP 石膏计算、未指定疫苗。没有重新设计 v4 正向词表或按 subject 批量删除。

各规则命中数可重叠：

'''+md(hits)+'''

完整 delta 正则、内联逻辑、旧规则 SHA256、问题清单 SHA256、冻结时间、全部历史样本清单均在 targeted_frozen_rules.json。旧规则及旧候选集仍保留，targeted_removed_examples.csv 保存全部 145 条本轮删除行及理由。规则按最终规则冻结后才抽新样本。

## 2. 真正按样本 ID held-out 的新验证

每域 seed=2026 无放回抽取 300；排除原 analysis_samples/*_200.csv、pilot_v1_*_300.csv、final_audit_*_300.csv 和合并 manual_taxonomy_audit.csv 中全部已审计 ID。历史 ID 清单及文件 hash 已校验；每域 overlap=0，且审计中候选 Parquet 与规则文件 hash 均未变。

'''+md(pd.DataFrame(checks))+'''

这里的 held-out 是**相对之前规则开发样本 ID 的独立验证**，不是未来实验 test split。全量清洗与重复检查当然处理过这些原始行，但先前未用于人工调规则。相似问题不同 ID 可能仍共享语义，不声称与开发集不存在任何语义关系。

逐题实际阅读 question/options，未依据 subject/topic、源 answer 标签或自动规则判 taxonomy。审计为助手内容判断，不是另一位盲审标注者或领域专家认证；没有测量标注者一致性。依沿用 taxonomy，医学基础知识计入 Ayur，农业支撑生物学与明确农场经济知识计入 Krishi；泛金融理论、金钱名言、公共卫生等未计作农业。

## 3. 最终纯度与 Wilson 95% CI

knowledge=recall / 300；joint=(recall 且 self_contained=yes) / 300；strict=(recall 且 self_contained=yes 且 quality_issue=no) / 300。**所有分母均为 300**，不是对 recall 子集条件化。区间使用双侧 Wilson 95%，不把区间下限达到 95% 当作用户没有要求的额外硬门槛。

'''+md(table)+'''

数值后方方括号为 Wilson 95% CI 的百分比端点。区间描述有限抽样不确定性，不覆盖助手误判、知识正确性或数据源偏差。strict 是保守“未观察到问题”的估计：疑似多个正确答案也记 quality_issue=yes，故不能把该数字当作专家确认的错误率。

'''+md(tax)+'\n\n'+md(stats[['domain','knowledge_point_pass','joint_point_pass','strict_point_pass','no_systematic_leakage_pass','overall_pass']])+'''

## 4. 新验证发现（冻结后只记录，不反向调参）

**Ayur：** #1 孕妇血红蛋白情境、#86 Rh 血型情境及 #104/#257 认知例子仍是 application；#205 的计数选项 T、#242 疾病题对应治疗程序、#295 数量题对应作者、#296 空白缺少阶段/单位仍残缺；#25 与 #144 有同义选项，#213/#235 的政策时间/地区缺少锚点。临床故事与结构损坏不只是一道孤例。

**Legal：** #58 是另一种数字编码 aptitude；#111/#186 的案例放在选项中，题干规则无法屏蔽；#106/#125/#180 的 either (a) or (b) 组合式选项未覆盖；#250/#267 的人事时事使用 chairman/presently 表述而非旧模式；#117 期限题没有给出所问法律行为，#200 summons 没有指定法典。多个条件、范围和选项歧义需进一步核验。

**Krishi：** #14 株行距、#48 F2 基因型计数、#140 温度转换、#274 灌溉效率仍是 calculation；#58/#149/#150/#161/#230 需要规则应用或实验推断；#25/#91/#217/#280 是泛金融/名言/银行/公共卫生；#102 酶与激素不匹配、#111 液氮温度缺负号、#223 比值转为时间格式、#245 营养元素缺失、#288 太阳光与粒径混杂。多个 options 直接附带干扰项的知识解释，提供不统一答题提示。

上述 index 为新 held-out CSV 的 heldout_audit_index（从 0 开始），每题含 source_row_index 与原生 id，可定位源数据。详细异常全部在 heldout_validation_issues.csv，而非仅列典型例子。

**语义重复残留：** 复核发现 Legal #33/#262、#53/#63 及 Krishi #56/#155、#128/#226 是同一知识题的重排/选项内嵌版本，已标为 quality_issue。全局 normalized QA、stem、cluster 和 word-trigram Jaccard≥0.85 near 检查仍为 0；这一数值只保证该定义阈值，不保证所有语义近重复都消失。不能把自动去重通过解释为“无近重复”。本轮不再为这些发现改阈值。

## 5. 准入和共同规模

A. 最终 Ayur **2,451**、Legal **4,030**、Krishi **4,264**。

B. 新的按历史 ID 排除的 held-out 审计确实完成，但**三个领域均未通过完整准入条件**。

C. **不能进入正式 federated split。** 达到 knowledge 点估计 95% 仍不足以抵消 joint/strict 不达标及系统性漏判。当前冻结版本用于记录本轮失败与人工复查，而不是声明训练集已达标。

D. 目前**不推荐共同实验规模**。最小域 2,451，因此 2,400 仅是当前条数上的容量参考，不是纯度合格后的承诺。不能用平均 purity 乘全量数量当作已经识别出来的可训练题数。未来若另行开展清洗，应把本次 900 题加入开发样本排除清单，再使用另一批未参与规则开发的样本验证，最后以合格数据的最小域决定规模；本任务没有启动下一轮调规则。

## 6. 复现与冻结证据

先执行 scripts/targeted_bhashabench_cleanup.py，随后执行 scripts/review_bhashabench_heldout.py。后者重放已实际完成的内容判断，并严格校验新样本内容 hash、历史文件 hash、候选文件 hash 与规则脚本 hash；样本改变必须重新审计，不能复用旧行号。fd Python 环境，种子 2026。

taxonomy 与质量检查相互独立：self_contained=no 仅用于必要信息、题意或选项不可完整解释；仍可读的 OCR、疑似答案歧义、时间口径、提示与语义重复另列 quality_issue。没有系统核对全量 answer key 正确性；候选集仍保留源 test 元信息，后续若转作训练须声明为 derived benchmark。

```json
'''+json.dumps({k:v for k,v in frozen.items() if k!='historical_audits'},ensure_ascii=False,indent=2)+'''
```
'''
    (R/'final_cleanup_summary.md').write_text(report,encoding='utf8')
    print(table.to_string(index=False))
    print('Held-out: 300/domain; all prior-audited ID overlaps=0; no post-validation rule tuning.')
    print('A: 2451 / 4030 / 4264. B: all FAILED full readiness. C: NO federated split. D: no approved common size.')
    for path in [R/'final_cleanup_summary.md',*[R/f'heldout_validation_{d}_300.csv' for d in DOMAINS],*[ROOT/f'data/bhashabench_final/{d}.parquet' for d in DOMAINS],R/'heldout_validation_statistics.csv',R/'heldout_validation_issues.csv']:
        assert path.exists() and path.stat().st_size>0
        print(f'OUTPUT {path} {path.stat().st_size:,} bytes')

if __name__=='__main__':main()
