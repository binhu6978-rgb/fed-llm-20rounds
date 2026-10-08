"""Rebuild the 800-row content audit from explicit per-question annotations.

This script joins and summarizes annotations; it never infers labels from native
subject_domain, classifies the full dataset, filters it, or constructs splits.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import unicodedata
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'reports/bhashabench_evidence'
DOMAINS=['ayur','legal','finance','krishi']
LABELS={'K':'knowledge_recall','R':'reasoning_application','C':'calculation_aptitude','G':'general_offdomain'}

def table(headers,rows):
    def esc(v):return str(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(esc(v) for v in row)+' |' for row in rows])

def wilson(k,n):
    z=1.959963984540054
    p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half

def norm(s):
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s.lower()).split())

def main():
    frames=[];provenance=[];summaries=[];estimates=[];duplicate_rows=[]
    for domain in DOMAINS:
        source=ROOT/f'analysis_samples/{domain}_200.csv'
        before=hashlib.sha256(source.read_bytes()).hexdigest()
        sample=pd.read_csv(source,keep_default_na=False)
        annotation=EVIDENCE/f'{domain}_content_labels.tsv'
        labels=pd.read_csv(annotation,sep='\t',header=None,names=['sample_index','code','self_contained','short_reason'],keep_default_na=False)
        assert len(sample)==200 and len(labels)==200
        assert labels.sample_index.tolist()==list(range(200)),domain
        assert labels.code.isin(LABELS).all() and labels.self_contained.isin(['yes','no']).all()
        assert labels.short_reason.str.strip().ne('').all()
        rawpath=ROOT/f'data/bhashabench/{domain}/test.parquet'
        full=pd.read_parquet(rawpath)
        manifest=json.loads((EVIDENCE/'download_manifest.json').read_text(encoding='utf-8'))
        original=next(x for x in manifest if x['path']==rawpath.relative_to(ROOT).as_posix())
        assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==original['sha256']
        expected=full.sample(n=200,random_state=42)
        assert sample.audit_source_row_index.tolist()==expected.index.tolist()
        for c in full.columns:assert sample[c].tolist()==expected[c].tolist(),(domain,c)
        annotated=sample.copy()
        annotated.insert(0,'domain',domain)
        annotated.insert(1,'sample_index',labels.sample_index)
        annotated['taxonomy']=labels.code.map(LABELS)
        annotated['self_contained']=labels.self_contained
        annotated['short_reason']=labels.short_reason
        annotated['review_method']='assistant_content_review_question_and_options'
        assert not annotated.duplicated(['domain','audit_source_row_index']).any()
        frames.append(annotated)
        counts=annotated.taxonomy.value_counts()
        summaries.append([domain,200,*[f'{counts.get(v,0)} ({counts.get(v,0)/200:.1%})' for v in LABELS.values()],int((annotated.self_contained=='no').sum())])
        keep=(annotated.taxonomy=='knowledge_recall')&(annotated.self_contained=='yes')
        k=int(keep.sum());N=len(full);lo,hi=wilson(k,200)
        estimates.append([domain,N,k,f'{k/200:.1%}',round(N*k/200),f'{round(N*lo):,}–{round(N*hi):,}'])
        grouped={}
        for i,q in enumerate(sample.question):grouped.setdefault(norm(q),[]).append(i)
        for q,indices in grouped.items():
            if len(indices)>1:duplicate_rows.append([domain,', '.join(map(str,indices)),q])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==before
        provenance.append({'domain':domain,'sample_csv':source.relative_to(ROOT).as_posix(),'sample_sha256':before,
             'annotation_tsv':annotation.relative_to(ROOT).as_posix(),'annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),
             'population_parquet_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest(),'sample_rows':200,'population_rows':N})
    # CSV has one union schema; a source-absent column (Krishi topic) stays blank.
    audit=pd.concat(frames,ignore_index=True).fillna('')
    assert len(audit)==800
    dest=ROOT/'reports/manual_taxonomy_audit.csv'
    audit.to_csv(dest,index=False,encoding='utf-8-sig')
    reread=pd.read_csv(dest,keep_default_na=False)
    pd.testing.assert_frame_equal(audit,reread,check_dtype=False)
    finance=audit[audit.domain=='finance'].copy()
    excluded=finance.subject_domain.isin(['Problem Solving','Mathematics for Finance'])
    remainder=finance[~excluded]
    finfull=pd.read_parquet(ROOT/'data/bhashabench/finance/test.parquet')
    full_remainder=finfull[~finfull.subject_domain.isin(['Problem Solving','Mathematics for Finance'])]
    fkeep=(remainder.taxonomy=='knowledge_recall')&(remainder.self_contained=='yes')
    fk=int(fkeep.sum());fn=len(remainder);fl,fh=wilson(fk,fn)
    finance_rows=[]
    for name,subset in [('全部Finance样本',finance),('两个原生标签以内（仅比较）',finance[excluded]),('两个原生标签以外（仅比较）',remainder)]:
        n=len(subset);c=subset.taxonomy.value_counts()
        finance_rows.append([name,n,*[f'{c.get(v,0)} ({c.get(v,0)/n:.2%})' for v in LABELS.values()],
                             int(((subset.taxonomy=='knowledge_recall')&(subset.self_contained=='yes')).sum())])
    subject_rows=[]
    for subject,g in remainder.groupby('subject_domain',sort=True):
        c=g.taxonomy.value_counts()
        subject_rows.append([subject,len(g),*[int(c.get(v,0)) for v in LABELS.values()],
            int(((g.taxonomy=='knowledge_recall')&(g.self_contained=='yes')).sum())])
    krishi=audit[audit.domain=='krishi']
    kg=krishi[krishi.subject_domain=='General Knowledge & Reasoning']
    kgc=kg.taxonomy.value_counts()
    rules=[
      '# BhashaBench 800题内容审计',
      '## 1. 审读方法与分类边界',
      '逐题读取四个 seed=42 样本 CSV 的完整 question 和 A/B/C/D options，共800题。标签是本助手的逐题内容判断，**不是人类专家标注、双人一致性审核或验证过的金标准**。代码仅连接显式逐题标签、核验与汇总，不用关键词模型或 subject_domain 自动分类。原生 subject_domain 仅在完成内容标签后用于 Finance 子组比较及 Krishi 标签诊断。题目正确答案未用于确定任务类别；本次不系统核验所有答案的事实正确性。',
      table(['标签','操作口径'],[
        ['knowledge_recall','直接记忆专业术语、配方、法条、判例对应、植物动物基础、制度事实或标准流程；数字只是记忆值、配对或年代排序不自动算计算。'],
        ['reasoning_application','需要将领域概念或规则用于案例、解释机制关系、判别断言与理由的因果关系。'],
        ['calculation_aptitude','数量计算、数列、符号不等式、编码、座位/楼层/亲属关系谜题或通用论证评价；即使借用金钱或法律故事也属于aptitude。'],
        ['general_offdomain','英语词汇语法、一般阅读理解、无领域联系的通用常识/时事、通用计算机操作及明显其他领域内容。'],
        ['self_contained=yes','题干及选项在当前行足够理解并完成该单选任务；允许需要外部的领域知识，无需另取缺失图表/材料，也无明显妨碍唯一选择的文本损坏。'],
        ['self_contained=no','引用材料/图表缺失、关键约束缺损、选项缺字或重复正确选项、明显不存在可行选项或问句与输出任务不匹配。']]),
      '基础知识边界：Ayur 按传统医学及相关基础医学/临床/公共卫生内容审读；Krishi 包含植物、土壤、农业生物技术、昆虫与畜牧相关基础知识；Finance 包含银行、支付、金融产品、经济/贸易政策及相关制度事实；Legal 包含法学、司法机构、宪法及法律历史。泛化的统计、管理、计算机、英语、人物生日或无农业联系的银行事实不因原生标签就算该领域知识。尤其把新闻式任命、当年目标/报告封面、近期产业签约和年度纪念主题归为offdomain，即使新闻带有金融或农业词汇；制度术语、金融产品规则与法制历史仍可为recall。微贷款/机会成本等在 Krishi 中无农业情境的样本也按 offdomain 计；这是一种偏严格边界，领域专家复核可能调整。',
      '优先区分通用形式化 aptitude 和领域规则应用；再判断内容是否在来源领域内。多项真伪、匹配、流程顺序题如果只是熟悉事实的组合，仍可标 knowledge_recall。例如 Krishi样本95是减数分裂定义的重述，而样本108须判断复制方向与滞后链的解释关系；原生题型本身不决定标签。self_contained 也不是 closed-book 的同义词：内嵌材料完整可为 yes，但材料抽取题仍需另行检查是否符合后续闭卷设计。',
      '## 2. 每领域四类数量与比例',
      table(['领域','样本数',*LABELS.values(),'self_contained=no'],summaries),
      '所有比例的分母均为各域200条，不删除损坏题再重新计算比例；每题仅有一个四分类标签。损坏题依据其可辨认的任务意图分类，同时标no。',
      '## 3. Finance：排除两个原生学科标签后的实际内容',
      table(['比较组','样本数',*LABELS.values(),'recall且self-contained=yes'],finance_rows),
      f'全量Finance共{len(finfull):,}题；原生 Problem Solving / Mathematics for Finance 以外共{len(full_remainder):,}题。样本中该标签以外有{fn}题，其中knowledge_recall为{sum(remainder.taxonomy=="knowledge_recall")}题；完整可用的知识记忆题为{fk}题（{fk/fn:.2%}）。**单纯排除这两个标签后，剩余样本仍然不是主要的Finance knowledge。**剩余题目中的计算/aptitude、英语/其他领域内容占据多数。',
      table(['Finance剩余原生subject','样本n','recall','application','calculation/aptitude','offdomain','recall且yes'],subject_rows),
      '以下样本支持该判断（sample_index是CSV中从0开始的行次序）：Finance0 原生Banking Services实为缺图表的市值计算；Finance8 原生Corporate Finance & Investment实为船速题；Finance128 原生Banking Services实为生活开支比例运算；Finance161 原生Behavioral Finance实为恐惧尖叫的生物心理阅读。反例中的真正知识题包括Finance16破产流程缩写、Finance169非银行ATM类型、Finance199黄金债券条款。分类依据的是问句与选项，原生标签仅帮助定位错误映射。',
      f'Finance仍存在可构建独立金融知识子集的证据，但**不能直接把当前全量集合或上述标签排除后的集合当成纯金融知识域**。{sum(finance.taxonomy=="knowledge_recall")}条recall样本还含机构人物和项目年份等；若未来只要求金融概念机制，需要更严格的边界审读，保留量可能进一步减少。不能用当前少量每subject样本对所有小subject作可靠排名。',
      '## 4. Legal、Krishi、Ayur 的内容差异',
      'Legal知识题主要是法条编号、时效、程序权限、判例与命题的直接对应；Legal29代理保管珠宝、Legal109私密录像传播、Legal140同意杀人以及Legal191卖房撤销属于案例/规则应用。Legal4心理治疗、Legal43文学材料、Legal64座位谜题、Legal153石膏化学都不适合当法律知识记忆题。',
      f'Krishi原生General Knowledge & Reasoning的样本共有{len(kg)}题，内容分类为'+', '.join(f'{name}={kgc.get(name,0)}' for name in LABELS.values())+'。其中Krishi114推广教育哲学与Krishi182农业推广项目历史仍属农业知识，说明不能把该原生标签整类自动当作offdomain；Krishi166银行总部和Krishi194部长时事则明显偏离农业知识。',
      'Ayur主要为术语、经典出处、配方、解剖与医学事实直接记忆；Ayur100中位数是计算，Ayur127前置胎盘分级是应用。Ayur46/114是规范化题干重复，审计仍分别保留。Ayur102缺少分类对象、Ayur178仅剩In PCOS，结构审计中的非空与有效A/B/C/D不能发现这些内容缺损。',
      '## 5. 仅保留 knowledge_recall + self_contained=yes 的全量保留估计',
      table(['领域','现有English test全量N','样本符合条件k/200','样本保留率','估计保留N×k/200','95% Wilson抽样区间换算题数'],estimates),
      '估计公式为N×(样本中recall且yes的数量/200)，结果四舍五入。样本已核验为全量原始行的固定种子42简单随机抽样。Wilson区间仅表示单次200条抽样的比例不确定性（未作有限总体校正，差异很小），不涵盖助手分类误差、领域边界分歧、事实错误或重复相关性；不应当作最终可用数据规模保证。估计按行计数，未去重；同一道题的重复版本也计入预计保留量。',
      f'Finance若未来先讨论排除两个原生标签后的{len(full_remainder):,}题，再按该子组样本的{fk}/{fn}估计，条件估计为约{round(len(full_remainder)*fk/fn):,}题，Wilson换算约{round(len(full_remainder)*fl):,}–{round(len(full_remainder)*fh):,}题。它与全量直接估计存在抽样组成差异，是另一估计口径，不能相加。这里仅计算统计量，没有导出筛选后的全量数据。',
      '## 6. 是否适合统一 knowledge-oriented MCQ benchmark',
      '四域可以统一为题干+四选项→选项字母的接口，但内容审计不支持直接使用全量数据进行严格的纯知识比较。Ayur、Legal、Krishi的recall比例较高，可以作为进一步核验的候选；Finance需要大量内容筛查，仅凭原生学科标签不足。若严格保留recall且yes，四域仍有规模可行的初步信号，但Finance明显成为规模瓶颈，领域平衡、去重、独立评估以及真实专业知识边界都需要下一阶段再设计。',
      '跨领域同任务仍需注意：Legal记忆大量地区性法条与判例，Ayur含传统医学经典命题，Krishi含农业基础生物学，Finance残留可能更偏金融机构与时事记忆；即使都属于recall，也不意味着难度、时效性、知识粒度完全相同。',
      '## 7. 内容缺损、重复及下一步复核',
      table(['domain','sample_index（0起）','标签','why self-contained=no'],[[r.domain,r.sample_index,r.taxonomy,r.short_reason] for r in audit[audit.self_contained=='no'].itertuples()]),
      table(['domain','样本规范化重复行次序','题干'],duplicate_rows),
      'self_contained包含最小的题目可用性判断，并非完整答案正确性检查。少数no基于显著算式不匹配或选项损坏；未穷尽求解全部复杂谜题，yes不能保证答案正确。应由领域专家复核所有边界项、明显缺损、近重复和答案泄露题；Finance91/123/146包含解题过程或提示，本次未新增第五标签。',
      '原始English组并不保证每条纯英语：Ayur132/176含印地语选项，Krishi186整段题干为印地语。可理解且完整的此类知识题仍按本次两条件计入估计，未擅自加语言过滤；若未来要求严格英文，需单独核验并估计语言过滤损失。',
      '## 8. 复现与输出验证',
      'manual_taxonomy_audit.csv逐行保留原样本的全部字段，另加domain、sample_index、taxonomy、self_contained、short_reason与review_method。audit_source_row_index和原生id可追溯全量行。四份源样本CSV及全部parquet均未修改；源文件SHA256、显式逐题标签和统计结果保存于reports/bhashabench_evidence/manual_taxonomy_provenance.json。显式判断文件为*_content_labels.tsv，汇总脚本scripts/summarize_manual_taxonomy.py只重建输出，不重新做内容判断。',
      "```powershell\n& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/summarize_manual_taxonomy.py\n```",
      '已验证800条均只有一个允许的四分类标签，self_contained均为yes/no，理由非空；每域200条，样本ID/行次序/题干/选项与原始全量seed42抽样一致；输出CSV重新读取与内存表相等。未修改原始数据、未正式过滤全量、未划分train/dev/test、未训练或分配客户端。']
    report=ROOT/'reports/manual_taxonomy_summary.md'
    report.write_text('\n\n'.join(rules)+'\n',encoding='utf-8')
    stat={'reviewer':'assistant content judgment, not human expert gold labels','seed':42,'sources':provenance,
          'sample_summary':summaries,'retention_estimates':estimates,'finance_comparison':finance_rows,
          'input_data_unchanged':True,'all_800_rows_verified':True}
    (EVIDENCE/'manual_taxonomy_provenance.json').write_text(json.dumps(stat,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(table(['DOMAIN','N','recall','application','calculation/aptitude','offdomain','self-contained=no'],summaries))
    print('\nRETENTION ESTIMATES');print(table(['DOMAIN','FULL N','k/200','rate','estimated retained','95% sampling interval'],estimates))
    print('\nFINANCE AFTER TWO NATIVE LABEL EXCLUSIONS');print(table(['group','n','recall','application','calculation/aptitude','offdomain','recall+yes'],finance_rows))
    for p in [dest,report,EVIDENCE/'manual_taxonomy_provenance.json']:
        assert p.exists() and p.stat().st_size>0
        print('VERIFIED',p.resolve(),p.stat().st_size,'bytes')

if __name__=='__main__':main()
