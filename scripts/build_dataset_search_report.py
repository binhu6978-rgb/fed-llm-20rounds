"""Build the research comparison from measured evidence; no model training."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import platform

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/dataset_search'
E=OUT/'evidence'

def read(name):
    return json.loads((E/(name+'.json')).read_text(encoding='utf-8'))

def counts(name,field):
    return {json.loads(k):v for k,v in read(name)['fields'][field]['counts'].items()}

def table(headers,rows):
    def cell(x):
        return str(x).replace('|','\\|').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |',
                       '| '+' | '.join(['---']*len(headers))+' |']+
                      ['| '+' | '.join(map(cell,row))+' |' for row in rows])

def csv_write(name,headers,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(headers); w.writerows(rows)

def main():
    q=counts('QANTA_guesstrain_audit','category')
    qb=counts('QANTA_buzztrain_audit','category')
    qu=Counter(q)+Counter(qb)
    ex=counts('ExpertQA_rand_train_audit','native_field')
    exp=counts('ExpertQA_domain_train_audit','native_field')
    diag=read('additional_diagnostics')
    sx=counts('SuperGPQA_train_audit','discipline')
    tb=counts('TextbookReasoning_train_audit','subject')
    sc=counts('ScienceQA_train_audit','topic')
    em=counts('EXAMS_train_audit','native_subject')
    med=counts('MedMCQA_train_audit','subject_name')
    shortlist=[
      ['QANTA / Quizbowl','EMNLP 2014；2018 扩展版 / 2019 技术报告','短答案实体事实 QA','是；guesstrain 96,221 + buzztrain 16,706；category 全覆盖','11 category；History、Literature、Science、Fine Arts、Social Science 等','不需外部文章；题目自身是多线索段落','最接近统一 closed-book factual QA，已有训练与评测 folds','标签至少部分由分类器补全；CS/medicine/law 不能作为独立完整大类宣称','优先进一步核验'],
      ['ExpertQA','NAACL 2024（2023 预印本）','专家长答案 / attributed QA','是；原始 rand_train 1,741 / domain_train 1,728；metadata.field 全覆盖','32 field，含 Healthcare / Medicine、Law、Economics、Chemistry、History；CS 包含在工程技术细分中','官方 LFQA 全部 context 为空；答案仍有引用及外部事实依赖','人类专家领域元数据；有真实训练/验证/测试和专家修订答案','简化 QA 文件无标签；意见/建议/情景等类型混合，非纯事实 QA','优先进一步核验'],
      ['QBLink','EMNLP 2018','连续三问的短答案知识 QA','是；22,818 train 序列 / 68,454 问答；category 有 2 条 Empty','11 有名大类 + Empty；主要 History、Literature、Science、Fine Arts','需要 lead_in，部分后续问需历史 QA；外部文章不是固定输入','有原生大类和真实 train/dev/test；知识 QA 形式相近','序列标签不保证每问领域纯；wiki_page 缺失/错误；下载版本数量与论文不同','优先进一步核验（强条件）'],
      ['ScienceQA','NeurIPS 2022','K–12 多选 QA；部分图文','是；train 12,726，subject/topic/category 完整','3 subject / 26 train topic / 119 train category；physics、chemistry、biology、economics、history 等','6,218 train 有图、6,079 有 hint；无图无 hint 4,373','同一数据来源、原生层次标签、正式 train/val/test','仅 3 最上层大类；topic 混学科与语言技能；不能直接当无上下文大学知识 QA','条件备选'],
      ['TextbookReasoning','MegaScience，arXiv 2025；作者仓库称 COLM 2026 接收','教材推理 QA，短参考答案 + 生成式推理','是；train 651,840，subject 全覆盖','7：math、medicine、biology、physics、chemistry、cs、economics','无独立 context 字段；作者旨在修订成自足问题，仍需查残余指代','统一 question/answer，含独立 medicine、cs、economics','学科由作者 LLM 判教材；推理/计算混杂；math 占约 65%；无原生独立 test','接受作者自动标签时的条件备选'],
      ['EXAMS','EMNLP 2020','高中考试多选 QA','是；所核 multilingual train 7,961，info.subject 完整','26 原始 subject 值；Biology、History、Chemistry、Physics、Philosophy、Business 等','原始版无固定检索文章；另有 with_para 版本','原生学科、正式训练协议、统一 MCQA','所核 train 11 种语言无英文；学科与语言分布混淆；有职业细分类','接受非英文时的条件备选'],
      ['MedMCQA','CHIL 2022','医学考试多选 QA','是；train 182,822，subject_name 非空，但 Unknown 3,045','21 原始值＝20 有名医学学科 + Unknown','不需固定外部文章；exp 是解释字段','原生专科标签和正式训练集，统一 MCQA','只有 medicine 内部专科，不能代替 law/history/economics 等跨大领域','仅医学内部实验 / 单领域组件'],
      ['SuperGPQA','NeurIPS 2025 Datasets and Benchmarks','研究生考试多选 QA','物理容器 train 有标签；原研究为评测题库，无正式训练划分','26,529 题；13 discipline / 72 field / 285 subfield','无固定 context 字段，question + options','大领域层次最贴近目标，领域标签全覆盖','Hub 自动 train 名称误导；知识与计算混杂；用作训练须重新设计独立评测','评测 / 改造后备选，严格 train 条件不满足'],
    ]
    headers=['数据集','论文 / 年份','QA 类型','真实 train 与领域标签','领域数与主要名称','context','最大优点','最大问题','本研究定位']
    csv_write('comparison.csv',headers,shortlist)
    allcounts=[]
    for file in sorted(E.glob('*_audit.json')):
        d=json.loads(file.read_text(encoding='utf-8'))
        for field,v in d['fields'].items():
            for label,n in v['counts'].items():
                allcounts.append([d['name'],d['split'],field,json.loads(label),n,d['rows'],v['null_or_absent'],v['blank'],d['source'].get('revision','')])
    csv_write('verified_domain_counts.csv',['dataset','split','field','raw_value','count','split_rows','null_or_absent','blank','revision'],allcounts)

    parts=[]
    def add(s): parts.append(s.strip())
    add('''# 公开多领域 QA 数据集检索与严格比较

检索与实证核验日期：2026-09-27。研究目标：Federated LLM fine-tuning 中按领域训练的本地模型经参数聚合后能否整合知识，以及是否出现干扰/遗忘。此次只检索、下载核验和比较，未训练模型、未划分联邦客户端、未重新标注领域。使用指定的 `fd` 环境。

## 1. 结论与筛选口径

**优先继续验证 QANTA、ExpertQA、QBLink，三者均有真实训练文件中的原生领域元数据；目前没有一个候选无条件满足全部偏好。** QANTA 最贴近短答案 closed-book factual QA；ExpertQA 的人类专家大领域覆盖最贴近 medicine/law/economics 等目标，但属于长答案且问题类型混合；QBLink 属于知识问答但必须保留连续上下文，并检查标签与答案映射。

下表列出 8 个值得审视的候选，不意味着 8 个都合格：只有前三个进入优先核验；其余明确标为条件备选、单领域组件或评测资源。样本规模没有作为淘汰门槛，也没有强行凑够 8–10 个大领域。小规模的人类专家数据仍保留。

严格区分四种概念：①真实发布的训练文件；②数据加载器把单文件默认命名为 train；③验证集/少样本演示可在另一个实验协议下用于训练；④论文称 multi-domain，但实际训练样本没有领域字段。另将“数据发布时自带标签”与“人类原始标签”分别说明，作者自动标注也不能隐去来源。

## 2. 检索范围与证据强度

以 ACL Anthology、NeurIPS/ICML/PMLR、OpenReview、arXiv、作者 GitHub/Hugging Face 与大学存档为来源，检索 multi-domain / multi-subject / factual / knowledge / expert / exam QA，覆盖经典数据及 2020–2026 年发布或更新。不能保证穷尽所有公开数据；没有把检索结果或二手综述直接当作训练标签证据。

**A级：整份文件审计。** 对高潜力候选读完整训练文件，逐行统计标签空值和所有名称；对 ExpertQA、QBLink、ScienceQA 也检查正式验证/测试文件。**B级：真实 schema / 样本与 split inventory。** 对字段已能明确排除的候选，读取 dataset-server 实际 schema 与前两行；这不构成全量标签覆盖保证。**C级：官方论文/代码与下载入口。** 下载不可得或已在任务/语言上不匹配者，明确未做全量训练审计。

审计代码只把嵌套的 `metadata.field`、`info.subject` 等投影为 `native_field` / `native_subject` 方便计数；这些 `native_*` 是审计列，不是原文件新标签。真实字段映射列于下文。全量原始名称、各 split 计数及版本在 [verified_domain_counts.csv](verified_domain_counts.csv)、[evidence/](evidence/) 和 [verification_manifest.json](verification_manifest.json)。''')
    add('## 3. 最值得考虑的 8 个数据集\n\n'+table(headers,shortlist))
    add('''上述论文与官方发布来源：QANTA [EMNLP 2014](https://aclanthology.org/D14-1070/)、[2019 Quizbowl 技术报告](https://arxiv.org/abs/1904.04792)、[官方资源页](https://sites.google.com/view/qanta/resources)；ExpertQA [NAACL 2024](https://aclanthology.org/2024.naacl-long.167/)；QBLink [EMNLP 2018](https://aclanthology.org/D18-1134/)；ScienceQA [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf)；TextbookReasoning [2025 技术报告](https://arxiv.org/abs/2507.16812)、[作者当前发布状态](https://github.com/GAIR-NLP/MegaScience)；EXAMS [EMNLP 2020](https://aclanthology.org/2020.emnlp-main.438/)；MedMCQA [CHIL 2022](https://proceedings.mlr.press/v174/pal22a.html)；SuperGPQA [NeurIPS 2025 官方论文页](https://proceedings.nips.cc/paper_files/paper/2025/hash/a3c5af1f56fc73eef1ba0f442739f5ca-Abstract-Datasets_and_Benchmarks_Track.html)。COLM 2026 状态来自作者仓库，未另行核验会议正式 proceedings；不写成 ICLR 已接收论文。

## 4. 最有希望候选的实际文件核验

### 4.1 QANTA：原生 category 可用，原生 subcategory 不完整

此次核验明确指 Hugging Face `qanta` 的 **`mode=full,char_skip=25` 版本**，不是把不同年份或增量句子配置混在一起。`guesstrain=96,221`，`buzztrain=16,706`，全量 `category` 为空/null 的样本均为 0。两训练 folds 的 `qanta_id` 交集实算为 0，总计 112,927 条独立 ID。所有这些训练行 `text==full_question`，因此该配置按完整问题计数；不能把 `mode=incremental` 中的问题片段当作独立 QA。

实际列包含 `text, full_question, answer, page, raw_answer, fold, category, subcategory` 及 ID、句子索引、赛事与年份信息。推荐后续核验的 Question 映射为 `full_question`；Answer 应比较 `raw_answer` 别名与 `page/answer` 规范实体，不应先验认为三者完全等价。Wikipedia 页面或额外 evidence 文件可用于检索任务，但这个版本的完整题目本身可作为 closed-book 输入。

完整训练大类计数如下（原样保留名称，不把 `Trash` 改名）：''')
    add(table(['category','guesstrain','buzztrain','训练合计'],[[k,q.get(k,0),qb.get(k,0),n] for k,n in qu.most_common()]))
    qmeta=read('hf_qanta')
    conf=next(x for x in qmeta['card']['dataset_info'] if x['config_name']=='mode=full,char_skip=25')
    add('其他 folds 的数量来自固定版本 Hub 元数据，此次未整份检查它们的领域覆盖：\n\n'+table(['fold','数量','证据'],[[x['name'],x['num_examples'],'全量文件' if x['name'] in ['guesstrain','buzztrain'] else '固定版本 Hub 元数据'] for x in conf['splits']]))
    add('''**标签来源限制：** Hub card 明确写 `annotations_creators: machine-generated`。作者当前 ingestion 的 [`classifier.py`](https://github.com/Pinafore/qb/blob/a5216348fadc0f2bee7d0d7919068a34a311e2cf/qanta/ingestion/classifier.py) 使用 TF-IDF / MultinomialNB；[`normalization.py`](https://github.com/Pinafore/qb/blob/a5216348fadc0f2bee7d0d7919068a34a311e2cf/qanta/ingestion/normalization.py) 中存在利用题目和答案预测 category/subcategory 的调用。这足以否定“所有标签都已证明由人工给定”，但当前代码不能还原每一条 2018 标签的历史来源。它满足“发布数据自带标签、无需我们用 LLM 分类”，标签准确率仍需下一阶段人工抽查。

`subcategory` 在 guesstrain 有 53,890 个空串，在 buzztrain 有 11,169 个空串，且还有字符串 `None`；它不能支撑完整的 CS/medicine/law 等细学科训练划分。11 个 category 不等于 11 个严格学术大域：`Trash`、`Current Events` 尤其需单独考虑。QANTA 2021 与 2026 多模态版本另有官方入口，但此次结论只针对已审计的固定 HF 版本，未将新版标签覆盖视作已验证。数据 license 在该 card 为 `unknown`。

证据：[guesstrain 全量审计](evidence/QANTA_guesstrain_audit.json)、[buzztrain 全量审计](evidence/QANTA_buzztrain_audit.json)、[训练 folds 独立性检查](evidence/QANTA_train_combination_check.json)、[固定版本 HF 数据](https://huggingface.co/datasets/qanta/tree/e3c5602229e5c0c8572636a81cb366eb9ccfa890)。

### 4.2 ExpertQA：真实 train 有 32 个专家领域，但简化 QA 删除标签

实际下载作者 GitHub `data/lfqa/` 下全部 12 个随机/领域划分、原始/简化 QA 文件，以及主集合 `r2_compiled_anon.jsonl`。主集合 2,177 行，32 个 `metadata.field`，472 个未经标准化的 `metadata.specific_field`。字段是专家所申报的专业范围；不能假设每一个问题只涉及其唯一领域。''')
    add(table(['版本','train','validation','test','train field 数','缺失 field'],[
      ['rand 原始',1741,217,219,32,0],['domain 原始',1728,206,243,32,0],
      ['rand_lfqa 简化',1734,216,219,'字段被删','全部'],['domain_lfqa 简化',1721,205,243,'字段被删','全部']]))
    add('''原始 schema：`question, annotator_id, answers, metadata`。领域映射为 `metadata.field`，专业细分为 `metadata.specific_field`，问题类型为 `metadata.question_type`；目标答案应检查每行系统对应的 `answers.<system>.revised_answer_string`，这是经专家修订的版本，而非自动采用原始 `answer_string`。

简化 schema 只有 `example_id, context, question, answer`。文件名 `*_lfqa_*.json` 实际是 **JSONL**，逐行读取才能正常处理。两个简化 train 分别比原始 train 少 7 行；实算原始训练行非空修订答案数量恰好为 1,734 / 1,721，差异与缺失修订答案一致，但此次未重写作者文件。所有简化 train/val/test 的 `context` 都为空。

**完整唯一关联已实测：** rand 简化 train 的 1,734 个 question 全部在原始 rand_train 中精确且唯一匹配，domain 简化 train 的 1,721 个也全部唯一匹配；无须 LLM 分类即可关联回同版作者标签。这只是验证可行性，此次没有写出新的训练集。''')
    add(table(['原生 field','rand 原始 train','domain 原始 train','rand 简化 QA 可唯一关联数'],[
        [k,n,exp.get(k,0),diag['ExpertQA']['rand_train']['converted_fields_via_unique_exact_question_join'].get(k,0)]
        for k,n in sorted(ex.items(),key=lambda x:-x[1])]))
    add('''32 个名称中有 `Other`，不能当明确学科；`Engineering and Technology` 含 CS 类专业，**没有独立原生 `Computer Science` 大域**。不得自行把细分专业合并成新的 CS 标签并称作原生标签。

论文 §6 的 domain split 是每个领域内部按约 80/10/10 分，不是把领域整体留出。实际 domain train/test 都有 32 个 field，吻合此定义。[论文](https://aclanthology.org/2024.naacl-long.167/)

问题类型也不是全部 factual：原始 rand_train 中仅标签恰好为 `Directed question that has a single unambiguous answer` 的有 251 条；纯 hypothetical scenario 有 539 条，另有多标签组合、建议、意见和资源请求。这里没有筛掉或改写任何题。答案有出处和引用，生成/修订过程可能使用检索；空 context 只证明作者 LFQA 输入无需提供文章，并不证明所有问题是时序稳定的闭卷知识或答案引用能独立使用。后续应按原生 question_type 做领域×任务类型交叉表、核验修订事实和引用，避免将建议能力或输出长度误当知识整合。

证据：[官方代码与发布](https://github.com/chaitanyamalaviya/ExpertQA/tree/1aa7ba81bd4c083c12ef1aa2fee338f6cb60fa87)、[原始随机 train 审计](evidence/ExpertQA_rand_train_audit.json)、[原始 domain train 审计](evidence/ExpertQA_domain_train_audit.json)、[简化 train 审计](evidence/ExpertQA_rand_lfqa_train_audit.json)、[精确关联与答案/context 诊断](evidence/additional_diagnostics.json)。项目 MIT 声明需连同数据与引用材料的具体适用范围确认，不据代码许可证推断外部出处全部同许可。

### 4.3 QBLink：作者大学存档有完整训练标签，不能简单拆三条独立问题

旧官网三个 Google Drive 下载入口本次返回 404。已从作者机构 [University of Maryland DRUM 存档](https://drum.lib.umd.edu/items/fec38294-978e-4892-ae32-f56f08337d86) 获取 train/dev/test；存档 DOI 为 [10.13016/t92u-mpwn](https://doi.org/10.13016/t92u-mpwn)。文件 SHA256 留存于审计 JSON。''')
    add(table(['split','三问序列数','QA 对数','空 question','空 raw_answer','空 wiki_page'],[
      [r['split'],r['sequences'],r['qa_pairs'],r['empty_question'],r['empty_raw_answer'],r['empty_wiki_page']] for r in diag['QBLink']]))
    add('''实际根字段：`id, tournament, lead_in, category, sub_category, q1, q2, q3`；每问是 `quetsion_text, raw_answer, wiki_page, t_id`，注意作者真实字段拼写是 **`quetsion_text`**。Question 映射不能写成不存在的 `question_text`。输入通常应保留 lead_in 和所需前序问答，目标答案需审查 `raw_answer` 中 accept/prompt/格式标记。

train 共 12 个 category 原始值，其中 11 个有名大类及 2 条序列的字符串 `Empty`；空串/null 都为 0。以下按序列计数，不是把每问重新分类：''')
    bc=counts('QBLink_train_audit','category'); bd=counts('QBLink_dev_audit','category'); bt=counts('QBLink_test_audit','category')
    add(table(['category','train 序列','dev 序列','test 序列'],[[k,n,bd.get(k,0),bt.get(k,0)] for k,n in sorted(bc.items(),key=lambda x:-x[1])]))
    add('''**版本差异与质量限制：** 此存档合计 27,834 序列 / 83,502 QA，官网介绍约 18,644 序列 / 56k QA。没有证据证明两者只是某个已知过滤规则的区别，因此明确记录发布/过滤版本差异，未自行删行使数量贴论文。[官网](https://sites.google.com/view/qanta/projects/qblink)、[论文](https://aclanthology.org/D18-1134/)

训练 `wiki_page` 空缺 8,658 / 68,454 = 12.65%；抽查的前两序列已有 anatomy 问题 `optic disc` 映射为 `Spinal_disc_herniation`、`intraocular pressure` 映射为 `Blood_pressure` 的显见异常。不能把 Wikipedia 标题直接当金标准答案。样例中 `History` 序列也有神话实体题，说明序列级标签不保证单问学科纯度；这只是可观察反例，未把两条样本的错误率外推至全量。与 QANTA 同属 Quizbowl 生态，**不能默认二者相互独立或联合后没有重题**。存档无明确数据授权字段；未把代码许可证当数据许可证。

证据：[train 全量审计及真实样例](evidence/QBLink_train_audit.json)、[下载失败入口记录](evidence/QBLink_official_download_links.json)、[三个 split 质量诊断](evidence/additional_diagnostics.json)。

## 5. 条件备选的真实标签与局限

### 5.1 ScienceQA：要检查上下文和标签层次

作者 `problems.json + pid_splits.json` 实算 train/val/test = 12,726 / 4,241 / 4,241。Question=`question`，输入候选=`choices`，Answer=`answer` 指向候选索引；`lecture/solution` 是讲解/解答，应防止作为输入泄漏。`subject` 最上层只 3 类：natural science 6,873、language science 3,243、social science 2,610。`topic` 有 26 个，`category` 在训练中有 119 个，不把整集合的类数套在 train 上。

train 有图 6,218、有 hint 6,079，这两集合重叠，不能相加当总上下文数。无图且无 hint 4,373 条，分布在 24 个 topic；其中 physics 272、chemistry 332、biology 329、economics 112。这个客观字段条件仍不是逐题证明 self-contained，不能删除上下文强行变 closed-book。topic 混杂 biology 与 punctuation/reference-skills 等技能，按 topic 做客户端会引入任务差异。[官方代码](https://github.com/lupantech/ScienceQA)、[全量 train 审计](evidence/ScienceQA_train_audit.json)

全部 train topic：''')
    add(table(['原生 topic','train','无图且无 hint'],[[k,n,read('ScienceQA_train_audit')['no_image_no_hint_topics'].get(k,0)] for k,n in sorted(sc.items(),key=lambda x:-x[1])]))
    add('''### 5.2 TextbookReasoning：字段完整，但作者自动领域分类与计算混杂

完整训练 parquet 的真实列是 `question, answer, subject, reference_answer`。`answer` 是推理回复，`reference_answer` 是教材参考答案；同为 Answer 字段但用途不同。全部 651,840 行 subject 非空。''')
    add(table(['原生 subject','train','占 train'],[[k,n,f'{n/sum(tb.values()):.2%}'] for k,n in sorted(tb.items(),key=lambda x:-x[1])]))
    add('''论文数据构建阶段使用 Llama-3.3 分类教材领域与学术层次，并进行 QA 提取和修订；因此这是**作者发布的 LLM 领域标签**，不能称为纯人工原生教材标签。若硬性要求避免任何自动领域推断，应排除；若仅要求不由本研究自行分类，则可以保留条件候选。统一 QA 外壳不能消除 math/physics 计算与 medicine factual QA 的任务差异。[构建过程](https://arxiv.org/html/2507.16812v1)、[固定版本数据](https://huggingface.co/datasets/MegaScience/TextbookReasoning/tree/ca7ecbec76d01bff2e99f3dc17735b02f87d4e96)、[全量审计](evidence/TextbookReasoning_train_audit.json)

`MegaScience/MegaScience` 完整混合库此次只核验真实 schema/前两条和 split，不做全量 1.25M 标签覆盖声明；含 `source` 字段不等于每行存在学科标签。TextbookReasoning 的全量统计不能直接推广至整个 MegaScience。此版本仅 train，无正式独立 held-out test；此次未创建任何新 split。数据声明 CC-BY-NC-SA-4.0，教材来源许可仍需按用途核验。

### 5.3 EXAMS：真实学科训练标签，但本次核验 train 没有英文

实际审计作者仓库 `data/exams/multilingual/train.jsonl.tar.gz`，7,961 行。原始 Question=`question.stem`，选项=`question.choices`，Answer=`answerKey`；领域=`info.subject`，语言=`info.language`。所有 subject 非空，26 个未经合并的 raw values；11 个训练语言为 Bulgarian、Croatian、Vietnamese、North Macedonian、Turkish、Polish、Hungarian、Serbian、Albanian、Italian、Portuguese，**不包含英文**。不要把“可 cross-lingual QA”写成“英文训练可用”。论文全数据描述与当前 train 的 raw subject 计数口径不同，以下只报告实际文件。

若接受非英文，可在同一语言内比较原生学科；目前未验证各语言内是否都有所需 8–10 个学科，不能直接推荐覆盖充分。固定发布可选带检索段落版本，与原始题目版本须区分。[官方协议](https://github.com/mhardalov/exams-qa)、[全量训练审计](evidence/EXAMS_train_audit.json)''')
    add(table(['原生 info.subject','train'],sorted(em.items(),key=lambda x:-x[1])))
    add('''### 5.4 MedMCQA：20 个有名医学专科，并非 20 个大知识领域

真实 train 182,822 行；Question=`question`，选项=`opa/opb/opc/opd`，Answer=`cop` 的选项索引；`exp` 是解释而非必需 context。`subject_name` 共 21 raw values，包含 `Unknown=3,045`（1.67%）。其余全为医学内部分科；适合研究医学内部知识整合，单独不能满足 Medicine/Law/History/CS/Economics 的跨大领域设定。[官方代码](https://github.com/medmcqa/medmcqa)、[实际训练文件](https://huggingface.co/datasets/openlifescienceai/medmcqa/tree/91c6572c454088bf71b679ad90aa8dffcd0d5868)、[审计](evidence/MedMCQA_train_audit.json)''')
    add(table(['原生 subject_name','train'],sorted(med.items(),key=lambda x:-x[1])))
    add('''### 5.5 SuperGPQA：完整标签验证通过，正式训练协议验证不通过

已读完整 `SuperGPQA-all.jsonl`：26,529 行，字段 `question, options, answer, answer_letter, discipline, field, subfield, difficulty, is_calculation, uuid`。三级标签全非空，分别 13 / 72 / 285 个。Dataset-server 当前 split inventory 确实只返回 `default/train`，但作者论文定位是评测 benchmark，单文件默认 train **不是额外监督训练数据**。因此准确结论是“train 容器里的标签真实存在，正式训练/独立测试协议没有”。若未来允许从评测池重新制定研究 split，可条件考虑，此次没有执行。

`is_calculation` 为后续检查各领域知识题与计算题比例提供原生字段，但本次不把统一 MCQA 自动视为任务完全一致。[作者发布](https://github.com/SuperGPQA/SuperGPQA)、[固定版本数据](https://huggingface.co/datasets/m-a-p/SuperGPQA/tree/4430d4458112c7d4497fdcf94d7cc223313d6acf)、[全量标签审计](evidence/SuperGPQA_train_audit.json)、[实际 split 返回](evidence/SuperGPQA_server_splits.json)''')
    add(table(['原生 discipline','题数（评测池）'],sorted(sx.items(),key=lambda x:-x[1])))
    add('''## 6. 必查及其他不匹配数据集

下表中“无学科标签”指核验版本的 schema 无相应字段；“未证实”指无法核验真实训练文件，二者不能混同。关系类型、来源网站、实体类型不冒充 academic domain。''')
    rejects=[
      ['MMLU','ICLR 2021','A级：all/auxiliary_train 99,842 行 subject 全是空字符串；dev 285 行有 57 subject','存在 auxiliary_train，但没有可用学科标签；dev 为每学科 5 条 few-shot 演示','严格官方带标签 train 不满足；不能把 57 test subjects 套到 auxiliary_train。若另立“小样本 dev 用作训练”协议则可考虑，不因其规模小而排除','[论文](https://arxiv.org/abs/2009.03300)；[全量 aux 审计](evidence/MMLU_auxiliary_train_audit.json)；[dev 审计](evidence/MMLU_dev_audit.json)'],
      ['MMLU-Pro','NeurIPS 2024 D&B','A级：validation 70，14 category，每类 5；Hub inventory 只有 validation/test','无正式 train；test 12,032 为固定版本 metadata','领域强但评测数据，不满足既有训练要求；允许另立协议才可将 demonstrations 转训','[论文](https://arxiv.org/abs/2406.01574)；[官方](https://github.com/TIGER-AI-Lab/MMLU-Pro)；[验证审计](evidence/MMLU-Pro_validation_audit.json)'],
      ['TriviaQA','ACL 2017','B级：rc.nocontext 实际训练 schema/首行，question、answer、question_source、entity_pages、search_results','有 train，无 academic subject/domain；question_source 为采集来源','适合短答案知识 QA，可不输入证据；缺原生训练领域标签，不能据网站或 answer.type 做领域划分','[论文](https://aclanthology.org/P17-1147/)；[实际 schema](evidence/TriviaQA_server_evidence.json)'],
      ['Natural Questions / NQ-open','TACL 2019；NQ-open 后续 open-domain 协议','B级 original NQ schema；A级 NQ-open 全 train 87,925 行，仅 question、answer','真实 train 存在；两种版本均无学科标签','original 是文档中的长/短答案 span；NQ-open 可闭卷多参考短答案，但仍无法按原生大域训练','[论文](https://aclanthology.org/Q19-1026/)；[官方](https://github.com/google-research-datasets/natural-questions)；[NQ-open 文件审计](evidence/NQ-open_train_audit.json)'],
      ['EntityQuestions','EMNLP 2021','C级：官方 relation_query_templates.json 实际有 17 个关系模板；未下载整份训练 zip','作者发布训练问题；模板和关系名可见，未对完整 train 做标签审计','KG subject 是头实体，relation 是关系；不是医学/历史/法律等学科，本研究不使用其 relation 代替 domain','[论文](https://aclanthology.org/2021.emnlp-main.496/)；[官方](https://github.com/princeton-nlp/EntityQuestions)；[模板实物](evidence/source_code/EntityQuestions_relation_query_templates.json)'],
      ['zsRE / KILT structured_zeroshot','CoNLL 2017；KILT 2021 派生协议','B级：KILT train 的 id/input/meta/output 真实嵌套 schema；未下载 original 1.3GB 全包','有 train；slot/relation 信息不是学科标签','原任务把关系抽取变为带 context 的 QA；KILT 可用检索答案/provenance。模型编辑的 zsRE 子集也是不同版本，均不能宣称大域标签已验证','[论文](https://aclanthology.org/K17-1034/)；[官方](https://nlp.cs.washington.edu/zeroshot/)；[派生文件 schema](evidence/zsRE-KILT_server_evidence.json)'],
      ['ExamQA','Findings of EMNLP 2021','C级：固定 GitHub tree 仅 README/license；数据在 Weiyun，本次未取到真实训练文件','README 示例为中文 question/choice/answer/doc/id；未证实训练领域标签','非英文；论文的考试科目描述不证明发布字段存在。不虚称全量无标签，也不将其作为已验证推荐；不要和 EXAMS 混淆','[论文](https://aclanthology.org/2021.findings-emnlp.6/)；[官方](https://github.com/nlpdata/examqa)；[存下的格式描述](evidence/source_code/ExamQA_README.md)'],
      ['NaturalReasoning','2025 技术报告 / 发布','B级：真实 train schema 为 question/reference_answer/responses','有 train；实际 schema 没有 subject/domain','论文称多领域不能代替逐行标签；reasoning QA 也可能混不同任务','[作者发布](https://huggingface.co/datasets/facebook/natural_reasoning)；[实际 schema](evidence/NaturalReasoning_server_evidence.json)'],
      ['PopQA','ACL 2023','B级：实际 test.tsv/schema，有 prop/subj/obj 和 question/possible_answers','只发布 test；属性与实体不是学科','可作长尾事实问答评测；不满足正式带大域标签训练','[论文](https://aclanthology.org/2023.acl-long.546/)；[实际 schema](evidence/PopQA_server_evidence.json)'],
      ['GPQA','COLM 2024（2023 预印本）','C级官方发布/论文，未全量下载 gated HF','专家 science 评测，不把加载器 train 名当正式训练','biology/physics/chemistry 仅三域，原始正式训练协议不匹配；不是因规模小而淘汰','[论文](https://arxiv.org/abs/2311.12022)；[作者代码](https://github.com/idavidrein/gpqa)'],
      ['SciBench','ICML 2024','C级官方论文/代码，未全量训练审计','大学级科学计算评测','physics/chemistry/math 多为计算题，无本次已证实的领域监督训练；知识与计算混淆','[官方论文](https://proceedings.mlr.press/v235/wang24z.html)；[代码](https://github.com/mandyyyyii/scibench)'],
      ['SciKnowEval','2024 预印本；2025 更新','B级实际 inventory：v2/test，无 train','四科学大域但跨众多 task','任务类型混杂，评测为主；不能充当统一 factual QA 训练源','[论文](https://arxiv.org/abs/2406.09098)；[官方](https://github.com/HICAI-ZJU/SciKnowEval)；[split 证据](evidence/SciKnowEval_server_evidence.json)'],
      ['MultiReQA','AdaptNLP 2021 workshop','C级官方论文的 retrieval QA 协议','多个来源数据集；不是原生 8 个学术大域','比较跨数据集检索泛化，需要 answer candidates/context；source domain 不等于 knowledge discipline','[官方论文](https://aclanthology.org/2021.adaptnlp-1.10/)'],
      ['M2QA','2024 多语言多领域研究','C级作者发布；此处未下载全量','German/Turkish/Chinese；reviews/news/creative writing','SQuAD 风格 context extraction 和不可回答题；所谓 domains 是文体，非学科，且无英文','[作者发布](https://github.com/UKPLab/m2qa)'],
      ['Multi-subject-RLVR','2025 发布；本项目之前全量核验','A级已有本地完整 train/test 分析','train 573,002 行 subject 全是字符串 None；test 6,000 行有学科','不能用测试标签替代训练标签；不可直接用于本研究按领域训练','[已有实测报告](../multi_subject_rlvr/REPORT.md)'],
    ]
    add(table(['数据集','论文 / 年份','实际核验与强度','train / label 判断','不适合的原因 / 可保留用途','来源'],rejects))
    add('''MMLU 的空串是实际数据值，不是 Python `None`；Multi-subject-RLVR 的 `None` 是字符串，也不是 null。报告将这些情况分开表述。MMLU dev 的 57 个原生学科和 MMLU-Pro validation 的 14 类全部名称/数量保存在 CSV，不能声称没有任何可改作小样本训练的标签数据，但这会改变官方 split 用途。

## 7. 最值得进一步下载验证的三个候选

三个是下一阶段核验优先级，**不是唯一数据集选择，也不是把三者拼成训练集**。其必要训练文件已在此次搜索中实际下载；“进一步下载验证”指补齐官方版本、评测/别名/出处并做题级质量与任务交叉核验。''')
    add(table(['候选','为什么进入前三','下一阶段需要验证（未实施）','决定它能否用于本研究的条件'],[
      ['QANTA','短答案 factual / closed-book 与任务一致性最贴近；train 有完整大类','补齐同版 dev/test 与原始 2018 官方 JSON；题/答案去重泄漏、别名和自动分类标签人工抽查；比对新版','接受类别部分机器来源及有限 STEM 细分；不能要求已经具备独立 medicine/law/CS 三大域'],
      ['ExpertQA','人类专家原生 field，覆盖 medicine/law/economics/chemistry/history；真实训练协议','验证修订答案与原生字段精确关联；逐领域 question_type、输出长度、引用可用性、时间依赖；题/标注者跨 split 泄漏','接受专家长答案 QA；限定一致 factual 类型后仍保留所需原生领域，不把细分重命名为新大类'],
      ['QBLink','正式三 split；多个原生大类，统一短答案知识问答','解释存档与论文版本差异；保留必要 lead_in/前序；核验 raw_answer 别名、wiki_page 错链；检查每问与序列领域一致性及与 QANTA 重题','接受序列问答 context 和题级质量成本；若必须独立 Q→A，则先证明足够 self-contained 样本，而非直接拆分'],
    ]))
    add('''如果“无上下文的独立 Q→A”是绝对硬门槛，QBLink 不应无条件进入最终训练选型；应暂时只保留 QANTA 与 ExpertQA，再根据是否接受作者自动分类决定要不要验证 TextbookReasoning。若“标签必须全部人工且客观问题学科”也是绝对硬门槛，QANTA 也需降级，ExpertQA 的专业归属仍需题级确认。此次没有找到已实证无争议地同时具有英文、统一 factual QA、人工 8–10 大域、正式训练/独立测试的完美候选。

ScienceQA 是下一顺位上下文/层次标签备选；TextbookReasoning 是接受作者 LLM 标签与 reasoning 任务时的下一顺位；SuperGPQA 与 MMLU-Pro 更适合评测协议讨论。不要为了凑够域数跨来源拼接 MedMCQA、法律分类和金融计算数据，再将 task/source 差异解释为领域知识干扰。

## 8. 可复现材料与重要边界

使用 `D:\\software\\minicoonda\\envs\\fd\\python.exe`；只使用数据审计与 HTTP/Arrow 读取，没有载入或训练模型。统计为完整文件计数，不使用随机抽样；未来如需抽样，固定 seed 42。前两行是确定性 schema 核验示例，不是随机代表性质量评估。

运行脚本：

```powershell
# 网络 discovery 会获取运行时最新 commit；每次结果都会保存实际 SHA。
# 要复用本次版本，请保留 evidence 中 metadata，直接运行具体 audits。
& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/audit_qa_dataset_candidates.py --discover
& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/audit_qa_dataset_candidates.py --full-audits --secondary --qblink-archive --expertqa
& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/diagnose_qa_dataset_candidates.py --network
& 'D:\\software\\minicoonda\\envs\\fd\\python.exe' scripts/build_dataset_search_report.py
```

`--expertqa` 会重新获取作者当时最新 commit 并记录；若要求逐字复现此次版本，使用 manifest 的固定 URL/SHA 和已下载文件，而不要覆盖旧 evidence 后再声称同版本。Dataset-server 辅助返回按日期记录，通常未固定 revision；凡做全量结论均以下载的 pinned 文件为准。

输出说明：

| 文件 | 内容 |
| --- | --- |
| REPORT.md | 本报告，来源、结论、逐候选 schema 和局限 |
| comparison.csv | 8 个重点候选的严格比较 |
| verified_domain_counts.csv | 所有整份审计文件的原生字段值、split 与计数，包括 null/空串 |
| verification_manifest.json | commit、文件大小/SHA256、审计元数据、运行环境 |
| evidence/downloads/ | 本次实际取得的训练文件及重点候选其他 splits；不是所有搜索对象的完整数据 |
| evidence/*_audit.json | 全量审计 schema、前两条、字段覆盖/计数 |
| evidence/*_server_evidence.json | 次级候选 schema/样例/split 或失败记录；不能代替全量审计 |
| scripts/audit_qa_dataset_candidates.py | 网络发现、下载和全量标签计数脚本（项目根 scripts 目录） |
| scripts/diagnose_qa_dataset_candidates.py | 唯一标签关联、上下文和 QBLink 目标诊断 |
| scripts/build_dataset_search_report.py | 从证据生成报告/CSV/manifest |

本次没有计算所有候选的全量答案准确率，没有完成下一阶段的所有去重/跨来源泄漏审计；因此推荐均是核验候选，而非可直接开训的最终批准数据。下载失败、未审计训练全量的对象和论文/文件数量差异均保留记录，不自动补标签、不将未知视为已合格。许可按具体数据发布与来源材料确认，公开可下载不等于所有用途同一许可。

## 9. 固定版本清单

以下 SHA 与真实文件统计对应；完整下载 SHA256 在 manifest。''')
    versions=[]
    for f in sorted(E.glob('hf_*.json'))+sorted(E.glob('github_*.json')):
        d=json.loads(f.read_text(encoding='utf-8'))
        versions.append([d.get('repo_id',d.get('repo',f.stem)),d.get('revision','未取得'),'[元数据](evidence/'+f.name+')'])
    add(table(['源','revision / commit','证据'],versions))
    (OUT/'REPORT.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    manifest={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'environment':{'python':platform.python_version(),'executable':__import__('sys').executable},
              'scope':'Search and data audit only; no training, client allocation, or inferred labels',
              'download_files':[],'evidence':[]}
    for p in sorted((E/'downloads').rglob('*')):
        if p.is_file() and '.cache' not in p.parts:
            manifest['download_files'].append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for p in sorted(E.glob('*.json')):
        d=json.loads(p.read_text(encoding='utf-8'))
        manifest['evidence'].append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
              'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'name':d.get('name'),
              'revision':d.get('revision',d.get('source',{}).get('revision') if isinstance(d.get('source'),dict) else None),'source':d.get('source'),
              'scope':d.get('audit_scope',d.get('scope')),'rows':d.get('rows'),'error':d.get('error')})
    (OUT/'verification_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for p in [OUT/'REPORT.md',OUT/'comparison.csv',OUT/'verified_domain_counts.csv',OUT/'verification_manifest.json']:
        assert p.is_file() and p.stat().st_size>0
        print(f'{p.resolve()}  {p.stat().st_size:,} bytes')
    print('Top verification candidates: QANTA, ExpertQA, QBLink')
    print('Actual downloaded files:',len(manifest['download_files']),'Bytes:',sum(x['bytes'] for x in manifest['download_files']))

if __name__=='__main__':
    main()
