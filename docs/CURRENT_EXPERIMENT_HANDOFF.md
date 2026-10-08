# fed-llm 当前实验交接（2026-10-08）

## 交接范围与证据边界

本次仅学习代码、读取现有输出、核对统计并新增本文；没有启动训练、模型推理或评测，没有改动算法、参数、数据、已有结果，也没有新增或执行文件哈希检测。下文的路径除另行说明外均相对于 `C:/Users/admin/Desktop/fed-llm/new-code/`。

**四种服务器方法已由完成标记、状态、审计、20轮轨迹、resume元数据以及新增轮次的预测文件交叉核实完成20轮。FedAvg20的统计记录完整且内部一致，但当前副本只有三份JSON，不能据此声称其checkpoint与预测已在本目录完整归档。**

核验方式：读取JSON/JSONL/CSV和checkpoint的ZIP及受限pickle元数据；检查轮次、样本ID、预算、评测来源、张量名称/形状/类型与storage长度。只比较已有receipt中的状态标识来核对衔接，未重新计算任何文件或张量摘要。当前检查解释器未安装PyTorch，因此没有通过 `torch.load` 重建张量或核对张量数值；元数据核验不等于重新验证数值内容或重新评测模型。

## 目录与实际入口

根目录包含 `alg/`、`configs/`、`dataset/`、`docs/`、`models/`、`outputs/`、`scripts/`、`tests/`、`utils/`，以及 `four_methods20.log`、`requirements-fedlora20-server.txt`、`server_fedlora20_manifest.json`。当前根目录没有 `.git`，不能提供当前Git提交号。根目录、适用上级目录和项目内未发现AGENTS.md。

用户指定的五个入口/算法、训练器与评测器文件均存在，已阅读。说明文档情况：

- `docs/FEDLORA20_SERVER_TRANSFER.md` 存在，但其“--check核验文件哈希”说法已过时：实际 `preflight()` 及 `transfer_check.json` 明确 `file_hash_validation=false`。本文优先依据当前代码和输出。
- `docs/FEDAVG20_CONTINUATION.md` 未在当前项目及相邻项目搜索到。
- 当前项目没有 `docs/MIGRATION.md`；在相邻旧项目 `C:/Users/admin/Desktop/fed-llm/code/docs/MIGRATION.md` 找到并阅读了早期四域说明（含LogiQA、不同test数量和0.25权重），不适用于本次五域20轮。相邻旧项目未发现本次 `outputs/` 归档。
- 当前项目未发现通用 `main.py`；历史相邻项目虽有该文件，本次没有运行它。

| 文件 | 实际职责 |
|---|---|
| `scripts/run_cosmosqa_five_experiments.py` | 原五域协议入口，复用Base/Independent Local端点，原FedAvg10与Centralized10；通过定制Experiment配置共享框架 |
| `scripts/run_three_dataset_lora.py` | 被五域入口复用的模型、Client构造、Trainer调用、score、诊断与保存工具；五域入口覆盖其data/DOMAINS/encoding，不能据文件名退回三域协议 |
| `scripts/run_fedlora_baseline.py` | 四种方法原1–10轮；绑定已有算法 `Client.run` 和 `Server.aggregate`，保存raw/upload和可变backbone |
| `scripts/evaluate_fedlora_local_before_upload.py` | 历史1–10轮本地模型的事后补评与汇总实现，本文只读其代码和已有统计，未运行 |
| `scripts/run_four_baselines20_server.py` | 四种方法从各自第10轮继续11–20轮，跨平台锁、恢复、raw本地评测、完整功能状态保存和正式汇总 |
| `scripts/continue_five_source_fedavg20.py` | FedAvg从原第10轮继续11–20轮；继承原五域循环；Windows专用 `msvcrt/ctypes`，不是Linux通用入口 |
| `configs/cosmosqa_five_client4000_seed42.yaml` / `configs/fedlora_baselines_5source_seed42.yaml` | 保留原10轮协议；续训入口明确覆盖至20，不能仅凭YAML的rounds=10判定训练未完成 |

## 固定数据与协议

正式数据在 `dataset/cosmosqa_five_client4000_seed42/`，包含 `train/`、`test/`、`manifest.json`、`sequence_statistics.json/csv`。本次实际读取十份数据文件，数量、去重后的native ID及ID顺序均与manifest一致，答案标签均为A/B/C/D。按原代码的空白规范化规则比较context、question与排序后的选项，跨域train/test内容交集为0；没有重新抽样、分词或生成统计。

| 数据集/客户端ID | Train | 本协议完整Test | Test中保留的source_split |
|---|---:|---:|---|
| CosmosQA / 0 | 4000 | 1500 | validation：1500 |
| OpenBookQA / 1 | 4000 | 1000 | validation：500；test：500 |
| SciQ / 2 | 4000 | 1998 | validation：1000；test：998 |
| HellaSwag / 3 | 4000 | 1500 | validation：1500 |
| RACE / 4 | 4000 | 1500 | test：1500 |
| 总计 | 20000 | 7498 | 五域等权Macro |

“完整测试集”指完整评测已经冻结的上述7498条协议test，不代表所有域均使用其全部官方test。代码显示CosmosQA/HellaSwag/RACE的1500条来自已有固定选择，OpenBookQA/SciQ使用清洗后的validation+test；不得改成早期500/998/4934等数量。

实际入口、配置和protocol一致支持以下协议：

- Llama-3.2-1B，离线 `models/models--llama3.2-1B/`，BF16基础权重、SDPA；本地训练时backbone无梯度且不变。FedEx/FedMomentum在服务器聚合时会显式修改目标backbone，故“frozen base”只描述本地优化行为。
- `q_proj/v_proj` LoRA，r=8、alpha=32、dropout=0.05、bias=none。CAUSAL_LM的bias沿用LoraConfig默认none，原共享构造器显式断言该值。共同初始LoRA为 `models/initial_lora/seed42_r8_alpha32_qv.pt`；64个FP32 LoRA张量，851968个参数。
- lr=1e-4，batch size=1，gradient accumulation=8，完整1 epoch，step=0，drop_last=False。实际走 `Trainer._train_mcq`，不走非MCQ的衰减学习率/step截断循环。
- 原AdamW：betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01。每次客户端训练重新创建optimizer，不延续其momentum状态，也没有scheduler。
- seed=42；原 `client_round_seed(base_seed,client_id,round_index) = base_seed + round_index*1000003 + client_id*10007`。保存轮次t对应 `server.round=t-1`，11–20轮用10–19；shuffle、dropout等使用该客户端轮次种子。
- 每轮5客户端全员参与，id顺序0–4；每客户端完整4000条、500次optimizer updates，按样本数聚合后各权重0.2。
- FedRot lam=0.5；Flex s=2；FedMomentum residual_threshold=0.9999；FedEx无额外参数。无异构rank、无跨域本地5×5矩阵。
- 原1–10轮直接复用，11–20轮从各方法自己的第10轮最终共享状态继续。FedRot旧最佳为第9轮，但其续训来源仍是第10轮。没有重新训练20轮，没有从最佳模型启动续训。

每方法20轮：100个客户端epoch，50000次optimizer updates、400000次样本访问。新增11–20轮：50个客户端epoch，25000次updates、200000次样本访问。四种服务器方法的原 `exposure.jsonl` 各50条、轮次1–10；续训目录各50条、轮次11–20。逐条样本ID集合与固定train相同，各条4000访问/500更新；两段合计才是总预算。FedAvg当前缺exposure，但20轮诊断中100份客户端预算与已有audit的总数一致。

## Prompt、训练、聚合与评测流程

数据 → `utils/cosmosqa_five_mcq.py` → 原encoder → `TrainingDataset` → 原MCQ Trainer → raw/upload LoRA → 原Server.aggregate → 保存共享功能状态 → choice-likelihood评测 → 原子提交resume和统计。

CosmosQA经 `utils/cosmosqa_mcq.py` 调用 `utils/race_mcq.py::encode_race`；RACE经 `utils/five_client_mcq.py` 调用同一passage encoder，必须保留 `dataset/mcq_balanced4000/race/manifest.json`。两者模板为 `Passage: ...\n\nQuestion: ...\nA. ...\nB. ...\nC. ...\nD. ...\n\nAnswer:`。OpenBookQA/SciQ/HellaSwag使用 `utils/mcq_utils.py` 原模板 `Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\n\nAnswer:`；HellaSwag已在数据准备时把原ctx放入question。

前缀加BOS，目标为正确字母的空格continuation加EOS；prompt标签均为-100，只监督目标。encode检查continuation没有改变前缀token边界。passage encoder预留四种continuation与EOS空间，只在超过位置上限时截断passage；RACE manifest和模型config实际上限均131072。已有sequence_statistics记录五域train/test的passage截断均为0；该统计本次未重新计算。

`utils/mcq_eval.py::MCQEvaluator.evaluate` 用同一加载模型对四个字母continuation分别求完整条件log-prob之和，取最高者；不是自由生成，也不是对选项正文整体求likelihood。若continuation均为单token，单次prompt forward计算四分数；否则计算完整continuation。eval_batch_size=8、model.eval、BF16 autocast、无梯度。域accuracy=correct/域样本数，Macro为五域accuracy等权平均，不按7498总样本加权。

| 方法 | 实际本地调用 | 实际聚合调用 | Backbone行为 |
|---|---|---|---|
| FedAvg | Client继承FTBaseClient，但定制入口的 `Experiment.train` **直接调用 `client.trainer.train`**，提取adapter并把pending LoRA赋给client；循环不直接调用 `FTBaseClient.run` | `FTBaseServer.aggregate(server)` | 全程保持canonical base，只加权平均A/B |
| FedEx-LoRA | `alg.fedexlora.Client.run` → `FTBaseClient.run` → 原Trainer | `alg.fedexlora.Server.aggregate` | 本地不改base；服务器把产品聚合残差加到q/v base权重 |
| FedRot-LoRA | `alg.fedrotlora.Client.run` → 原Trainer → 旋转对齐upload；RecordingTrainer仅记录loss | 继承的 `FTBaseServer.aggregate` | base不变；raw留在model，upload为对齐后的A/B；t=0跳过对齐，之后按零基round奇偶交替对齐A/B |
| FlexLoRA | `alg.flexlora.Client.run` → `FTBaseClient.run` | `alg.flexlora.Server.aggregate` | base不变；原SVD分解产生新A/B |
| FedMomentum | `alg.fedmomentum.Client.run` → `FTBaseClient.run` | `alg.fedmomentum.Server.aggregate` | 本地不改base；随机SVD主要分量生成LoRA，残差并入q/v base |

四方法的正式入口显式调用各Client.run与Server.aggregate，自行控制保存和评测；并未把算法替换成统一factor FedAvg，也不依赖通用Server.run循环。

四方法每轮开始恢复上一轮完整共享状态，每个客户端训练前再次恢复同一状态；训练后保存 `raw_lora.pt`（model中的训练结束A/B）和 `upload_lora.pt`（client.lora）。**本地评测使用该轮开始的backbone+该客户端raw LoRA，FedRot使用旋转前raw模型。** 11–20轮在聚合前评各自域；1–10轮本地数字来自历史事后补评统计，其脚本同样恢复上一轮backbone而非本轮聚合后的backbone。本文没有再次补评。

聚合前再次恢复该轮共同起点，恢复保存的aggregation RNG，再调用原aggregate。聚合后先保存 `global/complete_state.pt` 与pending_global，再评同一个共享模型的五域，随后提交records、completed_round及清空pending。FedAvg保留其原global_before→五个local_after→aggregate→global_after循环，base不变，用canonical base+LoRA恢复。

`complete_state.pt` 是**恢复模型功能所需的状态**，并非复制全部1B基础参数：含64个LoRA张量、canonical model引用；FedEx/FedMomentum另含32个已修改的q/v BF16基础权重，其余基础权重仍依赖canonical模型文件。FedRot/Flex的base_weights为空。恢复FedEx/FedMomentum或解释其模型时不能只看A/B。恢复第t轮local raw时，backbone应来自t−1全局（t=11来自旧round_10），不能拿t轮聚合后状态配raw。

断点恢复：四方法resume含completed_round、global_state、pending客户端、pending_global、records、aggregation_rng；已经原子提交的客户端及评测复用，未提交客户端从本轮共同起点重做完整epoch，已聚合未评测可从pending_global继续。文件锁防止同一worker/队列重复启动，异常停止后续方法。不要把optimizer内步数当作可直接恢复的训练位置。

## 完成状态与实际核验

| 方法 | 轮次/完成证据 | 原1–10与11–20衔接 | 最终/最佳checkpoint及评测 |
|---|---|---|---|
| FedAvg | 20条诊断、completed_round=20、audit passed=true；当前无status/completed/resume | 连续状态标识及global_before/上一轮global_after一致；缺源目录，无法再与原10轮文件直接比对 | 当前缺全部轮次checkpoint、预测和provenance；只有统计可核实 |
| FedEx-LoRA | completed.txt、status complete/round20、audit passed；resume round20且pending为空 | 前10条与源trajectory及源resume完全相同；第11轮接第10轮，后续连续 | 最终R20、最佳R19及新增各轮完整状态/评测均存在 |
| FedRot-LoRA | 同上 | 同上；明确接R10而非旧最佳R9 | 最终R20、最佳R19及新增各轮状态/评测均存在 |
| FlexLoRA | 同上 | 同上 | 最终R20、最佳R18及新增各轮状态/评测均存在 |
| FedMomentum | 同上 | 同上 | 最终/最佳均R20；新增各轮完整状态/评测均存在 |

四方法trajectory/local_global_rounds均含且仅含1–20轮。续训protocol中的original_protocol逐字段等于原protocol，正式smoke=false；source/target=10/20。四份源resume均round10、四份新resume均round20；pending为空、pending_global和aggregation_rng为None，records等于相应trajectory。所有新增global和raw/upload checkpoint的元数据结构、r8形状、dtype与storage长度通过只读核验；FedEx/FedMomentum每个完整状态确含32个BF16 base张量，另两方法为空。

对四方法11–20轮，核验240组评测目录、599840条已有预测（每方法60组/149960条，global与local各覆盖7498×10）：测试ID/金标、域数量、correct标记、四logprob有限性及argmax、CSV算出的accuracy均与metrics.jsonl/result/trajectory一致；global provenance指向本轮complete_state，local指向本轮raw，域集合、round和backbone_included正确。本地模型的起始backbone由代码恢复顺序与记录中的start状态共同支持；本次未数值重构模型验证来源。

1–10轮global统计与源trajectory完全一致；本地40条round_metrics的五域均值与续训合表一致，但原本地预测/identity/audit在本副本缺失，无法重新逐题核对。源目录保留FedEx/Flex/FedMomentum的R10 global、FedRot的R9/R10 global（result/provenance与trajectory一致），其他35个历史global checkpoint未迁入；保留的这5个global也无逐题预测。旧缺失属于当前归档范围限制，不能据此认定旧轮没有训练。

总队列 `outputs/fedlora20_server_seed42/sequence_status.json` 为complete，记录时间换算北京时间为2026-10-08 07:48。`smoke_<method>/` 的trajectory虽有12条（复用10条+新增两条），audit标记smoke=true、每客户端仅train8，是隔离检查，未纳入以下正式表；GPU probe也不属于正式实验，当前未找到gpu_probe_receipt.json。

## 三张正式结果表

准确率单位为%，差距单位为pp。计算、差值和最佳轮次选择均使用JSON未四舍五入原值，最后显示两位小数。

本地均值：五个本轮上传前raw本地模型，各评自己的完整test后等权平均。全局Macro：本轮聚合后的**同一个共享模型**评五域后等权平均。前者不是Independent Local，也不是一个共享模型的Macro。FedAvg所有轮次及四方法1–10轮为保留统计（存在上述原始归档限制）；四方法11–20轮已核对现有逐题预测。

### 表一：每轮“本地均值 / 聚合后全局Macro”

| 轮次 | FedAvg | FedEx-LoRA | FedRot-LoRA | FlexLoRA | FedMomentum |
|---:|---:|---:|---:|---:|---:|
| 1 | 68.68 / 53.47 | 68.68 / 54.21 | 68.68 / 53.47 | 68.68 / 60.78 | 68.68 / 56.45 |
| 2 | 70.38 / 60.96 | 70.83 / 61.25 | 70.38 / 60.95 | 71.65 / 62.44 | 71.27 / 62.82 |
| 3 | 71.62 / 64.33 | 70.93 / 64.39 | 71.30 / 64.21 | 71.72 / 64.50 | 72.15 / 65.36 |
| 4 | 73.02 / 65.77 | 73.29 / 65.69 | 73.25 / 65.85 | 71.95 / 65.65 | 72.95 / 66.53 |
| 5 | 73.03 / 66.79 | 73.20 / 66.72 | 73.04 / 66.60 | 70.70 / 66.38 | 72.78 / 67.43 |
| 6 | 74.08 / 67.25 | 73.99 / 67.44 | 73.93 / 67.27 | 70.36 / 66.81 | 72.99 / 67.66 |
| 7 | 73.07 / 68.23 | 73.70 / 68.27 | 73.15 / 68.45 | 69.77 / 67.39 | 72.58 / 68.42 |
| 8 | 73.37 / 68.60 | 73.20 / 68.93 | 73.59 / 68.63 | 70.66 / 67.81 | 73.15 / 68.79 |
| 9 | 72.75 / 68.78 | 72.85 / 68.73 | 73.10 / 69.00 | 70.11 / 67.90 | 73.41 / 69.20 |
| 10 | 73.01 / 69.37 | 72.93 / 69.07 | 73.51 / 68.95 | 69.90 / 68.36 | 72.67 / 69.33 |
| 11 | 72.91 / 69.30 | 72.82 / 69.26 | 72.85 / 69.48 | 69.74 / 68.19 | 72.05 / 69.38 |
| 12 | 72.95 / 69.63 | 73.31 / 69.36 | 73.21 / 69.39 | 69.47 / 68.01 | 72.92 / 69.91 |
| 13 | 72.57 / 69.82 | 73.04 / 69.71 | 73.17 / 70.14 | 69.48 / 68.61 | 72.73 / 69.99 |
| 14 | 72.96 / 69.66 | 72.04 / 69.83 | 73.23 / 69.83 | 69.95 / 68.49 | 72.13 / 70.01 |
| 15 | 72.68 / 70.11 | 72.35 / 70.06 | 73.21 / 70.08 | 69.72 / 68.25 | 72.80 / 69.94 |
| 16 | 72.73 / 69.94 | 72.15 / 69.72 | 73.27 / 70.15 | 70.04 / 68.71 | 72.23 / 70.36 |
| 17 | 72.70 / 69.46 | 72.53 / 69.45 | 73.10 / 70.24 | 69.61 / 68.78 | 71.76 / 70.11 |
| 18 | 72.96 / 69.52 | 72.56 / 69.61 | 73.26 / 70.10 | 69.94 / 68.78 | 72.53 / 70.24 |
| 19 | 72.16 / 70.00 | 72.16 / 70.60 | 72.99 / 70.34 | 69.85 / 68.71 | 72.24 / 70.06 |
| 20 | 73.24 / 69.99 | 72.13 / 70.15 | 72.32 / 70.22 | 69.66 / 68.75 | 72.28 / 70.50 |

### 表二：最终第20轮

| 方法 | CosmosQA | OpenBookQA | SciQ | HellaSwag | RACE | 全局Macro | 本地均值 | 本地−全局 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 64.80 | 65.50 | 82.43 | 72.13 | 65.07 | 69.99 | 73.24 | +3.25 |
| FedEx-LoRA | 66.27 | 66.00 | 82.13 | 70.87 | 65.47 | 70.15 | 72.13 | +1.98 |
| FedRot-LoRA | 66.33 | 64.80 | 82.88 | 72.20 | 64.87 | 70.22 | 72.32 | +2.11 |
| FlexLoRA | 62.93 | 63.10 | 80.73 | 72.47 | 64.53 | 68.75 | 69.66 | +0.91 |
| FedMomentum | 65.20 | 65.90 | 82.98 | 72.60 | 65.80 | 70.50 | 72.28 | +1.78 |

### 表三：1–20轮中的最佳单个共享模型

| 方法 | 最佳轮次 | CosmosQA | OpenBookQA | SciQ | HellaSwag | RACE | 最佳全局Macro | 最终R20 Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 15 | 66.07 | 64.10 | 82.83 | 72.73 | 64.80 | 70.11 | 69.99 |
| FedEx-LoRA | 19 | 66.27 | 65.20 | 82.98 | 72.47 | 66.07 | 70.60 | 70.15 |
| FedRot-LoRA | 19 | 65.13 | 65.50 | 83.33 | 71.87 | 65.87 | 70.34 | 70.22 |
| FlexLoRA | 18 | 64.53 | 63.30 | 81.28 | 71.27 | 63.53 | 68.78 | 68.75 |
| FedMomentum | 20 | 65.20 | 65.90 | 82.98 | 72.60 | 65.80 | 70.50 | 70.50 |

选择规则：按每轮同一共享checkpoint的五域global Macro最大值选择，并列取最早轮。重算结果与各result/best_result一致，没有拼接各域oracle最高值。Flex R17=0.6878278278278278、R18=0.6878292292292292，虽然均显示68.78%，实际R18更高。FedAvg R15来自统计记录，其实体checkpoint当前缺失。

这是使用已有测试结果做的回顾性最佳轮次选择，不是独立验证集选择。Base、Independent Local、Centralized正式结果目录在本副本缺失，故不新增可误解为完整核验的参照表。代码规定Base无训练，Independent Local为各域独立10 epoch固定最终端点，原Centralized为混合20000条的10 epoch固定最终端点（2500 updates/epoch），它们没有跟随联邦方法续训至20轮。早期四域Base/Local数字不得混入本表。

## Checkpoint、预测与统计路径

四方法公共根为 `outputs/fedlora20_server_seed42/`，目录名依次为 `fedexlora`、`fedrotlora`、`flexlora`、`fedmomentum`。最终与最佳文件均已检查存在：

| 方法 | 最终checkpoint | 最佳checkpoint |
|---|---|---|
| FedAvg | `outputs/fedavg20_five_client4000_seed42/round_20/global/lora_weights.pt`（预期路径，当前缺失） | `outputs/fedavg20_five_client4000_seed42/round_15/global/lora_weights.pt`（result引用，当前缺失） |
| FedEx-LoRA | `outputs/fedlora20_server_seed42/fedexlora/round_20/global/complete_state.pt` | `outputs/fedlora20_server_seed42/fedexlora/round_19/global/complete_state.pt` |
| FedRot-LoRA | `outputs/fedlora20_server_seed42/fedrotlora/round_20/global/complete_state.pt` | `outputs/fedlora20_server_seed42/fedrotlora/round_19/global/complete_state.pt` |
| FlexLoRA | `outputs/fedlora20_server_seed42/flexlora/round_20/global/complete_state.pt` | `outputs/fedlora20_server_seed42/flexlora/round_18/global/complete_state.pt` |
| FedMomentum | `outputs/fedlora20_server_seed42/fedmomentum/round_20/global/complete_state.pt` | 同最终第20轮 |

对四方法，记其方法目录为 `<method>`、轮次为两位数字`NN`，实际文件布局：

- 共享预测：`outputs/fedlora20_server_seed42/<method>/round_NN/global/evaluation/round_NN_predictions.csv`，含id/domain/gold/prediction/logprob_A–D/correct；最终NN=20，最佳NN分别19/19/18/20。
- 共享评测：同一global目录下 `result.json`、`evaluation_provenance.json`、`evaluation/metrics.jsonl`。
- 本地raw/upload：`outputs/fedlora20_server_seed42/<method>/round_NN/client_<domain>/raw_lora.pt` 和 `upload_lora.pt`。
- 本地预测：上述client目录下 `evaluation/round_NN_predictions.csv`；另有 `result.json`、`evaluation_provenance.json`、`evaluation/metrics.jsonl`。
- 续训状态：各方法目录 `resume.pt`、`status.json`、`completed.txt`、`protocol.json`、`protocol_audit.json`、`exposure.jsonl`。
- 统计：各方法目录 `round_trajectory.json`、`local_global_rounds.json/csv`、`result.json`、`report.md`；四方法总报告 `outputs/fedlora20_server_seed42/report.md`，队列状态 `sequence_status.json`，实际worker日志在 `logs/<method>.log`。
- 原10轮来源：`outputs/fedlora_baselines_5source_seed42/baseline_<method>_5source_seed42/` 的resume/protocol/audit/trajectory/exposure/result、保留的旧global目录；本地统计为 `outputs/fedlora_baselines_5source_seed42/local_before_upload/round_metrics.json`。
- FedAvg当前可读文件仅 `outputs/fedavg20_five_client4000_seed42/{best_result.json,protocol_audit.json,round_diagnostics.json}`，应据这三份汇总而不能虚构预测路径已存在。原 `outputs/cosmosqa_five_client4000_seed42/` 在当前副本整体缺失。
- 数据统计：`dataset/cosmosqa_five_client4000_seed42/sequence_statistics.json/csv` 与manifest。基础模型safetensors/config/tokenizer及共同初始LoRA实际存在。

路径中的历史Windows反斜杠在跨平台读取时需替换为 `/`，不要把它当Linux文件名的一部分。

## 实现差异、实际环境与现象

FedEx沿用 `res = Σw_i B_iA_i − mean(B_i)mean(A_i)` 并直接 `.add_(res)` 到base；A/B分别取非加权mean，在本协议五客户端等样本数下等于0.2加权mean。**残差合并未乘alpha/r=4**，因此不是已验证的全缩放LoRA函数精确聚合，并有BF16合并舍入。

FedMomentum沿用FP32产品聚合、sketch size=5×8=40的随机SVD、能量阈值0.9999；主要分量平衡sqrt(Sigma)拆为A/B，残差直接加到BF16 base。**该残差合并同样缺少alpha/r=4缩放**。该实现也不等于跨轮保留AdamW动量；Trainer仍每客户端每轮重置。

Flex沿用仓库原分解：`delta_W=s*Σw_i B_iA_i`；`B=U_r sqrt(Sigma_r)/s`、`A=Vh_r`。A侧没有再乘sqrt(Sigma)，没有替换成另一种论文式分解。FedRot保留原soft Procrustes及lam0.5；本地指标严格来自raw，不用对齐upload冒充raw。

以上均为本次读到的实际实现与已有protocol中记录的差异；没有修正，也不描述为已经验证的理想论文实现。`code_state.json` 存在，当前 `implementation_snapshot/` 未迁入；本次不重新计算摘要，不能额外保证历史代码快照与当前代码逐字/逐位相同。

| 环境 | 可核实Python/PyTorch/CUDA | GPU及限制 |
|---|---|---|
| 原本机训练receipt | Python3.12.11；`D:/sofrware/anaconda3/envs/wzm/python.exe`；torch `2.10.0.dev20251013+cu128`（CUDA构建标识12.8） | receipt未记录GPU型号/driver，不能用当前设备反推原训练设备；该解释器路径当前不存在 |
| 服务器续训receipt及日志 | Linux Python3.12.0；torch `2.7.1+cu118`（CUDA构建标识11.8）；transformers4.57.6、peft0.20.0、numpy2.1.2、datasets4.2.0、safetensors0.6.2、PyYAML6.0.3、accelerate1.14.0 | 正式worker日志保留 `CUDA_VISIBLE_DEVICES=1`；未记录物理GPU型号、driver及独立CUDA运行时详情，待服务器环境输出确认 |
| 本次当前本机只读检查 | `D:/software/minicoonda/python.exe`，Python3.14.7；当前解释器无torch/transformers/peft/datasets/safetensors/accelerate，numpy2.5.3、PyYAML6.0.3 | `nvidia-smi` 实测RTX4080 SUPER，driver572.47，16376 MiB；当前解释器无PyTorch，未测试torch CUDA/BF16能力 |

原本机环境来源 `outputs/fedlora_baselines_5source_seed42/environment_receipt.json` 及transfer_check内source_environment；服务器环境来源 `outputs/fedlora20_server_seed42/transfer_check.json`、`four_methods20.log` 和各worker日志。本次没有可调用的服务器终端，故GPU型号与driver缺口保持待确认，不安装环境、不启动GPU探测。两次训练的PyTorch构建不同，`cross_machine_bitwise_identity_guaranteed=false`，不承诺跨机器逐位一致。

仅描述本seed现象：第20轮Macro最高为FedMomentum70.50%，回顾性最佳共享Macro最高为FedEx70.60%（R19）。五种方法本地均值都高于同轮全局Macro，但它们来自不同数量的模型，差值不能直接解释成算法机制。Flex与其他方法存在已记录的实现差异；单seed结果不能用于统计显著性、因果解释或论文实现优劣的普遍判断。

## 缺失信息与后续入口

已经核实：固定20000/7498数据、原prompt/evaluator、四方法正式20轮完成状态与恢复元数据、源/续训预算、正确R10→R11衔接、四方法11–20轮已有预测与统计一致、五方法三张表和未舍入最佳选择、四方法最终/最佳功能状态文件存在。FedAvg20可核实的是保留统计与audit的内部一致性。

仍待确认或取回现有归档（本次未补训/补评）：

1. FedAvg20的resume、protocol、exposure、completed/status、所有轮次checkpoint/预测/provenance，以及原 `outputs/cosmosqa_five_client4000_seed42/`。当前无法直接核实FedAvg旧1–10轮与原目录相同，也无法重建其最终/最佳模型。
2. 四方法旧1–10轮全部客户端raw/upload和预测、旧本地evaluation identity/audit/client_metrics、未迁入的35个旧global checkpoint及全部旧global预测。原source status也未迁入；source completed标记、resume和audit仍在。
3. Base、Independent Local、Centralized原正式结果和选择记录，缺完整归档时不添加历史参照准确率。
4. 原训练机器GPU记录、服务器物理GPU型号/driver/运行时详情；本次未独立核对checkpoint张量数值。`implementation_snapshot/`、FEDAVG20_CONTINUATION说明和共享旧构造器引用的 `docs/migration_files_sha256.json` 当前缺失。

`server_fedlora20_manifest.json` 所列215项文件均存在（仅检查存在性，未做文件摘要检测）；清单完整不意味着包含全部历史实验归档。缺少上述文件与精简迁移包目的相符，但它们在原机器/服务器的实际保存位置仍需确认，不能把迁移文档“原机仍保留”当作本次已经核实的事实。

未来恢复四方法应使用 `scripts/run_four_baselines20_server.py` 及各方法自己的完整resume，在合适的CUDA/BF16环境和本项目canonical模型下进行；当前目标已经完成20轮，同一默认入口不会自动开始21轮。正式汇总优先只读取 `result.json`、`round_trajectory.json`、`local_global_rounds.json/csv` 与现有评测目录；现有 `combined_report()` 负责旧格式汇总，但调用队列main会写状态并检查smoke，不宜把无参数入口当纯只读汇总工具。

未来恢复FedAvg应使用 `scripts/continue_five_source_fedavg20.py` 并先取回完整原/续训归档及所需原环境文件。它依赖Windows、旧源resume及原五域Base结果，当前精简目录不能据三份JSON直接恢复。原10轮实验或其历史报告来源应看 `run_cosmosqa_five_experiments.py` / `run_fedlora_baseline.py`，不运行通用main，也不重跑旧结果来填补归档。

本次到此停止，后续训练轮数、新seed、实现修正、额外评测或数据协议变更均等待下一步实验安排。
