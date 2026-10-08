# fed-llm

五数据集 Federated LoRA 实验代码与结果归档：FedAvg、FedEx-LoRA、FedRot-LoRA、FlexLoRA、FedMomentum，seed=42，正式记录覆盖1–20轮。

完整项目交接、协议、三张结果表、实现差异及核验边界见 [docs/CURRENT_EXPERIMENT_HANDOFF.md](docs/CURRENT_EXPERIMENT_HANDOFF.md)。

## 仓库内容

- `alg/`：仓库算法实现。
- `scripts/`：数据准备、原实验、续训、评测及汇总入口。
- `utils/`：模型加载、训练器、五域prompt编码、choice-likelihood评测。
- `configs/`、`tests/`：已有配置与检查代码。
- `docs/`：当前实验交接与服务器迁移说明。
- `outputs/`：已有正式实验的轻量统计、结果、评测来源和完成记录。

本仓库不上传 `models/`、`dataset/`、任何权重或checkpoint、逐题预测、样本exposure、日志、锁文件、缓存和smoke输出。`.gitignore` 固定这些排除规则；没有改动已有实验数据或结果。

这是当前工作目录的代码与轻量结果归档，不是可直接恢复训练的完整迁移包。交接文档描述的checkpoint/预测/数据路径属于实验机器上的布局；它们不因此成为GitHub已收录文件。`server_fedlora20_manifest.json` 同样是历史完整迁移包清单，不是本Git仓库的文件清单。

## 当前结果

准确率单位为%；最佳轮次按未舍入的五域全局Macro选择，并列取最早轮次，五域均来自同一个共享模型。

| 方法 | 第20轮全局Macro | 最佳轮次 | 最佳全局Macro |
|---|---:|---:|---:|
| FedAvg | 69.99 | 15 | 70.11 |
| FedEx-LoRA | 70.15 | 19 | 70.60 |
| FedRot-LoRA | 70.22 | 19 | 70.34 |
| FlexLoRA | 68.75 | 18 | 68.78 |
| FedMomentum | 70.50 | 20 | 70.50 |

四种服务器方法已在原实验目录核对完成20轮；其11–20轮已有逐题预测与统计一致。FedAvg20在当前工作目录仅保留三份统计JSON，checkpoint、预测和resume缺失，故只能核对统计内部一致性。历史1–10轮也有归档缺口，详见交接文档。本仓库不声称收录完整模型归档。

## 协议与入口

CosmosQA、OpenBookQA、SciQ、HellaSwag、RACE各固定4000条训练数据，总计20000；完整协议test分别1500、1000、1998、1500、1500，总计7498。Llama-3.2-1B BF16 base，本地训练只优化q_proj/v_proj LoRA，r8/alpha32/dropout0.05；lr1e-4、bs1/acc8、原AdamW每客户端每轮重建。每轮5客户端全员参与，各500次更新、权重0.2。20轮合计50000次更新、400000次样本访问。

- 原五域FedAvg10/Centralized10：`scripts/run_cosmosqa_five_experiments.py`。
- 四种方法原1–10轮：`scripts/run_fedlora_baseline.py`。
- 四种方法从各自第10轮继续11–20轮：`scripts/run_four_baselines20_server.py`。
- FedAvg续训：`scripts/continue_five_source_fedavg20.py`（Windows入口）。

现有配置保留原rounds=10，续训入口覆盖目标为20。不要直接运行早期三域/四域入口来替代本协议。未补齐canonical模型、固定数据、源状态、相应环境和历史依赖文件前，不能凭本仓库直接恢复实验。

## 实现与环境限制

FedEx/FedMomentum在聚合时修改backbone，原残差合并缺少alpha/r=4缩放；FlexLoRA沿用仓库原分解。本归档保留原实现，没有修正，也不将它们宣称为已验证的理想论文实现。

原本机训练receipt为Python3.12.11、torch2.10.0.dev20251013+cu128；服务器续训为Python3.12.0、torch2.7.1+cu118。其他依赖见 `requirements-fedlora20-server.txt` 与已有环境receipt；不承诺跨机器逐位一致。

所有结果仅为单seed描述，不作机制、因果或统计显著性推断。上传准备过程没有启动训练或重新评测。
