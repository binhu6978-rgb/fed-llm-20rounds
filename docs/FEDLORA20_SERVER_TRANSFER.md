# 四种 Federated LoRA 方法：服务器从第10轮续训至第20轮

## 直接上传这个包

项目根目录的 `server_transfer/fedlora20_server_seed42.tar` 是本次迁移包。上传到服务器，解压到独立项目目录后，进入解压后的根目录运行。包内保留相同相对目录，不需要修改旧 YAML 中的 rounds=10；新入口明确覆盖目标为20，且只运行11–20轮。

若重新生成包，在当前电脑根目录运行：

```powershell
& 'D:/sofrware/anaconda3/envs/wzm/python.exe' -X utf8 -u scripts/package_fedlora20_server.py --archive
```

## 包里包含哪些东西

- `alg/`、`utils/`、`scripts/`、`configs/` 和相关测试的文本代码文件。训练器、四种算法、模型加载、五域 prompt 编码及 choice evaluator 保持原文件和哈希。
- `models/models--llama3.2-1B/` 的直接文件：模型 safetensors、config、tokenizer 等；以及 `models/initial_lora/seed42_r8_alpha32_qv.pt`。模型离线加载，不依赖服务器重新下载。
- `dataset/cosmosqa_five_client4000_seed42/`：五域固定4000 train、完整原 test、选中 ID、统计与 manifest。
- `dataset/mcq_balanced4000/race/manifest.json`：原 RACE encoder 用它读取 max_length，不能遗漏。
- 每种方法 `outputs/fedlora_baselines_5source_seed42/baseline_<method>_5source_seed42/` 中的 `resume.pt`、protocol/audit、round trajectory、exposure、result，以及第10轮完整 global checkpoint/receipt。若原最佳轮不是10，额外包含其完整 global checkpoint/receipt。
- `outputs/fedlora_baselines_5source_seed42/local_before_upload/round_metrics.json`：已完成的1–10轮本地统计，服务器直接复用。
- 原代码 snapshot、环境 receipt，以及 FedAvg20 的结果/逐轮统计（仅作为参照，不继续训练 FedAvg）。
- 新启动脚本、依赖记录、本说明及 `server_fedlora20_manifest.json`。

完整逐文件清单是当前电脑的 `server_transfer/fedlora20_files.txt`。不需要原始下载数据、其他历史实验的模型、所有旧客户端 checkpoint 或整个 outputs。包只包括恢复与比较所必需的旧全局权重；原1–10轮全部逐题预测仍保留在当前电脑。

**FedEx/FedMomentum 必须传完整 `resume.pt` 和 `complete_state.pt`，其中包含已更新的 backbone。单独的 LoRA A/B 无法恢复当前模型。** FedRot 从自己的第10轮续训，不从最佳第9轮续训。

## 在 Linux 服务器启动

先用服务器的 CUDA PyTorch 环境（支持 BF16），建议 Python3.12 以匹配原环境。原机器 PyTorch 是 Windows 的 `2.10.0.dev20251013+cu128`；Linux 上不能复制 Windows conda 环境，应使用服务器适用的 PyTorch/CUDA 构建。其余实际依赖版本记录在 requirements。不同 GPU/构建不承诺逐位相同，脚本会记录服务器实际环境并检查两轮 GPU smoke。

```bash
mkdir -p fedlora20
tar -xf fedlora20_server_seed42.tar -C fedlora20
cd fedlora20
python -m pip install -r requirements-fedlora20-server.txt
python scripts/run_four_baselines20_server.py --check
```

`--check` 仅在 CPU 核验迁移文件哈希、四种第10轮完整状态、固定 train/test ID、原算法参数，不训练。

后台启动全部流程：

```bash
nohup python -X utf8 -u scripts/run_four_baselines20_server.py > four_methods20.log 2>&1 &
```

如需指定 GPU，可加 `--gpu 0`；默认保留调度器已有的 `CUDA_VISIBLE_DEVICES`，未设置时使用0。若服务器使用作业调度器，将同一个 Python 命令放入其作业脚本即可。

默认流程是 **四种方法各两轮 train8/client 的隔离 GPU smoke（仍使用完整原 test）→ FedEx-LoRA 11–20 → FedRot-LoRA 11–20 → FlexLoRA 11–20 → FedMomentum 11–20**。smoke 使用复制的第10轮状态，正式训练仍从原第10轮开始；smoke 结果不会混入正式20轮结果。可以单独运行 `python scripts/run_four_baselines20_server.py --smoke` 做 smoke 后退出。

Windows 服务器也可在项目根目录运行同一 Python 文件，入口不依赖旧的 Windows 启动脚本。离开会话时需按该服务器的方式保留进程。

## 配置与记录

每轮5客户端全员参与、固定各4000条 train、完整1 epoch（500 updates/client）、权重0.2。Llama-3.2-1B frozen BF16 base；q/v LoRA r8/alpha32/dropout0.05/bias none；lr1e-4、bs1/acc8、原 AdamW 每客户端每轮重建；原 `client_round_seed(42,source_id,round-1)` 使用11–20轮对应的10–19；原 prompt/evaluator 不变。FedRot lam0.5、Flex s2、FedMomentum threshold0.9999。

每个客户端训练完成后原子保存 raw/upload LoRA、训练 loss、RNG 和预算；上传前用 raw 模型在自己的完整测试集上评测（FedRot 使用旋转对齐前模型）。每轮聚合前恢复该轮完整共享 backbone；FedEx/FedMomentum 的合并权重保存在全局完整状态。聚合后原子保存状态，再评共享模型的五个完整 test。中断时可重跑同一入口；完整客户端、已完成聚合与评测复用，未完成客户端从本轮共同模型重跑一整个 epoch。失败会停止后续方法，不调参数重试。

每种方法新增25000 updates、200000 sample visits；20轮累计50000 updates、400000 sample visits。训练预算与 FedAvg20 相同，是旧10轮实验的两倍。报告固定Round20和20轮中的单个共享最佳 Macro checkpoint，五域取自同一轮。

原算法文件完全保留：FedEx/FedMomentum 的残差合并仍缺少 alpha/r=4 缩放；Flex 原分解也原样使用。新脚本仅负责跨平台加载、恢复、评测和记录，不悄悄修算法。

## 查看进度与恢复

```bash
cat outputs/fedlora20_server_seed42/sequence_status.json
tail -f outputs/fedlora20_server_seed42/logs/fedexlora.log
```

各方法状态在 `outputs/fedlora20_server_seed42/<method>/status.json`；失败细节在对应日志。同一命令恢复，入口使用跨平台文件锁防止重复队列/worker。恢复时不要修改配置、代码、manifest 或源checkpoint。

最终汇总：`outputs/fedlora20_server_seed42/report.md`。每种方法还有 `report.md`、`local_global_rounds.csv/json`（所有20轮本地均值/全局Macro/差距）、`round_trajectory.json`（每客户端 loss、准确率、上传状态和共享结果）、`result.json`（最佳与最终）、`protocol_audit.json`，以及各新增轮次的完整 checkpoint、逐题 predictions/choice logprob 和 evaluation provenance。源10轮目录只读，新结果单独保存。
