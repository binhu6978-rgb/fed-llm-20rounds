"""Continue the completed original five-source FedAvg from round 10 through 20."""
import ctypes
import msvcrt
import os
import shutil
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_cosmosqa_five_experiments as original
from scripts import fedlora_baseline_data as baseline_data

data, shared = original.data, original.shared
DOMAINS = original.DOMAINS
SOURCE = original.MASTER / 'fedavg'
OUTPUT = ROOT / 'outputs/fedavg20_five_client4000_seed42'
REPORT = ROOT / 'reports/fedavg20_five_client4000_seed42.md'


def source_state():
    import torch
    state = torch.load(SOURCE / 'resume.pt', map_location='cpu', weights_only=True)
    assert state['completed_round']==10 and not state['pending'] and len(state['records'])==10
    assert [r['round'] for r in state['records']]==list(range(1,11))
    assert state['records']==data.read(SOURCE / 'round_diagnostics.json')['rounds']
    assert shared.tensor_hash(state['global_lora'])==state['records'][-1]['global_after_tensor_sha256']
    assert shared.tensor_hash(torch.load(SOURCE / 'round_10/global/lora_weights.pt',map_location='cpu',weights_only=True))==shared.tensor_hash(state['global_lora'])
    return state


def verify_inputs():
    files = dict(data.read(baseline_data.MASTER / 'code_state.json')['file_sha256'])
    cfg = data.config()
    files.update(data.read(ROOT / cfg['prepared_dir'] / 'manifest.json')['files'])
    for name in ('resume.pt','exposure.jsonl','round_diagnostics.json','protocol.json','result.json','round_10/global/lora_weights.pt'):
        p=SOURCE/name; key=str(p.relative_to(ROOT));digest=data.sha(p)
        if key in files:
            assert files[key]==digest, 'Protected original input changed: '+key
        files[key]=digest
    files[str(Path(__file__).relative_to(ROOT))]=data.sha(Path(__file__))
    for path,digest in files.items():
        assert data.sha(ROOT/path)==digest, 'Original input changed: '+path
    target=OUTPUT/'input_snapshot.json'
    if target.exists():
        assert data.read(target)['file_sha256']==files, 'Continuation inputs/code changed'
    else:
        data.dump(target,dict(file_sha256=files,source_round10_hash=shared.tensor_hash(source_state()['global_lora'])))


def bootstrap():
    state=source_state()
    resume=OUTPUT/'resume.pt'
    if not resume.exists():
        # Seed ONLY the new output; never modify or retrain the original ten rounds.
        shutil.copyfile(SOURCE/'exposure.jsonl',OUTPUT/'exposure.jsonl')
        shared.save_state(resume,state)
    else:
        import torch
        current=torch.load(resume,map_location='cpu',weights_only=True)
        assert 10<=current['completed_round']<=20
        assert current['records'][:10]==state['records']
    return state


def write_report(records):
    best=max(records,key=lambda r:(r['global_after']['macro_accuracy'],-r['round']))
    last=records[-1]
    folder=SOURCE if best['round']<=10 else OUTPUT
    best_checkpoint=folder/f"round_{best['round']:02d}/global/lora_weights.pt"
    lines=['# FedAvg 五数据集：原第10轮继续训练至第20轮','',
           '原第1–10轮复用；第11–20轮从原第10轮全局权重继续训练。每域固定4000条，5客户端全员参与，每客户端每轮完整1 epoch（500次更新），权重0.2。',
           '相同 Llama-3.2-1B、frozen base、LoRA q/v r8/alpha32/dropout0.05、lr1e-4、bs1/acc8、逐客户端每轮重置 AdamW、原 prompt 与 evaluator。seed=42，client_round_seed 使用实际 round−1，继续至19。',
           '20轮累计50000次 optimizer updates、400000次样本访问；新增10轮25000次更新、200000次样本访问。与10轮方法相比训练预算加倍。','',
           f"目前已完成 {last['round']}/20 轮；最佳全局 Macro 为 {100*best['global_after']['macro_accuracy']:.2f}%（Round {best['round']}）。",'',
           '| 轮次 | CosmosQA global | OpenBookQA global | SciQ global | HellaSwag global | RACE global | 本地均值 | 全局 Macro | 本地−全局 (pp) |',
           '|---:|'+'---:|'*8]
    for r in records:
        local=sum(r['local_after'][d+'_accuracy'] for d in DOMAINS)/5
        macro=r['global_after']['macro_accuracy']
        lines.append('| '+str(r['round'])+' | '+' | '.join('%.2f'%(100*r['global_after'][d+'_accuracy']) for d in DOMAINS)+
                     ' | %.2f | %.2f | %+.2f |'%(100*local,100*macro,100*(local-macro)))
    lines+=['', '所有准确率单位为 %；本地均值由五个上传前本地模型各评自己域后等权平均，全局 Macro 来自同一个共享模型评五域。',
            '', '最佳 checkpoint：`'+str(best_checkpoint.relative_to(ROOT))+'`。',
            '', '本地逐域准确率、训练 loss 和同轮全局逐域结果见 `outputs/fedavg20_five_client4000_seed42/round_diagnostics.json`；原1–10轮预测在原实验目录，11–20轮预测在本目录各轮 client/global evaluation 子目录。']
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    data.dump(OUTPUT/'best_result.json',dict(completed_round=last['round'],best_round=best['round'],
             best_metrics=best['global_after'],best_checkpoint=str(best_checkpoint.relative_to(ROOT)),
             latest_metrics=last['global_after'],round10_reference=records[9]['global_after'],
             delta_best_vs_round10_pp=100*(best['global_after']['macro_accuracy']-records[9]['global_after']['macro_accuracy']),
             training_budget_doubled_at_round20=True))


class Experiment(original.Experiment):
    def __init__(self):
        super().__init__(smoke=False)
        self.rounds=20
        self.outputs={'fedavg':OUTPUT}
        shared.MASTER=OUTPUT
        self.base_keys=tuple(k.replace('.lora_A.default.weight','.base_layer.weight') for k in self.initial if 'lora_A' in k)
        self.base_hash=shared.tensor_hash({k:self.model.state_dict()[k].detach().cpu() for k in self.base_keys})
        self.protocol.update(rounds=20,continuation_from_round=10,additional_rounds=10,
                             source_protocol=str((SOURCE/'protocol.json').relative_to(ROOT)),
                             source_protocol_sha256=data.sha(SOURCE/'protocol.json'),
                             source_round10_hash=data.read(OUTPUT/'input_snapshot.json')['source_round10_hash'],
                             continuation_runner_sha256=data.sha(Path(__file__)),
                             optimizer_behavior='Same original MCQ Trainer AdamW reset every client/round',
                             checkpoint_selection='Run all 20 rounds; report Round20 and shared best Macro, earliest tie',
                             fresh_methods=['fedavg_rounds11_to20'],reuse_all_base_local=True)
    def base(self):
        # Original fedavg() calls base(); reuse the existing endpoint without any writes/evaluation.
        return data.read(original.OUTPUTS['base']/'result.json')
    def train(self,client,domain):
        assert 10<=client.server.round<=19
        assert shared.tensor_hash({k:self.model.state_dict()[k].detach().cpu() for k in self.base_keys})==self.base_hash
        result=super().train(client,domain)
        assert shared.tensor_hash({k:self.model.state_dict()[k].detach().cpu() for k in self.base_keys})==self.base_hash
        return result
    def fedavg(self):
        delegate=shared.write_diagnostics
        def diagnostics(output,records):
            delegate(output,records)
            write_report(records)
        with patch.object(shared,'write_diagnostics',diagnostics):
            return super().fedavg()


def audit(ex,records):
    original_records=source_state()['records']
    assert len(records)==20 and records[:10]==original_records
    exposure=data.rows(OUTPUT/'exposure.jsonl')
    assert len(exposure)==100
    assert {(r['round'],r['client']) for r in exposure}=={(t,c) for t in range(1,21) for c in range(5)}
    for row in exposure:
        d=DOMAINS[row['client']]
        assert row['samples']==len(row['ids'])==len(set(row['ids']))==4000
        assert set(row['ids'])=={data.native_key(r) for r in ex.train_rows[d]}
        assert row['optimizer_updates']==500 and row['supervised_tokens']==ex.supervised_counts[d]
    import torch
    for i,r in enumerate(records):
        if i:
            assert r['common_start_tensor_sha256']==records[i-1]['global_after_tensor_sha256']
            assert r['global_before']==records[i-1]['global_after']
        assert r['weights']=={d:.2 for d in DOMAINS}
        if r['round']<=10:
            continue
        directory=OUTPUT/f"round_{r['round']:02d}"
        for kind,domains in [('global',DOMAINS),*[(f'client_{d}',[d]) for d in DOMAINS]]:
            p=directory/kind
            weights=torch.load(p/'lora_weights.pt',map_location='cpu',weights_only=True)
            receipt=data.read(p/'evaluation_provenance.json')
            assert receipt['lora_tensor_sha256']==shared.tensor_hash(weights)
            assert receipt['domains']==list(domains)
    verify_inputs()
    data.dump(OUTPUT/'protocol_audit.json',dict(passed=True,rounds=20,reused_rounds=10,new_rounds=10,
              client_epochs=100,optimizer_updates=50000,sample_visits=400000,
              original_rounds_unchanged=True,source_files_unchanged=True,full_original_test_sets=True,
              local_own_domain_before_upload_each_round=True,absolute_round_seed_continuation=True,
              canonical_frozen_base=True,previous_global_is_next_start=True))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'run.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            shared.MASTER=OUTPUT
            shared.status('checking_continuation_inputs')
            verify_inputs();source=bootstrap();write_report(source['records'])
            ex=Experiment()
            records=ex.fedavg()
            audit(ex,records);write_report(records)
            shared.status('complete',round=20,report=str(REPORT))
        except BaseException as error:
            shared.MASTER=OUTPUT
            shared.status('failed',error=str(error),traceback=traceback.format_exc())
            traceback.print_exc();raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    main()
