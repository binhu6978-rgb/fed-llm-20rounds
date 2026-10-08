"""Default repository algorithms on the exact successful five-source FedAvg protocol."""
import argparse
import copy
import csv
import ctypes
import importlib
import json
import msvcrt
import os
import random
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import fedlora_baseline_data as data
from scripts import run_cosmosqa_five_experiments as original
shared=original.shared
DOMAINS,MASTER=data.DOMAINS,data.MASTER
MUTATORS=('fedexlora','fedmomentum')
WARNINGS={
    'fedexlora':['The unchanged implementation adds sum(w_i B_i A_i) - mean(B_i)mean(A_i) to base weights without alpha/r=4. It uses product aggregation, but is not mathematically exact for the full scaled LoRA function under this protocol (also subject to BF16 merge rounding).'],
    'fedrotlora':[],
    'flexlora':['Literal repository decomposition is preserved: delta_W=s*sum(w_i B_i A_i); B=U_r sqrt(Sigma_r)/s; A=Vh_r. No alternative SVD factorization or hyperparameter tuning is substituted.'],
    'fedmomentum':['The unchanged implementation merges residual B@A components into base without alpha/r=4. The complete mutated backbone is saved; the mathematical implementation is not changed.']}


def cpu_dict(state): return {k:v.detach().cpu().clone() for k,v in state.items()}
def device_dict(state,device): return {k:v.to(device).clone() for k,v in state.items()}
def function_hash(state):
    return shared.tensor_hash({**{'adapter/'+k:v for k,v in state['lora'].items()},
        **{'backbone/'+k:v for k,v in state['base_weights'].items()}})


def rng_state():
    import torch
    import numpy as np
    n=np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python=random.getstate(),numpy=dict(name=n[0],keys=n[1].tolist(),position=n[2],has_gauss=n[3],cached_gaussian=n[4]))


def restore_rng(state):
    import torch
    import numpy as np
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']: torch.cuda.set_rng_state_all(state['torch_cuda'])
    random.setstate(state['python']); n=state['numpy']
    np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))


def bind_server(module,args,model,clients,lora):
    from alg.base import BaseServer
    server=module.Server.__new__(module.Server)
    BaseServer.__init__(server,args,clients)
    server.model,server.global_lora=model,device_dict(lora,next(model.parameters()).device)
    server.round,server.sample_rate,server.wall_clock_time=0,1.,0.
    # These are precisely the constructors' method-specific constant fields.
    if module.__name__.endswith('.flexlora'): server.r,server.s=args.lora_rank,args.s
    if module.__name__.endswith('.fedmomentum'): server.tau=args.residual_threshold
    return server


class RecordingTrainer:
    """Record loss for FedRot's run() without changing its alignment or training."""
    def __init__(self,delegate,client): self.delegate,self.client=delegate,client
    def train(self,model):
        loss=self.delegate.train(model); self.client.last_train_loss=loss; return loss
    def __getattr__(self,key): return getattr(self.delegate,key)


class Experiment(original.Experiment):
    def __init__(self,method,smoke=False):
        super().__init__(smoke=False)
        shared.MASTER=MASTER
        self.method,self.smoke=method,smoke
        self.module=importlib.import_module('alg.'+method)
        cfg=data.config()
        self.rounds=cfg['smoke_rounds'] if smoke else 10
        self.loaded_counts=dict(self.counts)
        if smoke:
            n=cfg['smoke_train_samples_per_client']
            self.train_rows={d:self.train_rows[d][:n] for d in DOMAINS}
            self.datasets={d:original.encoding.TrainingDataset(self.train_rows[d],self.tokenizer) for d in DOMAINS}
            self.counts={d:len(self.datasets[d]) for d in DOMAINS}
            self.supervised_counts={d:sum(sum(v!=-100 for v in x['labels']) for x in self.datasets[d].encoded) for d in DOMAINS}
            original.encoding.clear_cache()
        self.args.alg=method
        self.args.lam,self.args.s,self.args.residual_threshold=.5,2.,.9999
        self.args.rnd,self.args.sr,self.args.deterministic=self.rounds,1.,True
        self.base_keys=tuple(k.replace('.lora_A.default.weight','.base_layer.weight') for k in self.initial if 'lora_A' in k)
        assert len(self.base_keys)==32
        self.canonical_base=cpu_dict({k:self.model.state_dict()[k] for k in self.base_keys})
        self.canonical_base_hash=shared.tensor_hash(self.canonical_base)
        self.shape_check(shared.adapter(self.model))
        self.output=data.output(method)/'smoke' if smoke else data.output(method)
        self.outputs={method:self.output}
        self.protocol.update(method=method,smoke=smoke,rounds=self.rounds,counts=self.counts,
            loaded_full_counts=self.loaded_counts,weights={d:.2 for d in DOMAINS},
            fresh_methods=[method],reuse_all_base_local=True,
            aggregation='Verbatim alg.'+method+'.Server.aggregate and alg.'+method+'.Client.run',
            algorithm_parameters=cfg['algorithms'][method],algorithm_sha256=data.sha(ROOT/'alg'/(method+'.py')),
            baseline_runner_sha256=data.sha(Path(__file__)),baseline_data_sha256=data.sha(Path(data.__file__)),
            canonical_mutable_backbone_sha256=self.canonical_base_hash,
            mutated_backbone_in_checkpoint=method in MUTATORS,smoke_full_tests=True,
            evaluation_capabilities=5,shared_models=1,heterogeneous_rank=False,
            test_counts={d:len(self.test_rows[d]) for d in DOMAINS},
            checkpoint_selection='One shared best test Macro round; earliest tie; final10 also reported',
            baseline_implementation_modified=False,warnings=WARNINGS[method],
            command=data.command(method,smoke),common_original_command=cfg['original_command'])
        self.protocol.pop('centralized_optimizer_reset',None)
        print('METHOD',method,'PARAMETERS',cfg['algorithms'][method],flush=True)
        print('LOADED FROZEN TRAIN',self.loaded_counts,'UNCHANGED FULL TEST',self.protocol['test_counts'],flush=True)
        print('FORMAL BUDGET 5 x 500 x 10 = 25000 optimizer updates; 5 x 4000 x 10 = 200000 visits; weights .2',flush=True)
        assert all(n==4000 for n in self.loaded_counts.values())
        for warning in WARNINGS[method]: print('IMPLEMENTATION WARNING:',warning,flush=True)

    def shape_check(self,lora):
        assert lora.keys()==self.initial.keys()
        assert all(v.shape==self.initial[k].shape and v.dtype==self.initial[k].dtype for k,v in lora.items())
        assert all((v.shape[0] if 'lora_A' in k else v.shape[1])==8 for k,v in lora.items())

    def base_hash(self): return shared.tensor_hash(cpu_dict({k:self.model.state_dict()[k] for k in self.base_keys}))

    def capture(self):
        return dict(format_version=1,lora=shared.adapter(self.model),
            base_weights=cpu_dict({k:self.model.state_dict()[k] for k in self.base_keys}) if self.method in MUTATORS else {},
            canonical_model=data.original.config()['model_path'],canonical_initial_sha256=self.protocol['initial_lora_sha256'])

    def restore(self,state):
        self.shape_check(state['lora'])
        base=state['base_weights'] if self.method in MUTATORS else self.canonical_base
        assert set(base)==set(self.base_keys)
        self.model.load_state_dict(base,strict=False)
        self.load(state['lora'])
        assert function_hash(self.capture())==function_hash(state)

    def client(self,domain,output,server,method):
        from alg.base import BaseClient
        from utils.train_utils import Trainer
        args=copy.copy(self.args); args.suffix=str(output)
        algorithm_client=self.module.Client
        class Client(algorithm_client):
            def __init__(client):
                BaseClient.__init__(client,DOMAINS.index(domain),args)
                client.server,client.dataset=server,{'train':self.datasets[domain]}
                client.tokenizer,client.lora,client.delay=self.tokenizer,{},1.
                client.evaluator,client.last_train_loss=None,None
                delegate=Trainer(args,client.dataset,client)
                assert len(delegate.train_loader)==self.counts[domain] and not delegate.train_loader.drop_last
                delegate.train_loader=shared.ProgressLoader(delegate.train_loader,client,self.method)
                client.trainer=RecordingTrainer(delegate,client)
        return Client()

    def score_full(self,directory,t,checkpoint,state):
        digest=function_hash(state)
        assert function_hash(self.capture())==digest
        identity=dict(global_function_state_sha256=digest,canonical_model=self.protocol['config']['model_path'],
            complete_checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=data.sha(checkpoint),
            canonical_mutable_backbone_sha256=self.canonical_base_hash,
            mutated_backbone_included=self.method in MUTATORS)
        identity_path=directory/'full_function_identity.json'
        if identity_path.exists(): assert data.read(identity_path)==identity,'Cached evaluation has a different backbone or adapter'
        else: data.dump(identity_path,identity)
        scores=super().score(directory,DOMAINS,t,checkpoint)
        receipt=data.read(directory/'evaluation_provenance.json'); receipt.update(identity)
        data.dump(directory/'evaluation_provenance.json',receipt)
        return scores

    def run(self):
        import torch
        output=self.init_output(self.method); resume=output/'resume.pt'
        state=torch.load(resume,map_location='cpu',weights_only=True) if resume.exists() else dict(
            completed_round=0,global_state=self.capture(),pending={},pending_global=None,records=[],aggregation_rng=None)
        initial_path=output/'initial_complete_state.pt'
        if not initial_path.exists():
            assert state['completed_round']==0 and not state['pending']
            shared.save_state(initial_path,state['global_state'])
        self.trim_exposure(output/'exposure.jsonl',state['completed_round'],state['pending'])
        self.restore(state['global_state'])
        server=bind_server(self.module,self.args,self.model,[],state['global_state']['lora'])
        clients=[self.client(d,output,server,self.method) for d in DOMAINS]
        server.clients=clients
        for t in range(state['completed_round']+1,self.rounds+1):
            directory=output/('round_%02d'%t); directory.mkdir(parents=True,exist_ok=True)
            server.round=t-1; server.sample()
            assert [c.id for c in server.sampled_clients]==list(range(5))
            start=function_hash(state['global_state'])
            self.restore(state['global_state']); start_base=self.base_hash()
            server.global_lora=device_dict(state['global_state']['lora'],self.model.device)
            if state['pending_global'] is None:
                for client,d in zip(clients,DOMAINS):
                    path=directory/('client_'+d)/'upload_lora.pt'
                    if client.id not in state['pending']:
                        self.restore(state['global_state'])
                        assert self.base_hash()==start_base
                        shared.status('baseline_local_training',method=self.method,round=t,dataset=d,smoke=self.smoke)
                        client.run(self.model)  # Original algorithm-specific training/alignment, unchanged.
                        assert client.last_seen_count==self.counts[d]
                        assert client.last_optimizer_updates==(self.counts[d]+7)//8
                        assert self.base_hash()==start_base,'A local client mutated the frozen backbone'
                        raw,upload=shared.adapter(self.model),cpu_dict(client.lora)
                        self.shape_check(raw); self.shape_check(upload)
                        assert all(torch.isfinite(v).all() for v in upload.values())
                        shared.save_state(path,upload); shared.save_state(path.parent/'raw_lora.pt',raw)
                        state['pending'][client.id]=dict(lora=upload,start_function_hash=start,start_base_hash=start_base,
                            raw_hash=shared.tensor_hash(raw),upload_hash=shared.tensor_hash(upload),
                            budget=dict(train_loss=client.last_train_loss,samples=client.last_seen_count,optimizer_updates=client.last_optimizer_updates))
                        state['aggregation_rng']=rng_state(); shared.save_state(resume,state)
                        shared.event(output,t,'client_committed',dataset=d,start_function_hash=start,start_base_hash=start_base)
                    item=state['pending'][client.id]
                    assert item['start_function_hash']==start and item['start_base_hash']==start_base
                    assert shared.tensor_hash(torch.load(path,map_location='cpu',weights_only=True))==item['upload_hash']
                    client.lora=device_dict(item['lora'],self.model.device)
                assert len(state['pending'])==5
                self.restore(state['global_state'])
                restore_rng(state['aggregation_rng'])
                shared.status('baseline_aggregation',method=self.method,round=t,smoke=self.smoke)
                server.aggregate()  # The repository's actual algorithm, never replaced with factor FedAvg.
                value=self.capture(); self.shape_check(value['lora'])
                assert all(torch.isfinite(v).all() for v in [*value['lora'].values(),*value['base_weights'].values()])
                if self.method not in MUTATORS: assert self.base_hash()==self.canonical_base_hash
                state['pending_global']=value
                shared.save_state(directory/'global/complete_state.pt',value)
                shared.save_state(resume,state)
                shared.event(output,t,'aggregation_committed',global_function_state_sha256=function_hash(value))
            value=state['pending_global']; self.restore(value)
            path=directory/'global/complete_state.pt'
            assert function_hash(torch.load(path,map_location='cpu',weights_only=True))==function_hash(value)
            shared.status('baseline_global_evaluation',method=self.method,round=t,smoke=self.smoke)
            scores=self.score_full(directory/'global',t,path,value)
            state['records'].append(dict(round=t,metrics=scores,checkpoint=str(path.relative_to(ROOT)),
                global_function_state_sha256=function_hash(value),lora_tensor_sha256=shared.tensor_hash(value['lora']),
                start_function_hash=start,start_base_hash=start_base,weights={d:.2 for d in DOMAINS},
                clients={d:{k:v for k,v in state['pending'][i].items() if k!='lora'} for i,d in enumerate(DOMAINS)}))
            state.update(completed_round=t,global_state=value,pending={},pending_global=None,aggregation_rng=None)
            shared.save_state(resume,state); write_trajectory(output,state['records'])
            print('ROUND',self.method,t,'MACRO',scores['macro_accuracy'],flush=True)
        self.restore(state['global_state']); write_trajectory(output,state['records'])
        result=summarize(state['records']); result.update(method=self.method,warnings=WARNINGS[self.method])
        data.dump(output/'result.json',result)
        audit=self.audit(state['records'])
        data.dump(output/'protocol_audit.json',audit)
        (output/'completed.txt').write_text('All method rounds and full five-source evaluations completed.\n',encoding='utf-8')
        return result

    def audit(self,records):
        import torch
        exposure=data.rows(self.output/'exposure.jsonl')
        assert len(records)==self.rounds and len(exposure)==5*self.rounds
        assert len({(r['round'],r['client']) for r in exposure})==len(exposure)
        for r in exposure:
            d=DOMAINS[r['client']]; ids={data.original.native_key(x) for x in self.train_rows[d]}
            assert r['samples']==len(r['ids'])==len(set(r['ids']))==self.counts[d]
            assert set(r['ids'])==ids and r['optimizer_updates']==(self.counts[d]+7)//8
            assert r['supervised_tokens']==self.supervised_counts[d]
        visits=sum(r['samples'] for r in exposure); updates=sum(r['optimizer_updates'] for r in exposure)
        assert visits==sum(self.counts.values())*self.rounds
        assert updates==sum((n+7)//8 for n in self.counts.values())*self.rounds
        if not self.smoke: assert visits==200000 and updates==25000
        initial=torch.load(self.output/'initial_complete_state.pt',map_location='cpu',weights_only=True)
        assert shared.tensor_hash(initial['lora'])==self.protocol['canonical_initial_tensor_sha256']
        prior=function_hash(initial)
        for r in records:
            assert r['start_function_hash']==prior
            assert all(x['start_function_hash']==prior and x['start_base_hash']==r['start_base_hash'] for x in r['clients'].values())
            checkpoint=ROOT/r['checkpoint']
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert function_hash(saved)==r['global_function_state_sha256']
            assert bool(saved['base_weights'])==(self.method in MUTATORS)
            receipt=data.read(checkpoint.parent/'evaluation_provenance.json')
            assert receipt['domains']==list(DOMAINS) and receipt['global_function_state_sha256']==function_hash(saved)
            assert receipt['checkpoint_sha256']==data.sha(checkpoint)
            assert abs(sum(r['metrics'][d+'_accuracy'] for d in DOMAINS)/5-r['metrics']['macro_accuracy'])<1e-12
            prior=function_hash(saved)
        return dict(passed=True,smoke=self.smoke,rounds=self.rounds,client_epochs=len(exposure),sample_visits=visits,
            optimizer_updates=updates,canonical_initial=True,rank8=True,base_frozen_during_local=True,
            same_complete_global_start_per_client=True,merged_backbone_persisted=self.method in MUTATORS,
            evaluation_includes_complete_backbone=True,one_shared_model=True,heterogeneous_rank=False,
            test_counts=self.protocol['test_counts'],full_test_sets=True,algorithm_source_unchanged=True)


def write_trajectory(output,records):
    data.dump(output/'round_trajectory.json',records)
    fields=['round',*[d+'_accuracy' for d in DOMAINS],'macro_accuracy','worst_domain_accuracy']
    path=output/'round_trajectory.csv'; temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        writer.writerows(dict(round=r['round'],**r['metrics']) for r in records)
    os.replace(temporary,path)


def summarize(records):
    assert records and [r['round'] for r in records]==list(range(1,len(records)+1))
    best=max(records,key=lambda r:(r['metrics']['macro_accuracy'],-r['round']))
    return dict(best_round=best['round'],best_metrics=best['metrics'],best_checkpoint=best['checkpoint'],
        final_round=records[-1]['round'],final_metrics=records[-1]['metrics'],final_checkpoint=records[-1]['checkpoint'],
        oracle_diagnostics={d:dict(round=max(records,key=lambda r:(r['metrics'][d+'_accuracy'],-r['round']))['round'],
            accuracy=max(r['metrics'][d+'_accuracy'] for r in records)) for d in DOMAINS})


def report():
    preflight=data.prepare(); reference=preflight['original_reference']; base=reference['metrics']
    code=data.read(MASTER/'code_state.json')
    lines=['# Federated LoRA baselines: five sources, seed42','',
        'Actual original command: `'+data.config()['original_command']+'`. Original config: `configs/cosmosqa_five_client4000_seed42.yaml`. '
        'The successful 69.37% run uses the custom Experiment runner, not main.py. Baselines bind its exact model/data/encoder/evaluator '
        'to the existing algorithm Client.run and Server.aggregate. Generic main.py dataset/model defaults are not guessed.','',
        'Code state: '+code['git_status']+' Snapshot: `outputs/fedlora_baselines_5source_seed42/code_state.json` and `implementation_snapshot/`.','',
        '## Common protocol','',
        'Original frozen train4000 per source; tests CosmosQA1500, OpenBookQA1000, SciQ1998, HellaSwag1500, RACE1500. '
        'Identical IDs, prompts, preprocessing, evaluator, canonical base and initial LoRA. Llama-3.2-1B frozen BF16 base; '
        'q_proj/v_proj, r8, alpha32, dropout0.05, bias none. Ten full-participation rounds, source/client weights0.2, '
        'one complete epoch per client/round, batch1, accumulation8, lr1e-4, original AdamW betas0.9/0.999 eps1e-8 weight_decay0.01 '
        'reset once per client epoch. Original client_round_seed(42,source_id0..4,round-1). Each method25000 updates/200000 visits. '
        'One global model evaluated on five full tests after every aggregation.','',
        '## Implementation and persistence audit','',
        'Baseline algorithm files are unchanged (minimal algorithm diff: empty). New plumbing only binds the existing custom datasets/model, '
        'records FedRot training loss, restores the complete global start before every client, saves all mutable target backbone weights '
        'alongside A/B for FedEx/FedMomentum, and preserves aggregation RNG across interruptions. '
        'The canonical model is referenced by its frozen source hashes; unchanged backbone tensors need not be duplicated. '
        'Snapshots are complete necessary function states, not adapter-only exports.','']
    for m in data.METHODS:
        lines += ['- '+data.NAMES_METHOD[m]+': '+json.dumps(data.config()['algorithms'][m])+'.']
        for warning in WARNINGS[m]: lines += ['  Implementation warning: '+warning]
    lines += ['', 'In particular, this report evaluates the literal provided FedEx implementation; product construction is verified, '
        'but it must not be described as a proven exact full-function baseline at alpha/r=4. No scaling fix is silently applied.','',
        '## Exact commands','', '```powershell', *[data.command(m) for m in data.METHODS], '```','',
        'Interpreter used: `D:/sofrware/anaconda3/envs/wzm/python.exe` (UTF-8). Serial queue: `scripts/run_fedlora_baseline_sequence.py`.','',
        '## Shared best-Macro comparison','',
        '| Method | CosmosQA | OpenBookQA | SciQ | HellaSwag | RACE | Peak Macro | Best Round | Delta Macro vs FedAvg (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| FedAvg (reused) | '+' | '.join('%.2f'%(base[d+'_accuracy']*100) for d in DOMAINS)+' | %.2f | 10 | 0.00 |'%(base['macro_accuracy']*100)]
    results={}; complete=True
    for m in data.METHODS:
        folder=data.output(m); path=folder/'result.json'
        if not (folder/'completed.txt').exists():
            complete=False; lines.append('| '+data.NAMES_METHOD[m]+' | pending | pending | pending | pending | pending | pending | pending | pending |'); continue
        result=data.read(path); assert data.read(folder/'protocol_audit.json')['passed']
        scores=result['best_metrics']; results[m]=result
        lines.append('| '+data.NAMES_METHOD[m]+' | '+' | '.join('%.2f'%(scores[d+'_accuracy']*100) for d in DOMAINS)+
            ' | %.2f | %d | %+.2f |'%(scores['macro_accuracy']*100,result['best_round'],(scores['macro_accuracy']-base['macro_accuracy'])*100))
    lines += ['', 'BestRound is selected by the unrounded five-source test Macro (earliest tie); all five task values come from that single shared checkpoint. '
        'Per-task oracle peaks are stored only as diagnostics and never combined into the main Macro.','',
        '## Task deltas at the shared best round (pp)','',
        '| Method | '+' | '.join(data.NAMES[d] for d in DOMAINS)+' | Macro |','|---|'+'---:|'*6]
    for m,result in results.items():
        scores=result['best_metrics']
        lines.append('| '+data.NAMES_METHOD[m]+' | '+' | '.join('%+.2f'%((scores[d+'_accuracy']-base[d+'_accuracy'])*100) for d in DOMAINS)+
            ' | %+.2f |'%((scores['macro_accuracy']-base['macro_accuracy'])*100))
    for m,result in results.items():
        folder=data.output(m); records=data.read(folder/'round_trajectory.json')
        lines += ['', '## '+data.NAMES_METHOD[m]+' full trajectory','',
            '| Round | '+' | '.join(data.NAMES[d] for d in DOMAINS)+' | Macro |','|---:|'+'---:|'*6]
        for r in records:
            scores=r['metrics']; lines.append('| '+str(r['round'])+' | '+' | '.join('%.2f'%(scores[d+'_accuracy']*100) for d in DOMAINS)+' | %.2f |'%(scores['macro_accuracy']*100))
        lines += ['', f"Final round {result['final_round']} Macro: {result['final_metrics']['macro_accuracy']*100:.2f}%. "
            f"Best round {result['best_round']} Macro: {result['best_metrics']['macro_accuracy']*100:.2f}%.",
            '', 'Best complete checkpoint: `'+result['best_checkpoint']+'`; final complete checkpoint: `'+result['final_checkpoint']+'`. '
            'Each checkpoint is accompanied by predictions, choice log probabilities and complete function/adapter/backbone provenance.']
    lines += ['', '## Warnings and errors','']
    for m in data.METHODS:
        path=data.output(m)/'status.json'
        if path.exists() and data.read(path).get('stage')=='failed': lines += ['- '+m+': '+data.read(path).get('error','unknown failure')]
        for warning in WARNINGS[m]: lines += ['- '+m+': '+warning]
    lines += ['', 'All comparisons are descriptive for one fixed seed and literal repository implementations; no mechanism or causal conclusion is inferred.']
    path=ROOT/'reports/fedlora_baselines_5source_seed42.md'; path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    data.dump(MASTER/'results_summary.json',dict(complete=complete,reference=reference,methods=results,report=str(path),warnings=WARNINGS))
    return path,complete


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alg',choices=data.METHODS)
    parser.add_argument('--suffix')
    parser.add_argument('--lam',type=float,default=.5)
    parser.add_argument('--s',type=float,default=2.)
    parser.add_argument('--residual_threshold',type=float,default=.9999)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    assert (args.lam,args.s,args.residual_threshold)==(.5,2.,.9999),'Default baseline parameters only'
    data.prepare()
    if args.report: report(); return
    assert args.alg and args.suffix==data.suffix(args.alg),'Explicit canonical method/suffix required'
    output=data.output(args.alg)/'smoke' if args.smoke else data.output(args.alg)
    output.mkdir(parents=True,exist_ok=True)
    with (data.output(args.alg)/'run.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: raise SystemExit('This baseline is already running')
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            if not args.smoke:
                smoke=data.read(data.output(args.alg)/'smoke/protocol_audit.json')
                assert smoke['passed'] and smoke['smoke']
                assert data.read(data.output(args.alg)/'smoke/protocol.json')['baseline_runner_sha256']==data.sha(Path(__file__))
            ex=Experiment(args.alg,args.smoke); ex.run()
            data.dump(output/'status.json',dict(stage='complete',pid=os.getpid(),updated_at=shared.now()))
        except BaseException as error:
            data.dump(output/'status.json',dict(stage='failed',pid=os.getpid(),updated_at=shared.now(),error=str(error),traceback=traceback.format_exc()))
            shared.status('failed',method=args.alg,smoke=args.smoke,error=str(error))
            traceback.print_exc(); raise SystemExit(1)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__': main()
