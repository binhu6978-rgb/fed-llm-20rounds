"""Portable serial continuation: original four methods, absolute rounds 11-20."""
import argparse
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
METHODS=('fedexlora','fedrotlora','flexlora','fedmomentum')
NAMES=dict(zip(METHODS,('FedEx-LoRA','FedRot-LoRA','FlexLoRA','FedMomentum')))
DOMAINS=('cosmosqa','openbookqa','sciq','hellaswag','race')
COUNTS=dict(zip(DOMAINS,(1500,1000,1998,1500,1500)))
MUTATORS=('fedexlora','fedmomentum')
MASTER=ROOT/'outputs/fedlora20_server_seed42'
MANIFEST=ROOT/'server_fedlora20_manifest.json'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def relative(path): return str(Path(path).relative_to(ROOT)).replace('\\','/')
def portable_path(text): return ROOT/str(text).replace('\\','/')
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)
def save(path,state):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(state,temp);os.replace(temp,path)
def load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)
def cpu(state): return {k:v.detach().cpu().clone() for k,v in state.items()}
def adapter(model): return cpu({k:v for k,v in model.state_dict().items() if 'lora_' in k})
def tensor_hash(state):
    import torch
    h=hashlib.sha256()
    for key in sorted(state):
        v=state[key].detach().cpu().contiguous()
        h.update(json.dumps([key,str(v.dtype),list(v.shape)]).encode())
        h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def function_hash(state):
    return tensor_hash({**{'adapter/'+k:v for k,v in state['lora'].items()},
                        **{'backbone/'+k:v for k,v in state['base_weights'].items()}})
def status(path,stage,**kw): dump(path,dict(stage=stage,pid=os.getpid(),updated_at=now(),**kw))


@contextmanager
def file_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if f.tell()==0: f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def rng_state():
    import torch,numpy as np
    n=np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),
                numpy=dict(name=n[0],keys=n[1].tolist(),position=n[2],has_gauss=n[3],cached_gaussian=n[4]))
def restore_rng(state):
    import torch,numpy as np
    torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state_all(state['torch_cuda'])
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))


def validate_source(method):
    source=ROOT/f'outputs/fedlora_baselines_5source_seed42/baseline_{method}_5source_seed42'
    state=load(source/'resume.pt');protocol=read(source/'protocol.json')
    assert state['completed_round']==10 and not state['pending'] and state['pending_global'] is None
    records=state['records'];assert len(records)==10 and [r['round'] for r in records]==list(range(1,11))
    assert records==read(source/'round_trajectory.json')
    assert function_hash(state['global_state'])==records[-1]['global_function_state_sha256']
    assert function_hash(load(source/'round_10/global/complete_state.pt'))==function_hash(state['global_state'])
    assert read(source/'protocol_audit.json')['passed'] and read(source/'protocol_audit.json')['rounds']==10
    assert bool(state['global_state']['base_weights'])==(method in MUTATORS)
    assert state['global_state']['canonical_initial_sha256']==protocol['initial_lora_sha256']
    assert protocol['algorithm_parameters']==dict(fedexlora={},fedrotlora={'lam':.5},flexlora={'s':2},fedmomentum={'residual_threshold':.9999})[method]
    previous=None
    for record in records:
        if previous is not None: assert record['start_function_hash']==previous
        assert all(x['start_function_hash']==record['start_function_hash'] for x in record['clients'].values())
        assert all(x['budget']['samples']==4000 and x['budget']['optimizer_updates']==500 for x in record['clients'].values())
        previous=record['global_function_state_sha256']
    return source,state,protocol


def preflight():
    assert MANIFEST.exists(), 'Run scripts/package_fedlora20_server.py locally first; upload its bundle.'
    m=read(MANIFEST)
    assert m['methods']==list(METHODS) and m['target_rounds']==20
    for path in m['files']:
        assert portable_path(path).is_file(), 'Required transfer file missing: '+path
    from scripts import cosmosqa_five_data as data
    cfg=data.config();prepared=ROOT/cfg['prepared_dir'];manifest=read(prepared/'manifest.json')
    for path in manifest['files']:
        assert portable_path(path).is_file(), 'Required dataset file missing: '+path
    for d in DOMAINS:
        train=rows(prepared/'train'/f'{d}.jsonl');test=rows(prepared/'test'/f'{d}.jsonl')
        assert len(train)==4000 and len(test)==COUNTS[d]
        assert [data.native_key(r) for r in train]==manifest['selected_ids'][d]['train']
        assert [data.native_key(r) for r in test]==manifest['selected_ids'][d]['test']
        assert not {data.qa_fingerprint(r) for r in train}&{data.qa_fingerprint(r) for r in test}
    for method in METHODS:
        source,state,protocol=validate_source(method)
    packages={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','numpy','datasets','safetensors','PyYAML','accelerate')}
    MASTER.mkdir(parents=True,exist_ok=True)
    dump(MASTER/'transfer_check.json',dict(passed=True,file_hash_validation=False,methods=list(METHODS),
         source_rounds=10,target_rounds=20,train=20000,test=7498,packages=packages,python=sys.version,platform=sys.platform,
         source_environment=m['source_environment'],cross_machine_bitwise_identity_guaranteed=False))
    print('TRANSFER CHECK PASSED (file hashes disabled): four complete round10 states; 20000 train / 7498 tests',flush=True)
    print('SERVER ENVIRONMENT',packages,flush=True)


class ProgressLoader:
    def __init__(self,loader,ex,domain): self.loader,self.ex,self.domain=loader,ex,domain;self.drop_last=loader.drop_last
    def __len__(self): return len(self.loader)
    def __iter__(self):
        for count,batch in enumerate(self.loader,1):
            yield batch
            if count%200==0 or count==len(self):
                print('TRAIN',self.ex.method,self.ex.current_round,self.domain,count,'/',len(self),flush=True)


class RecordingTrainer:
    def __init__(self,delegate,client): self.delegate,self.client=delegate,client
    def train(self,model):
        result=self.delegate.train(model);self.client.last_train_loss=result;return result
    def __getattr__(self,k): return getattr(self.delegate,k)


class Experiment:
    def __init__(self,method,smoke=False):
        import torch
        from peft import get_peft_model
        from scripts import cosmosqa_five_data as data
        from utils import cosmosqa_five_mcq as encoding
        from utils.model_utils import load_model,load_tokenizer,load_lora_config
        from utils.seed_utils import set_global_seed
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'Requires a CUDA GPU with BF16 support'
        self.method,self.smoke=method,smoke
        self.output=MASTER/(('smoke_'+method) if smoke else method)
        self.output.mkdir(parents=True,exist_ok=True)
        self.target=12 if smoke else 20
        self.source,self.source_state,self.source_protocol=validate_source(method)
        cfg=data.config();self.cfg=cfg;self.module=importlib.import_module('alg.'+method)
        self.args=SimpleNamespace(model_path=str(ROOT/cfg['model_path']),model='llama32_1b_base',device=0,mcq=True,
              task_type='CAUSAL_LM',seed=42,bs=1,grad_accum=8,lr=1e-4,epoch=1,step=0,cn=5,sr=1.,rnd=20,
              lora_rank=8,lora_alpha=32,lora_dropout=.05,eval_batch_size=8,mcq_domains=list(DOMAINS),
              mcq_dataset_dir=str(ROOT/cfg['prepared_dir']),suffix=str(self.output),alg=method,lam=.5,s=2.,
              residual_threshold=.9999,deterministic=True)
        set_global_seed(42,device=0,deterministic=True)
        self.tokenizer=load_tokenizer(self.args)
        prepared=ROOT/cfg['prepared_dir'];self.manifest=read(prepared/'manifest.json')
        full={d:rows(prepared/'train'/f'{d}.jsonl') for d in DOMAINS}
        self.train_rows={d:(full[d][:8] if smoke else full[d]) for d in DOMAINS}
        self.datasets={d:encoding.TrainingDataset(self.train_rows[d],self.tokenizer) for d in DOMAINS}
        self.test_rows={d:rows(prepared/'test'/f'{d}.jsonl') for d in DOMAINS}
        self.test_encoded={d:[(encoding.encode_prompt(self.tokenizer,r),[encoding.continuation_ids(self.tokenizer,r,l) for l in 'ABCD']) for r in self.test_rows[d]] for d in DOMAINS}
        self.counts={d:len(self.datasets[d]) for d in DOMAINS}
        self.supervised={d:sum(sum(v!=-100 for v in x['labels']) for x in self.datasets[d].encoded) for d in DOMAINS}
        if not smoke:
            stats=read(prepared/'sequence_statistics.json')
            for d in DOMAINS:
                assert sum(len(x['input_ids']) for x in self.datasets[d].encoded)==stats[d]['train']['token_total']
                assert self.supervised[d]==stats[d]['train']['supervised_token_total']
        encoding.clear_cache()
        set_global_seed(42,device=0,deterministic=True)
        self.model=get_peft_model(load_model(self.args),load_lora_config(self.args))
        self.initial=load(ROOT/cfg['initial_lora']);self.model.load_state_dict(self.initial,strict=False)
        self.base_keys=tuple(k.replace('.lora_A.default.weight','.base_layer.weight') for k in self.initial if 'lora_A' in k)
        self.canonical_base=cpu({k:self.model.state_dict()[k] for k in self.base_keys})
        self.canonical_base_hash=tensor_hash(self.canonical_base)
        assert self.canonical_base_hash==self.source_protocol['canonical_mutable_backbone_sha256']
        assert tensor_hash(adapter(self.model))==self.source_protocol['canonical_initial_tensor_sha256']
        assert all('lora_' in k for k,v in self.model.named_parameters() if v.requires_grad)
        self.shape_check(self.initial)
        protocol=dict(method=method,smoke=smoke,source_round=10,target_round=self.target,additional_rounds=self.target-10,
             original_protocol=self.source_protocol,
             canonical_base_hash=self.canonical_base_hash,train_counts=self.counts,test_counts=COUNTS,
             original_algorithms_unchanged=True,local_model_evaluation='raw training end before upload alignment',
             absolute_round_seed=True,optimizer_reset_each_client_epoch=True,weights={d:.2 for d in DOMAINS})
        path=self.output/'protocol.json'
        if path.exists():
            old=read(path)
            # Old receipts contain file fingerprints. They no longer participate in resume checks.
            old={k:v for k,v in old.items() if k in protocol}
            assert old==protocol,'Incompatible method resume'
        else: dump(path,protocol)
        self.local_previous={}
        for r in read(ROOT/'outputs/fedlora_baselines_5source_seed42/local_before_upload/round_metrics.json'):
            if r['method']==method: self.local_previous[r['round']]=r

    def shape_check(self,lora):
        import torch
        assert lora.keys()==self.initial.keys()
        assert all(v.shape==self.initial[k].shape and v.dtype==self.initial[k].dtype and torch.isfinite(v).all() for k,v in lora.items())
        assert all((v.shape[0] if 'lora_A' in k else v.shape[1])==8 for k,v in lora.items())
    def base_hash(self): return tensor_hash(cpu({k:self.model.state_dict()[k] for k in self.base_keys}))
    def capture(self):
        return dict(format_version=1,lora=adapter(self.model),base_weights=cpu({k:self.model.state_dict()[k] for k in self.base_keys}) if self.method in MUTATORS else {},
                    canonical_model=self.cfg['model_path'],canonical_initial_sha256=self.source_protocol['initial_lora_sha256'])
    def restore(self,state):
        self.shape_check(state['lora']);base=state['base_weights'] if self.method in MUTATORS else self.canonical_base
        assert set(base)==set(self.base_keys)
        self.model.load_state_dict(base,strict=False);self.model.load_state_dict(state['lora'],strict=False)
        assert function_hash(self.capture())==function_hash(state)
    def bind(self,start):
        from alg.base import BaseServer,BaseClient
        from utils.train_utils import Trainer
        server=self.module.Server.__new__(self.module.Server)
        BaseServer.__init__(server,self.args,[])
        server.model,server.round,server.sample_rate,server.wall_clock_time=self.model,10,1.,0.
        if self.method=='flexlora': server.r,server.s=8,2.
        if self.method=='fedmomentum': server.tau=.9999
        ex=self;algorithm_client=self.module.Client
        clients=[]
        for domain in DOMAINS:
            class Client(algorithm_client):
                def __init__(client,d):
                    args=copy.copy(ex.args)
                    BaseClient.__init__(client,DOMAINS.index(d),args)
                    client.server,client.dataset=server,{'train':ex.datasets[d]}
                    client.tokenizer,client.lora,client.delay=ex.tokenizer,{},1.
                    client.evaluator,client.last_train_loss=None,None
                    delegate=Trainer(args,client.dataset,client)
                    assert not delegate.train_loader.drop_last and len(delegate.train_loader)==ex.counts[d]
                    delegate.train_loader=ProgressLoader(delegate.train_loader,ex,d)
                    client.trainer=RecordingTrainer(delegate,client)
            clients.append(Client(domain))
        server.clients=clients
        return server,clients
    def score(self,directory,domains,t,state,checkpoint):
        from utils.mcq_eval import MCQEvaluator
        directory.mkdir(parents=True,exist_ok=True)
        assert function_hash(self.capture())==function_hash(state)
        identity=dict(function_state_sha256=function_hash(state),checkpoint=relative(checkpoint),
                      domains=list(domains),round=t,
                      canonical_base_sha256=self.canonical_base_hash,backbone_included=self.method in MUTATORS)
        path=directory/'evaluation_provenance.json'
        if path.exists():
            old={k:v for k,v in read(path).items() if k in identity}
            assert old==identity,'Cached evaluation has a different complete model'
        else: dump(path,identity)
        output=directory/'evaluation';output.mkdir(parents=True,exist_ok=True)
        self.model.eval()
        if (output/'metrics.jsonl').exists():
            metric=rows(output/'metrics.jsonl');assert len(metric)==1 and metric[0]['round']==t
            result={k:v for k,v in metric[0].items() if k!='round'}
        else:
            evaluator=MCQEvaluator.__new__(MCQEvaluator)
            evaluator.args,evaluator.tokenizer,evaluator.domains=self.args,self.tokenizer,tuple(domains)
            evaluator.rows={d:self.test_rows[d] for d in domains};evaluator.encoded={d:self.test_encoded[d] for d in domains};evaluator.output=output
            result=evaluator.evaluate(self.model,t)
        with (output/f'round_{t:02d}_predictions.csv').open(encoding='utf-8',newline='') as f: predictions=list(csv.DictReader(f))
        for d in domains:
            current=[r for r in predictions if r['domain']==d]
            assert len(current)==len({r['id'] for r in current})==COUNTS[d]
            assert {r['id'] for r in current}=={str(r['id']) for r in self.test_rows[d]}
            gold={str(r['id']):r['correct_answer'] for r in self.test_rows[d]}
            assert all(r['gold']==gold[r['id']] and int(r['correct'])==int(r['gold']==r['prediction']) for r in current)
            assert abs(sum(int(r['correct']) for r in current)/len(current)-result[d+'_accuracy'])<1e-12
        assert function_hash(self.capture())==identity['function_state_sha256']
        dump(directory/'result.json',result);return result

    def report(self,records):
        best=max(records,key=lambda r:(r['metrics']['macro_accuracy'],-r['round']))
        imported=self.source_state['records']
        assert records[:10]==imported
        combined=[]
        for r in records:
            local=(self.local_previous[r['round']]['local_own_domain_mean'] if r['round']<=10 else sum(x['local_after'][d+'_accuracy'] for d,x in r['clients'].items())/5)
            combined.append(dict(round=r['round'],local_mean=local,global_macro=r['metrics']['macro_accuracy'],
                                 local_minus_global_pp=100*(local-r['metrics']['macro_accuracy'])))
        dump(self.output/'round_trajectory.json',records);dump(self.output/'local_global_rounds.json',combined)
        with (self.output/'local_global_rounds.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
        dump(self.output/'result.json',dict(method=self.method,smoke=self.smoke,completed_round=records[-1]['round'],best_round=best['round'],best_metrics=best['metrics'],
             best_checkpoint=str(best['checkpoint']).replace('\\','/'),final_metrics=records[-1]['metrics'],
             round10_metrics=records[9]['metrics'],delta_best_vs_round10_pp=100*(best['metrics']['macro_accuracy']-records[9]['metrics']['macro_accuracy'])))
        lines=['# '+NAMES[self.method]+' continuation', '', 'Local mean / global Macro use identical full original tests. '+('SMOKE ONLY: train8/client, not formal results.' if self.smoke else 'Round1–10 reused; new full rounds11–20. Total training budget doubles.'),'',
               '| Round | Local own-domain mean (%) | Global Macro (%) | Local − Global (pp) |','|---:|---:|---:|---:|']
        for r in combined: lines.append('| %d | %.2f | %.2f | %+.2f |'%(r['round'],r['local_mean']*100,r['global_macro']*100,r['local_minus_global_pp']))
        lines+=['',f"Best shared round: {best['round']}; Macro {best['metrics']['macro_accuracy']*100:.2f}%.",'',
                 'Original algorithm implementation warnings remain unchanged; see source protocol warnings. Imported old rounds have provenance in the transfer manifest.']
        (self.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def run(self):
        import torch
        resume=self.output/'resume.pt'
        if resume.exists(): state=load(resume)
        else:
            state=copy.deepcopy(self.source_state)
            state.update(pending={},pending_global=None,aggregation_rng=None)
            save(resume,state)
        assert state['records'][:10]==self.source_state['records']
        exposure=self.output/'exposure.jsonl'
        if exposure.exists():
            kept=[r for r in rows(exposure) if r['round']<=state['completed_round'] or (r['round']==state['completed_round']+1 and r['client'] in state['pending'])]
            exposure.write_text(''.join(json.dumps(r)+'\n' for r in kept),encoding='utf-8')
        self.restore(state['global_state']);server,clients=self.bind(state['global_state']);self.report(state['records'])
        for t in range(state['completed_round']+1,self.target+1):
            self.current_round=t;server.round=t-1;server.sample()
            assert [c.id for c in server.sampled_clients]==list(range(5))
            self.restore(state['global_state']);start=function_hash(state['global_state']);start_base=self.base_hash()
            server.global_lora={k:v.to(self.model.device).clone() for k,v in state['global_state']['lora'].items()}
            directory=self.output/f'round_{t:02d}'
            if state['pending_global'] is None:
                for client,d in zip(clients,DOMAINS):
                    current=directory/f'client_{d}';raw_path=current/'raw_lora.pt';upload_path=current/'upload_lora.pt'
                    if client.id not in state['pending']:
                        self.restore(state['global_state']);assert self.base_hash()==start_base
                        status(self.output/'status.json','local_training',method=self.method,round=t,domain=d,smoke=self.smoke)
                        client.run(self.model)
                        assert client.last_seen_count==self.counts[d] and client.last_optimizer_updates==(self.counts[d]+7)//8
                        assert self.base_hash()==start_base,'Local trainer mutated frozen backbone'
                        raw,upload=adapter(self.model),cpu(client.lora);self.shape_check(raw);self.shape_check(upload)
                        save(raw_path,raw);save(upload_path,upload)
                        state['pending'][client.id]=dict(lora=upload,start_function_hash=start,start_base_hash=start_base,
                            raw_hash=tensor_hash(raw),upload_hash=tensor_hash(upload),
                            budget=dict(train_loss=client.last_train_loss,samples=client.last_seen_count,optimizer_updates=client.last_optimizer_updates))
                        state['aggregation_rng']=rng_state();save(resume,state)
                    item=state['pending'][client.id]
                    assert item['start_function_hash']==start and item['start_base_hash']==start_base
                    assert tensor_hash(load(upload_path))==item['upload_hash'] and tensor_hash(load(raw_path))==item['raw_hash']
                    if 'local_after' not in item:
                        local_state=dict(state['global_state'],lora=load(raw_path));self.restore(local_state)
                        status(self.output/'status.json','local_evaluation',method=self.method,round=t,domain=d,smoke=self.smoke)
                        item['local_after']=self.score(current,[d],t,local_state,raw_path);save(resume,state)
                    client.lora={k:v.to(self.model.device).clone() for k,v in item['lora'].items()}
                assert len(state['pending'])==5
                self.restore(state['global_state']);restore_rng(state['aggregation_rng'])
                status(self.output/'status.json','aggregation',method=self.method,round=t,smoke=self.smoke)
                server.aggregate()
                value=self.capture();self.shape_check(value['lora'])
                assert all(torch.isfinite(v).all() for v in value['base_weights'].values())
                if self.method not in MUTATORS: assert self.base_hash()==self.canonical_base_hash
                state['pending_global']=value;save(directory/'global/complete_state.pt',value);save(resume,state)
            value=state['pending_global'];self.restore(value);checkpoint=directory/'global/complete_state.pt'
            assert function_hash(load(checkpoint))==function_hash(value)
            status(self.output/'status.json','global_evaluation',method=self.method,round=t,smoke=self.smoke)
            scores=self.score(directory/'global',DOMAINS,t,value,checkpoint)
            state['records'].append(dict(round=t,metrics=scores,checkpoint=relative(checkpoint),global_function_state_sha256=function_hash(value),
                 lora_tensor_sha256=tensor_hash(value['lora']),start_function_hash=start,start_base_hash=start_base,weights={d:.2 for d in DOMAINS},
                 clients={d:{k:v for k,v in state['pending'][i].items() if k!='lora'} for i,d in enumerate(DOMAINS)}))
            state.update(completed_round=t,global_state=value,pending={},pending_global=None,aggregation_rng=None)
            save(resume,state);self.report(state['records'])
            print('ROUND',self.method,t,'GLOBAL MACRO',scores['macro_accuracy'],flush=True)
        self.audit(state)
        (self.output/'completed.txt').write_text('All continuation rounds and local/global evaluations audited.\n',encoding='utf-8')
        status(self.output/'status.json','complete',method=self.method,round=self.target,smoke=self.smoke)

    def audit(self,state):
        from scripts import cosmosqa_five_data as data
        records=state['records'];new=records[10:];assert len(new)==self.target-10
        visits=rows(self.output/'exposure.jsonl')
        assert len(visits)==5*len(new) and len({(r['round'],r['client']) for r in visits})==len(visits)
        for r in visits:
            d=DOMAINS[r['client']]
            assert set(r['ids'])=={data.native_key(x) for x in self.train_rows[d]}
            assert r['samples']==len(r['ids'])==len(set(r['ids']))==self.counts[d]
            assert r['optimizer_updates']==(self.counts[d]+7)//8 and r['supervised_tokens']==self.supervised[d]
        prior=self.source_state['records'][-1]['global_function_state_sha256']
        for r in new:
            assert r['start_function_hash']==prior
            assert all(x['start_function_hash']==prior for x in r['clients'].values())
            p=portable_path(r['checkpoint']);assert function_hash(load(p))==r['global_function_state_sha256']
            assert read(p.parent/'evaluation_provenance.json')['function_state_sha256']==r['global_function_state_sha256']
            assert read(p.parent/'result.json')==r['metrics']
            for d in DOMAINS:
                current=p.parent.parent/f'client_{d}'
                assert read(current/'result.json')==r['clients'][d]['local_after']
                assert tensor_hash(load(current/'raw_lora.pt'))==r['clients'][d]['raw_hash']
                assert tensor_hash(load(current/'upload_lora.pt'))==r['clients'][d]['upload_hash']
            prior=r['global_function_state_sha256']
        dump(self.output/'protocol_audit.json',dict(passed=True,smoke=self.smoke,reused_rounds=10,new_rounds=len(new),
             new_sample_visits=sum(r['samples'] for r in visits),new_optimizer_updates=sum(r['optimizer_updates'] for r in visits),
             total_formal_updates=None if self.smoke else 50000,total_formal_visits=None if self.smoke else 400000,
             complete_backbone_restored=True,absolute_round_seed=True,raw_local_test_each_round=True,full_test_sets=True,
             original_algorithm_unchanged=True,file_hash_validation=False,previous_global_is_next_start=True))


def combined_report():
    lines=['# Four-method continuation to 20 rounds','',
           'All methods reuse the original ten rounds; additional ten full rounds double the original training budget. Compare shared best checkpoints and fixed Round20 separately.','',
           '| Method | Completed | Best round | Best global Macro (%) | Latest global Macro (%) | Latest local mean (%) |','|---|---:|---:|---:|---:|---:|']
    reference=ROOT/'outputs/fedavg20_five_client4000_seed42/best_result.json'
    if reference.exists():
        r=read(reference);trajectory=read(reference.parent/'round_diagnostics.json')['rounds']
        local=sum(trajectory[-1]['local_after'][d+'_accuracy'] for d in DOMAINS)/5
        lines.append('| FedAvg20 (reused reference) | 20/20 | %d | %.2f | %.2f | %.2f |'%(r['best_round'],r['best_metrics']['macro_accuracy']*100,r['latest_metrics']['macro_accuracy']*100,local*100))
    for method in METHODS:
        folder=MASTER/method
        if not (folder/'result.json').exists(): lines.append('| '+NAMES[method]+' | pending | — | — | — | — |');continue
        r=read(folder/'result.json');local=read(folder/'local_global_rounds.json')[-1]
        lines.append('| %s | %d/20 | %d | %.2f | %.2f | %.2f |'%(NAMES[method],r['completed_round'],r['best_round'],r['best_metrics']['macro_accuracy']*100,r['final_metrics']['macro_accuracy']*100,local['local_mean']*100))
    for method in METHODS:
        p=MASTER/method/'local_global_rounds.json'
        if p.exists():
            lines+=['','## '+NAMES[method],'','| Round | Local mean (%) | Global Macro (%) | Gap (pp) |','|---:|---:|---:|---:|']
            for r in read(p): lines.append('| %d | %.2f | %.2f | %+.2f |'%(r['round'],r['local_mean']*100,r['global_macro']*100,r['local_minus_global_pp']))
    (MASTER/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def child(method,smoke,args):
    stage=('smoke_' if smoke else '')+method
    folder=MASTER/stage
    # A prior queue can die while its worker survives. Wait for its lock, never run it twice.
    while True:
        try:
            with file_lock(folder/'run.lock'): pass
            break
        except OSError as error:
            if error.errno not in (11,13,36): raise
            status(MASTER/'sequence_status.json','waiting_existing_worker',method=method,smoke=smoke)
            time.sleep(10)
    if (folder/'completed.txt').exists():
        assert read(folder/'protocol_audit.json')['passed']
        return
    logdir=MASTER/'logs';logdir.mkdir(exist_ok=True)
    command=[sys.executable,'-X','utf8','-u',str(Path(__file__)), '--worker',method]
    if smoke: command+=['--smoke']
    if args.gpu is not None: command+=['--gpu',args.gpu]
    with (logdir/(stage+'.log')).open('a',encoding='utf-8') as f:
        f.write('\nSTART '+now()+'\n');f.flush()
        process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                   env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'))
        status(MASTER/'sequence_status.json','running',method=method,smoke=smoke,child_pid=process.pid,log=relative(logdir/(stage+'.log')))
        code=process.wait()
    if code: raise RuntimeError(stage+' failed; remaining methods stopped. See '+str(logdir/(stage+'.log')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='CPU-only transfer/input validation, no training')
    parser.add_argument('--smoke',action='store_true',help='Only isolated two-round train8/client/full-test GPU checks for all four')
    parser.add_argument('--worker',choices=METHODS,help=argparse.SUPPRESS)
    parser.add_argument('--gpu',help='Optional CUDA_VISIBLE_DEVICES; otherwise preserve scheduler environment')
    args=parser.parse_args()
    if args.gpu is not None: os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    MASTER.mkdir(parents=True,exist_ok=True)
    if args.worker:
        folder=MASTER/(('smoke_' if args.smoke else '')+args.worker)
        with file_lock(folder/'run.lock'):
            try:
                if (folder/'completed.txt').exists() and read(folder/'protocol_audit.json')['passed']: return
                # Parent verified the transfer. Direct worker invocations must also verify it.
                preflight()
                ex=Experiment(args.worker,args.smoke);ex.run()
            except BaseException as error:
                status(folder/'status.json','failed',error=str(error),traceback=traceback.format_exc());raise
        return
    with file_lock(MASTER/'sequence.lock'):
        try:
            status(MASTER/'sequence_status.json','checking_inputs');preflight()
            if args.check: status(MASTER/'sequence_status.json','checked');return
            for method in METHODS:
                folder=MASTER/('smoke_'+method)
                if not (folder/'completed.txt').exists(): child(method,True,args)
                assert read(folder/'protocol_audit.json')['passed']
            if args.smoke: status(MASTER/'sequence_status.json','smoke_complete');return
            combined_report()
            for method in METHODS:
                folder=MASTER/method
                if not (folder/'completed.txt').exists(): child(method,False,args)
                assert read(folder/'protocol_audit.json')['passed'];combined_report()
            status(MASTER/'sequence_status.json','complete',report=relative(MASTER/'report.md'))
        except BaseException as error:
            status(MASTER/'sequence_status.json','failed',error=str(error),traceback=traceback.format_exc())
            traceback.print_exc();raise


if __name__=='__main__': main()
