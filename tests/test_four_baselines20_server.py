"""Actual algorithm tests: interruption must not retrain clients or double-merge a backbone."""
import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import MethodType,SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from peft import LoraConfig,get_peft_model
from scripts import run_four_baselines20_server as r


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.q_proj=torch.nn.Linear(24,24,bias=False);self.v_proj=torch.nn.Linear(24,24,bias=False)
    @property
    def device(self): return next(self.parameters()).device


class Loader:
    drop_last=False
    def __len__(self): return 8


class RecoveryTests(unittest.TestCase):
    def exercise(self,method,crash):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';source=root/'source';source.mkdir();out.mkdir()
            torch.manual_seed(42)
            model=get_peft_model(Tiny(),LoraConfig(r=8,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none'))
            ex=r.Experiment.__new__(r.Experiment)
            ex.method,ex.smoke,ex.output,ex.source,ex.target=method,True,out,source,12
            ex.model=model;ex.tokenizer=None;ex.module=r.importlib.import_module('alg.'+method)
            ex.initial=r.adapter(model);ex.cfg={'model_path':'canonical'};ex.source_protocol={'initial_lora_sha256':'canonical'}
            ex.base_keys=tuple(k.replace('.lora_A.default.weight','.base_layer.weight') for k in ex.initial if 'lora_A' in k)
            ex.canonical_base=r.cpu({k:model.state_dict()[k] for k in ex.base_keys});ex.canonical_base_hash=r.tensor_hash(ex.canonical_base)
            ex.args=SimpleNamespace(seed=42,lora_rank=8,s=2.,lam=.5,residual_threshold=.9999)
            ex.datasets={d:[{'dataset':d,'id':i} for i in range(8)] for d in r.DOMAINS}
            ex.train_rows=ex.datasets;ex.counts={d:8 for d in r.DOMAINS};ex.supervised={d:8 for d in r.DOMAINS}
            initial=ex.capture();digest=r.function_hash(initial)
            records=[dict(round=t,metrics={**{d+'_accuracy':.4 for d in r.DOMAINS},'macro_accuracy':.4},
                          checkpoint='imported.pt',global_function_state_sha256=digest) for t in range(1,11)]
            ex.source_state=dict(completed_round=10,global_state=initial,pending={},pending_global=None,aggregation_rng=None,records=records)
            ex.local_previous={t:dict(local_own_domain_mean=.5) for t in range(1,11)}
            calls=[];fired=[]
            class Trainer:
                def __init__(self,args,dataset,client): self.args,self.dataset,self.client=args,dataset,client;self.train_loader=Loader()
                def train(self,m):
                    from utils.seed_utils import client_round_seed
                    index=self.client.server.round;calls.append((index,self.client.id))
                    torch.manual_seed(client_round_seed(42,self.client.id,index))
                    with torch.no_grad():
                        for name,p in m.named_parameters():
                            if p.requires_grad: p.add_(torch.randn_like(p)*.02)
                    self.client.last_seen_count,self.client.last_optimizer_updates=8,1
                    with (out/'exposure.jsonl').open('a',encoding='utf-8') as f:
                        d=r.DOMAINS[self.client.id]
                        f.write(json.dumps(dict(round=index+1,client=self.client.id,samples=8,optimizer_updates=1,supervised_tokens=8,ids=[d+':'+str(i) for i in range(8)]))+'\n')
                    return .2
            def score(owner,path,domains,t,state,checkpoint):
                self.assertEqual(r.function_hash(owner.capture()),r.function_hash(state))
                is_global=len(domains)==5
                if crash and not fired and t==11 and ((crash=='global' and is_global) or (crash=='local' and domains==['sciq'])):
                    fired.append(True);raise RuntimeError('injected interruption')
                result={d+'_accuracy':.6 for d in domains};result.update(macro_accuracy=.6,worst_domain_accuracy=.6)
                r.dump(path/'result.json',result);r.dump(path/'evaluation_provenance.json',dict(function_state_sha256=r.function_hash(state)))
                return result
            ex.score=MethodType(score,ex)
            manifest=root/'manifest.json';r.dump(manifest,dict(files={}))
            with patch.object(r,'ROOT',root),patch.object(r,'MANIFEST',manifest),patch('utils.train_utils.Trainer',Trainer),\
                 patch.object(r,'rng_state',lambda:dict(cpu=torch.get_rng_state())),\
                 patch.object(r,'restore_rng',lambda value:torch.set_rng_state(value['cpu'])),redirect_stdout(StringIO()):
                if crash:
                    with self.assertRaisesRegex(RuntimeError,'injected'): ex.run()
                ex.run()
                state=r.load(out/'resume.pt');audit=r.read(out/'protocol_audit.json')
                self.assertTrue(audit['passed']);self.assertEqual(len(calls),10)
                self.assertEqual(set(calls),{(t,c) for t in (10,11) for c in range(5)})
                self.assertEqual(state['completed_round'],12)
                return r.function_hash(state['global_state'])

    def test_actual_four_algorithms_resume_before_local_and_after_aggregation(self):
        for method in r.METHODS:
            with self.subTest(method=method):
                reference=self.exercise(method,None)
                self.assertEqual(self.exercise(method,'local'),reference)
                self.assertEqual(self.exercise(method,'global'),reference)

    def test_windows_manifest_paths_resolve_to_local_root(self):
        self.assertEqual(r.portable_path('dataset\\train\\x.jsonl'),r.ROOT/'dataset/train/x.jsonl')


if __name__=='__main__': unittest.main()
