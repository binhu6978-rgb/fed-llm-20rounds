"""Use real baseline classes to validate full-function persistence and recovery."""
import copy
import importlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import MethodType,SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig,get_peft_model
from scripts import run_fedlora_baseline as r


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.q_proj=torch.nn.Linear(24,24,bias=False); self.v_proj=torch.nn.Linear(24,24,bias=False)
    @property
    def device(self): return next(self.parameters()).device
    def forward(self,x): return self.q_proj(x)+self.v_proj(x)


def model():
    torch.manual_seed(42)
    return get_peft_model(Tiny(),LoraConfig(r=8,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none'))


class AggregationTests(unittest.TestCase):
    def probe(self,method):
        module=importlib.import_module('alg.'+method); m=model()
        initial=r.shared.adapter(m); clients=[]
        for i in range(5):
            torch.manual_seed(50+i)
            lora={k:torch.randn_like(v)*.1 for k,v in initial.items()}
            clients.append(SimpleNamespace(id=i,lora=lora,dataset={'train':list(range(8))}))
        args=SimpleNamespace(lora_rank=8,s=2.,lam=.5,residual_threshold=.9999)
        server=r.bind_server(module,args,m,clients,initial); server.sampled_clients=clients
        before={k:v.clone() for k,v in m.state_dict().items() if '.base_layer.weight' in k}
        with redirect_stdout(StringIO()): server.aggregate()
        return m,initial,clients,before,server

    def test_fedex_products_and_explicit_scaling_discrepancy(self):
        m,initial,clients,before,server=self.probe('fedexlora')
        for a in (k for k in initial if 'lora_A' in k):
            b=a.replace('lora_A','lora_B'); base=a.replace('.lora_A.default.weight','.base_layer.weight')
            exact=sum(c.lora[b]@c.lora[a]/5 for c in clients)
            mean_a=sum(c.lora[a]/5 for c in clients); mean_b=sum(c.lora[b]/5 for c in clients)
            residual=exact-mean_b@mean_a
            torch.testing.assert_close(m.state_dict()[base]-before[base],residual,atol=1e-7,rtol=1e-5)
            effective=m.state_dict()[base]-before[base]+4*(server.global_lora[b]@server.global_lora[a])
            torch.testing.assert_close(4*exact-effective,3*residual,atol=1e-7,rtol=1e-5)
            self.assertGreater(torch.linalg.norm(4*exact-effective).item(),.01)

    def test_rotation_preserves_products_at_both_round_parities(self):
        from alg.fedrotlora import _align_factors
        torch.manual_seed(1); a=torch.randn(8,24); b=torch.randn(24,8)
        for t in (1,2):
            aa,bb=_align_factors(a,b,a+.1,b+.1,t,.5)
            torch.testing.assert_close(bb@aa,b@a,atol=2e-5,rtol=2e-5)

    def test_all_original_aggregations_keep_configured_rank(self):
        for method in r.data.METHODS:
            with self.subTest(method=method):
                m,initial,clients,before,server=self.probe(method)
                self.assertEqual(r.shared.adapter(m).keys(),initial.keys())
                self.assertTrue(all(v.shape==initial[k].shape for k,v in r.shared.adapter(m).items()))
                if method in r.MUTATORS: self.assertTrue(any(not torch.equal(m.state_dict()[k],v) for k,v in before.items()))
                else: self.assertTrue(all(torch.equal(m.state_dict()[k],v) for k,v in before.items()))


class RecoveryTests(unittest.TestCase):
    def exercise(self,method,crash=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/method
            ex=r.Experiment.__new__(r.Experiment)
            ex.model=model(); ex.method=method; ex.module=importlib.import_module('alg.'+method)
            ex.tokenizer=None
            ex.smoke,ex.rounds=True,2; ex.output=output; ex.outputs={method:output}
            ex.initial=r.shared.adapter(ex.model)
            ex.base_keys=tuple(k.replace('.lora_A.default.weight','.base_layer.weight') for k in ex.initial if 'lora_A' in k)
            ex.canonical_base=r.cpu_dict({k:ex.model.state_dict()[k] for k in ex.base_keys})
            ex.canonical_base_hash=r.shared.tensor_hash(ex.canonical_base)
            ex.counts={d:8 for d in r.DOMAINS}; ex.supervised_counts={d:16 for d in r.DOMAINS}
            ex.train_rows={d:[dict(dataset=d,id=str(i)) for i in range(8)] for d in r.DOMAINS}
            class Dataset:
                def __init__(self,rows): self.rows=rows
                def __len__(self): return len(self.rows)
            ex.datasets={d:Dataset(rows) for d,rows in ex.train_rows.items()}
            ex.args=SimpleNamespace(lora_rank=8,s=2.,lam=.5,residual_threshold=.9999,seed=42,bs=1,cn=5)
            ex.parameter_info={}; ex.protocol=dict(config=dict(model_path='canonical'),
                initial_lora_sha256='fixture',canonical_initial_tensor_sha256=r.shared.tensor_hash(ex.initial),
                test_counts={d:3 for d in r.DOMAINS})
            failures={crash} if crash else set(); attempts=[]; aggregate_count=[]
            class Loader:
                drop_last=False
                def __len__(self): return 8
            class Trainer:
                def __init__(self,args,dataset,client):
                    self.args,self.dataset,self.client=args,dataset['train'],client; self.train_loader=Loader()
                def train(self,m):
                    from utils.seed_utils import seed_rngs,client_round_seed
                    c=self.client; epoch=c.server.round+1
                    seed_rngs(client_round_seed(42,c.id,c.server.round)); attempts.append((epoch,c.id))
                    with torch.no_grad():
                        for k,p in m.named_parameters():
                            if 'lora_A' in k: p.add_(torch.randn_like(p)*.02)
                            if 'lora_B' in k: p.add_(torch.randn_like(p)*.01)
                    c.last_seen_count,c.last_optimizer_updates=8,1
                    row=dict(round=epoch,client=c.id,samples=8,optimizer_updates=1,supervised_tokens=16,
                        ids=[r.data.original.native_key(x) for x in self.dataset.rows])
                    with (Path(self.args.suffix)/'exposure.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                    if ('train',epoch,c.id) in failures:
                        failures.remove(('train',epoch,c.id)); raise RuntimeError('simulated interruption')
                    return .1
            def score(self,directory,t,checkpoint,state):
                if ('score',t) in failures:
                    failures.remove(('score',t)); raise RuntimeError('simulated interruption')
                value={d+'_accuracy':.5+t/100 for d in r.DOMAINS}; value.update(macro_accuracy=.5+t/100,worst_domain_accuracy=.5+t/100)
                r.data.dump(directory/'evaluation_provenance.json',dict(domains=list(r.DOMAINS),
                    global_function_state_sha256=r.function_hash(state),checkpoint_sha256=r.data.sha(checkpoint)))
                return value
            ex.score_full=MethodType(score,ex)
            original_aggregate=ex.module.Server.aggregate
            def aggregate(server):
                aggregate_count.append(server.round); return original_aggregate(server)
            with patch.object(r,'ROOT',root),patch.object(r.shared,'ROOT',root),patch.object(r.shared,'MASTER',root), \
                 patch('utils.train_utils.Trainer',Trainer),patch.object(ex.module.Server,'aggregate',aggregate),redirect_stdout(StringIO()):
                if crash:
                    with self.assertRaisesRegex(RuntimeError,'simulated'): ex.run()
                ex.run(); final=ex.capture(); count=len(attempts)
                ex.restore({**final,'lora':ex.initial}) # Intentionally disrupt adapter state before completed resume.
                ex.run(); self.assertEqual(r.function_hash(ex.capture()),r.function_hash(final))
                self.assertEqual(len(attempts),count)
                self.assertEqual(count,11 if crash and crash[0]=='train' else 10)
                self.assertEqual(len(aggregate_count),2) # Pending evaluated global is restored, not aggregated twice.
                self.assertEqual(len(r.data.rows(output/'exposure.jsonl')),10)
                if method in r.MUTATORS:
                    self.assertNotEqual(r.shared.tensor_hash(final['base_weights']),ex.canonical_base_hash)
                    ex.restore(final); ex.model.eval(); actual=ex.model(torch.ones(1,24)).detach()
                    adapter_only={**final,'base_weights':ex.canonical_base}
                    ex.restore(adapter_only); other=ex.model(torch.ones(1,24)).detach()
                    self.assertGreater(torch.linalg.norm(actual-other).item(),1e-6)
                    ex.restore(final); torch.testing.assert_close(ex.model(torch.ones(1,24)),actual)
            return r.function_hash(final)

    def test_full_states_roundtrip_all_algorithms(self):
        for method in r.data.METHODS:
            with self.subTest(method=method): self.exercise(method)

    def test_fedmomentum_rng_and_backbone_recovery_after_client_failure(self):
        self.assertEqual(self.exercise('fedmomentum'),self.exercise('fedmomentum',('train',1,3)))

    def test_mutating_global_evaluation_resume_no_double_merge(self):
        for method in r.MUTATORS:
            with self.subTest(method=method): self.assertEqual(self.exercise(method),self.exercise(method,('score',1)))

    def test_rotated_upload_resume_preserves_original_client_behavior(self):
        self.assertEqual(self.exercise('fedrotlora'),self.exercise('fedrotlora',('train',2,2)))


class ProtocolTests(unittest.TestCase):
    def test_original_protocol_frozen_and_no_guessing_main_command(self):
        audit=r.data.prepare()
        self.assertTrue(audit['passed']); self.assertEqual(audit['original_reference']['round'],10)
        self.assertEqual(audit['budget']['total_updates'],25000)
        self.assertEqual(audit['budget']['total_sample_visits'],200000)
        for method in r.data.METHODS:
            self.assertIn('run_fedlora_baseline.py',r.data.command(method))

    def test_shared_best_round_not_task_oracle(self):
        records=[dict(round=t,metrics={**{d+'_accuracy':a for d in r.DOMAINS},'macro_accuracy':a},checkpoint=str(t))
                 for t,a in ((1,.5),(2,.7),(3,.7))]
        v=r.summarize(records); self.assertEqual(v['best_round'],2); self.assertEqual(v['final_round'],3)

    def test_cached_evaluation_rejects_changed_backbone_with_same_adapter(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); directory=root/'evaluation'; checkpoint=root/'complete.pt'
            ex=r.Experiment.__new__(r.Experiment)
            ex.model=model(); ex.method='fedexlora'; ex.initial=r.shared.adapter(ex.model)
            ex.base_keys=tuple(k.replace('.lora_A.default.weight','.base_layer.weight') for k in ex.initial if 'lora_A' in k)
            ex.canonical_base=r.cpu_dict({k:ex.model.state_dict()[k] for k in ex.base_keys})
            ex.canonical_base_hash=r.shared.tensor_hash(ex.canonical_base)
            ex.protocol=dict(initial_lora_sha256='fixture',config=dict(model_path='fixture'),prepared_manifest_sha256='fixture')
            ex.tokenizer=None; ex.args=SimpleNamespace(eval_batch_size=8)
            ex.test_rows={d:[dict(id='one',correct_answer='A')] for d in r.DOMAINS}
            ex.test_encoded={d:[None] for d in r.DOMAINS}
            class Evaluator:
                def evaluate(self,m,t):
                    scores={d+'_accuracy':1. for d in self.domains}; scores.update(macro_accuracy=1.,worst_domain_accuracy=1.)
                    prediction=[dict(id='one',domain=d,gold='A',prediction='A',correct=1) for d in self.domains]
                    with (self.output/('round_%02d_predictions.csv'%t)).open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(prediction[0])); w.writeheader(); w.writerows(prediction)
                    r.data.write_rows(self.output/'metrics.jsonl',[dict(round=t,**scores)])
                    return scores
            value=ex.capture(); r.shared.save_state(checkpoint,value)
            with patch.object(r,'ROOT',root),patch.object(r.shared,'ROOT',root),patch('utils.mcq_eval.MCQEvaluator',Evaluator):
                scores=ex.score_full(directory,1,checkpoint,value); self.assertEqual(scores['macro_accuracy'],1.)
                with torch.no_grad(): ex.model.state_dict()[ex.base_keys[0]].add_(.1)
                changed=ex.capture(); self.assertEqual(r.shared.tensor_hash(value['lora']),r.shared.tensor_hash(changed['lora']))
                r.shared.save_state(checkpoint,changed)
                with self.assertRaisesRegex(AssertionError,'different backbone or adapter'):
                    ex.score_full(directory,1,checkpoint,changed)


if __name__=='__main__': unittest.main()
