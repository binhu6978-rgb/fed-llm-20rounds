"""Real partitions and aggregation, interrupted Local/FedAvg commits and capability summaries."""
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import MethodType,SimpleNamespace
from unittest.mock import patch

import torch
from scripts import run_fragmentation_experiment as r


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.lora_factor=torch.nn.Parameter(torch.zeros(1))


class PartitionTests(unittest.TestCase):
    def test_frozen_partition_and_unmodified_tests(self):
        manifest=r.data.prepare(); audit=r.data.verify(manifest)
        self.assertTrue(audit['passed']); self.assertEqual(audit['test_total'],7498)
        for d in r.DOMAINS:
            source=r.data.rows(r.ROOT/r.data.config()['source_prepared_dir']/'train'/(d+'.jsonl'))
            a,b=r.data.partition_indices(4000,42)
            self.assertEqual((a,b),r.data.partition_indices(4000,42))
            self.assertFalse(set(a)&set(b)); self.assertEqual(set(a)|set(b),set(range(4000)))
            for c,indices in ((d+'_a',a),(d+'_b',b)):
                self.assertEqual(manifest['clients'][c]['source_train_indices'],indices)
                split=r.data.rows(r.ROOT/r.data.config()['prepared_dir']/'clients'/(c+'.jsonl'))
                self.assertEqual(split,[source[i] for i in indices])
        budget=r.data.budgets()
        self.assertEqual(budget['local_updates'],25000); self.assertEqual(budget['fedavg_updates'],25000)
        self.assertEqual(budget['local_sample_visits'],200000); self.assertEqual(budget['fedavg_sample_visits'],200000)
        self.assertTrue(all(w==.2 for w in budget['source_weights'].values()))

    def test_indexed_encoder_preserves_original_objects(self):
        from utils.model_utils import load_tokenizer
        tokenizer=load_tokenizer(SimpleNamespace(model_path=str(r.ROOT/'models/models--llama3.2-1B'),mcq=True))
        for d in r.DOMAINS:
            rows=r.data.rows(r.ROOT/r.data.config()['source_prepared_dir']/'train'/(d+'.jsonl'))[:4]
            source=r.encoding.TrainingDataset(rows,tokenizer)
            indexed=r.IndexedDataset(source,[0,3])
            self.assertEqual(indexed.rows,[rows[0],rows[3]])
            self.assertIs(indexed[0],source[0]); self.assertIs(indexed[1],source[3])
        r.encoding.clear_cache()

    def test_capability_mean_and_single_best_global_round(self):
        curves={c:[dict(epoch=10,accuracy=(i+1)/20)] for i,c in enumerate(r.CLIENTS)}
        summary=r.local_summary(curves,10)
        self.assertAlmostEqual(summary['macro_accuracy'],.275)
        self.assertAlmostEqual(summary['sources']['cosmosqa']['absolute_difference'],.05)
        records=[dict(round=i,metrics={**{d+'_accuracy':v for d in r.DOMAINS},'macro_accuracy':v}) for i,v in ((1,.3),(2,.4),(3,.4))]
        fed=r.fed_summary(records)
        self.assertEqual(fed['best_macro_round'],2); self.assertEqual(fed['final_round'],3)


class RecoveryTests(unittest.TestCase):
    def exercise(self,crash=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); master=root/'master'
            ex=r.Experiment.__new__(r.Experiment)
            ex.model=TinyModel(); ex.initial=r.shared.adapter(ex.model)
            ex.smoke,ex.epochs,ex.rounds=True,2,2
            ex.run_root=master; ex.outputs={m:master/m for m in ('clientlocal','fedavg')}
            ex.train_rows={c:[dict(dataset=r.SOURCE[c],id=str(i)) for i in range(8)] for c in r.CLIENTS}
            ex.counts={c:8 for c in r.CLIENTS}; ex.weights={c:.1 for c in r.CLIENTS}
            ex.supervised_counts={c:16 for c in r.CLIENTS}
            ex.protocol=dict(budget=r.data.budgets(ex.counts,2,2)); ex.parameter_info={}
            attempts=[]; failures={crash} if crash else set(); score_attempts=[]

            def client(self,c,directory,server,method):
                return SimpleNamespace(id=r.CLIENTS.index(c),name=c,directory=directory,server=server,
                    method=method,dataset={'train':self.train_rows[c]})
            def train(self,c,name):
                epoch=c.server.round+1; attempts.append((c.method,name,epoch))
                with torch.no_grad(): self.model.lora_factor.add_(c.id+1)
                row=dict(round=epoch,client=c.id,samples=8,optimizer_updates=1,supervised_tokens=16,
                    ids=[r.data.native_key(x) for x in self.train_rows[name]])
                with (c.directory/'exposure.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                key=(c.method,'train',name,epoch)
                if key in failures:
                    failures.remove(key); raise RuntimeError('simulated interruption')
                return r.shared.adapter(self.model),dict(train_loss=.2,samples=8,optimizer_updates=1)
            def score(self,directory,domains,epoch,checkpoint):
                key=('local' if len(domains)==1 else 'fedavg','score',directory.parent.name,epoch)
                if key in failures:
                    failures.remove(key); raise RuntimeError('simulated interruption')
                if (directory/'result.json').exists(): return r.data.read(directory/'result.json')
                score_attempts.append((len(domains),epoch))
                scores={d+'_accuracy':self.model.lora_factor.item()/100 for d in domains}
                scores.update(macro_accuracy=sum(scores.values())/len(domains),worst_domain_accuracy=min(scores.values()))
                r.data.dump(directory/'evaluation_provenance.json',dict(domains=list(domains),lora_tensor_sha256=r.shared.tensor_hash(r.shared.adapter(self.model))))
                r.data.dump(directory/'result.json',scores)
                return scores
            ex.client,ex.train,ex.score=(MethodType(fn,ex) for fn in (client,train,score))
            with patch.object(r,'ROOT',root),patch.object(r,'MASTER',master),patch.object(r.shared,'ROOT',root), \
                 patch.object(r.shared,'MASTER',master),patch.object(r.shared,'data',r.data),redirect_stdout(StringIO()):
                if crash and crash[0] in ('clientlocal','local'):
                    with self.assertRaisesRegex(RuntimeError,'simulated'): ex.local()
                ex.local()
                if crash and crash[0]=='fedavg':
                    with self.assertRaisesRegex(RuntimeError,'simulated'): ex.fedavg()
                ex.fedavg()
                self.assertAlmostEqual(ex.model.lora_factor.item(),11.,places=5)
                self.assertTrue(ex.audit()['passed'])
                before=len(attempts); ex.local(); ex.fedavg()
                self.assertEqual(len(attempts),before)
                expected=41 if crash and crash[1]=='train' else 40
                self.assertEqual(before,expected)
                self.assertEqual(len(score_attempts),22) # 20 source-only Local + two five-source global.
                self.assertEqual(sum(k==5 for k,_ in score_attempts),2)
                self.assertEqual(len(r.data.rows(ex.outputs['fedavg']/'exposure.jsonl')),20)

    def test_real_aggregation_and_no_extra_global_evaluations(self): self.exercise()
    def test_local_completed_epoch_pending_evaluation(self): self.exercise(('local','score','sciq_a',1))
    def test_local_uncommitted_epoch_rollback(self): self.exercise(('clientlocal','train','cosmosqa_a',1))
    def test_fedavg_saved_clients_resume(self): self.exercise(('fedavg','score','round_01',1))
    def test_fedavg_uncommitted_epoch_rollback(self): self.exercise(('fedavg','train','hellaswag_b',1))


class SequenceTests(unittest.TestCase):
    def test_queue_order_and_failure_stop(self):
        from scripts import run_fragmentation_sequence as q
        with tempfile.TemporaryDirectory() as tmp:
            master=Path(tmp); calls=[]
            with patch.object(q,'QUEUE',master),patch.object(q.data,'MASTER',master),patch.object(q.data,'prepare'),patch.object(q,'status'), \
                 patch.object(q,'child',side_effect=lambda stage:calls.append(stage)):
                q.sequence()
                self.assertEqual(calls,['smoke','local','fedavg','report'])
            calls=[]
            def child(stage):
                calls.append(stage)
                if stage=='local': raise RuntimeError('simulated failure')
            with patch.object(q,'QUEUE',master),patch.object(q.data,'MASTER',master),patch.object(q.data,'prepare'),patch.object(q,'status'),patch.object(q,'child',side_effect=child):
                with self.assertRaises(RuntimeError): q.sequence()
            self.assertEqual(calls,['smoke','local'])


class ReportTests(unittest.TestCase):
    def test_complete_trajectories_reused_references_and_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            master=Path(tmp); report=master/'report.md'
            curves={c:[dict(epoch=e,accuracy=.5+i/100,train_loss=1/e) for e in range(1,11)]
                    for i,c in enumerate(r.CLIENTS)}
            metrics=lambda a:{**{d+'_accuracy':a for d in r.DOMAINS},'macro_accuracy':a}
            records=[dict(round=e,metrics=metrics(.5+e/100)) for e in range(1,11)]
            refs=dict(base=metrics(.3),original_local_epoch10=metrics(.7),
                original_fedavg_round10=metrics(.65),centralized=metrics(.75),centralized_epoch10=metrics(.72))
            integrity=dict(sources={d:dict(test=100) for d in r.DOMAINS})
            r.data.dump(master/'protocol_audit.json',dict(passed=True,smoke=False))
            r.data.dump(master/'clientlocal/epoch_trajectories.json',curves)
            r.data.dump(master/'fedavg/round_trajectory.json',records)
            with patch.object(r,'MASTER',master),patch.object(r,'REPORT',report), \
                 patch.object(r.data,'verify',return_value=integrity),patch.object(r,'reused_references',return_value=refs):
                self.assertEqual(r.report(),report)
            text=report.read_text(encoding='utf-8')
            self.assertIn('Base and Centralized results are reused',text)
            self.assertIn('Epoch10',text); self.assertIn('cosmosqa_b',text)
            result=r.data.read(master/'final_results.json')
            self.assertEqual(result['fedavg']['best_macro_round'],10)
            self.assertAlmostEqual(result['macro_integration_gap_pp'],15.)
            self.assertAlmostEqual(result['original_integration_gap_pp']['race'],10.)
            self.assertAlmostEqual(result['local']['macro_accuracy'],.545)


if __name__=='__main__': unittest.main()
