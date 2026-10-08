"""Cross-domain arithmetic, source fidelity and interruption recovery fixtures."""
import csv
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import MethodType,SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from alg.ftbase import FTBaseServer
from scripts import run_five_client_fedlora as r


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_factor=torch.nn.Parameter(torch.zeros(1))


class DataAndMatrixTests(unittest.TestCase):
    def test_delta_subtracts_target_global_before_not_global_after(self):
        before={d+'_accuracy':(j+1)/10 for j,d in enumerate(r.DOMAINS)}
        matrix={s:{t:(i-j)/20 for j,t in enumerate(r.DOMAINS)} for i,s in enumerate(r.DOMAINS)}
        delta=r.transfer_matrix(matrix,before)
        for i,s in enumerate(r.DOMAINS):
            for j,t in enumerate(r.DOMAINS):
                self.assertAlmostEqual(delta[s][t],(i-j)/20-(j+1)/10)
        second={s:{t:matrix[s][t]+.1 for t in r.DOMAINS} for s in r.DOMAINS}
        mean=r.mean_transfer([dict(local_cross_domain=matrix,global_before=before),
                              dict(local_cross_domain=second,global_before=before)])
        self.assertAlmostEqual(mean['race']['sciq'],delta['race']['sciq']+.05)

    def test_existing_five_client_sample_weighted_aggregation(self):
        model=TinyModel()
        clients=[SimpleNamespace(id=i,dataset={'train':range(4000)},
            lora={'lora_factor':torch.tensor([float(i+1)])}) for i in range(5)]
        server=SimpleNamespace(model=model,sampled_clients=clients)
        with redirect_stdout(StringIO()): FTBaseServer.aggregate(server)
        self.assertAlmostEqual(model.lora_factor.item(),3.,places=6)

    def test_inputs_and_native_mapping_are_reproducible(self):
        import pyarrow.parquet as pq
        result=r.input_checks()
        self.assertTrue(result['passed'])
        folder=r.ROOT/r.data.config()['prepared_dir']
        for d in r.data.NEW:
            for split in ('train','test'):
                rows=r.data.rows(folder/split/f'{d}.jsonl')
                for row in (rows[0],rows[len(rows)//2],rows[-1]):
                    raw=pq.read_table(r.ROOT/row['source_file']).slice(row['source_file_row'],1).to_pylist()[0]
                    if d=='hellaswag':
                        self.assertEqual(row['original_id'],raw['ind'])
                        self.assertEqual(row['question'],raw['ctx'])
                        self.assertEqual(row['correct_answer'],'ABCD'[int(raw['label'])])
                        self.assertEqual(row['source_split'],'train' if split=='train' else 'validation')
                    else:
                        self.assertEqual(row['original_id'],raw['example_id'])
                        self.assertEqual(row['context'],raw['article'])
                        self.assertEqual(row['question'],raw['question'])
                        self.assertEqual(row['correct_answer'],raw['answer'].strip())
                    options=raw['endings'] if d=='hellaswag' else raw['options']
                    self.assertEqual([row['option_'+letter] for letter in 'abcd'],options)


class ResumeTests(unittest.TestCase):
    def exercise(self,crash=None):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); master=root/'master'; old=root/'old'
            outputs={m:master/m for m in r.OUTPUTS}
            ex=r.Experiment.__new__(r.Experiment)
            ex.smoke,ex.rounds,ex.epochs=True,2,1
            ex.model=TinyModel(); ex.initial=r.shared.adapter(ex.model)
            ex.outputs=outputs; ex.protocol={'fixture':True}; ex.parameter_info={}
            ex.counts=dict(zip(r.DOMAINS,(9,8,17,9,8)))
            ex.weights={d:n/sum(ex.counts.values()) for d,n in ex.counts.items()}
            ex.train_rows={d:[dict(dataset=d,id=str(i)) for i in range(n)] for d,n in ex.counts.items()}
            ex.test_rows={d:[dict(dataset=d,id='test_'+str(i)) for i in range(3)] for d in r.DOMAINS}
            ex.training_statistics={d:dict(supervised_token_total=2*n) for d,n in ex.counts.items()}
            r.data.dump(old/'clientlocal/result.json',{d+'_accuracy':.3 for d in r.data.OLD})
            for d in r.data.OLD:
                folder=old/'clientlocal'/('client_'+d); folder.mkdir(parents=True)
                state={'lora_factor':torch.tensor([1.])}
                torch.save(state,folder/'final_lora.pt')
                r.data.dump(folder/'evaluation_provenance.json',dict(lora_tensor_sha256=r.shared.tensor_hash(state)))
            attempts,failures=[],{crash} if crash else set()

            def client(self,domain,directory,server,method):
                return SimpleNamespace(id=r.DOMAINS.index(domain),server=server,domain=domain,directory=directory,
                    method=method,dataset={'train':self.train_rows[domain]})

            def train(self,c,domain):
                t=c.server.round+1; attempts.append((c.method,t,domain))
                with torch.no_grad(): self.model.lora_factor.add_(c.id+1)
                n=self.counts[domain]
                with (c.directory/'exposure.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(round=t,client=c.id,samples=n,optimizer_updates=math.ceil(n/8),
                        supervised_tokens=2*n,ids=[r.data.native_key(row) for row in self.train_rows[domain]]))+'\n')
                if ('train',t,domain) in failures and 'fedavg' in c.method:
                    failures.remove(('train',t,domain)); raise RuntimeError('simulated incomplete client commit')
                return r.shared.adapter(self.model),dict(train_loss=.1,samples=n,optimizer_updates=math.ceil(n/8))

            def score(self,directory,domains,t,checkpoint):
                if directory.parent.name.startswith('round_') and ('score',t,directory.name) in failures:
                    failures.remove(('score',t,directory.name)); raise RuntimeError('simulated evaluation interruption')
                values={d+'_accuracy':self.model.lora_factor.item()/100+(r.DOMAINS.index(d)+1)/1000 for d in domains}
                values.update(macro_accuracy=sum(values.values())/len(domains),
                              worst_domain_accuracy=min(values.values()))
                r.data.dump(directory/'evaluation_provenance.json',dict(domains=list(domains),
                    lora_tensor_sha256=r.shared.tensor_hash(r.shared.adapter(self.model))))
                r.data.dump(directory/'result.json',values)
                return values

            ex.client=MethodType(client,ex); ex.train=MethodType(train,ex); ex.score=MethodType(score,ex)
            with patch.object(r,'ROOT',root),patch.object(r,'MASTER',master),patch.object(r.data.previous,'MASTER',old), \
                 patch.object(r.shared,'ROOT',root),patch.object(r.shared,'MASTER',master), \
                 patch.object(r.shared,'DOMAINS',r.DOMAINS),patch.object(r.shared,'NAMES',r.NAMES),redirect_stdout(StringIO()):
                ex.base(); ex.local()
                self.assertTrue(all(d in r.data.NEW for _,_,d in attempts))
                if crash:
                    with self.assertRaisesRegex(RuntimeError,'simulated'): ex.fedavg()
                records=ex.fedavg()
                expected=sum(ex.weights[d]*(i+1) for i,d in enumerate(r.DOMAINS))
                self.assertAlmostEqual(ex.model.lora_factor.item(),2*expected,places=5)
                final_hash=r.shared.tensor_hash(r.shared.adapter(ex.model))
                ex.fedavg()
                self.assertEqual(r.shared.tensor_hash(r.shared.adapter(ex.model)),final_hash)
                ex.lifecycle_audit(); ex.budget_audit()
                expected_attempts=13 if crash and crash[0]=='train' else 12
                self.assertEqual(len(attempts),expected_attempts)
                self.assertEqual(len(r.data.rows(outputs['fedavg']/'exposure.jsonl')),10)
                for stem,count in [('round_local_cross_domain_accuracy',50),('round_cross_domain_delta',50),
                                   ('total_diagnostics',50),('round_aggregation_gap',10)]:
                    with (outputs['fedavg']/(stem+'.csv')).open() as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))),count)
                self.assertEqual(records[1]['common_start_tensor_sha256'],records[0]['global_after_tensor_sha256'])
                self.assertEqual(r.data.read(outputs['fedavg']/'round_02/local_accuracy_matrix.json')['matrix'],
                                 records[-1]['local_cross_domain'])

    def test_normal_five_client_lifecycle_and_only_two_independent_locals(self): self.exercise()
    def test_resume_saved_local_before_cross_domain_evaluation(self): self.exercise(('score',1,'client_hellaswag'))
    def test_resume_after_aggregation_before_global_evaluation(self): self.exercise(('score',1,'global'))
    def test_resume_uncommitted_epoch_removes_partial_exposure(self): self.exercise(('train',1,'race'))


class ReportTests(unittest.TestCase):
    def test_complete_five_tables_mean_delta_and_exported_heatmaps(self):
        # Synthetic values remain in a temporary directory; real run output is never edited.
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); outputs={m:root/m for m in r.OUTPUTS}
            values={d+'_accuracy':.2+.01*i for i,d in enumerate(r.DOMAINS)}
            values['macro_accuracy']=sum(values.values())/5
            for method,directory in outputs.items(): r.data.dump(directory/'result.json',values)
            cross={s:{t:values[t+'_accuracy']+.01*(i+1) for t in r.DOMAINS} for i,s in enumerate(r.DOMAINS)}
            record=dict(round=10,global_before=values,global_after=values,
                        local_after={d+'_accuracy':cross[d][d] for d in r.DOMAINS},local_cross_domain=cross)
            r.data.dump(outputs['fedavg']/'round_diagnostics.json',dict(rounds=[dict(record,round=t) for t in range(1,11)]))
            budgets=[dict(method='FedAvg',dataset=r.NAMES[d],reused_previous=False,effective_epochs=10,
                total_sample_visits=40000,steps_per_epoch=500,total_optimizer_steps=5000) for d in r.DOMAINS]
            ex=SimpleNamespace(outputs=outputs,test_rows={d:[{}]*1500 for d in r.DOMAINS},
                budget_audit=lambda:budgets,lifecycle_audit=lambda:dict(passed=True))
            with patch.object(r,'REPORT',root/'report.md'),patch.object(r,'input_checks',return_value=dict(passed=True)), \
                 patch.object(r.shared,'figures'),redirect_stdout(StringIO()):
                r.final_report(ex)
            report=(root/'report.md').read_text(encoding='utf-8')
            for table in range(1,6): self.assertIn('Table '+str(table),report)
            matrix=r.data.read(outputs['fedavg']/'mean_cross_domain_delta_matrix.json')['matrix']
            self.assertAlmostEqual(matrix['race']['logiqa'],.05)
            self.assertEqual(len(list((outputs['fedavg']/'figures').glob('*'))),9)


if __name__=='__main__': unittest.main()
