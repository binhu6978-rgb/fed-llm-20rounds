"""Reuse 3 Base/Local results; new HellaSwag/RACE Local10; fresh 5-client FedAvg10."""
import argparse
import csv
import ctypes
import gc
import hashlib
import json
import math
import msvcrt
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import five_client_data as data
from scripts import run_three_dataset_lora as shared

DOMAINS, NAMES = data.DOMAINS, data.NAMES
MASTER = data.MASTER
OUTPUTS = {method: MASTER/method for method in ('base','clientlocal','fedavg')}
REPORT = ROOT/'reports/five_client_fedlora_seed42.md'
SharedExperiment = shared.Experiment


def configure():
    shared.data, shared.MASTER, shared.OUTPUTS = data, MASTER, OUTPUTS
    shared.DOMAINS, shared.NAMES = DOMAINS, NAMES


def write_csv(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    os.replace(temporary,path)


def transfer_matrix(accuracy, before):
    return {source: {target: accuracy[source][target]-before[target+'_accuracy'] for target in DOMAINS}
            for source in DOMAINS}


def round_matrices(output, record):
    directory = output/f"round_{record['round']:02d}"
    for stem, matrix, units in (
        ('local_accuracy_matrix',record['local_cross_domain'],'accuracy fraction'),
        ('cross_domain_delta_matrix',transfer_matrix(record['local_cross_domain'],record['global_before']),
         'accuracy change fraction relative to common Global Before')):
        data.dump(directory/(stem+'.json'),dict(round=record['round'],row_clients=list(DOMAINS),
            column_test_domains=list(DOMAINS),units=units,matrix=matrix,checkpoints=record['checkpoints']))
        write_csv(directory/(stem+'.csv'),['client',*DOMAINS],
                  [dict(client=d,**matrix[d]) for d in DOMAINS])


def write_diagnostics(output, records):
    # Keep the original own-domain schema and units in addition to the full matrices.
    shared.write_diagnostics(output,records)
    before_rows, after_rows, accuracy_rows, delta_rows, gap_rows, combined = [],[],[],[],[],[]
    for record in records:
        t = record['round']
        round_matrices(output,record)
        for target in DOMAINS:
            before, after = record['global_before'][target+'_accuracy'], record['global_after'][target+'_accuracy']
            before_rows.append(dict(round=t,dataset=target,accuracy=before,
                checkpoint=record['checkpoints']['global_before'],lora_tensor_sha256=record['common_start_tensor_sha256']))
            after_rows.append(dict(round=t,dataset=target,accuracy=after,
                checkpoint=record['checkpoints']['global_after'],lora_tensor_sha256=record['global_after_tensor_sha256']))
        for source in DOMAINS:
            for target in DOMAINS:
                accuracy = record['local_cross_domain'][source][target]
                before, after = record['global_before'][target+'_accuracy'], record['global_after'][target+'_accuracy']
                common = dict(round=t,client=source,test_domain=target,
                    local_checkpoint=record['checkpoints'][source],local_tensor_sha256=record['local_tensor_sha256'][source])
                accuracy_rows.append(dict(common,accuracy=accuracy))
                delta_rows.append(dict(common,accuracy_change=accuracy-before,
                    global_before_checkpoint=record['checkpoints']['global_before']))
                budget = record['clients'][source]
                combined.append(dict(common,global_before=before,local_accuracy=accuracy,global_after=after,
                    cross_domain_delta=accuracy-before,
                    aggregation_gap=accuracy-after if source==target else '',
                    train_loss=budget['train_loss'],train_samples=budget['samples'],
                    optimizer_updates=budget['optimizer_updates'],
                    sequence_statistics='dataset/five_client_balanced4000_seed42/sequence_statistics.json',
                    global_before_checkpoint=record['checkpoints']['global_before'],
                    global_after_checkpoint=record['checkpoints']['global_after'],
                    common_start_tensor_sha256=record['common_start_tensor_sha256']))
            own, after = record['local_after'][source+'_accuracy'], record['global_after'][source+'_accuracy']
            gap_rows.append(dict(round=t,dataset=source,local_own_accuracy=own,global_after_accuracy=after,
                aggregation_gap=own-after,local_checkpoint=record['checkpoints'][source],
                global_checkpoint=record['checkpoints']['global_after']))
    sets = [('round_global_before',before_rows,['round','dataset','accuracy','checkpoint','lora_tensor_sha256']),
            ('round_global_after',after_rows,['round','dataset','accuracy','checkpoint','lora_tensor_sha256']),
            ('round_local_cross_domain_accuracy',accuracy_rows,['round','client','test_domain','local_checkpoint','local_tensor_sha256','accuracy']),
            ('round_cross_domain_delta',delta_rows,['round','client','test_domain','local_checkpoint','local_tensor_sha256','accuracy_change','global_before_checkpoint']),
            ('round_aggregation_gap',gap_rows,['round','dataset','local_own_accuracy','global_after_accuracy','aggregation_gap','local_checkpoint','global_checkpoint']),
            ('total_diagnostics',combined,['round','client','test_domain','local_checkpoint','local_tensor_sha256',
                'global_before','local_accuracy','global_after','cross_domain_delta','aggregation_gap','train_loss',
                'train_samples','optimizer_updates','sequence_statistics','global_before_checkpoint',
                'global_after_checkpoint','common_start_tensor_sha256'])]
    for stem, values, fields in sets: write_csv(output/(stem+'.csv'),fields,values)
    data.dump(output/'cross_domain_diagnostics.json',dict(domains=list(DOMAINS),units='fractions; plots use percent/pp',
        interpretation='Local-update-associated cross-domain transfer; no causal or mechanism identification.',rounds=records))


class Experiment(SharedExperiment):
    def __init__(self,smoke=False):
        from utils import mcq_utils, five_client_mcq as encoding
        assert shared.data is data, 'Call configure before allocating the experiment'
        with patch.object(mcq_utils,'MCQTrainingDataset',encoding.TrainingDataset), \
             patch.object(mcq_utils,'encode_prompt',encoding.encode_prompt), \
             patch.object(mcq_utils,'continuation_ids',encoding.continuation_ids):
            # Shared allocation, frozen model, LoRA and checkpoint checks are unchanged.
            super().__init__(smoke=False)
            if smoke:
                self.train_rows = {d:self.train_rows[d][:n] for d,n in zip(DOMAINS,(9,8,17,9,8))}
                self.test_rows = {d:self.test_rows[d][:3] for d in DOMAINS}
                self.datasets = {d:encoding.TrainingDataset(self.train_rows[d],self.tokenizer) for d in DOMAINS}
                self.test_encoded = {d:[(encoding.encode_prompt(self.tokenizer,row),
                    [encoding.continuation_ids(self.tokenizer,row,l) for l in 'ABCD']) for row in self.test_rows[d]] for d in DOMAINS}
        encoding.race_encoding.cache_clear()
        self.smoke,self.rounds,self.epochs = smoke,2 if smoke else 10,1 if smoke else 10
        self.args.cn = 5
        self.counts = {d:len(self.datasets[d]) for d in DOMAINS}
        self.weights = {d:n/sum(self.counts.values()) for d,n in self.counts.items()}
        import numpy as np
        self.training_statistics = {}
        prepared_stats = data.read(ROOT/data.config()['prepared_dir']/'sequence_statistics.json')
        for d in DOMAINS:
            lengths = [len(row['input_ids']) for row in self.datasets[d].encoded]
            self.training_statistics[d] = dict(samples=len(lengths),token_total=sum(lengths),
                supervised_token_total=sum(sum(label!=-100 for label in row['labels']) for row in self.datasets[d].encoded),
                mean_input_length=float(np.mean(lengths)),median_input_length=float(np.median(lengths)),
                p90_input_length=float(np.percentile(lengths,90)),max_input_length=max(lengths))
            if not smoke:
                assert all(prepared_stats[d]['train'][key]==value for key,value in self.training_statistics[d].items())
        self.protocol.update(smoke=smoke,rounds=self.rounds,clientlocal_epochs=self.epochs,
            counts=self.counts,weights=self.weights,client_count=5,
            five_runner_sha256=data.sha(Path(__file__)),choice_encoding_sha256=data.sha(Path(encoding.__file__)),
            previous_sampler_sha256=data.sha(Path(data.previous.__file__)),
            old_base_local_reuse_sha256=self.manifest['audit']['preserved_sha256'],
            cross_domain_evaluation='Every true local checkpoint on all 5 domains before aggregation',
            client_seed_rule='Existing client_round_seed(42, zero-based client id, zero-based round)',
            race_max_length=encoding.max_length(),experiment='five_client_balanced4000_seed42')
        if smoke:
            version = hashlib.sha256(json.dumps({k:v for k,v in self.protocol.items() if k.endswith('_sha256')},
                                                sort_keys=True).encode()).hexdigest()[:12]
            self.outputs = {m:MASTER/'smoke'/version/m for m in OUTPUTS}
        else:
            assert self.counts == {d:4000 for d in DOMAINS}
            assert self.weights == {d:.2 for d in DOMAINS}

    def train(self,client,domain):
        value,budget = super().train(client,domain)
        budget['sequence_statistics'] = self.training_statistics[domain]
        return value,budget

    def base(self):
        if self.smoke: return super().base()
        import torch
        output = self.init_output('base')
        self.load(self.initial)
        checkpoint = output/'initial_lora.pt'
        shared.save_state(checkpoint,self.initial)
        shared.status('base_new_domains',domains=list(data.NEW))
        old_root = data.previous.MASTER/'base'
        old = data.read(old_root/'result.json')
        assert shared.tensor_hash(torch.load(old_root/'initial_lora.pt',map_location='cpu',weights_only=True)) == shared.tensor_hash(self.initial)
        original_receipt = data.read(old_root/'evaluation_provenance.json')
        assert original_receipt['lora_tensor_sha256'] == shared.tensor_hash(self.initial)
        scores = self.score(output/'new_domains',data.NEW,0,checkpoint)
        result = {d+'_accuracy':old[d+'_accuracy'] for d in data.OLD}
        result.update({d+'_accuracy':scores[d+'_accuracy'] for d in data.NEW})
        result.update(macro_accuracy=sum(result[d+'_accuracy'] for d in DOMAINS)/5,
                      worst_domain_accuracy=min(result[d+'_accuracy'] for d in DOMAINS))
        data.dump(output/'reused_previous.json',dict(domains=list(data.OLD),source=str(old_root.relative_to(ROOT)),
            old_manifest_sha256=self.manifest['audit']['existing_manifest_sha256'],original_receipt=original_receipt))
        data.dump(output/'result.json',result)
        (output/'completed.txt').write_text('Old 3 Base results reused; new 2 evaluated from canonical initial LoRA.\n',encoding='utf-8')
        return result

    def local(self):
        import torch
        output = self.init_output('clientlocal')
        old = data.read(data.previous.MASTER/'clientlocal/result.json')
        result = {d+'_accuracy':old[d+'_accuracy'] for d in data.OLD}
        reuse = {}
        for domain in data.OLD:
            directory = data.previous.MASTER/'clientlocal'/('client_'+domain)
            checkpoint = directory/'final_lora.pt'
            receipt = data.read(directory/'evaluation_provenance.json')
            assert receipt['lora_tensor_sha256'] == shared.tensor_hash(torch.load(checkpoint,map_location='cpu',weights_only=True))
            reuse[domain] = dict(checkpoint=str(checkpoint.relative_to(ROOT)),sha256=data.sha(checkpoint),
                                accuracy=result[domain+'_accuracy'],evaluation_receipt=receipt)
        data.dump(output/'reused_specialists.json',reuse)
        # The existing three Independent Local specialists are never retrained, including smoke.
        for domain in data.NEW:
            directory = output/('client_'+domain); directory.mkdir(parents=True,exist_ok=True)
            checkpoint = directory/'resume.pt'
            state = (torch.load(checkpoint,map_location='cpu',weights_only=True) if checkpoint.exists()
                     else dict(epoch=0,lora=self.initial,curves=[]))
            self.load(state['lora']); self.trim_exposure(directory/'exposure.jsonl',state['epoch'])
            server = SimpleNamespace(round=0)
            client = self.client(domain,directory,server,'smoke_clientlocal' if self.smoke else 'clientlocal')
            for epoch in range(state['epoch']+1,self.epochs+1):
                server.round = epoch-1
                shared.status('smoke_clientlocal' if self.smoke else 'clientlocal',dataset=domain,epoch=epoch)
                value,record = self.train(client,domain)
                state['curves'].append(dict(epoch=epoch,**record)); state.update(epoch=epoch,lora=value)
                shared.save_state(checkpoint,state)
                data.dump(directory/'training_epochs.json',state['curves'])
            final = directory/'final_lora.pt'; shared.save_state(final,state['lora']); self.load(state['lora'])
            scores = self.score(directory,(domain,),self.epochs,final)
            result[domain+'_accuracy'] = scores[domain+'_accuracy']
            (directory/'completed.txt').write_text('New independent specialist completed.\n',encoding='utf-8')
        result['macro_accuracy'] = sum(result[d+'_accuracy'] for d in DOMAINS)/5
        data.dump(output/'result.json',result)
        (output/'completed.txt').write_text('Old 3 specialists reused; new HellaSwag and RACE specialists completed.\n',encoding='utf-8')
        return result

    def fedavg(self):
        import torch
        from alg.ftbase import FTBaseServer
        output = self.init_output('fedavg'); checkpoint = output/'resume.pt'
        state = (torch.load(checkpoint,map_location='cpu',weights_only=True) if checkpoint.exists()
                 else dict(completed_round=0,global_lora=self.initial,pending={},records=[]))
        self.trim_exposure(output/'exposure.jsonl',state['completed_round'],state['pending'])
        write_diagnostics(output,state['records'])
        server = SimpleNamespace(round=0,model=self.model,global_lora=state['global_lora'])
        clients = [self.client(d,output,server,'smoke_fedavg' if self.smoke else 'fedavg') for d in DOMAINS]
        server.sampled_clients = clients
        base_result = self.base()
        for t in range(state['completed_round']+1,self.rounds+1):
            directory = output/f'round_{t:02d}'; directory.mkdir(parents=True,exist_ok=True)
            server.round,server.global_lora = t-1,state['global_lora']
            self.load(server.global_lora); common_hash = shared.tensor_hash(server.global_lora)
            if t==1: assert common_hash == shared.tensor_hash(self.initial), 'FedAvg did not start fresh from Base'
            before_path = directory/'global_before/lora_weights.pt'; shared.save_state(before_path,server.global_lora)
            shared.status('smoke_global_before' if self.smoke else 'global_before',round=t)
            before = self.score(directory/'global_before',DOMAINS,t,before_path)
            previous = base_result if t==1 else state['records'][-1]['global_after']
            assert before == previous
            shared.event(output,t,'global_before',lora_tensor_sha256=common_hash)
            cross,own,hashes,paths = {},{},{},dict(global_before=str(before_path.relative_to(ROOT)))
            for client,domain in zip(clients,DOMAINS):
                current = directory/('client_'+domain); current.mkdir(parents=True,exist_ok=True)
                local_path = current/'lora_weights.pt'
                if client.id not in state['pending']:
                    self.load(server.global_lora)
                    assert shared.tensor_hash(shared.adapter(self.model)) == common_hash
                    shared.status('smoke_local_training' if self.smoke else 'local_training',round=t,dataset=domain)
                    value,record = self.train(client,domain); shared.save_state(local_path,value)
                    state['pending'][client.id] = dict(lora=value,budget=record,start_hash=common_hash)
                    shared.save_state(checkpoint,state)
                    shared.event(output,t,'local_checkpoint_saved',dataset=domain,checkpoint=str(local_path.relative_to(ROOT)),
                                 common_start_tensor_sha256=common_hash)
                item = state['pending'][client.id]
                assert item['start_hash'] == common_hash
                assert shared.tensor_hash(torch.load(local_path,map_location='cpu',weights_only=True)) == shared.tensor_hash(item['lora'])
                self.load(item['lora'])
                shared.status('smoke_local_cross_evaluation' if self.smoke else 'local_cross_evaluation',round=t,dataset=domain)
                scores = self.score(current,DOMAINS,t,local_path)
                cross[domain] = {target:scores[target+'_accuracy'] for target in DOMAINS}
                own[domain+'_accuracy'] = scores[domain+'_accuracy']
                item['cross_scores'] = scores; client.lora = item['lora']; shared.save_state(checkpoint,state)
                hashes[domain] = shared.tensor_hash(item['lora']); paths[domain] = str(local_path.relative_to(ROOT))
                shared.event(output,t,'local_after',dataset=domain,domains=list(DOMAINS),lora_tensor_sha256=hashes[domain],
                             common_start_tensor_sha256=common_hash)
            assert len(state['pending'])==5 and all('cross_scores' in item for item in state['pending'].values())
            preliminary = dict(round=t,local_cross_domain=cross,global_before=before,checkpoints=paths)
            round_matrices(output,preliminary)  # A and Delta committed before aggregation.
            shared.event(output,t,'cross_matrices_saved')
            shared.event(output,t,'aggregation_start',weights=self.weights)
            FTBaseServer.aggregate(server)  # Original sample-count weighted A/B aggregation.
            global_lora = shared.adapter(self.model)
            assert all(torch.isfinite(v).all() for v in global_lora.values())
            global_path = directory/'global/lora_weights.pt'; shared.save_state(global_path,global_lora)
            global_hash = shared.tensor_hash(global_lora); paths['global_after'] = str(global_path.relative_to(ROOT))
            shared.event(output,t,'aggregation_complete',lora_tensor_sha256=global_hash)
            shared.status('smoke_global_after' if self.smoke else 'global_after',round=t)
            after = self.score(directory/'global',DOMAINS,t,global_path)
            shared.event(output,t,'global_after',lora_tensor_sha256=global_hash)
            record = dict(round=t,global_before=before,local_after=own,local_cross_domain=cross,global_after=after,
                common_start_tensor_sha256=common_hash,global_after_tensor_sha256=global_hash,
                local_tensor_sha256=hashes,checkpoints=paths,weights=self.weights,
                clients={d:state['pending'][i]['budget'] for i,d in enumerate(DOMAINS)})
            data.dump(directory/'round_metadata.json',record)
            state['records'].append(record); state.update(completed_round=t,global_lora=global_lora,pending={})
            shared.save_state(checkpoint,state); write_diagnostics(output,state['records'])
            print(f'Five-client FedAvg round {t}/{self.rounds} committed; macro={after["macro_accuracy"]:.6f}',flush=True)
        self.load(state['global_lora']); assert len(state['records'])==self.rounds
        data.dump(output/'result.json',state['records'][-1]['global_after'])
        (output/'completed.txt').write_text('All five-client rounds and pre-aggregation matrices completed.\n',encoding='utf-8')
        return state['records']

    def budget_audit(self):
        output = self.outputs['fedavg']; exposure = data.rows(output/'exposure.jsonl')
        assert len(exposure)==self.rounds*5
        assert {(r['round'],r['client']) for r in exposure} == {(t,c) for t in range(1,self.rounds+1) for c in range(5)}
        budgets = []
        for i,domain in enumerate(DOMAINS):
            sources = [('FedAvg',self.rounds,[r for r in exposure if r['client']==i],False)]
            if domain in data.NEW:
                sources.append(('Independent Local',self.epochs,
                    data.rows(self.outputs['clientlocal']/('client_'+domain)/'exposure.jsonl'),False))
            elif not self.smoke:
                sources.append(('Independent Local',10,
                    data.rows(data.previous.MASTER/'clientlocal'/('client_'+domain)/'exposure.jsonl'),True))
            for method,epochs,current,reused in sources:
                ids = {data.native_key(row) for row in self.train_rows[domain]}
                assert len(current)==epochs
                for t,row in enumerate(current,1):
                    assert row['round']==t and row['client']==i
                    assert row['samples']==len(row['ids'])==len(set(row['ids']))==len(ids)
                    assert set(row['ids'])==ids
                    assert row['optimizer_updates']==math.ceil(len(ids)/8)
                    assert row['supervised_tokens']==self.training_statistics[domain]['supervised_token_total']
                budgets.append(dict(method=method,dataset=NAMES[domain],reused_previous=reused,
                    train_samples=len(ids),effective_epochs=epochs,steps_per_epoch=math.ceil(len(ids)/8),
                    total_optimizer_steps=sum(r['optimizer_updates'] for r in current),
                    total_sample_visits=sum(r['samples'] for r in current),
                    supervised_tokens=sum(r['supervised_tokens'] for r in current)))
        data.dump(output/'training_budget_audit.json',dict(passed=True,rows=budgets))
        write_csv(output/'training_budget.csv',list(budgets[0]),budgets)
        return budgets

    def lifecycle_audit(self):
        import torch
        output = self.outputs['fedavg']; records = data.read(output/'round_diagnostics.json')['rounds']
        assert len(records)==self.rounds
        events = data.rows(output/'events.jsonl')
        for i,record in enumerate(records):
            t = i+1; current = [event for event in events if event['round']==t]
            starts = [j for j,event in enumerate(current) if event['phase']=='aggregation_start']
            assert starts and current[-1]['phase']=='global_after'
            for start in starts:
                assert {e['dataset'] for e in current[:start] if e['phase']=='local_after'}==set(DOMAINS)
                assert any(e['phase']=='cross_matrices_saved' for e in current[:start])
                assert all(e['domains']==list(DOMAINS) and e['common_start_tensor_sha256']==record['common_start_tensor_sha256']
                           for e in current[:start] if e['phase']=='local_after')
            for name,expected in [('global_before',record['common_start_tensor_sha256']),('global',record['global_after_tensor_sha256'])] + \
                                 [('client_'+d,record['local_tensor_sha256'][d]) for d in DOMAINS]:
                directory = output/f'round_{t:02d}'/name
                actual = shared.tensor_hash(torch.load(directory/'lora_weights.pt',map_location='cpu',weights_only=True))
                receipt = data.read(directory/'evaluation_provenance.json')
                assert receipt['domains']==list(DOMAINS) and receipt['lora_tensor_sha256']==actual==expected
            if i:
                assert record['common_start_tensor_sha256']==records[i-1]['global_after_tensor_sha256']
                assert record['global_before']==records[i-1]['global_after']
            else: assert record['common_start_tensor_sha256']==shared.tensor_hash(self.initial)
            actual_matrix = data.read(output/f'round_{t:02d}'/'local_accuracy_matrix.json')['matrix']
            assert actual_matrix==record['local_cross_domain']
            for source in DOMAINS:
                assert record['local_after'][source+'_accuracy']==actual_matrix[source][source]
                result = data.read(output/f'round_{t:02d}'/('client_'+source)/'result.json')
                assert all(actual_matrix[source][target]==result[target+'_accuracy'] for target in DOMAINS)
        assert len(data.read(output/'round_diagnostics.json')['rows'])==self.rounds*5
        with (output/'total_diagnostics.csv').open(encoding='utf-8',newline='') as stream:
            assert len(list(csv.DictReader(stream)))==self.rounds*25
        result = dict(passed=True,rounds=self.rounds,own_domain_rows=self.rounds*5,cross_domain_values=self.rounds*25,
            all_local_checkpoints_before_evaluation=True,all_5x5_evaluations_before_aggregation=True,
            all_clients_same_global_before=True,next_round_uses_previous_global=True,
            fedavg_started_from_canonical_base=True,unchanged_sample_weighted_aggregation=True)
        data.dump(output/'lifecycle_audit.json',result)
        return result


def input_checks():
    manifest = data.read(ROOT/data.config()['prepared_dir']/'manifest.json')
    for path,digest in manifest['files'].items(): assert data.sha(ROOT/path)==digest,path
    data.verify_previous()
    folder = ROOT/data.config()['prepared_dir']
    train = {d:data.rows(folder/'train'/f'{d}.jsonl') for d in DOMAINS}
    test = {d:data.rows(folder/'test'/f'{d}.jsonl') for d in DOMAINS}
    assert all(len(train[d])==4000 for d in DOMAINS)
    assert all(len(test[d])==1500 for d in data.NEW)
    assert not {data.qa_fingerprint(r) for v in train.values() for r in v}.intersection(
               {data.qa_fingerprint(r) for v in test.values() for r in v})
    videos = {r['source_id'] for r in train['hellaswag']} & {r['source_id'] for r in test['hellaswag']}
    passages = {' '.join(r['context'].split()) for r in train['race']} & {' '.join(r['context'].split()) for r in test['race']}
    assert not videos and not passages
    for d in data.OLD:
        old_folder = ROOT/data.previous.config()['prepared_dir']
        for split in ('train','test'):
            assert data.sha(folder/split/f'{d}.jsonl')==data.sha(old_folder/split/f'{d}.jsonl')
    result = dict(passed=True,train_counts={d:len(v) for d,v in train.items()},test_counts={d:len(v) for d,v in test.items()},
        qa_overlap=0,hellaswag_shared_video_sources=0,race_shared_passage_texts=0,
        old_three_data_byte_identical=True,old_base_local_preserved=True,seed=42)
    data.dump(MASTER/'input_integrity.json',result)
    return result


def protocol_audit(smoke):
    lifecycle = smoke.lifecycle_audit(); budgets = smoke.budget_audit(); inputs = input_checks()
    result = dict(passed=True,completed_at=shared.now(),input_integrity=inputs,smoke_lifecycle=lifecycle,
        smoke_budget=budgets,smoke_train_counts=smoke.counts,
        smoke_outputs={k:str(v.relative_to(ROOT)) for k,v in smoke.outputs.items()},
        formal_weights={d:.2 for d in DOMAINS},old_independent_local_not_retrained=True)
    data.dump(MASTER/'protocol_audit.json',result)
    with data.AUDIT_REPORT.open('a',encoding='utf-8') as stream:
        stream.write('\n## Runtime smoke protocol audit\n\nPASS: 2 rounds, all 5 clients, 9/8/17/9/8 '
            'train rows and 3 test rows/domain; 50 real pre-aggregation cross-domain evaluations, '
            '10 saved local LoRAs, verified canonical initialization/common round starts, '
            'next-global continuity, sample-weighted aggregation, complete epochs and tail updates. '
            'Independent smoke Local trains only the new HellaSwag/RACE clients. '
            'Old three Base/Local/data artifacts are byte-identical. '
            'HellaSwag video-source overlap and RACE passage-text overlap are zero.\n')
    print('Five-client real smoke, 5x5 lifecycle and immutable old artifact checks PASS.',flush=True)


def run_smoke():
    shared.status('smoke'); ex = Experiment(smoke=True)
    ex.base(); ex.local(); ex.fedavg(); protocol_audit(ex)


def mean_transfer(records):
    matrices = [transfer_matrix(r['local_cross_domain'],r['global_before']) for r in records]
    return {source:{target:sum(m[source][target] for m in matrices)/len(matrices) for target in DOMAINS}
            for source in DOMAINS}


def matrix_figures(output,records):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = output/'figures'; destination.mkdir(parents=True,exist_ok=True)
    last = records[-1]
    matrices = [(f"round{last['round']:02d}_local_accuracy",last['local_cross_domain'],'Local accuracy (%)',False),
                (f"round{last['round']:02d}_cross_domain_delta",transfer_matrix(last['local_cross_domain'],last['global_before']),
                 f"Round {last['round']} cross-domain transfer (pp)",True),
                ('mean_cross_domain_delta',mean_transfer(records),f'Mean transfer ({len(records)} rounds, pp)',True)]
    for stem,matrix,title,diverging in matrices:
        values = np.array([[100*matrix[s][t] for t in DOMAINS] for s in DOMAINS])
        fig = plt.figure(figsize=(7.1,5.8)); ax = fig.add_axes([.20,.25,.67,.65])
        if diverging:
            limit=max(float(np.abs(values).max()),.01)
            art=ax.imshow(values,cmap='RdBu',vmin=-limit,vmax=limit)
        else: art=ax.imshow(values,cmap='Blues',vmin=0,vmax=100)
        ax.set_xticks(range(5),[NAMES[d] for d in DOMAINS],rotation=35,ha='right')
        ax.set_yticks(range(5),[NAMES[d]+' local' for d in DOMAINS])
        ax.set(title=title,xlabel='Test domain',ylabel='Local update source')
        for i in range(5):
            for j in range(5):
                value=values[i,j]
                white=abs(value)>.65*limit if diverging else value>60
                ax.text(j,i,f'{value:+.2f}' if diverging else f'{value:.2f}',ha='center',va='center',
                        fontsize=9,color='white' if white else 'black')
        fig.colorbar(art,ax=ax,fraction=.045,pad=.04)
        for extension in ('png','svg','pdf'): fig.savefig(destination/f'{stem}.{extension}',dpi=300)
        plt.close(fig)


def final_report(ex):
    budgets=ex.budget_audit(); lifecycle=ex.lifecycle_audit(); inputs=input_checks()
    output=ex.outputs['fedavg']; records=data.read(output/'round_diagnostics.json')['rounds']; last=records[-1]
    results={method:data.read(directory/'result.json') for method,directory in ex.outputs.items()}
    mean=mean_transfer(records)
    data.dump(output/'mean_cross_domain_delta_matrix.json',dict(units='accuracy change fraction',matrix=mean,
        definition='mean over all 10 rounds of A_ij(t) minus common G_j(t) before local training'))
    write_csv(output/'mean_cross_domain_delta_matrix.csv',['client',*DOMAINS],[dict(client=s,**mean[s]) for s in DOMAINS])
    shared.figures(output); matrix_figures(output,records)
    stats=data.read(ROOT/data.config()['prepared_dir']/'sequence_statistics.json')
    audit=data.read(MASTER/'data_audit.json')
    lines=['# Five-client FedLoRA (fixed4000, seed42)','',
        '## Table 1 — Base / Independent Local / FedAvg','',
        '| Method | LogiQA | OpenBookQA | SciQ | HellaSwag | RACE | Macro |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method,label in [('base','Base'),('clientlocal','Independent Local'),('fedavg','FedAvg Round 10')]:
        value=results[method]
        lines.append('| '+label+' | '+' | '.join(f"{100*value[d+'_accuracy']:.2f}%" for d in DOMAINS)
                     +f" | {100*value['macro_accuracy']:.2f}% |")
    lines += ['', 'Original three Base results and independently trained Local specialists are reused '
        'from `outputs/three_dataset_balanced4000_seed42`. Only HellaSwag and RACE Base tests and '
        'Independent Local10 were newly computed. The five-client FedAvg is a fresh ten-round run from '
        'the canonical initial LoRA. Macro is an unweighted average of the five domain accuracies.', '',
        '| Client | Train | Test | Test source | Weight |', '| --- | ---: | ---: | --- | ---: |']
    for d in DOMAINS:
        lines.append(f"| {NAMES[d]} | 4000 | {len(ex.test_rows[d])} | {audit['counts'][d]['test_definition']} | 0.20 |")
    lines += ['', 'HellaSwag uses labeled official validation; unused validation and official test are not used. '
        'RACE uses official all/test and keeps the full passage. New selections use seed42 uniform sampling '
        'without replacement after quality filtering, with original native IDs/file indices preserved. '
        'Old three prepared inputs are byte-identical, including all Combined Test rows.', '',
        'Frozen BF16 Llama-3.2-1B; q_proj/v_proj LoRA r8/alpha32/dropout0.05/bias=none; constant lr1e-4, '
        'bs1, accumulation8, drop_last=False, step0. AdamW resets each full client epoch. '
        'The original Trainer, four-choice likelihood evaluator and sample-count A/B FedAvg are reused '
        'without modification. Internal client IDs are 0–4 in the requested order; seed42 uses the '
        'existing client/round seed offsets. No tuning, checkpoint selection or performance-based sampling.', '',
        '## Table 2 — Round 10 Local → Global','',
        '| Client | Pre-aggregation Local own | Post-aggregation Global | Aggregation Gap (pp) |',
        '| --- | ---: | ---: | ---: |']
    for d in DOMAINS:
        local,after=last['local_after'][d+'_accuracy'],last['global_after'][d+'_accuracy']
        lines.append(f'| {NAMES[d]} | {100*local:.2f}% | {100*after:.2f}% | {100*(local-after):+.2f} |')
    local_macro=sum(last['local_after'][d+'_accuracy'] for d in DOMAINS)/5
    lines += ['', f"Round10 pre-aggregation own-domain Macro = {100*local_macro:.2f}%; "
        f"post-aggregation global Macro = {100*last['global_after']['macro_accuracy']:.2f}%.", '',
        '## Table 3 — Every round own-domain Local → Global','',
        '| Round | LogiQA | OpenBookQA | SciQ | HellaSwag | RACE |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        lines.append('| '+str(r['round'])+' | '+' | '.join(
            f"{100*r['local_after'][d+'_accuracy']:.2f}% → {100*r['global_after'][d+'_accuracy']:.2f}%" for d in DOMAINS)+' |')
    for title,matrix,change in [
        ('Table 4 — Round10 local cross-domain accuracy',last['local_cross_domain'],False),
        ('Table 5 — Round10 cross-domain transfer',transfer_matrix(last['local_cross_domain'],last['global_before']),True),
        ('Mean cross-domain transfer over all 10 rounds',mean,True)]:
        lines += ['', '## '+title, '', 'Rows are local update sources; columns are test domains. '
            +('Values are percentage-point changes from the common Global Before.' if change else 'Values are accuracy percentages.'), '',
            '| Local source | LogiQA | OpenBookQA | SciQ | HellaSwag | RACE |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for s in DOMAINS:
            lines.append('| '+NAMES[s]+' | '+' | '.join(f'{100*matrix[s][t]:+.2f}' if change else
                f'{100*matrix[s][t]:.2f}%' for t in DOMAINS)+' |')
    lines += ['', 'Delta_ij(t) = A_ij(t) − G_j(t). These are descriptive, local-update-associated '
        'cross-domain transfers. Values are displayed without a near-neutral threshold, causal attribution '
        'or mechanism claims. Local models are evaluated before any aggregation; later Global After '
        'scores are not used to construct Delta.', '',
        '## Actual sequence statistics', '',
        'Input length counts BOS + original prompt + gold answer continuation + EOS, without padding. '
        'Prompt and supervised token totals are separately available in the JSON. No token matching.', '',
        '| Domain | Split | Samples | Total tokens | Mean length | Median | P90 | Labels A/B/C/D |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for d in DOMAINS:
        for split,value in stats[d].items():
            labels='/'.join(str(value['label_distribution'].get(l,0)) for l in 'ABCD')
            lines.append(f"| {NAMES[d]} | {split} | {value['samples']} | {value['token_total']} | "
                         f"{value['mean_input_length']:.2f} | {value['median_input_length']:.2f} | "
                         f"{value['p90_input_length']:.2f} | {labels} |")
    lines += ['', '## Actual training budget', '',
        '| Method | Domain | Reused | Full epochs | Visits | Steps/epoch | Total updates |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for b in budgets:
        lines.append(f"| {b['method']} | {b['dataset']} | {b['reused_previous']} | {b['effective_epochs']} | "
                     f"{b['total_sample_visits']} | {b['steps_per_epoch']} | {b['total_optimizer_steps']} |")
    lines += ['', 'Fresh FedAvg budget is 200000 visits / 25000 optimizer updates. New Independent Local '
        'budget is 80000 visits / 10000 updates. The existing three Local specialists have 120000 '
        'previous visits / 15000 previous updates; these were audited and not rerun.', '',
        '## Quality and integrity findings', '']
    for d in data.NEW:
        detail=audit['cleaning'][d]
        lines.append(f"{NAMES[d]}: train filtering = {json.dumps(detail['train']['rejection_reasons'])}; "
            f"test filtering = {json.dumps(detail['test']['rejection_reasons'])}; "
            f"train/test duplicate removals = {detail['train_test_removed']}.")
        lines.append('')
    lines += ['RACE is located at `data/raw_mcq/race/all`, rather than a root `race` directory. '
        'RACE example_id identifies a passage; the original question row index disambiguates questions. '
        'HellaSwag 0–3 labels map to A–D without changing ending order. Selected HellaSwag video-source '
        'overlap and RACE normalized-passage overlap are zero, as is all-domain normalized full-QA overlap.', '',
        'Every local checkpoint and all five evaluation receipts are linked by tensor SHA256. '
        'All clients use the same round-start hash; each Global After is the next Global Before. '
        'Old Base/Local/data/source/report hashes remain unchanged. Completed lifecycle and budget audits PASS. '
        'No silent training/scoring/aggregation protocol changes were detected.', '',
        '## Key artifacts', '',
        f'- Run root: `{MASTER.relative_to(ROOT)}`; shared baseline summary: `base/result.json`; new Base predictions: `base/new_domains/evaluation/`.',
        '- New independent weights: `clientlocal/client_hellaswag/final_lora.pt`, `clientlocal/client_race/final_lora.pt`; '
        'old weight references: `clientlocal/reused_specialists.json`.',
        '- Final global: `fedavg/round_10/global/lora_weights.pt`; each round has `client_{domain}/lora_weights.pt` '
        'and `global_before/lora_weights.pt`, corresponding metrics, predictions and evaluation receipts.',
        '- Per-round matrices: `fedavg/round_01` through `round_10` contain `local_accuracy_matrix.csv/json` '
        'and `cross_domain_delta_matrix.csv/json`, with source checkpoint paths.',
        '- Global dynamics: `fedavg/round_global_before.csv`, `round_global_after.csv`, '
        '`round_aggregation_gap.csv`, `round_diagnostics.csv/json` (50 own-domain rows).',
        '- Cross-domain dynamics: `fedavg/round_local_cross_domain_accuracy.csv`, `round_cross_domain_delta.csv`, '
        '`total_diagnostics.csv` (250 rows), `cross_domain_diagnostics.json`, `mean_cross_domain_delta_matrix.csv/json`.',
        '- Sequence statistics and native indices: `dataset/five_client_balanced4000_seed42/sequence_statistics.csv/json`, '
        '`manifest.json`; quality exclusions: `outputs/five_client_balanced4000_seed42/quality_exclusions.jsonl`.',
        '- Figures: `fedavg/figures/`; audits: `fedavg/lifecycle_audit.json`, `training_budget_audit.json`, '
        '`training_budget.csv`, plus run-root `input_integrity.json` and `protocol_audit.json`.']
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    data.dump(output/'final_results.json',dict(results=results,budgets=budgets,lifecycle=lifecycle,
        input_integrity=inputs,mean_delta=mean,round10=last,sequence_statistics=stats))
    print('\n'.join(lines[:10]),flush=True)
    return REPORT


def run_all():
    data.prepare(); input_checks(); run_smoke()
    import torch
    gc.collect(); torch.cuda.empty_cache()
    ex = Experiment()
    print('Formal Client | Train | Test | Sample weight | Updates/epoch',flush=True)
    for d in DOMAINS: print(f'{NAMES[d]} | {ex.counts[d]} | {len(ex.test_rows[d])} | {ex.weights[d]} | 500',flush=True)
    print(json.dumps(data.config()),flush=True)
    ex.base(); ex.local(); ex.fedavg()
    shared.status('reporting'); path = final_report(ex)
    shared.status('complete',report=str(path))
    (MASTER/'completed.txt').write_text('New Base2/Local2, fresh FedAvg5, matrices, budgets and report completed.\n',encoding='utf-8')


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',choices=('audit','prepare','smoke','all','report'),default='all')
    args = parser.parse_args(); os.chdir(ROOT); configure(); MASTER.mkdir(parents=True,exist_ok=True)
    if args.job in ('audit','prepare'):
        (data.audit if args.job=='audit' else data.prepare)(); return
    with (MASTER/'run.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0');lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: raise SystemExit('Five-client experiment already running; duplicate launch refused.')
        try:
            if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001): raise ctypes.WinError()
            if args.job=='all': run_all()
            elif args.job=='smoke': data.prepare(); input_checks(); run_smoke();shared.status('smoke_complete')
            else: final_report(Experiment())
        except BaseException as error:
            shared.status('failed',error=str(error),traceback=traceback.format_exc());traceback.print_exc();raise SystemExit(1)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__': main()
