"""Setting B: independent split specialists, then standard ten-client FedAvg."""
import argparse
import copy
import csv
import ctypes
import hashlib
import json
import msvcrt
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import fragmentation_data as data
from scripts import run_cosmosqa_five_experiments as original

shared, encoding = original.shared, original.encoding
DOMAINS, CLIENTS, SOURCE, MASTER = data.DOMAINS, data.CLIENTS, data.SOURCE, data.MASTER
REPORT = ROOT / 'reports/fragmentation_5source_10client_seed42.md'
AGGREGATION = 'standard FTBaseServer.aggregate: sample-weighted LoRA A/B factors; no exact aggregation'


def configure():
    shared.MASTER, shared.data = MASTER, data
    shared.DOMAINS, shared.NAMES = DOMAINS, data.NAMES


def write_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    os.replace(temporary, path)


def local_summary(curves, epoch):
    assert set(curves) == set(CLIENTS)
    values = {c: next(r['accuracy'] for r in curves[c] if r['epoch'] == epoch) for c in CLIENTS}
    sources = {d: dict(client_a=values[d+'_a'], client_b=values[d+'_b'],
        mean=(values[d+'_a']+values[d+'_b'])/2, absolute_difference=abs(values[d+'_a']-values[d+'_b'])) for d in DOMAINS}
    return dict(epoch=epoch, clients=values, sources=sources,
        macro_accuracy=sum(sources[d]['mean'] for d in DOMAINS)/5)


def fed_summary(records):
    assert records and [r['round'] for r in records] == list(range(1, len(records)+1))
    best = max(records, key=lambda r: (r['metrics']['macro_accuracy'], -r['round']))
    return dict(best_macro_round=best['round'], best_macro_metrics=best['metrics'],
        final_round=records[-1]['round'], final_metrics=records[-1]['metrics'],
        dataset_peaks={d: dict(round=max(records, key=lambda r: (r['metrics'][d+'_accuracy'], -r['round']))['round'],
            accuracy=max(r['metrics'][d+'_accuracy'] for r in records),
            final=records[-1]['metrics'][d+'_accuracy']) for d in DOMAINS})


class IndexedDataset:
    def __init__(self, dataset, indices):
        self.rows = [dataset.rows[i] for i in indices]
        self.encoded = [dataset.encoded[i] for i in indices]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.encoded[index]


class ProgressLoader:
    def __init__(self, loader, client, name, method):
        self.loader, self.client, self.name, self.method = loader, client, name, method
        self.drop_last = loader.drop_last
    def __len__(self): return len(self.loader)
    def __iter__(self):
        started = time.monotonic()
        for count, batch in enumerate(self.loader, 1):
            yield batch
            if count % 200 == 0 or count == len(self):
                value = dict(method=self.method, client=self.name, source=SOURCE[self.name],
                    round_or_epoch=self.client.server.round+1, samples_finished=count, samples_total=len(self),
                    optimizer_updates_finished=count//8, elapsed_seconds=time.monotonic()-started)
                data.dump(MASTER / 'live_progress.json', value)
                print('PROGRESS', json.dumps(value), flush=True)


class Experiment(original.Experiment):
    def __init__(self, smoke=False):
        # Allocate the identical model/LoRA and encode the same frozen five sources.
        super().__init__(smoke=False)
        configure()
        cfg = data.config(); folder = ROOT / cfg['prepared_dir']
        self.manifest = data.read(folder / 'manifest.json')
        self.source_datasets = self.datasets
        self.datasets = {}
        for c in CLIENTS:
            indices = self.manifest['clients'][c]['source_train_indices']
            if smoke: indices = indices[:8]
            self.datasets[c] = IndexedDataset(self.source_datasets[SOURCE[c]], indices)
            if not smoke: assert self.datasets[c].rows == data.rows(folder / 'clients' / (c+'.jsonl'))
        self.train_rows = {c: self.datasets[c].rows for c in CLIENTS}
        self.smoke, self.rounds, self.epochs = smoke, 2 if smoke else 10, 2 if smoke else 10
        if smoke:
            self.test_rows = {d: self.test_rows[d][:3] for d in DOMAINS}
            self.test_encoded = {d: self.test_encoded[d][:3] for d in DOMAINS}
        else:
            assert all(self.test_rows[d] == data.rows(folder / 'test' / (d+'.jsonl')) for d in DOMAINS)
        self.counts = {c: len(self.datasets[c]) for c in CLIENTS}
        self.weights = {c: n/sum(self.counts.values()) for c,n in self.counts.items()}
        self.supervised_counts = {c: sum(sum(v != -100 for v in row['labels']) for row in self.datasets[c].encoded) for c in CLIENTS}
        self.args.cn, self.args.suffix = 10, str(MASTER)
        self.args.mcq_dataset_dir = str(folder)
        old_protocol = copy.deepcopy(self.protocol)
        self.protocol.update(config=cfg, original_5client_config=old_protocol['config'],
            prepared_manifest_sha256=data.sha(folder/'manifest.json'),
            original_prepared_manifest_sha256=old_protocol['prepared_manifest_sha256'],
            setting='B', variable='client fragmentation / local-isolation granularity',
            client_order=list(CLIENTS), client_source=SOURCE, client_count=10, evaluation_sources=list(DOMAINS),
            counts=self.counts, weights=self.weights, split_seed=cfg['split_seed'], smoke=smoke,
            rounds=self.rounds, clientlocal_epochs=self.epochs,
            aggregation=AGGREGATION, aggregation_sha256=data.sha(ROOT/'alg/ftbase.py'),
            suite_runner_sha256=data.sha(Path(__file__)), data_helper_sha256=data.sha(Path(data.__file__)),
            original_runner_sha256=data.sha(Path(original.__file__)),
            reuse_all_base_local=False, fresh_methods=['clientlocal','fedavg'],
            reused_methods=['base','centralized'], own_domain_local_evaluation_only=True,
            centralized_reference_epoch=cfg['centralized_reference_epoch'],
            centralized_epochs=None, centralized_samples=None,
            checkpoint_selection='Local epoch10; FedAvg final10 plus retrospective best test Macro',
            budget=data.budgets(self.counts, self.epochs, self.rounds),
            client_seed_rule='client_round_seed(42, interleaved_client_id_0_to_9, epoch_or_round-1)',
            global_evaluation='once per round on five full test sets; no duplicate A/B tests')
        self.protocol.pop('centralized_optimizer_reset', None)
        version = hashlib.sha256(json.dumps(self.protocol,sort_keys=True).encode()).hexdigest()[:12]
        root = MASTER / 'smoke' / version if smoke else MASTER
        self.outputs = {m: root/m for m in ('clientlocal','fedavg')}
        self.run_root = root
        print('AGGREGATION:', AGGREGATION, flush=True)
        print('BUDGET:', json.dumps(self.protocol['budget']), flush=True)

    def base(self): raise RuntimeError('Base is reused; no Base evaluation in Setting B')
    def centralized(self): raise RuntimeError('Centralized is reused; no centralized training in Setting B')

    def client(self, name, output, server, method):
        from alg.base import BaseClient
        from alg.ftbase import FTBaseClient
        from utils.train_utils import Trainer
        args = copy.copy(self.args); args.suffix = str(output)
        class Client(FTBaseClient):
            def __init__(client):
                BaseClient.__init__(client, CLIENTS.index(name), args)
                client.server, client.dataset = server, {'train': self.datasets[name]}
                client.tokenizer, client.lora, client.delay = self.tokenizer, {}, 1.
                client.evaluator, client.last_train_loss = None, None
                client.trainer = Trainer(args, client.dataset, client)
                assert len(client.trainer.train_loader) == self.counts[name]
                assert not client.trainer.train_loader.drop_last
                client.trainer.train_loader = ProgressLoader(client.trainer.train_loader, client, name, method)
        return Client()

    def local(self):
        import torch
        output = self.init_output('clientlocal'); all_curves = {}
        for c in CLIENTS:
            current = output/c; current.mkdir(parents=True,exist_ok=True)
            resume = current/'resume.pt'
            state = torch.load(resume,map_location='cpu',weights_only=True) if resume.exists() else dict(epoch=0,lora=self.initial,curves=[])
            self.trim_exposure(current/'exposure.jsonl',state['epoch'])
            server = SimpleNamespace(round=0); client = self.client(c,current,server,'clientlocal')
            for epoch in range(1,self.epochs+1):
                epoch_path = current/('epoch_%02d_lora.pt' % epoch)
                if epoch > state['epoch']:
                    assert epoch == state['epoch']+1
                    self.load(state['lora']); start_hash = shared.tensor_hash(state['lora'])
                    if epoch == 1: assert start_hash == shared.tensor_hash(self.initial)
                    server.round = epoch-1
                    shared.status('local_training',client=c,source=SOURCE[c],epoch=epoch,smoke=self.smoke)
                    value,budget = self.train(client,c)
                    shared.save_state(epoch_path,value)
                    state['curves'].append(dict(epoch=epoch,**budget,start_tensor_sha256=start_hash,
                        lora_tensor_sha256=shared.tensor_hash(value)))
                    state.update(epoch=epoch,lora=value); shared.save_state(resume,state)
                value = torch.load(epoch_path,map_location='cpu',weights_only=True)
                curve = state['curves'][epoch-1]
                assert shared.tensor_hash(value) == curve['lora_tensor_sha256']
                self.load(value)
                shared.status('local_evaluation',client=c,source=SOURCE[c],epoch=epoch,smoke=self.smoke)
                scores = self.score(current/('epoch_%02d' % epoch),(SOURCE[c],),epoch,epoch_path)
                curve['accuracy'] = scores[SOURCE[c]+'_accuracy']
                shared.save_state(resume,state); data.dump(current/'training_epochs.json',state['curves'])
                print('LOCAL',c,epoch,curve['accuracy'],flush=True)
            final = current/'final_lora.pt'; shared.save_state(final,state['lora'])
            data.dump(current/'result.json',dict(source=SOURCE[c],epoch=self.epochs,accuracy=state['curves'][-1]['accuracy'],
                checkpoint=str(final.relative_to(ROOT))))
            (current/'completed.txt').write_text('Independent Local; every epoch scored on the single full source test.\n',encoding='utf-8')
            all_curves[c] = state['curves']
            data.dump(output/'epoch_trajectories.json',all_curves)
            flat = [dict(client=name,source=SOURCE[name],epoch=r['epoch'],accuracy=r['accuracy'],train_loss=r['train_loss'])
                    for name,curve in all_curves.items() for r in curve]
            write_csv(output/'epoch_trajectories.csv',flat,['client','source','epoch','accuracy','train_loss'])
        result = local_summary(all_curves,self.epochs)
        data.dump(output/'result.json',result)
        (output/'completed.txt').write_text('Ten split specialists completed.\n',encoding='utf-8')
        return result

    def write_rounds(self, output, records):
        data.dump(output/'round_trajectory.json',records)
        flat = [dict(round=r['round'],**r['metrics']) for r in records]
        write_csv(output/'round_trajectory.csv',flat,['round',*[d+'_accuracy' for d in DOMAINS],
            'macro_accuracy','worst_domain_accuracy'])

    def fedavg(self):
        import torch
        from alg.ftbase import FTBaseServer
        output = self.init_output('fedavg'); resume = output/'resume.pt'
        state = torch.load(resume,map_location='cpu',weights_only=True) if resume.exists() else dict(completed_round=0,global_lora=self.initial,pending={},records=[])
        self.trim_exposure(output/'exposure.jsonl',state['completed_round'],state['pending'])
        self.write_rounds(output,state['records'])
        server = SimpleNamespace(round=0,model=self.model,global_lora=state['global_lora'])
        clients = [self.client(c,output,server,'fedavg') for c in CLIENTS]
        server.sampled_clients = clients
        for t in range(state['completed_round']+1,self.rounds+1):
            directory = output/('round_%02d' % t); directory.mkdir(parents=True,exist_ok=True)
            server.round,server.global_lora = t-1,state['global_lora']
            common_hash = shared.tensor_hash(server.global_lora)
            if t == 1: assert common_hash == shared.tensor_hash(self.initial)
            for client,c in zip(clients,CLIENTS):
                path = directory/c/'lora_weights.pt'
                if client.id not in state['pending']:
                    self.load(server.global_lora)
                    assert shared.tensor_hash(shared.adapter(self.model)) == common_hash
                    shared.status('fedavg_local_training',round=t,client=c,source=SOURCE[c],smoke=self.smoke)
                    value,budget = self.train(client,c); shared.save_state(path,value)
                    state['pending'][client.id] = dict(lora=value,budget=budget,start_hash=common_hash)
                    shared.save_state(resume,state)
                    shared.event(output,t,'client_committed',client=c,start_tensor_sha256=common_hash,
                        lora_tensor_sha256=shared.tensor_hash(value))
                item = state['pending'][client.id]
                assert item['start_hash'] == common_hash
                assert shared.tensor_hash(torch.load(path,map_location='cpu',weights_only=True)) == shared.tensor_hash(item['lora'])
                client.lora = item['lora']
            assert len(state['pending']) == 10
            shared.event(output,t,'aggregation_start',implementation=AGGREGATION,weights=self.weights)
            print('WEIGHTS',json.dumps(self.weights),'SOURCE_WEIGHTS',json.dumps(self.protocol['budget']['source_weights']),flush=True)
            FTBaseServer.aggregate(server)  # Deliberately use the original implementation verbatim.
            value = shared.adapter(self.model)
            assert all(torch.isfinite(v).all() for v in value.values())
            path = directory/'global/lora_weights.pt'; shared.save_state(path,value)
            digest = shared.tensor_hash(value)
            shared.event(output,t,'aggregation_complete',lora_tensor_sha256=digest)
            shared.status('fedavg_global_evaluation',round=t,smoke=self.smoke)
            scores = self.score(directory/'global',DOMAINS,t,path)
            state['records'].append(dict(round=t,metrics=scores,common_start_tensor_sha256=common_hash,
                global_tensor_sha256=digest,checkpoint=str(path.relative_to(ROOT)),weights=self.weights,
                clients={c:dict(**state['pending'][i]['budget'],start_hash=state['pending'][i]['start_hash'],
                    local_tensor_sha256=shared.tensor_hash(state['pending'][i]['lora'])) for i,c in enumerate(CLIENTS)}))
            state.update(completed_round=t,global_lora=value,pending={}); shared.save_state(resume,state)
            self.write_rounds(output,state['records'])
            print('FEDAVG',t,scores['macro_accuracy'],flush=True)
        self.load(state['global_lora'])
        data.dump(output/'result.json',fed_summary(state['records']))
        (output/'completed.txt').write_text('Ten-client standard FedAvg complete.\n',encoding='utf-8')
        return state['records']

    def audit(self):
        import torch
        local = self.outputs['clientlocal']; fed = self.outputs['fedavg']
        visits = {}; initial_hash = shared.tensor_hash(self.initial)
        trajectories = data.read(local/'epoch_trajectories.json')
        assert set(trajectories) == set(CLIENTS)
        for c in CLIENTS:
            current = local/c
            curves = trajectories[c]; assert [r['epoch'] for r in curves] == list(range(1,self.epochs+1))
            exposure = data.rows(current/'exposure.jsonl')
            assert len(exposure) == self.epochs
            prior = initial_hash
            for curve, row in zip(curves,exposure):
                assert curve['start_tensor_sha256'] == prior
                self.check_exposure(row,c,curve['epoch']); assert curve['optimizer_updates'] == row['optimizer_updates']
                receipt = data.read(current/('epoch_%02d' % curve['epoch'])/'evaluation_provenance.json')
                assert receipt['domains'] == [SOURCE[c]] and receipt['lora_tensor_sha256'] == curve['lora_tensor_sha256']
                assert shared.tensor_hash(torch.load(current/('epoch_%02d_lora.pt' % curve['epoch']),map_location='cpu',weights_only=True)) == curve['lora_tensor_sha256']
                assert curve['accuracy'] == data.read(current/('epoch_%02d' % curve['epoch'])/'result.json')[SOURCE[c]+'_accuracy']
                prior = curve['lora_tensor_sha256']
            visits[c] = dict(samples=sum(r['samples'] for r in exposure),updates=sum(r['optimizer_updates'] for r in exposure))
        records = data.read(fed/'round_trajectory.json'); exposure = data.rows(fed/'exposure.jsonl')
        assert [r['round'] for r in records] == list(range(1,self.rounds+1))
        assert len(exposure) == self.rounds*10
        assert len({(r['round'],r['client']) for r in exposure}) == len(exposure)
        for row in exposure: self.check_exposure(row,CLIENTS[row['client']],row['round'])
        prior = initial_hash
        for r in records:
            assert r['common_start_tensor_sha256'] == prior and r['weights'] == self.weights
            assert set(r['clients']) == set(CLIENTS)
            assert all(x['start_hash'] == prior for x in r['clients'].values())
            assert abs(r['metrics']['macro_accuracy']-sum(r['metrics'][d+'_accuracy'] for d in DOMAINS)/5) < 1e-12
            receipt = data.read(fed/('round_%02d' % r['round'])/'global/evaluation_provenance.json')
            assert receipt['domains'] == list(DOMAINS) and receipt['lora_tensor_sha256'] == r['global_tensor_sha256']
            assert shared.tensor_hash(torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True)) == r['global_tensor_sha256']
            prior = r['global_tensor_sha256']
        budget = self.protocol['budget']
        assert sum(r['samples'] for r in visits.values()) == budget['local_sample_visits']
        assert sum(r['updates'] for r in visits.values()) == budget['local_updates']
        assert sum(r['samples'] for r in exposure) == budget['fedavg_sample_visits']
        assert sum(r['optimizer_updates'] for r in exposure) == budget['fedavg_updates']
        audit = dict(passed=True,smoke=self.smoke,budget=budget,local_clients=visits,
            local_epoch_evaluations=len(CLIENTS)*self.epochs,global_evaluations=self.rounds,
            evaluation_capabilities=5,canonical_initial_tensor_sha256=initial_hash,
            optimizer_reset_per_client_epoch=True,aggregation=AGGREGATION,
            same_common_round_start=True,shared_test_per_source=True,base_or_centralized_rerun=False)
        data.dump(self.run_root/'protocol_audit.json',audit)
        return audit

    def check_exposure(self,row,c,epoch):
        ids = {data.native_key(r) for r in self.train_rows[c]}
        assert row['round'] == epoch and row['client'] == CLIENTS.index(c)
        assert row['samples'] == len(row['ids']) == len(set(row['ids'])) == self.counts[c]
        assert set(row['ids']) == ids
        assert row['optimizer_updates'] == (self.counts[c]+7)//8
        assert row['supervised_tokens'] == self.supervised_counts[c]


def reused_references():
    final_path = data.original.MASTER/'final_results.json'
    central_path = data.original.MASTER/'best_saved_analysis/centralized_best_result.json'
    final,central = data.read(final_path),data.read(central_path)
    assert central['selected_epoch'] == data.config()['centralized_reference_epoch'] == 6
    return dict(reused=True,base=final['results']['base'],original_local_epoch10=final['results']['clientlocal'],
        original_fedavg_round10=final['results']['fedavg'],centralized=central['selected_metrics'],
        centralized_epoch=6,centralized_selection='retrospective best fixed-test Macro from original ten epochs',
        centralized_epoch10=final['results']['centralized'],
        receipts={str(p.relative_to(ROOT)):data.sha(p) for p in (final_path,central_path)})


def report():
    integrity = data.verify()
    audit = data.read(MASTER/'protocol_audit.json'); assert audit['passed'] and not audit['smoke']
    curves = data.read(MASTER/'clientlocal/epoch_trajectories.json')
    local = local_summary(curves,10)
    records = data.read(MASTER/'fedavg/round_trajectory.json'); assert len(records) == 10
    fed = fed_summary(records); refs = reused_references()
    centralized, oldfed = refs['centralized'],refs['original_fedavg_round10']
    gap = {d:(centralized[d+'_accuracy']-fed['final_metrics'][d+'_accuracy'])*100 for d in DOMAINS}
    oldgap = {d:(centralized[d+'_accuracy']-oldfed[d+'_accuracy'])*100 for d in DOMAINS}
    macro_gap = (centralized['macro_accuracy']-fed['final_metrics']['macro_accuracy'])*100
    lines = ['# Setting B: 5-source / 10-client fragmentation experiment','',
        '**Aggregation: '+AGGREGATION+'.**','',
        'Variable: client fragmentation / local-isolation granularity. Fixed split seed42; no new examples or tasks.','',
        'Base and Centralized results are reused from the original 5-source experiment because the union of training examples, evaluation sets, and shared adapter class are unchanged.','',
        'Original Local uses epoch10. Primary FedAvg comparisons use final round10 in both settings. '
        'The reused centralized reference is the previously reported best shared epoch6 (test-Macro selected); '
        'its fixed epoch10 endpoint is also disclosed. Per-dataset peaks below are diagnostic and are not combined into a shared-model Macro.','',
        '## A. Data partition audit','',
        '| Source | Original train | A | B | Intersection | Union | Test | Split seed |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in DOMAINS:
        v=integrity['sources'][d]
        lines.append(f"| {data.NAMES[d]} | 4000 | 2000 | 2000 | 0 | 4000 | {v['test']} | 42 |")
    lines += ['', 'Every source union equals the original frozen subset. Tests are byte-identical and shared by A/B. '
        'Full sample ID/index mapping: `dataset/fragmentation_5source_10client_seed42/manifest.json`.','',
        '## B. Training budget and protocol','',
        '| Setting | Updates per client epoch | Clients | Epochs / rounds | Total updates per method | Sample visits per method |',
        '|---|---:|---:|---:|---:|---:|',
        '| Original | 500 | 5 | 10 | 25000 | 200000 |',
        '| Setting B | 250 | 10 | 10 | 25000 | 200000 |','',
        'Independent Local and FedAvg each have these budgets separately. Full participation; client weight0.1, source weight0.2. '
        'Same canonical initial LoRA, Llama-3.2-1B frozen BF16 base, q_proj/v_proj, r8, alpha32, dropout0.05, bias none, '
        'constant lr1e-4, AdamW (betas0.9/0.999, eps1e-8, weight_decay0.01), batch1, accumulation8, drop_last=False. '
        'Original Trainer resets AdamW once per client epoch. Original client_round_seed(42,client_id,epoch_or_round-1) '
        'with stable interleaved A/B IDs0..9. Original prompt/preprocessing and four-choice likelihood evaluator unchanged.','',
        '## C. Independent Local','', '| Client | Source | '+ ' | '.join('Epoch%d'%e for e in range(1,11))+' |',
        '|---|---|'+'---:|'*10]
    for c in CLIENTS: lines.append('| '+c+' | '+data.NAMES[SOURCE[c]]+' | '+' | '.join('%.2f%%'%(r['accuracy']*100) for r in curves[c])+' |')
    lines += ['', '| Source | A epoch10 | B epoch10 | Mean | Absolute A/B difference (pp) |', '|---|---:|---:|---:|---:|']
    for d in DOMAINS:
        v=local['sources'][d]
        lines.append(f"| {data.NAMES[d]} | {v['client_a']*100:.2f}% | {v['client_b']*100:.2f}% | {v['mean']*100:.2f}% | {v['absolute_difference']*100:.2f} |")
    lines += ['', f"Five-capability Local_split Macro = **{local['macro_accuracy']*100:.2f}%** (mean A/B first, then mean five sources).",'',
        'Training losses, checkpoints and predictions are retained for every client/epoch under `clientlocal/<client>/`.','',
        '## D. FedAvg global trajectory','',
        '| Round | '+' | '.join(data.NAMES[d] for d in DOMAINS)+' | Macro |','|---:|'+'---:|'*6]
    for r in records:
        lines.append('| '+str(r['round'])+' | '+' | '.join('%.2f%%'%(r['metrics'][d+'_accuracy']*100) for d in DOMAINS)+' | %.2f%% |'%(r['metrics']['macro_accuracy']*100))
    lines += ['', f"Best shared test-Macro round: **{fed['best_macro_round']}**, **{fed['best_macro_metrics']['macro_accuracy']*100:.2f}%**. "
        f"Final round10 Macro: **{fed['final_metrics']['macro_accuracy']*100:.2f}%**.",'',
        '| Source | Diagnostic peak round | Peak accuracy | Final10 accuracy |','|---|---:|---:|---:|']
    for d in DOMAINS:
        v=fed['dataset_peaks'][d]
        lines.append(f"| {data.NAMES[d]} | {v['round']} | {v['accuracy']*100:.2f}% | {v['final']*100:.2f}% |")
    lines += ['', '## E. Descriptive comparison','',
        '| Dataset | Original Local10 | Split Local10 mean | Original FedAvg10 | Split FedAvg10 | Local split − original (pp) | FedAvg split − original (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    comparison={}
    for d in (*DOMAINS,'macro'):
        key=d+'_accuracy'; a=refs['original_local_epoch10'][key]
        b=local['macro_accuracy'] if d=='macro' else local['sources'][d]['mean']
        c=oldfed[key]; e=fed['final_metrics'][key]
        comparison[d]=dict(original_local=a,split_local=b,original_fedavg=c,split_fedavg=e,
            local_delta_pp=(b-a)*100,fedavg_delta_pp=(e-c)*100)
        lines.append(f"| {data.NAMES.get(d,'Macro')} | {a*100:.2f}% | {b*100:.2f}% | {c*100:.2f}% | {e*100:.2f}% | {(b-a)*100:+.2f} | {(e-c)*100:+.2f} |")
    lines += ['', '## F. Reused references and integration gaps','',
        '| Dataset | Reused Base | Reused Central best6 | Reused Central final10 | Central best6 − Original FedAvg10 (pp) | Central best6 − Split FedAvg10 (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for d in (*DOMAINS,'macro'):
        k=d+'_accuracy'; g=(centralized[k]-fed['final_metrics'][k])*100; og=(centralized[k]-oldfed[k])*100
        lines.append(f"| {data.NAMES.get(d,'Macro')} | {refs['base'][k]*100:.2f}% | {centralized[k]*100:.2f}% | {refs['centralized_epoch10'][k]*100:.2f}% | {og:+.2f} | {g:+.2f} |")
    lines += ['', 'These are descriptive differences for one fixed split and seed. No automatic causal or mechanism explanation.','',
        'Audit: unchanged source/test examples and protected artifacts; identical per-method update/visit budgets; '
        '100 Local epoch evaluations, ten global evaluations on five sources; same round start for all ten clients; '
        'unchanged standard aggregation; no Base or Centralized rerun.','',
        'Artifacts: `outputs/fragmentation_5source_10client_seed42/`; protocol/configuration receipts, exposure IDs, '
        'epoch/round checkpoints, predictions with four choice log probabilities, JSON/CSV trajectories and audit files.']
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    data.dump(MASTER/'reused_references.json',refs)
    data.dump(MASTER/'final_results.json',dict(local=local,fedavg=fed,reused=refs,comparison=comparison,
        integration_gap_pp=gap,original_integration_gap_pp=oldgap,macro_integration_gap_pp=macro_gap,
        audit=audit,data_audit=integrity,report=str(REPORT)))
    return REPORT


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True,choices=('prepare','smoke','local','fedavg','report'))
    args=parser.parse_args(); configure(); MASTER.mkdir(parents=True,exist_ok=True)
    with (MASTER/'run.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: raise SystemExit('Fragmentation job already running')
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            data.prepare()
            print('FORMAL PREFLIGHT BUDGET',json.dumps(data.budgets()),flush=True)
            print('AGGREGATION',AGGREGATION,flush=True)
            if args.job=='prepare': shared.status('data_ready'); return
            if args.job=='report':
                path=report(); shared.status('complete',report=str(path))
                (MASTER/'completed.txt').write_text('Setting B Local10 -> FedAvg10 -> report complete.\n',encoding='utf-8'); return
            ex=Experiment(smoke=args.job=='smoke')
            if args.job=='smoke':
                ex.local(); ex.fedavg(); audit=ex.audit()
                data.dump(MASTER/'smoke_audit.json',dict(passed=audit['passed'],protocol=ex.protocol,
                    outputs={k:str(v.relative_to(ROOT)) for k,v in ex.outputs.items()},audit=audit))
                shared.status('smoke_complete'); return
            smoke=data.read(MASTER/'smoke_audit.json')
            assert smoke['passed'] and smoke['protocol']['suite_runner_sha256']==data.sha(Path(__file__))
            assert smoke['protocol']['data_helper_sha256']==data.sha(Path(data.__file__))
            if args.job=='local': ex.local(); shared.status('local_complete')
            else:
                assert (MASTER/'clientlocal/completed.txt').exists()
                ex.fedavg(); ex.audit(); shared.status('fedavg_complete')
        except BaseException as error:
            shared.status('failed',error=str(error),traceback=traceback.format_exc()); traceback.print_exc(); raise SystemExit(1)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__': main()
