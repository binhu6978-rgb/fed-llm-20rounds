import os
import random

from peft import get_peft_model
from utils.data_utils import load_data, load_global_test_data
from utils.sys_utils import device_config
from utils.train_utils import Trainer
from utils.eval_utils import Evaluator
from alg.base import BaseClient, BaseServer
from utils.model_utils import load_model, load_tokenizer, load_lora_config
from utils.time_utils import time_record


class FTBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.tokenizer = load_tokenizer(args)
        self.dataset = load_data(args=args, idx=self.id)
        self.lora = {}
        self.last_train_loss = None
        self.trainer = Trainer(args=args, dataset=self.dataset, client=self)
        self.evaluator = None if (args.global_test or getattr(args, 'cross_domain_qa', False) or getattr(args, 'mcq', False)) else Evaluator(args=args, dataset=self.dataset)

    @time_record
    def run(self, model):
        self.last_train_loss = self.trainer.train(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}

    def local_test(self, model):
        if self.evaluator is None:
            raise RuntimeError("Client-local evaluation is disabled for global_test or cross-domain QA runs")
        return self.evaluator.evaluate(model, round_idx=self.server.round, client_id=self.id)

class FTBaseServer(BaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.model = get_peft_model(load_model(args), load_lora_config(args))

        self.global_lora = {k: v.clone() for k, v in self.model.state_dict().items() if "lora_" in k}
        if getattr(args, 'mcq', False):
            import json
            import torch
            from pathlib import Path
            metadata = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert 'lora_' in name, name
                    metadata.append({'name': name, 'shape': list(param.shape),
                                     'dtype': str(param.dtype), 'numel': param.numel()})
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            output = Path(args.suffix)
            output.mkdir(parents=True, exist_ok=True)
            (output / 'parameters.json').write_text(json.dumps({
                'total_parameters': total, 'trainable_parameters': trainable,
                'base_dtype': str(next(p for n, p in self.model.named_parameters() if 'lora_' not in n).dtype),
                'trainable': metadata,
            }, indent=2), encoding='utf-8')
            torch.save({k: v.detach().cpu() for k, v in self.global_lora.items()}, output / 'initial_lora.pt')
            print(f'Total parameters: {total}; trainable parameters: {trainable}', flush=True)
            for item in metadata:
                print(f"LoRA {item['name']} {item['shape']} {item['dtype']}", flush=True)
        self.sample_rate = args.sr
        self.wall_clock_time = 0
        self.round = 0

        if getattr(args, 'mcq', False):
            from utils.mcq_eval import MCQEvaluator
            self.global_test_dataset = None
            # Final-only protocols must not read the official test during training.
            self.global_evaluator = None if (getattr(args, 'final_only_test', False) or getattr(args, 'skip_mcq_test', False)) else MCQEvaluator(args, clients[0].tokenizer)
        elif args.global_test:
            self.global_test_dataset = load_global_test_data(args)
            self.global_evaluator = Evaluator(args=args, dataset=self.global_test_dataset)
        elif getattr(args, 'cross_domain_qa', False):
            # This evaluator operates on raw named test files and performs
            # generation-based MRQA metrics independently for each domain.
            from utils.cross_domain_qa import CrossDomainQAEvaluator
            self.global_test_dataset = None
            self.global_evaluator = CrossDomainQAEvaluator(args)
        else:
            self.global_test_dataset = None
            self.global_evaluator = None

        for client, delay in zip(clients, device_config(args)): client.delay = delay

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def sample(self):
        sample_num = int(self.sample_rate * len(self.clients))
        self.sampled_clients = sorted(random.sample(self.clients, sample_num), key=lambda x: x.id)

    def local_run(self):
        for client in self.sampled_clients:
            client.run(self.model)
            self.model.load_state_dict(self.global_lora, strict=False)
        self.wall_clock_time += max([c.training_time for c in self.sampled_clients])

    def aggregate(self):
        from collections import defaultdict
        aggregated = defaultdict(lambda: 0)
        data_sum = sum(len(client.dataset['train']) for client in self.sampled_clients)
        client_weights = {
            client.id: len(client.dataset['train']) / data_sum
            for client in self.sampled_clients
        }
        for client in self.sampled_clients:
            model = client.lora
            for k, v in model.items():
                aggregated[k] = aggregated[k] + v * client_weights[client.id]

        self.global_lora = aggregated
        self.model.load_state_dict(self.global_lora, strict=False)
        print("Aggregated model updated.")

    def test_all(self):
        if getattr(self.args, 'mcq', False):
            if self.global_evaluator is None:
                from utils.mcq_eval import MCQEvaluator
                self.global_evaluator = MCQEvaluator(self.args, self.clients[0].tokenizer)
            return self.global_evaluator.evaluate(self.model, self.round + 1)
        if getattr(self.args, 'cross_domain_qa', False):
            print("Testing aggregated global model on named extractive-QA domain test sets ...")
            return self.global_evaluator.evaluate(
                self.model, self.clients[0].tokenizer, round_idx=self.round, method=self.args.alg
            )
        if self.global_evaluator is not None:
            print("Testing aggregated global model on shared global test set ...")
            metrics = self.global_evaluator.evaluate(
                self.model, round_idx=self.round, client_id=None
            )
            if self.args.round_generation_metrics:
                # Import lazily to keep the ordinary training path lightweight
                # and avoid loading generation evaluation code for other tasks.
                from eval import (
                    _ExactMatchEvaluator,
                    _F1Evaluator,
                    _generate_predictions,
                    _save_predictions,
                )

                predictions = _generate_predictions(
                    self.model, self.clients[0].tokenizer, self.args, client_idx=None
                )
                metrics.update(_ExactMatchEvaluator().evaluate(predictions))
                metrics.update(_F1Evaluator().evaluate(predictions))
                prediction_path = _save_predictions(
                    predictions,
                    self.args,
                    client_idx=None,
                    filename=f'global_round_{self.round}_predictions.jsonl',
                )
                print(f'Round {self.round} predictions saved to {prediction_path}')
            return metrics

        all_metrics = []
        for client in self.clients:
            print(f"Testing on client {client.id} ...")
            metrics = client.local_test(self.model)
            all_metrics.append(metrics)

        res_dict = {}
        for k in all_metrics[0].keys():
            res_dict[k] = sum(m[k] for m in all_metrics) / len(all_metrics)

        return res_dict

    def _adapter_path(self):
        args = self.args
        dataset_id = args.dataset.replace('/', '_').replace('\\', '_')
        model_id = args.model.replace('/', '_').replace('\\', '_').replace(':', '_')
        name = (
            f'{args.alg}_{dataset_id}_{model_id}_'
            f'{args.cn}c_{args.epoch}E_lr{args.lr}'
        )
        return os.path.join(args.suffix, 'adapter', name)

    def save_round_adapter(self, round_idx):
        """Persist the post-aggregation global adapter for audit/re-evaluation."""
        import torch
        adapter_path = os.path.join(self._adapter_path(), f'round_{round_idx:03d}')
        os.makedirs(adapter_path, exist_ok=True)
        torch.save(dict(self.global_lora), os.path.join(adapter_path, 'lora_weights.pt'))
        print(f'Round {round_idx} adapter saved to {adapter_path}')

    def save_client_updates(self, round_idx):
        """Persist each sampled client's post-training LoRA factors (Δ_i) for this round."""
        import torch
        for client in self.sampled_clients:
            client_dir = os.path.join(self._adapter_path(), f'round_{round_idx:03d}', 'clients')
            os.makedirs(client_dir, exist_ok=True)
            torch.save(dict(client.lora), os.path.join(client_dir, f'client_{client.id}.pt'))

    def save_adapter(self):
        import json
        import torch
        args = self.args
        adapter_path = self._adapter_path()
        os.makedirs(adapter_path, exist_ok=True)
        torch.save(dict(self.global_lora), os.path.join(adapter_path, 'lora_weights.pt'))
        with open(os.path.join(adapter_path, 'training_config.json'), 'w', encoding='utf-8') as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
        print(f'Adapter saved to {adapter_path}')
