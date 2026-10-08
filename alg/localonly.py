"""Four independent domain-local LoRA models with no server aggregation."""

from __future__ import annotations

import json
import os

import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


class Client(FTBaseClient):
    @time_record
    def run(self, model):
        super().run(model)


class Server(FTBaseServer):
    """Keep one persistent LoRA state per client and never aggregate them."""

    def __init__(self, args, clients):
        super().__init__(args, clients)
        if self.global_evaluator is None:
            raise ValueError("localonly cross-domain QA requires cross_domain_qa: true")
        self.domains = self.global_evaluator.domains
        if len(clients) != len(self.domains):
            raise ValueError(
                f"localonly cross-domain QA requires {len(self.domains)} clients, got {len(clients)}"
            )
        if float(args.sr) != 1.0:
            raise ValueError("localonly requires sr: 1.0 so every domain trains each round")
        self.client_loras = {
            client.id: {key: value.clone() for key, value in self.global_lora.items()}
            for client in clients
        }

    def run(self):
        self.sample()
        self.local_run()
        # Intentionally no call to aggregate(): local adapters remain separate.

    def local_run(self):
        for client in self.sampled_clients:
            self.model.load_state_dict(self.client_loras[client.id], strict=False)
            client.run(self.model)
            self.client_loras[client.id] = {
                key: value.clone() for key, value in client.lora.items()
            }
        # Leave the shared model container in its neutral initial state. Every
        # subsequent train/eval explicitly loads the intended client's LoRA.
        self.model.load_state_dict(self.global_lora, strict=False)
        self.wall_clock_time += max(client.training_time for client in self.sampled_clients)

    def aggregate(self):
        """No-op required by the common server interface."""

    def get_round_train_metrics(self) -> dict[str, float]:
        metrics = {}
        for client in self.sampled_clients:
            if client.last_train_loss is not None:
                metrics[f"{self.domains[client.id]}_train_loss"] = client.last_train_loss
        return metrics

    def test_all(self):
        per_domain = []
        for client in self.clients:
            domain = self.domains[client.id]
            self.model.load_state_dict(self.client_loras[client.id], strict=False)
            per_domain.append(
                self.global_evaluator.evaluate_domain(
                    self.model,
                    client.tokenizer,
                    domain=domain,
                    round_idx=self.round,
                    method=self.args.alg,
                )
            )
        self.model.load_state_dict(self.global_lora, strict=False)
        macro = self.global_evaluator.append_macro(per_domain, self.round, self.args.alg)

        metrics = {}
        for row in per_domain:
            domain = row["domain"]
            metrics[f"{domain}_em"] = row["em"]
            metrics[f"{domain}_f1"] = row["f1"]
            metrics[f"{domain}_containment"] = row["containment"]
        metrics.update({
            "macro_em": macro["em"],
            "macro_f1": macro["f1"],
            "macro_containment": macro["containment"],
        })
        return metrics

    def save_round_adapter(self, round_idx):
        """No global adapter exists; per-client snapshots are saved separately."""
        print(f"Round {round_idx}: local-only has no global adapter to save")

    def save_adapter(self):
        adapter_path = self._adapter_path()
        os.makedirs(adapter_path, exist_ok=True)
        for client_id, lora in self.client_loras.items():
            torch.save(dict(lora), os.path.join(adapter_path, f"client_{client_id}.pt"))
        with open(os.path.join(adapter_path, "training_config.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(self.args), handle, ensure_ascii=False, indent=2, default=str)
        print(f"Final local adapters saved to {adapter_path}")
