"""Independent LoRA state for each existing Domain-Skewed MCQ client."""

from pathlib import Path

import torch

from alg.ftbase import FTBaseClient, FTBaseServer


class Client(FTBaseClient):
    pass


class Server(FTBaseServer):
    def __init__(self, args, clients, initial_path):
        super().__init__(args, clients)
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        current = {k: v for k, v in self.model.state_dict().items() if 'lora_' in k}
        if set(initial) != set(current):
            raise ValueError('Initial LoRA keys do not match this model')
        for key in initial:
            if initial[key].shape != current[key].shape:
                raise ValueError(f'Initial LoRA shape differs: {key}')
        self.model.load_state_dict(initial, strict=False)
        self.initial_lora = {k: v.clone() for k, v in initial.items()}
        self.client_loras = {
            client.id: {k: v.clone() for k, v in initial.items()}
            for client in clients
        }
        torch.save(self.initial_lora, Path(args.suffix) / 'initial_lora.pt')

    def run(self):
        raise RuntimeError('Run ClientLocal via scripts/run_bhashabench_clientlocal.py')

    def aggregate(self):
        raise RuntimeError('ClientLocal never aggregates')

    def train_client_epoch(self, client, epoch):
        self.round = epoch - 1
        self.model.load_state_dict(self.client_loras[client.id], strict=False)
        current = self.model.state_dict()
        if not all(torch.equal(current[k].cpu(), v) for k, v in self.client_loras[client.id].items()):
            raise RuntimeError(f'Client {client.id} did not receive its own adapter')
        client.run(self.model)
        self.client_loras[client.id] = {k: v.detach().cpu().clone() for k, v in client.lora.items()}
        return client.last_train_loss

    def save_client_adapter(self, client_id):
        path = Path(self.args.suffix) / f'client_{client_id:02d}' / 'lora_weights.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.client_loras[client_id], path)
        return path
