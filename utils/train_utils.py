import time
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from utils.seed_utils import client_round_seed, seed_rngs


class Trainer:
    """
    Stateless trainer decoupled from model architecture.
    Instantiated once per client and reused across rounds.
    """

    def __init__(self, args, dataset, client):
        self.args = args
        self.client = client
        self.task_type = args.task_type
        self.shuffle_generator = torch.Generator()
        self.train_loader = DataLoader(
            dataset['train'],
            batch_size=self.args.bs,
            shuffle=True,
            drop_last=not getattr(self.args, 'mcq', False),
            generator=self.shuffle_generator,
        )

    def _set_round_seed(self):
        seed = client_round_seed(
            self.args.seed, self.client.id, self.client.server.round
        )
        seed_rngs(seed)
        self.shuffle_generator.manual_seed(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, model):
        """Run one round and return the mean unscaled loss over seen batches."""
        if getattr(self.args, 'mcq', False):
            return self._train_mcq(model)
        self._set_round_seed()
        model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.client.server.round)
        )
        scaler = GradScaler()

        train_loss = self._train_loop(model, optimizer, scaler)
        self._log(train_loss)
        return train_loss

    def _train_mcq(self, model):
        """One complete local epoch, constant LR and round-wise AdamW reset."""
        import json
        from pathlib import Path
        self._set_round_seed()
        model.train()
        model.config.use_cache = False
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
        )
        assert self.args.bs == 1 and self.args.epoch == 1
        accumulation = self.args.grad_accum
        count = len(self.train_loader)
        consumed, loss_sum, updates, supervised = [], 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(self.train_loader):
            ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
            labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
            mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
            window_start = (index // accumulation) * accumulation
            window_size = min(accumulation, count - window_start)
            with autocast('cuda', dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite MCQ loss')
            (loss / window_size).backward()
            loss_sum += loss.detach().float().item()
            consumed.extend(batch['native_key'])
            supervised += int((labels != -100).sum().item())
            if (index + 1) % accumulation == 0 or index + 1 == count:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        assert len(consumed) == len(self.client.dataset['train'])
        assert len(set(consumed)) == len(consumed)
        self.client.last_optimizer_updates = updates
        self.client.last_seen_count = len(consumed)
        output = Path(self.args.suffix) / 'exposure.jsonl'
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'round': self.client.server.round + 1, 'client': self.client.id,
                'samples': len(consumed), 'optimizer_updates': updates,
                'supervised_tokens': supervised, 'ids': consumed,
            }) + '\n')
        result = loss_sum / count
        print(f'Round {self.client.server.round + 1} Client {self.client.id}: '
              f'samples={len(consumed)} updates={updates} loss={result:.6f}', flush=True)
        return result

    def train_windowed(self, model, start_time, window_end_time):
        """Run local training but stop early when the simulated window closes.

        Args:
            model: the model to train.
            start_time: simulated wall-clock datetime when training is assigned.
            window_end_time: simulated datetime at which the device's window
                closes; training stops after the first gradient update whose
                real elapsed time exceeds this simulated budget.

        Returns:
            (loss, truncated): the last computed loss and a bool indicating
            whether the run was cut short by the window deadline.
        """
        self._set_round_seed()
        model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.client.server.round)
        )
        scaler = GradScaler()

        delay = getattr(self.client, 'delay', 1.0)
        loss, truncated, global_step = self._train_loop_windowed(model, optimizer, scaler, start_time, window_end_time, delay=delay)
        self._log(loss)
        return loss, truncated, global_step

    def train_resumed(self, model, start_step: int, start_time, window_end_time):
        """Resume windowed training from *start_step* completed gradient steps.

        Loads model weights from the caller (already restored externally) and
        continues for the remaining ``args.step - start_step`` optimizer steps,
        still subject to the window deadline.

        Returns:
            (loss, truncated, completed_steps)
        """
        self._set_round_seed()
        model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.client.server.round)
        )
        scaler = GradScaler()

        delay = getattr(self.client, 'delay', 1.0)
        loss, truncated, global_step = self._train_loop_windowed(
            model, optimizer, scaler, start_time, window_end_time, resume_step=start_step, delay=delay
        )
        self._log(loss)
        return loss, truncated, global_step

    # ------------------------------------------------------------------
    # Per-task implementations
    # ------------------------------------------------------------------

    def _forward(self, model, batch):
        if self.task_type == 'SEQ_CLS':
            return self._forward_seq_cls(model, batch)
        elif self.task_type == 'CAUSAL_LM':
            return self._forward_causal_lm(model, batch)
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _forward_seq_cls(self, model, batch):
        input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
        attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
        labels = torch.tensor(batch['labels']).to(model.device)
        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def _forward_causal_lm(self, model, batch):
        input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
        attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
        labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_loop(self, model, optimizer, scaler):
        accumulation_steps = self.args.grad_accum
        global_step = 0
        loss_sum = None
        batch_count = 0
        done = False

        optimizer.zero_grad()
        for _ in range(self.args.epoch):
            if done:
                break
            for step, batch in enumerate(self.train_loader):
                unscaled_loss = self._forward(model, batch)
                detached_loss = unscaled_loss.detach().float()
                loss_sum = detached_loss if loss_sum is None else loss_sum + detached_loss
                batch_count += 1
                loss = unscaled_loss / accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step >= self.args.step:
                    done = True
                    break

        if batch_count == 0:
            raise RuntimeError("Training DataLoader produced no batches")
        if (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Synchronise with the device only once per round, rather than calling
        # ``.item()`` for every micro-batch in the hot training loop.
        return float((loss_sum / batch_count).item())

    def _train_loop_windowed(self, model, optimizer, scaler, start_time, window_end_time, resume_step: int = 0, delay: float = 1.0):
        accumulation_steps = self.args.grad_accum
        global_step = resume_step
        loss = None
        done = False
        truncated = False

        # real_elapsed = sim_budget / delay  (because training_time = real_elapsed * delay)
        sim_budget = (window_end_time - start_time).total_seconds()
        real_deadline = time.time() + sim_budget / delay

        optimizer.zero_grad()
        for _ in range(self.args.epoch):
            if done:
                break
            for step, batch in enumerate(self.train_loader):
                loss = self._forward(model, batch) / accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if time.time() > real_deadline:
                        
                        truncated = True
                        done = True
                        break

                if global_step >= self.args.step:
                    done = True
                    break

        if not truncated and (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return loss, truncated, global_step

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, loss):
        # Ordinary training returns a float containing the round-average
        # unscaled loss. Windowed training still supplies its last scaled
        # tensor, so retain backward-compatible handling for that path.
        if torch.is_tensor(loss):
            loss_value = loss.item() * self.args.grad_accum
        else:
            loss_value = float(loss)
        print(
            f"Round {self.client.server.round} | "
            f"Client {self.client.id} | "
            f"Train Loss: {loss_value:.4f}"
        )
