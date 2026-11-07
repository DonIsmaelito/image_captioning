# Training loop with source credits in the project notice

import torch
from torch.nn import functional as F

from .data import sample_coco_minibatch


def caption_loss(logits, targets, null_token):
    # Zero padding losses and average over all positions

    # Keep padding positions in the denominator
    # This keeps the loss scale consistent between runs
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    )
    mask = targets.reshape(-1) != null_token
    return (losses * mask).mean()


class CaptionTrainer:
    def __init__(self, model, data, idx_to_word=None, learning_rate=0.001,
                 batch_size=25, num_epochs=100, print_every=10, verbose=True):
        if batch_size < 1 or num_epochs < 1 or print_every < 1:
            raise ValueError("batch_size, num_epochs and print_every must be positive")
        self.model = model
        self.data = data
        self.idx_to_word = idx_to_word
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.verbose = verbose
        self.optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_history = []
        self.epoch = 0

    def _step(self):
        captions, features, _ = sample_coco_minibatch(self.data, self.batch_size)
        parameter = next(self.model.parameters())
        features = torch.as_tensor(features, dtype=parameter.dtype, device=parameter.device)
        captions = torch.as_tensor(captions, dtype=torch.long, device=parameter.device)
        # Input: <START> a dog -> target: a dog <END>
        logits = self.model(features, captions[:, :-1])
        loss = caption_loss(logits, captions[:, 1:], self.model._null)
        if not torch.isfinite(loss):
            raise RuntimeError("Training produced a non-finite loss")
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.loss_history.append(loss.item())

    def train(self):
        self.model.train()
        steps_per_epoch = max(len(self.data["train_captions"]) // self.batch_size, 1)
        total_steps = self.num_epochs * steps_per_epoch
        for step in range(total_steps):
            self._step()
            if self.verbose and (step % self.print_every == 0 or step == total_steps - 1):
                print(f"Step {step + 1}/{total_steps}: loss {self.loss_history[-1]:.6f}", flush=True)
            if (step + 1) % steps_per_epoch == 0:
                self.epoch += 1
        return self.loss_history
