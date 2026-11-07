# Turn precomputed image features into a sentence, one word at a time

# Source credits are recorded in the project notice

import copy

import torch
from torch import nn

from .layers import PositionalEncoding, TransformerDecoderLayer


class CaptioningTransformer(nn.Module):
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>")
        self._end = word_to_idx.get("<END>")
        self.max_length = max_length
        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(self.vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, dropout, max_length)
        layer = TransformerDecoderLayer(wordvec_dim, num_heads, dim_feedforward, dropout)
        self.transformer = TransformerDecoder(layer, num_layers)
        self.apply(self._init_weights)
        # Create the output layer after initializing the other layers
        self.output = nn.Linear(wordvec_dim, self.vocab_size)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, features, captions):
        # (N, image_features), (N, words) -> (N, words, vocabulary)
        words = self.positional_encoding(self.embedding(captions))
        image = self.visual_projection(features).unsqueeze(1)
        length = captions.shape[1]
        # A word may see itself and earlier words, never the future answer
        mask = torch.ones(length, length, dtype=torch.bool, device=captions.device).tril()
        decoded = self.transformer(words, image, mask)
        return self.output(decoded)

    @torch.no_grad()
    def sample(self, features, max_length=30):
        # Greedily choose words; return NumPy IDs padded after <END>
        if self._start is None:
            raise ValueError("Sampling requires a <START> token")
        if not 1 <= max_length <= self.max_length:
            raise ValueError(f"max_length must be between 1 and {self.max_length}")
        parameter = next(self.parameters())
        features = torch.as_tensor(features, dtype=parameter.dtype, device=parameter.device)
        batch = features.shape[0]
        result = torch.full((batch, max_length), self._null, device=features.device, dtype=torch.long)
        prefix = torch.full((batch, 1), self._start, device=features.device, dtype=torch.long)
        finished = torch.zeros(batch, device=features.device, dtype=torch.bool)
        was_training = self.training
        self.eval()
        try:
            for step in range(max_length):
                next_word = self(features, prefix)[:, -1].argmax(dim=-1)
                next_word = next_word.masked_fill(finished, self._null)
                result[:, step] = next_word
                prefix = torch.cat([prefix, next_word[:, None]], dim=1)
                if self._end is not None:
                    finished = finished | (next_word == self._end)
                    if finished.all():
                        break
        finally:
            self.train(was_training)
        return result.cpu().numpy()


class TransformerDecoder(nn.Module):
    # Apply a short list of decoder layers in order

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask)
        return tgt
