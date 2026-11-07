# Published numerical examples plus behavior that matters for training

from pathlib import Path

import numpy as np
import pytest
import torch

from captioning.trainer import caption_loss
from captioning.model import CaptioningTransformer
from captioning.data import decode_captions
from captioning.layers import (
    MultiHeadAttention, PositionalEncoding, TransformerDecoderLayer,
)

torch.set_num_threads(2)


@pytest.mark.parametrize("name,tolerance", [
    ("attention", 7e-3), ("position", 1e-3),
    ("decoder", 1e-6), ("captioner", 1e-5),
])
def test_numerical_example(name, tolerance):
    errors = []
    absolute_errors = []

    def rel_error(expected, actual):
        error = np.max(np.abs(expected - actual) / np.maximum(1e-8, np.abs(expected) + np.abs(actual)))
        errors.append(error)
        absolute_errors.append(np.max(np.abs(expected - actual)))
        return error

    # Keep upstream numerical fixtures fixed as the implementation changes
    namespace = dict(globals(), rel_error=rel_error)
    fixture = Path(__file__).parent / "fixtures" / f"{name}.py"
    exec(compile(fixture.read_text(), str(fixture), "exec"), namespace)
    assert errors
    if name == "decoder":
        # Allow small float32 rounding differences across PyTorch and BLAS builds
        # Require absolute error below 5e-7 or the original relative threshold
        assert max(errors) < tolerance or max(absolute_errors) < 5e-7
    else:
        assert max(errors) < tolerance


def small_model():
    torch.manual_seed(7)
    vocabulary = {"<NULL>": 0, "<START>": 1, "<END>": 2, "cat": 3, "dog": 4}
    return CaptioningTransformer(vocabulary, 4, 8, num_heads=2, num_layers=1,
                                 max_length=6, dim_feedforward=16, dropout=0)


def test_future_words_cannot_change_earlier_predictions():
    model = small_model().eval()
    features = torch.randn(1, 4)
    first = model(features, torch.tensor([[1, 3, 4, 2]]))
    changed = model(features, torch.tensor([[1, 3, 2, 3]]))
    torch.testing.assert_close(first[:, :2], changed[:, :2])


def test_image_features_affect_predictions():
    model = small_model().eval()
    captions = torch.tensor([[1, 3]])
    first = model(torch.zeros(1, 4), captions)
    second = model(torch.ones(1, 4) * 10, captions)
    assert not torch.allclose(first, second)


def test_attention_matches_pytorch_reference_and_gradients():
    # Compare with PyTorch attention using different query and key lengths
    torch.manual_seed(23)
    attention = MultiHeadAttention(12, 3, dropout=0).double()
    query = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
    key = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
    value = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(3, 5, dtype=torch.bool).tril()
    actual = attention(query, key, value, mask)
    q = attention.query(query).reshape(2, 3, 3, 4).transpose(1, 2)
    k = attention.key(key).reshape(2, 5, 3, 4).transpose(1, 2)
    v = attention.value(value).reshape(2, 5, 3, 4).transpose(1, 2)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    expected = attention.proj(expected.transpose(1, 2).reshape(2, 3, 12))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    actual_grads = torch.autograd.grad(actual.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)


def test_padding_has_zero_gradient_and_full_sequence_denominator():
    logits = torch.zeros(1, 2, 5, requires_grad=True)
    loss = caption_loss(logits, torch.tensor([[3, 0]]), null_token=0)
    assert loss.item() == pytest.approx(np.log(5) / 2)
    loss.backward()
    assert logits.grad[0, 1].count_nonzero() == 0
    assert logits.grad[0, 0].count_nonzero() > 0


def test_sampling_stops_and_restores_training_mode():
    model = small_model().train()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output.bias[2] = 10  # Force END as the first prediction
    captions = model.sample(np.zeros((2, 4)), max_length=4)
    np.testing.assert_array_equal(captions, [[2, 0, 0, 0], [2, 0, 0, 0]])
    assert model.training
    with pytest.raises(ValueError, match="max_length"):
        model.sample(np.zeros((1, 4)), max_length=7)


def test_tiny_training_can_learn_two_different_captions():
    model = small_model()
    features = torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0]])
    captions = torch.tensor([[1, 3, 2], [1, 4, 2]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(120):
        loss = caption_loss(model(features, captions[:, :-1]), captions[:, 1:], 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05
    np.testing.assert_array_equal(model.sample(features, 3), [[3, 2, 0], [4, 2, 0]])


def test_decode_hides_special_tokens_and_stops_at_end():
    assert decode_captions(np.array([1, 3, 2, 4, 0]),
                           ["<NULL>", "<START>", "<END>", "cat", "dog"]) == "cat"
