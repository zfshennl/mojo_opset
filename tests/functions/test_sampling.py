import math
import random

from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

KS_ALPHA = 1e-4

PENALTY_SHAPES = [(20, 151936)]


REJECTION_CASES = [(15, 155136, 3)]


SAMPLING_CASES = [
    ("top_k_sampling", (20, 151936), dict(top_k=10, min_tokens_to_keep=1)),
    ("top_p_sampling", (20, 151936), dict(top_p=0.75, min_tokens_to_keep=1, rand_top_k=1000)),
    ("top_p_filter", (20, 151936), dict(top_p=0.75, min_tokens_to_keep=1, rand_top_k=1000)),
    ("top_p_filter", (60, 155136), dict(top_p=0.7, min_tokens_to_keep=1, rand_top_k=100)),
]


def accepted_prefix(result):
    tokens, lengths = result
    valid = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
    return tokens.masked_fill(~valid, 0).long(), lengths.long()


def kolmogorov_sf(x):
    total, factor, previous = 0.0, 2.0, 0.0
    for k in range(1, 101):
        term = factor * math.exp(-2.0 * k * k * x * x)
        total += term
        if abs(term) <= 0.001 * previous or abs(term) <= 1e-8 * total:
            return total
        factor = -factor
        previous = abs(term)
    return 1.0


def ks_2samp(actual, expected, rtol=1e-5):
    n_actual, n_expected = actual.shape[1], expected.shape[1]
    combined = torch.cat([actual, expected], dim=1)
    order = torch.argsort(combined, dim=1)
    sorted_values = torch.gather(combined, 1, order)
    cdf_actual = torch.cumsum((order < n_actual).to(actual.dtype), dim=1) / n_actual
    cdf_expected = torch.cumsum((order >= n_actual).to(actual.dtype), dim=1) / n_expected
    difference = (cdf_actual - cdf_expected).abs()
    # Different fp32 reductions make the same logical probability differ by ~1e-7
    # across implementations; merge adjacent values within the relative tolerance
    # into one tie group, or every group splits and D jumps by the group's mass.
    magnitude = torch.maximum(sorted_values[:, :-1].abs(), sorted_values[:, 1:].abs()).clamp_min(1e-30)
    group_end = torch.ones_like(difference, dtype=torch.bool)
    group_end[:, :-1] = (sorted_values[:, 1:] - sorted_values[:, :-1]) > rtol * magnitude
    statistic = (difference * group_end).amax(dim=1)
    en = math.sqrt(n_actual * n_expected / (n_actual + n_expected))
    p_value = torch.tensor([kolmogorov_sf(en * value) for value in statistic.tolist()])
    return statistic, p_value


def check_sampling(name, actual_fn, reference_fn, logits):
    if name == "top_p_filter":
        actual_probs, actual_ids = actual_fn(logits.clone())
        expected_probs, expected_ids = reference_fn(logits.clone())
        assert_close(actual_probs.float(), expected_probs.float(), torch.float32, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(actual_ids.sort(-1).values, expected_ids.sort(-1).values, rtol=0, atol=0)
        return
    # Different sampling algorithms need not draw the same token under one seed.
    # Draw 200 samples per row and compare the drawn-probability distributions with a
    # two-sample KS test, per row and pooled: per row catches localized errors, pooled
    # catches small biases shared by all rows. ks_2samp merges values within fp32
    # jitter so equivalent probabilities from different reductions stay one tie group.
    actual_probs_list, expected_probs_list = [], []
    for _ in range(200):
        expected_p, expected_ids = reference_fn(logits.clone())
        actual_p, actual_ids = actual_fn(logits.clone())
        assert actual_ids.shape == expected_ids.shape
        assert bool(((actual_ids >= 0) & (actual_ids < logits.shape[-1])).all())
        actual_probs_list.append(actual_p.float())
        expected_probs_list.append(expected_p.float())
    actual, expected = torch.cat(actual_probs_list, 1), torch.cat(expected_probs_list, 1)
    cases = [
        (f"row {index}", actual[index : index + 1], expected[index : index + 1]) for index in range(actual.shape[0])
    ]
    cases.append(("pooled rows", actual.reshape(1, -1), expected.reshape(1, -1)))
    # KS_ALPHA is the false-fail budget for the whole run; splitting it across all
    # checks (Bonferroni) keeps repeated runs with fresh seeds from accumulating flakes.
    threshold = KS_ALPHA / len(cases)
    for label, rows_actual, rows_expected in cases:
        statistic, p_value = ks_2samp(rows_actual, rows_expected)
        stat, p = statistic.item(), p_value.item()
        assert p >= threshold, f"{label}: sampling distributions differ (ks_statistic={stat:.4g}, p_value={p:.4g})"


def make_penalty_case(shape, device):
    batch, vocab = shape
    logits = torch.randn(shape, device=device)
    frequencies = torch.randint(0, 5, shape, device=device, dtype=torch.int32)
    frequencies[torch.rand(shape, device=device) > 0.05] = 0
    token_freqs = [row if bool(row.any()) else None for row in frequencies]
    rng = random.Random(42)
    frequency = [rng.uniform(-0.5, 0.5) for _ in range(batch)]
    presence = [rng.uniform(-0.5, 0.5) for _ in range(batch)]
    repetition = [rng.uniform(0.5, 3) for _ in range(batch)]
    temperatures = [rng.uniform(0.1, 2) for _ in range(batch)]
    return logits, (token_freqs, presence, frequency, repetition, temperatures)


def make_rejection_case(name, case, device):
    batch, vocab, steps = case
    target = torch.randn(batch, steps + 1, vocab, device=device)
    if name == "join_prob_reject_sampling":
        target = target.softmax(-1)
    return target, torch.randint(vocab, (batch, steps), device=device), torch.ones(batch, steps, device=device)


@pytest.mark.parametrize(
    "name,shape,options",
    [
        pytest.param(name, shape, options, marks=pytest.mark.api("functions." + name))
        for name, shape, options in SAMPLING_CASES
    ],
)
@pytest.mark.accuracy
def test_sampling(accuracy_backend, name, shape, options):
    implementation, _, device = accuracy_backend
    function = getattr(F, name)
    check_sampling(
        name,
        partial(function, **options, implementation=implementation),
        partial(function, **options, implementation="torch_reference"),
        torch.randn(shape, device=device),
    )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=pytest.mark.api("functions." + name))
        for name in ["reject_sampling", "join_prob_reject_sampling"]
    ],
)
@pytest.mark.parametrize("case", REJECTION_CASES)
@pytest.mark.accuracy
def test_rejection(accuracy_backend, name, case):
    implementation, _, device = accuracy_backend
    inputs = make_rejection_case(name, case, device)
    function = getattr(F, name)
    actual = accepted_prefix(function(*inputs, random_seed=42, implementation=implementation))
    expected = accepted_prefix(function(*inputs, random_seed=42, implementation="torch_reference"))
    for a, e in zip(actual, expected):
        assert_close(a, e, rtol=0, atol=0)


@pytest.mark.api("functions.apply_penalties_temperature")
@pytest.mark.parametrize("shape", PENALTY_SHAPES)
@pytest.mark.accuracy
def test_penalties(accuracy_backend, shape):
    implementation, _, device = accuracy_backend
    logits, options = make_penalty_case(shape, device)
    actual = F.apply_penalties_temperature(logits.clone(), *options, implementation=implementation)
    expected = F.apply_penalties_temperature(logits.clone(), *options, implementation="torch_reference")
    assert_close(actual, expected, logits.dtype)
