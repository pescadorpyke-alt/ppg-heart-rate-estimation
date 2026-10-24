"""Model I/O shapes, sizes against the edge budget, and the 2-D export copy."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.loso import INPUTS, MODELS, val_subject_for
from src.model import DSHeartRateCNN, PPGHeartRateCNN, count_params, to_2d
from src.profile_model import BUDGET, profile


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_registered_model_maps_a_batch_of_windows_to_one_value_each(name):
    torch.manual_seed(0)
    model = MODELS[name]().eval()
    x = torch.randn(5, *INPUTS[name]["shape"])
    with torch.no_grad():
        y = model(x)
    assert y.shape == (5,)
    assert torch.isfinite(y).all()


def test_parameter_counts_match_the_documented_numbers():
    assert count_params(PPGHeartRateCNN()) == 118_945
    assert count_params(DSHeartRateCNN()) == 46_753


def test_profiler_numbers_and_budget():
    v0 = profile(PPGHeartRateCNN(), (4, 512))
    assert v0["macs"] == 11_485_312
    assert not v0["fits_budget"]
    ds = profile(DSHeartRateCNN(), (4, 256))
    assert ds["macs"] == 1_609_856
    assert ds["fits_budget"]
    assert ds["macs"] <= BUDGET["macs"]


def _randomize_batchnorm(model):
    g = torch.Generator().manual_seed(1)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g))
            m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)


@pytest.mark.parametrize("cls,length", [(PPGHeartRateCNN, 512), (DSHeartRateCNN, 256)])
def test_to_2d_gives_identical_outputs(cls, length):
    torch.manual_seed(0)
    m1 = cls().eval()
    _randomize_batchnorm(m1)
    m2 = to_2d(m1)
    x = torch.randn(8, 4, length)
    with torch.no_grad():
        np.testing.assert_allclose(m1(x).numpy(), m2(x).numpy(), atol=1e-4)
        # also accepts the already-2-D export layout (B, C, 1, T)
        np.testing.assert_allclose(m2(x).numpy(), m2(x.unsqueeze(2)).numpy(), atol=1e-6)


def test_to_2d_turns_depthwise_conv1d_into_depthwise_conv2d():
    m2 = to_2d(DSHeartRateCNN())
    convs = [m for m in m2.modules() if isinstance(m, torch.nn.Conv2d)]
    assert not any(isinstance(m, torch.nn.Conv1d) for m in m2.modules())
    assert sum(1 for c in convs if c.groups == c.in_channels > 1) == 3


def test_loso_validation_subject_is_never_the_test_subject():
    vals = [val_subject_for(s) for s in range(1, 16)]
    assert all(v != s for s, v in zip(range(1, 16), vals, strict=True))
    assert sorted(vals) == list(range(1, 16))   # every subject validates exactly once
