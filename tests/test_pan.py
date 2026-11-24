# -*- coding: utf-8 -*-
"""声道 / pan_swap 测试。"""
import numpy as np
import pytest

from tingleloom.script import AUTO_PAN, parse_pan
from tingleloom.sfx import apply_pan


def test_explicit_pan_untouched():
    left = parse_pan("L70")
    assert left.explicit is True
    assert left.for_start(0, 150).value == pytest.approx(-0.7)
    assert left.for_start(150, 150).value == pytest.approx(-0.7)   # pan_swap 不影响显式声道
    assert parse_pan("C").for_start(150, 150).value == 0.0


def test_auto_pan_alternates_with_pan_swap():
    assert AUTO_PAN.explicit is False
    a = AUTO_PAN.for_start(0, 150)
    b = AUTO_PAN.for_start(150, 150)
    c = AUTO_PAN.for_start(300, 150)
    assert a.value < 0 and b.value > 0 and c.value < 0
    assert abs(a.value) == pytest.approx(abs(b.value))


def test_auto_pan_centered_without_pan_swap():
    assert AUTO_PAN.for_start(150, None).value == 0.0
    assert AUTO_PAN.for_start(0, None).value == 0.0


@pytest.mark.parametrize("pan", [parse_pan("C"), AUTO_PAN])
def test_center_preserves_native_stereo(pan):
    signal = np.sin(2 * np.pi * 10 * np.arange(1000) / 1000).astype(np.float32)
    data = np.column_stack((np.zeros_like(signal), signal))
    np.testing.assert_array_equal(apply_pan(data, pan), data)


def test_fixed_pan_relocates_right_only_source():
    signal = np.sin(2 * np.pi * 10 * np.arange(1000) / 1000).astype(np.float32)
    data = np.column_stack((np.zeros_like(signal), signal))
    output = apply_pan(data, parse_pan("L70"))
    np.testing.assert_array_equal(output[:, 0], signal)
    np.testing.assert_allclose(output[:, 1], signal * 0.3)


def test_lr_swaps_right_only_source_with_smooth_transition():
    sr = 1000
    signal = np.sin(2 * np.pi * 10 * np.arange(8 * sr) / sr).astype(np.float32)
    data = np.column_stack((np.zeros_like(signal), signal))
    output = apply_pan(data, parse_pan("LR4"), sr=sr)
    first = np.mean(output[sr:3 * sr] ** 2, axis=0)
    second = np.mean(output[5 * sr:7 * sr] ** 2, axis=0)
    assert first[0] > first[1]
    assert second[1] > second[0]
    np.testing.assert_array_equal(output[sr:3 * sr, 0], signal[sr:3 * sr])
    np.testing.assert_array_equal(output[5 * sr:7 * sr, 1], signal[5 * sr:7 * sr])
    np.testing.assert_allclose(output.sum(axis=1), signal * 1.25, atol=1e-7)
    # 非零平台音源直接呈现交叉淡化包络，而不是靠正弦过零掩盖硬切换。
    envelope = apply_pan(np.column_stack((np.zeros(8 * sr), np.ones(8 * sr))), parse_pan("LR4"), sr=sr)
    assert np.max(np.abs(np.diff(envelope[:, 0]))) < 0.001
