# test_realtime_buffers.py
import numpy as np
import torch
import pytest

import realtime


class DummyModel:
    """
    Simple fake model for testing OverlapAddEngine.

    infer_frame(x) returns x unchanged, so the engine's predicted frame
    should exactly match the current input_ring.
    """

    def __init__(self):
        self.reset_called = False

    def infer_frame(self, x):
        return x

    def reset_phase(self):
        self.reset_called = True


@pytest.fixture
def engine():
    model = DummyModel()
    device = torch.device("cpu")
    eng = realtime.OverlapAddEngine(model, device)
    return eng


def test_initial_buffers_are_zero(engine):
    assert engine.input_ring.shape == (realtime.FRAME_SIZE,)
    assert engine.output_ring.shape == (realtime.FRAME_SIZE,)
    assert engine.buf.shape == (1, realtime.FRAME_SIZE)

    assert np.allclose(engine.input_ring, 0.0)
    assert np.allclose(engine.output_ring, 0.0)
    assert torch.allclose(engine.buf, torch.zeros_like(engine.buf))


def test_reset_clears_buffers_and_calls_model_reset(engine):
    engine.input_ring[:] = 1.0
    engine.output_ring[:] = 2.0

    engine.reset()

    assert np.allclose(engine.input_ring, 0.0)
    assert np.allclose(engine.output_ring, 0.0)
    assert engine.model.reset_called is True


def test_process_hop_rejects_wrong_size(engine):
    bad_hop = np.zeros(realtime.HOP_SIZE - 1, dtype=np.float32)

    with pytest.raises(ValueError, match="Expected hop"):
        engine.process_hop(bad_hop)


def test_input_ring_shifts_new_hop_to_end(engine):
    hop1 = np.ones(realtime.HOP_SIZE, dtype=np.float32)
    hop2 = np.full(realtime.HOP_SIZE, 2.0, dtype=np.float32)

    engine.process_hop(hop1)

    # First hop should appear at the end of the input ring.
    assert np.allclose(engine.input_ring[-realtime.HOP_SIZE:], hop1)

    engine.process_hop(hop2)

    # Previous hop should shift left by one hop.
    assert np.allclose(
        engine.input_ring[-2 * realtime.HOP_SIZE : -realtime.HOP_SIZE],
        hop1,
    )

    # New hop should now be at the end.
    assert np.allclose(engine.input_ring[-realtime.HOP_SIZE:], hop2)


def test_output_hop_matches_expected_overlap_add_first_call(engine):
    """
    Since DummyModel returns input_ring unchanged, after the first call:

    input_ring = [zeros..., hop]
    output_ring += input_ring
    out_hop = first HOP_SIZE samples of output_ring

    Because the first samples are still zeros, the first output hop should be zero.
    """
    in_hop = np.ones(realtime.HOP_SIZE, dtype=np.float32)
    out_hop = engine.process_hop(in_hop)

    assert out_hop.shape == (realtime.HOP_SIZE,)
    assert np.allclose(out_hop, 0.0)


def test_output_hop_eventually_emits_delayed_input(engine):
    """
    The input hop is placed at the end of a FRAME_SIZE window.
    With identity model output, it takes FRAME_SIZE / HOP_SIZE - 1
    additional hops before that first hop reaches the output.
    """
    first_hop = np.ones(realtime.HOP_SIZE, dtype=np.float32)

    out = engine.process_hop(first_hop)
    assert np.allclose(out, 0.0)

    delay_hops = realtime.FRAME_SIZE // realtime.HOP_SIZE - 1

    for _ in range(delay_hops - 1):
        out = engine.process_hop(np.zeros(realtime.HOP_SIZE, dtype=np.float32))
        assert np.allclose(out, 0.0)

    out = engine.process_hop(np.zeros(realtime.HOP_SIZE, dtype=np.float32))

    assert np.allclose(out, first_hop)


def test_output_ring_tail_is_zeroed_after_shift(engine):
    in_hop = np.ones(realtime.HOP_SIZE, dtype=np.float32)

    engine.process_hop(in_hop)

    # After emitting one hop, the last HOP_SIZE region should be cleared.
    assert np.allclose(engine.output_ring[-realtime.HOP_SIZE:], 0.0)


def test_multiple_hops_produce_correct_delayed_sequence(engine):
    """
    Feed a sequence of unique constant hops and verify they come out
    in the same order after the frame delay.
    """
    delay_hops = realtime.FRAME_SIZE // realtime.HOP_SIZE - 1

    input_values = [1.0, 2.0, 3.0, 4.0]
    outputs = []

    for value in input_values:
        hop = np.full(realtime.HOP_SIZE, value, dtype=np.float32)
        outputs.append(engine.process_hop(hop))

    for _ in range(delay_hops):
        zero_hop = np.zeros(realtime.HOP_SIZE, dtype=np.float32)
        outputs.append(engine.process_hop(zero_hop))

    emitted_values = [
        float(np.mean(out))
        for out in outputs
        if not np.allclose(out, 0.0)
    ]

    assert emitted_values[: len(input_values)] == input_values