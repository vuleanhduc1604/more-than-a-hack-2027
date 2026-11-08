"""ball.py on synthetic data only. Runs on a fresh clone with no data/ folder."""

import hashlib
import os

import numpy as np
import pytest

import ball
import synthetic

SIGNATURE = {"source": "synthetic", "seed": synthetic.SEED}


@pytest.fixture(scope="module")
def case():
    tracking, truth = synthetic.make_tracking()
    out = ball.estimate_ball(tracking, truth["blocks"])
    return tracking, truth, out


def scored_mask(n_frames, blocks):
    mask = np.zeros(n_frames, dtype=bool)
    for lo, hi in blocks:
        mask[lo:hi] = True
    return mask


def test_shapes_and_dtypes(case):
    tracking, _, out = case
    n = len(tracking["ball_raw"])
    assert out["ball_xy"].shape == (n, 2) and out["ball_xy"].dtype == np.float32
    assert out["ball_conf"].shape == (n,) and out["ball_conf"].dtype == np.float32
    assert out["ball_method"].shape == (n,) and out["ball_method"].dtype == np.uint8


def test_no_nan_on_scored_frames(case):
    tracking, truth, out = case
    scored = scored_mask(len(tracking["ball_raw"]), truth["blocks"])
    assert not np.isnan(out["ball_xy"][scored]).any()


def test_conf_range_and_observed(case):
    tracking, truth, out = case
    conf, method = out["ball_conf"], out["ball_method"]
    assert not np.isnan(conf).any()
    assert conf.min() >= 0.0 and conf.max() <= 1.0
    clean, _ = ball.remove_outliers(tracking["ball_raw"], truth["blocks"])
    observed = ~np.isnan(clean[:, 0])
    assert np.all(conf[observed] == 1.0)
    scored = scored_mask(len(conf), truth["blocks"])
    assert np.all(method[observed & scored] == ball.OBSERVED)
    filled = scored & ~observed
    assert np.all(conf[filled] < 1.0) and np.all(conf[filled] >= ball.CONF_FLOOR)


def test_spikes_and_far_points_removed(case):
    tracking, truth, out = case
    _, n_removed = ball.remove_outliers(tracking["ball_raw"], truth["blocks"])
    assert n_removed == len(truth["spikes"]) + len(truth["far"])
    for frame in truth["spikes"] + truth["far"]:
        assert out["ball_method"][frame] == ball.LINEAR
        err = np.hypot(*(out["ball_xy"][frame] - truth["ball_true"][frame]))
        assert err < 2.0


def test_fast_passes_are_not_outliers():
    # A clean 30 m/s pass over 2 s (60 m, stays on the pitch), densely observed, must survive.
    n = 50
    x = np.linspace(10.0, 10.0 + 30.0 * (n - 1) / ball.FPS, n)
    raw = np.stack([x, np.full(n, 30.0)], axis=1).astype(np.float32)
    clean, n_removed = ball.remove_outliers(raw, [(0, n)])
    assert n_removed == 0 and np.array_equal(clean, raw)


def test_offpitch_inside_margin_kept(case):
    tracking, truth, out = case
    frames = truth["offpitch"]
    assert len(frames) > 0
    assert np.array_equal(out["ball_xy"][frames], tracking["ball_raw"][frames])
    assert np.all(out["ball_xy"][frames, 1] < 0.0)  # not clipped back onto the pitch
    assert np.all(out["ball_method"][frames] == ball.OBSERVED)


def test_no_fill_across_halftime(case):
    tracking, truth, out = case
    lo, hi = truth["halftime"]
    assert np.isnan(out["ball_xy"][lo:hi]).all()
    assert np.all(out["ball_conf"][lo:hi] == 0.0)
    assert np.all(out["ball_method"][lo:hi] == ball.UNSCORED)

    raw = tracking["ball_raw"]
    end_gap, start_gap = truth["edge_gaps"]
    # The end of block 1 holds its last observation, not a line toward block 2.
    last = raw[end_gap[0] - 1]
    assert np.allclose(out["ball_xy"][end_gap[0]:end_gap[1]], last)
    assert np.all(out["ball_method"][end_gap[0]:end_gap[1]] == ball.HOLD)
    # The start of block 2 holds its first observation.
    first = raw[start_gap[1]]
    assert np.allclose(out["ball_xy"][start_gap[0]:start_gap[1]], first)
    assert np.all(out["ball_method"][start_gap[0]:start_gap[1]] == ball.HOLD)


def test_linear_fill_is_a_straight_line(case):
    tracking, _, out = case
    start, end = 1000, 1100  # a planted interior gap
    a, b = tracking["ball_raw"][start - 1], tracking["ball_raw"][end]
    t = (np.arange(start, end) - (start - 1)) / (end - (start - 1))
    expected = a + t[:, None] * (b - a)
    assert np.allclose(out["ball_xy"][start:end], expected, atol=1e-4)
    assert np.all(out["ball_method"][start:end] == ball.LINEAR)


def test_conf_falls_with_distance_and_gap_length(case):
    _, _, out = case
    conf = out["ball_conf"]
    # Inside one gap: lowest in the middle, highest next to the observations.
    gap = conf[1000:1100]
    assert gap[0] > gap[50] and gap[-1] > gap[50]
    assert np.all(np.diff(gap[:50]) <= 0)
    # Longer gaps get a lower minimum.
    minima = [conf[s:s + n].min() for s, n in ((650, 3), (820, 18), (900, 40), (1200, 300))]
    assert all(x > y for x, y in zip(minima[:-1], minima[1:]))
    # The formula itself is monotone in both arguments.
    d = np.arange(1, 200)
    assert np.all(np.diff(ball.gap_confidence(d, 400)) <= 0)
    assert np.all(np.diff(ball.gap_confidence(5, d)) <= 0)


def test_outside_blocks_kept_not_filled(case):
    tracking, truth, out = case
    n = len(tracking["ball_raw"])
    outside = ~scored_mask(n, truth["blocks"])
    raw_obs = ~np.isnan(tracking["ball_raw"][:, 0])
    assert np.all(out["ball_method"][outside] == ball.UNSCORED)
    assert np.all(out["ball_conf"][outside & raw_obs] == 1.0)
    assert np.all(out["ball_conf"][outside & ~raw_obs] == 0.0)
    assert np.isnan(out["ball_xy"][outside & ~raw_obs]).all()


def test_find_gaps_half_open_inside_blocks(case):
    tracking, truth, _ = case
    gaps = ball.find_gaps(tracking["ball_raw"], truth["blocks"])
    for start, end in truth["gaps"]:
        assert (start, end) in gaps
    for start, end in gaps:
        assert any(lo <= start < end <= hi for lo, hi in truth["blocks"])
        assert np.isnan(tracking["ball_raw"][start:end, 0]).all()


def test_block_without_observation_has_no_nan():
    tracking, _ = synthetic.make_tracking()
    tracking["ball_raw"][3300:5600] = np.nan
    out = ball.estimate_ball(tracking, synthetic.BLOCKS)
    assert not np.isnan(out["ball_xy"][3300:5600]).any()
    assert np.all(out["ball_method"][3300:5600] == ball.PLAYERS)
    assert np.all(out["ball_conf"][3300:5600] == np.float32(ball.CONF_FLOOR))


def test_scored_blocks_are_half_open():
    blocks = ball.scored_blocks()
    assert blocks == [(11185, 90652), (110808, 190198)]


def _digest(folder):
    return {name: hashlib.sha256(open(os.path.join(folder, name + ".npy"), "rb").read())
            .hexdigest() for name in ball.ARRAYS}


def test_build_is_deterministic_and_cached(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for folder in (first, second):
        tracking, truth = synthetic.make_tracking()
        ball.build(tracking, cache_dir=str(folder), blocks=truth["blocks"], signature=SIGNATURE)
    assert _digest(str(first)) == _digest(str(second))

    # A current cache is served without recomputing; tracking=None would fail otherwise.
    cached = ball.build(None, cache_dir=str(first), blocks=synthetic.BLOCKS, signature=SIGNATURE)
    assert cached["ball_xy"].shape == (synthetic.N_FRAMES, 2)

    # A changed signature forces a rebuild.
    tracking, truth = synthetic.make_tracking()
    stale = dict(SIGNATURE, version="something-else")
    rebuilt = ball.build(tracking, cache_dir=str(first), blocks=truth["blocks"], signature=stale)
    assert np.array_equal(rebuilt["ball_xy"], cached["ball_xy"], equal_nan=True)
    with open(os.path.join(str(first), ball.SIGNATURE_FILE)) as handle:
        assert "something-else" in handle.read()


def test_report_has_no_frame_level_output(case):
    tracking, truth, out = case
    lines = ball.report_lines(tracking, out, truth["blocks"])
    assert lines[0].startswith("=== ball.py --report") and lines[-1] == "=== end ==="
    assert any("NaN in ball_xy on scored frames: 0" in line for line in lines)
    assert len(lines) < 40
