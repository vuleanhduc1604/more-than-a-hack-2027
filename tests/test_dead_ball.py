"""dead_ball.py on synthetic data only. Runs on a fresh clone with no data/ folder."""

import numpy as np

import ball
import dead_ball
import synthetic

FPS = dead_ball.FPS


def test_features_cover_every_frame():
    tracking, truth = synthetic.make_tracking()
    ball_est = ball.estimate_ball(tracking, truth["blocks"])
    X = dead_ball.features(tracking, ball_est, synthetic.PITCH)
    assert X.shape == (len(tracking["speeds"]), 26)
    assert np.isfinite(X).all()


def test_ball_out_is_remembered_for_a_few_seconds():
    n = 400
    xy = np.full((n, 2), [50.0, 30.0], dtype=np.float32)
    xy[100] = [50.0, -3.0]                      # one observed frame over the touchline
    xy[200] = [50.0, -0.3]                      # inside the margin: not out
    method = np.full(n, ball.OBSERVED, dtype=np.uint8)
    out = dead_ball.ball_out_recent({"ball_xy": xy, "ball_method": method}, (105.0, 68.0))
    recent = int(dead_ball.OUT_RECENT_S * FPS)
    assert out[:100].sum() == 0
    assert (out[100:100 + recent] == 1).all()
    assert out[100 + recent:].sum() == 0


def test_filled_ball_never_counts_as_out():
    xy = np.array([[50.0, -3.0]] * 10, dtype=np.float32)
    method = np.full(10, ball.LINEAR, dtype=np.uint8)
    assert dead_ball.ball_out_recent({"ball_xy": xy, "ball_method": method}, (105.0, 68.0)).sum() == 0


def test_logistic_regression_learns_a_threshold_deterministically():
    rng = np.random.default_rng(0)
    speed = rng.uniform(0, 4, size=(2000, 1))
    y = (speed[:, 0] < 1.5).astype(float)
    model = dead_ball.fit(speed, y)
    p = dead_ball.predict(model, np.array([[0.5], [3.5]]))
    assert p[0] > 0.9 and p[1] < 0.1
    again = dead_ball.fit(speed, y)
    assert np.array_equal(model["weights"], again["weights"])


def test_oracle_keeps_the_previous_team_through_a_missed_stoppage():
    H, A, D = 0, 1, 2
    labels = np.array([H, H, D, D, A, A], dtype=np.int8)
    dead = np.zeros(6, dtype=bool)              # we miss the stoppage entirely
    assert dead_ball.oracle_codes(labels, dead).tolist() == [H, H, H, H, A, A]
    assert dead_ball.oracle_codes(labels, labels == D).tolist() == labels.tolist()
