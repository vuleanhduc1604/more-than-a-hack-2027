"""team.py and decode.py on synthetic data only. Runs on a fresh clone with no data/ folder."""

import numpy as np

import ball
import decode
import synthetic
import team

H, A, D, X = decode.H, decode.A, decode.D, decode.X


def _logit(p):
    return np.log(p / (1 - p))


def test_nearest_distances_pick_each_teams_closest_player():
    players = np.array([[[0.0, 0.0], [10.0, 0.0], [3.0, 0.0], [50.0, 0.0]]])   # one frame
    ball_xy = np.array([[4.0, 0.0]])
    d_home, d_away = team.nearest_distances(players, ball_xy, np.array([0, 0, 1, 1]))
    assert d_home[0] == 4.0 and d_away[0] == 1.0


def test_no_ball_gives_nan_distance_and_zero_margin_feature():
    tracking, _ = synthetic.make_tracking()
    xy = np.full((len(tracking["ball_raw"]), 2), np.nan, dtype=np.float32)
    X_ = team.features(tracking, xy)
    assert np.isfinite(X_).all()
    assert (X_[:, 0] == 0).all()                    # margin is 0 when there is no ball


def test_features_cover_every_frame():
    tracking, truth = synthetic.make_tracking()
    est = ball.estimate_ball(tracking, truth["blocks"])
    X_ = team.features(tracking, est["ball_xy"])
    assert X_.shape == (len(tracking["ball_raw"]), 8)
    assert np.isfinite(X_).all()


def test_low_ball_confidence_pulls_p_home_to_one_half():
    model = {"mean": np.zeros(1), "scale": np.ones(1), "weights": np.array([0.0, 2.0])}
    X_ = np.full((3, 1), 3.0)
    conf = np.array([1.0, 0.5, 0.0])
    p = team.p_home_from_logit(team.scaled_logit(model, X_, conf))
    assert p[0] > 0.99 and 0.5 < p[1] < p[0] and abs(p[2] - 0.5) < 1e-6


def _two_teams(n, home_spread, away_spread, tracked=11):
    """n frames of two teams of 11 on a line along the pitch, with the given x spreads."""
    players = np.full((n, 46, 2), np.nan, dtype=np.float32)
    line = np.linspace(-1.0, 1.0, 11)
    players[:, :11, 0] = 50.0 + line * home_spread
    players[:, 23:34, 0] = 55.0 + line * away_spread
    players[:, :11, 1] = players[:, 23:34, 1] = 34.0 + line * 10.0
    players[:, tracked:11] = np.nan
    return {"players": players, "team_of_track": np.repeat([0, 1], 23).astype(np.int8)}


def test_shape_features_need_no_ball_and_see_which_team_is_stretched():
    tracking = _two_teams(200, home_spread=30.0, away_spread=10.0)
    X_, trust = team.shape_features(tracking)
    assert X_.shape == (200, 24) and np.isfinite(X_).all()
    assert (trust == 1.0).all()
    assert (X_[:, 0] > 0).all()                     # home is the more stretched side along x
    swapped, _ = team.shape_features(_two_teams(200, home_spread=10.0, away_spread=30.0))
    assert (swapped[:, 0] < 0).all()


def test_shape_is_not_trusted_when_a_team_is_barely_tracked():
    X_, trust = team.shape_features(_two_teams(100, 30.0, 10.0, tracked=4))
    assert np.isfinite(X_).all() and (trust == 0.0).all()
    model = {"mean": np.zeros(24), "scale": np.ones(24), "weights": np.ones(25)}
    assert (team.shape_logit(model, X_, trust) == 0.0).all()


def test_blend_hands_over_from_ball_to_shape_as_confidence_falls():
    conf = np.array([1.0, 0.5, 0.0])
    ball_logit = 4.0 * conf                          # scaled_logit has already applied ball_conf
    out = team.blend(ball_logit, np.full(3, -2.0), conf)
    assert out[0] == 4.0                             # ball seen: the shape has no say
    assert out[2] == -2.0 * team.SHAPE_WEIGHT        # ball unknown: the shape decides
    assert out[2] < out[1] < out[0]


def test_decoder_ignores_a_brief_dip_but_follows_a_sustained_change():
    n = 400
    logit = np.full(n, _logit(0.9))
    logit[100:105] = _logit(0.2)                    # a 5-frame deflection: stay with H
    logit[250:] = _logit(0.1)                       # sustained change: hand over to A
    codes = decode.decode(None, np.zeros(n), [(0, n)], logit=logit)
    assert (codes[:250] == H).all() and (codes[250:] == A).all()


def test_loose_ball_keeps_the_last_possessor():
    n = 300
    logit = np.zeros(n)                              # p_home 0.5: no evidence at all
    logit[:100] = _logit(0.9)
    codes = decode.decode(None, np.zeros(n), [(0, n)], logit=logit)
    assert (codes == H).all()


def test_dead_probability_moves_the_path_to_d_and_back():
    n = 400
    logit = np.full(n, _logit(0.9))
    p_dead = np.zeros(n)
    p_dead[150:250] = 0.95                           # a real stoppage
    p_dead[50] = 0.9                                 # a one-frame false alarm
    codes = decode.decode(None, p_dead, [(0, n)], logit=logit)
    assert (codes[150:250] == D).all()
    assert codes[50] == H
    assert (codes[:150][codes[:150] != D] == H).all() and (codes[260:] == H).all()


def test_frames_outside_the_blocks_are_x_and_blocks_are_independent():
    n = 600
    logit = np.concatenate([np.full(300, _logit(0.9)), np.full(300, _logit(0.1))])
    codes = decode.decode(None, np.zeros(n), [(50, 250), (350, 550)], logit=logit)
    assert (codes[:50] == X).all() and (codes[250:350] == X).all() and (codes[550:] == X).all()
    assert (codes[50:250] == H).all() and (codes[350:550] == A).all()


def test_viterbi_is_deterministic_and_costs_a_switch_per_handover():
    rng = np.random.default_rng(0)
    costs = rng.uniform(0, 2, size=(500, 3))
    one = decode.viterbi(costs, 10.0, 20.0)
    assert np.array_equal(one, decode.viterbi(costs, 10.0, 20.0))
    flicker = lambda path: int((np.diff(path) != 0).sum())  # noqa: E731
    assert flicker(decode.viterbi(costs, 1000.0, 1000.0)) == 0
    assert flicker(decode.viterbi(costs, 0.0, 0.0)) > flicker(one)


def test_to_letters():
    assert decode.to_letters(np.array([H, A, D, X], dtype=np.int8)).tolist() == list("HADX")
