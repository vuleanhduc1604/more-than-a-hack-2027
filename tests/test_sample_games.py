"""sample_games.py label rules, on a hand-made event table. No data needed."""

import numpy as np
import pandas as pd

import sample_games as sg

H, A, D, X = sg.H, sg.A, sg.D, sg.X


def events(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])


# One period: kick-off at 100, home passes, ball goes out (ends 300), away throw-in at 400,
# away passes, a foul at 600 with a card during the stoppage, home free kick at 700.
EVENTS = events([
    (1, "SET PIECE", "KICK OFF", 1, 100, 100),
    (1, "PASS", "", 1, 100, 150),
    (0, "RECOVERY", "", 1, 160, 160),
    (0, "PASS", "", 1, 160, 250),
    (0, "BALL OUT", "", 1, 260, 300),
    (1, "SET PIECE", "THROW IN", 1, 400, 400),
    (1, "PASS", "", 1, 400, 450),
    (0, "FAULT RECEIVED", "", 1, 600, 600),
    (1, "CARD", "YELLOW", 1, 650, 650),
    (0, "SET PIECE", "FREE KICK", 1, 700, 700),
    (0, "PASS", "", 1, 700, 900),
])


def test_labels_follow_the_events():
    labels, blocks, restarts = sg.derive_labels(EVENTS, 1000)
    assert blocks == [(0, 901)]
    assert (labels[:100] == D).all()            # before the kick-off
    assert (labels[100:160] == A).all()         # away kicked off
    assert (labels[160:301] == H).all()         # home recovered, then put the ball out
    assert (labels[301:400] == D).all()         # out until the throw-in
    assert (labels[400:601] == A).all()
    assert (labels[601:700] == D).all()         # foul to free kick; the card does not end it
    assert (labels[700:901] == H).all()
    assert (labels[901:] == X).all()            # after the last event: not graded
    assert restarts["kind"].tolist() == ["kick off", "throw in", "free kick"]
    assert restarts["dead_from"].tolist() == [0, 301, 601]


def test_goal_kicks_are_restarts():
    assert sg.restart_kind("PASS", "GOAL KICK") == "goal kick"
    assert sg.restart_kind("BALL LOST", "GOAL KICK-INTERCEPTION") == "goal kick"
    assert sg.restart_kind("PASS", "") is None
    assert sg.restart_kind("SET PIECE", "CORNER KICK") == "corner kick"


def test_speed_from_positions():
    t = np.arange(100) / sg.FPS
    players = np.zeros((100, 1, 2))
    players[:, 0, 0] = 3.0 * t                  # 3 m/s along x
    speeds = sg.speeds_from_positions(players)
    assert np.allclose(speeds[5:-5, 0], 3.0)
    assert np.isnan(speeds[0, 0]) and np.isnan(speeds[-1, 0])


def test_degrade_hides_live_ball_only_up_to_target():
    labels = np.r_[np.full(500, H), np.full(500, D)].astype(np.int8)
    game = {"labels": labels, "ball_raw": np.zeros((1000, 2), dtype=np.float32)}
    out = sg.degrade(game, seed=1, target_missing=0.4, gap_lengths=np.array([10, 30]))
    missing = np.isnan(out["ball_raw"][:, 0])
    assert 0.4 <= missing[:500].mean() < 0.6
    assert not np.isnan(game["ball_raw"]).any()   # the input is not modified
