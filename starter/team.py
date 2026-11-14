"""Section 4, step 1: p_home, how likely it is that the home team controls the ball.

    import team
    p_home = team.load()      # (199319,) float32 in [0, 1], no NaN; 0.5 means "no idea"

p_home assumes the ball is in play. Whether it is in play is p_dead's job (dead_ball.py), and the
decoder (decode.py) combines the two. Near 1 means the home side (Barcelona) looks in control,
near 0 the away side, 0.5 means the evidence says nothing.

What it reads, per frame, from players, team_of_track and the Section 2 ball estimate:

  * the distance from the ball to the nearest home player and to the nearest away player. The
    SIGNED MARGIN between them says more than "who is closest": a one-metre lead means little, a
    ten-metre lead means a lot.
  * that margin averaged over 0.5, 2 and 5 s: a team that stays closest is in control, one that
    is closest for a single frame is not ("how long a team has stayed closest").
  * how the margin is changing, which is who is moving towards the ball.
  * how close the nearest player is: a ball within a metre or two of someone is being played, a
    ball ten metres from everyone is loose and anybody's.

A plain logistic regression combines them into the BALL evidence. It is fitted on Metrica's three
public sample games,
where the events give the team on the ball, never on our match (RULES.md). The sample ball is
degraded like ours first (sample_games.degrade), so the model sees gap-filled balls too.

Then comes the part that matters for gaps. Our ball is missing on 41% of graded frames, in gaps
that average 17 s, so several hand-overs happen inside one gap and a filled-in ball position says
nothing about them (on held-out sample games the ball evidence names the right team on 90% of
observed-ball frames and on about 55% of gap frames, which is a coin toss). So a second, ball-free
model reads the TEAM SHAPE:

  * the team with the ball is stretched: its keeper stays home while the rest push up and spread
    out to offer passes. The team without it is compact, packed between the ball and its own
    goal. Per frame, the spread (standard deviation) of each team's players along and across the
    pitch, and the home-minus-away difference, averaged over 1 and 4 s and over 3 s just before
    and just after (a team opens up a moment after it wins the ball).

That needs no ball and no playing direction, only players, and on held-out sample games it names
the right team on about 84% of gap frames. It is a second logistic regression fitted on the same
sample games. A frame where either team has fewer than SHAPE_MIN_TRACKED tracked players has no
usable shape and contributes nothing.

The two are blended by ball_conf:

    logit = ball_conf * ball logit  +  (1 - ball_conf) * shape logit

Where the ball is seen it decides; where it is only estimated the shape takes over; where neither
says anything p_home is 0.5, which gives the decoder no reason to hand the ball over, so the last
possessor keeps it. The ball model is fitted only on frames with an observed ball (ball_conf 1),
the shape model on every live frame.

Smart Tagging (st_team, restart) is NOT used: the sample games have no tags to validate it on,
and an unvalidated weight is a guess. See docs/section4_handoff.md.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES  # noqa: E402
import ball  # noqa: E402
import dead_ball  # noqa: E402
import load_data  # noqa: E402
import sample_games  # noqa: E402

# Params
VERSION = "team-v2"
FPS = GAMES[GAME]["fps"]
MARGIN_CAP_M = 20.0                 # a lead beyond this is "far ahead"; no extra information
NO_PLAYER_M = 40.0                  # nearest-player distance when a team has nobody tracked
MARGIN_WINDOWS_S = (0.5, 2.0, 5.0)  # centred averages of the signed margin
TREND_S = 0.5                       # margin change over +/- this long
CLOSE_SCALE_M = 3.0                 # how fast "someone is on the ball" fades with distance
CONF_POWER = 1.0                    # logit is scaled by ball_conf ** CONF_POWER
SHAPE_WINDOWS_S = (1.0, 4.0)        # centred averages of the team spreads
SHAPE_EDGE_WINDOW_S = 3.0           # plus a window this long ...
SHAPE_EDGE_SHIFT_S = 2.0            # ... centred this far ahead, and this far behind
SHAPE_MIN_TRACKED = 7               # a team with fewer tracked players has no usable shape
SHAPE_WEIGHT = 1.0                  # shape logit weight where the ball is unknown (see decode.py
                                    # --tune-shape: 0.5 to 1.5 score the same on the sample games)
LOGIT_CAP = 8.0
L2 = 1e-2
TRAIN_GAMES = sample_games.GAME_NUMBERS
DEGRADE_SEED = 0
TRAIN_MIN_CONF = 0.99               # fit on observed ball frames only

PARAMS = {"margin_cap_m": MARGIN_CAP_M, "no_player_m": NO_PLAYER_M,
          "margin_windows_s": MARGIN_WINDOWS_S, "trend_s": TREND_S,
          "close_scale_m": CLOSE_SCALE_M, "conf_power": CONF_POWER, "l2": L2,
          "shape_windows_s": SHAPE_WINDOWS_S, "shape_edge_window_s": SHAPE_EDGE_WINDOW_S,
          "shape_edge_shift_s": SHAPE_EDGE_SHIFT_S, "shape_min_tracked": SHAPE_MIN_TRACKED,
          "train_games": TRAIN_GAMES, "degrade_seed": DEGRADE_SEED,
          "train_min_conf": TRAIN_MIN_CONF, "sample_games": sample_games.VERSION,
          "ball": ball.VERSION, "dead_ball": dead_ball.VERSION}

CACHE_DIR = load_data.CACHE_DIR
CACHE_FILE = "p_home.npy"
SIGNATURE_FILE = "team_signature.json"
SAMPLE_CACHE = "team_sample_{}.npz"


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------

def nearest_distances(players, ball_xy, team_of_track):
    """(d_home, d_away): distance in metres from the ball to each team's nearest tracked player.

    NaN where the ball has no position. A team with nobody tracked on a frame gets NO_PLAYER_M.
    """
    gap = np.linalg.norm(players - ball_xy[:, None, :], axis=2)         # (frames, tracks)
    gap = np.where(np.isnan(gap), np.inf, gap)
    out = []
    for team in (0, 1):
        nearest = gap[:, team_of_track == team].min(axis=1)
        out.append(np.where(np.isinf(nearest), NO_PLAYER_M, nearest))
    no_ball = np.isnan(ball_xy[:, 0])
    return tuple(np.where(no_ball, np.nan, d) for d in out)


def _smooth(values, seconds, fps=FPS):
    frames = max(int(seconds * fps), 1)
    return pd.Series(values).rolling(frames, center=True, min_periods=1).mean().to_numpy()


def features(tracking, ball_xy, fps=FPS):
    """(n_frames, 8) feature matrix. Same code for our match and the sample games."""
    d_home, d_away = nearest_distances(tracking["players"], ball_xy, tracking["team_of_track"])
    margin = np.clip(d_away - d_home, -MARGIN_CAP_M, MARGIN_CAP_M)   # > 0: home is closer
    margin = np.nan_to_num(margin, nan=0.0)
    nearest = np.nan_to_num(np.minimum(d_home, d_away), nan=NO_PLAYER_M)

    smooth = {s: _smooth(margin, s, fps) for s in MARGIN_WINDOWS_S}
    shift = max(int(TREND_S * fps), 1)
    base = smooth[MARGIN_WINDOWS_S[0]]
    ahead, behind = np.roll(base, -shift), np.roll(base, shift)
    ahead[-shift:], behind[:shift] = base[-1], base[0]
    closest = _smooth(np.sign(margin), MARGIN_WINDOWS_S[1], fps)      # share of time home is closer
    control = np.tanh(margin / 2.0) * np.exp(-nearest / CLOSE_SCALE_M)

    columns = [margin] + [smooth[s] for s in MARGIN_WINDOWS_S] + [ahead - behind, closest,
                                                                  control, nearest]
    return np.column_stack(columns).astype(np.float64)


def _window_mean(values, seconds, shift_s=0.0, fps=FPS):
    """Centred mean over `seconds`, read `shift_s` later (negative: earlier). NaN frames are
    left out of the mean; a window with none left stays NaN."""
    frames = max(int(seconds * fps), 1)
    series = pd.Series(values).rolling(frames, center=True, min_periods=1).mean()
    return series.shift(-int(shift_s * fps)).to_numpy()


def shape_features(tracking, fps=FPS):
    """(X, trust): the (n_frames, 24) team-shape matrix and how far to trust each row, 0 to 1.

    Reads players and team_of_track only: no ball, no playing direction, so it works through any
    ball gap and needs nothing match-specific. A team's spread is the standard deviation of its
    tracked players' x (along the pitch) and y (across it). Frames where a team has fewer than
    SHAPE_MIN_TRACKED players are left out of the averages, and trust is the share of the last
    second in which both teams had enough.
    """
    players, team_of_track = tracking["players"], tracking["team_of_track"]
    spreads, enough = [], np.ones(len(players), dtype=bool)
    for team in (0, 1):
        xy = players[:, team_of_track == team, :].astype(np.float64)
        present = ~np.isnan(xy[..., 0])
        count = present.sum(axis=1)
        mean = np.where(present[..., None], xy, 0.0).sum(axis=1) / np.maximum(count, 1)[:, None]
        square = np.where(present[..., None], (xy - mean[:, None, :]) ** 2, 0.0).sum(axis=1)
        spread = np.sqrt(square / np.maximum(count, 1)[:, None])
        spread[count < SHAPE_MIN_TRACKED] = np.nan
        enough &= count >= SHAPE_MIN_TRACKED
        spreads.append(spread)
    home, away = spreads
    series = [home[:, 0] - away[:, 0], home[:, 1] - away[:, 1],
              home[:, 0], away[:, 0], home[:, 1], away[:, 1]]
    columns = [_window_mean(v, s, fps=fps) for v in series for s in SHAPE_WINDOWS_S]
    columns += [_window_mean(v, SHAPE_EDGE_WINDOW_S, shift, fps) for v in series
                for shift in (SHAPE_EDGE_SHIFT_S, -SHAPE_EDGE_SHIFT_S)]
    X = pd.DataFrame(np.column_stack(columns)).ffill().bfill().fillna(0.0).to_numpy()
    trust = np.nan_to_num(_window_mean(enough.astype(float), 1.0, fps=fps), nan=0.0)
    return X.astype(np.float64), np.where(enough, trust, 0.0)


# ---------------------------------------------------------------------------------------------
# Models, reusing dead_ball's numpy logistic regression
# ---------------------------------------------------------------------------------------------

def _logit(model, X):
    p = np.clip(dead_ball.predict(model, X), 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1 - p)), -LOGIT_CAP, LOGIT_CAP)


def fit_model(prepared):
    """Fit on the live, observed-ball frames of prepared sample games (see sample_game)."""
    X, y = [], []
    for g in prepared:
        keep = (g["labels"] < sample_games.D) & (g["ball_conf"] >= TRAIN_MIN_CONF)
        X.append(g["features"][keep])
        y.append((g["labels"][keep] == sample_games.H).astype(float))
    return dead_ball.fit(np.vstack(X), np.concatenate(y), l2=L2)


def scaled_logit(model, X, ball_conf):
    """The model's logit, shrunk to 0 (p = 0.5) as ball_conf falls. This is the evidence the
    decoder uses: positive favours home."""
    return _logit(model, X) * np.clip(ball_conf, 0.0, 1.0) ** CONF_POWER


def fit_shape_model(prepared):
    """Fit the ball-free shape model on every live frame of prepared sample games."""
    X, y = [], []
    for g in prepared:
        keep = (g["labels"] < sample_games.D) & (g["shape_trust"] > 0.99)
        X.append(g["shape"][keep])
        y.append((g["labels"][keep] == sample_games.H).astype(float))
    return dead_ball.fit(np.vstack(X), np.concatenate(y), l2=L2)


def fit_models(prepared):
    prepared = list(prepared)
    return {"ball": fit_model(prepared), "shape": fit_shape_model(prepared)}


def shape_logit(model, X, trust):
    """The shape model's logit, shrunk to 0 where too few players are tracked."""
    return _logit(model, X) * np.clip(trust, 0.0, 1.0)


def blend(ball_logit, shape_logit_, ball_conf, shape_weight=SHAPE_WEIGHT):
    """The evidence the decoder uses. ball_logit is already scaled by ball_conf (scaled_logit);
    the shape fills in exactly the share of trust the ball does not have."""
    conf = np.clip(ball_conf, 0.0, 1.0) ** CONF_POWER
    return ball_logit + shape_weight * (1.0 - conf) * shape_logit_


def evidence(models, g, shape_weight=SHAPE_WEIGHT):
    """Blended logit for a prepared game dict (features, shape, shape_trust, ball_conf)."""
    return blend(scaled_logit(models["ball"], g["features"], g["ball_conf"]),
                 shape_logit(models["shape"], g["shape"], g["shape_trust"]),
                 g["ball_conf"], shape_weight)


def p_home_from_logit(logit):
    return (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)


def p_home(tracking, ball_est, models):
    """p_home for any match: tracking dict, the Section 2 ball dict and the fitted models."""
    shape, trust = shape_features(tracking)
    g = {"features": features(tracking, ball_est["ball_xy"]), "shape": shape,
         "shape_trust": trust, "ball_conf": ball_est["ball_conf"]}
    return p_home_from_logit(evidence(models, g))


# ---------------------------------------------------------------------------------------------
# Sample games: everything the decoder is tuned on, prepared once and cached.
# ---------------------------------------------------------------------------------------------

def sample_game(number, cache_dir=CACHE_DIR):
    """A degraded sample game with its ball estimate, features, labels and held-out p_dead.

    p_dead for game n comes from a dead-ball model trained on the OTHER two games, so tuning the
    decoder never sees a p_dead that was fitted on the game it is scored on.
    """
    path = os.path.join(cache_dir, SAMPLE_CACHE.format(number))
    signature = hashlib.sha256(json.dumps(PARAMS, sort_keys=True).encode()).hexdigest()[:16]
    if os.path.isfile(path):
        cached = np.load(path, allow_pickle=False)
        if str(cached["signature"]) == signature:
            return {name: cached[name] for name in cached.files if name != "signature"}

    prepared = {n: dead_ball.sample_game(n) for n in TRAIN_GAMES}
    others = [prepared[n] for n in TRAIN_GAMES if n != number]
    dead_model = dead_ball.train(others)
    game = prepared[number]
    blocks = [tuple(b) for b in game["blocks"]]
    est = ball.estimate_ball(game, blocks)
    out = {"players": game["players"], "speeds": game["speeds"],
           "team_of_track": game["team_of_track"], "labels": game["labels"],
           "blocks": game["blocks"], "ball_raw": game["ball_raw"],
           "ball_xy": est["ball_xy"], "ball_conf": est["ball_conf"],
           "ball_method": est["ball_method"],
           "p_dead": dead_ball.predict(dead_model, game["features"]).astype(np.float32),
           "features": features(game, est["ball_xy"])}
    out["shape"], out["shape_trust"] = shape_features(game)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, signature=signature, **out)
    return out


def sample_p_home(number, prepared):
    """Held-out p_home for sample game `number`: model fitted on the other games only."""
    models = fit_models(prepared[n] for n in TRAIN_GAMES if n != number)
    return p_home_from_logit(evidence(models, prepared[number]))


# ---------------------------------------------------------------------------------------------
# Our match, cached like ball.py and dead_ball.py
# ---------------------------------------------------------------------------------------------

def cache_signature():
    text = json.dumps(PARAMS, sort_keys=True)
    return {"tracking": load_data.source_signature(), "version": VERSION,
            "params": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]}


def build(refresh=False, cache_dir=CACHE_DIR):
    signature = json.loads(json.dumps(cache_signature(), sort_keys=True))
    path, sig_path = os.path.join(cache_dir, CACHE_FILE), os.path.join(cache_dir, SIGNATURE_FILE)
    if not refresh and os.path.isfile(path) and os.path.isfile(sig_path):
        with open(sig_path) as handle:
            if json.load(handle) == signature:
                return np.load(path)

    prepared = {n: sample_game(n, cache_dir) for n in TRAIN_GAMES}
    models = fit_models(prepared.values())
    out = p_home(load_data.load_tracking(), ball.load(), models)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, out)
    with open(sig_path, "w") as handle:     # written last: an interrupted build never looks done
        json.dump(signature, handle, sort_keys=True)
    return out


def load(refresh=False):
    """The one line for everyone: p_home, shape (199319,), float32 in [0, 1]."""
    return build(refresh=refresh)


if __name__ == "__main__":
    out = load(refresh="--refresh" in sys.argv)
    scored = np.zeros(len(out), dtype=bool)
    for lo, hi in ball.scored_blocks():
        scored[lo:hi] = True
    print("p_home {} {}, NaN {}".format(out.shape, out.dtype, int(np.isnan(out).sum())))
    print("graded frames: home-leaning {:.1%}, away-leaning {:.1%}, undecided (0.4-0.6) {:.1%}"
          .format((out[scored] > 0.6).mean(), (out[scored] < 0.4).mean(),
                  ((out[scored] >= 0.4) & (out[scored] <= 0.6)).mean()))
