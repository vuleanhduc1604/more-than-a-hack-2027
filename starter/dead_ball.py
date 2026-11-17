"""Section 3: is the ball in play? One number per frame, p_dead.

The baseline calls every frame without a ball dead. The ball is missing on 41% of graded frames,
mostly while play goes on, so that splits possessions everywhere. This module estimates whether
play is stopped from how the players move, plus one ball signal, and never from the ball being
missing.

    import dead_ball
    p_dead = dead_ball.load()      # (199319,) float32 in [0, 1], no NaN

0 means play is almost certainly on, 1 almost certainly stopped. It is a probability from a model
trained on Metrica's three public sample games (starter/sample_games.py), so 0.5 is a sensible
cut, and Section 4 can use it directly as the D evidence in its decoder.

The signals, all computed the same way on our match and on the sample games:

  1. Team speed. When the referee stops play, everyone slows to a walk; in play someone is always
     running. Per frame, the median and the 90th percentile of the tracked players' speeds, each
     averaged over centred windows of 0.5, 2, 5 and 10 s, and over 3 s just before and just
     after (the slowdown lags the whistle and the speed-up lags the restart by a second or two).
     Speeds in our match and in the sample games have nearly the same distribution (checked
     without labels), which is what lets a model trained there work here.
  2. Ball over a line. The observed ball was beyond the touchline or goal line (by more than
     OUT_MARGIN_M) in the last OUT_RECENT_S seconds, raw and spread over 5 s. Only observed
     positions count, after Section 2's outlier removal: a filled-in gap never puts the ball out.

The model is a logistic regression on those features, fitted on the sample games after
sample_games.degrade() has hidden their ball as often as ours. It has 26 inputs (two speeds x six
windows, each also as a log, plus the two ball signals) and no match-specific constants: nothing
here refers to a frame of our match.

Tried and dropped, because they did not improve held-out sample games: a kick-off formation
signal (both teams in their own halves; kick-offs are too rare there to show a gain), and a
two-state smoothing pass over the output (the 10 s windows already smooth it). Not used: the
Smart Tagging `restart` hint, because the sample games have no tagging to validate it on.

    python starter/dead_ball.py            # build or load the cache, print a summary
    python starter/dead_ball.py --report   # held-out results on the sample games + checks on ours
"""

import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (os.path.join(ROOT, "evaluation"), ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)
from games import GAME, GAMES  # noqa: E402
import ball  # noqa: E402
import load_data  # noqa: E402
import sample_games  # noqa: E402

# Params
VERSION = "dead-ball-v1"
FPS = GAMES[GAME]["fps"]
SPEED_WINDOWS_S = (0.5, 2.0, 5.0, 10.0)   # centred averages of the team speed
EDGE_WINDOW_S = 3.0                       # plus a window this long ...
EDGE_SHIFT_S = 2.0                        # ... centred this far ahead, and this far behind
OUT_MARGIN_M = 0.5                        # tracker error allowed before the ball counts as out
OUT_RECENT_S = 3.0                        # "the ball went out" stays true this long
OUT_SPREAD_S = 5.0                        # and is also averaged over this window
L2 = 1e-3                                 # ridge penalty of the logistic regression
TRAIN_GAMES = sample_games.GAME_NUMBERS
DEGRADE_SEED = 0
THRESHOLD = 0.5                           # only for reports; the output is the probability

PARAMS = {"speed_windows_s": SPEED_WINDOWS_S, "edge_window_s": EDGE_WINDOW_S,
          "edge_shift_s": EDGE_SHIFT_S, "out_margin_m": OUT_MARGIN_M,
          "out_recent_s": OUT_RECENT_S, "out_spread_s": OUT_SPREAD_S, "l2": L2,
          "train_games": TRAIN_GAMES, "degrade_seed": DEGRADE_SEED,
          "sample_games": sample_games.VERSION, "ball": ball.VERSION}

CACHE_DIR = load_data.CACHE_DIR
CACHE_FILE = "p_dead.npy"
SIGNATURE_FILE = "dead_ball_signature.json"


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------

def _rolling_mean(values, frames, shift=0):
    """Centred rolling mean over `frames`, read `shift` frames later (negative: earlier)."""
    series = pd.Series(values).rolling(max(int(frames), 1), center=True, min_periods=1).mean()
    return series.shift(-int(shift)).ffill().bfill().to_numpy()


def team_speed(speeds):
    """(median, 90th percentile) of the tracked players' speeds per frame, gaps carried over."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # frames with no speed at all
        median = np.nanmedian(speeds, axis=1)
        top = np.nanpercentile(speeds, 90, axis=1)
    fill = lambda x: pd.Series(x).ffill().bfill().fillna(0.0).to_numpy()  # noqa: E731
    return fill(median), fill(top)


def ball_out_recent(ball_est, pitch, fps=FPS):
    """1.0 where the observed ball was beyond a pitch line within the last OUT_RECENT_S."""
    observed = ball_est["ball_method"] == ball.OBSERVED
    x, y = ball_est["ball_xy"][:, 0], ball_est["ball_xy"][:, 1]
    length, width = pitch
    m = OUT_MARGIN_M
    with np.errstate(invalid="ignore"):
        out = observed & ((x < -m) | (x > length + m) | (y < -m) | (y > width + m))
    recent = pd.Series(out.astype(float)).rolling(int(OUT_RECENT_S * fps), min_periods=1).max()
    return recent.to_numpy()


def features(tracking, ball_est, pitch, fps=FPS):
    """(n_frames, 26) feature matrix. Same code for our match and the sample games."""
    columns = []
    for speed in team_speed(tracking["speeds"]):
        windows = [_rolling_mean(speed, s * fps) for s in SPEED_WINDOWS_S]
        windows += [_rolling_mean(speed, EDGE_WINDOW_S * fps, shift)
                    for shift in (EDGE_SHIFT_S * fps, -EDGE_SHIFT_S * fps)]
        columns += windows + [np.log(w + 0.1) for w in windows]
    out = ball_out_recent(ball_est, pitch, fps)
    columns += [out, _rolling_mean(out, OUT_SPREAD_S * fps)]
    return np.column_stack(columns).astype(np.float64)


# ---------------------------------------------------------------------------------------------
# Model: a plain logistic regression, fitted by Newton's method. numpy only.
# ---------------------------------------------------------------------------------------------

def fit(X, y, l2=L2, max_iter=50):
    """dict(weights, mean, scale) for P(dead | x). Deterministic: same data, same model."""
    mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])
    w = np.zeros(Z.shape[1])
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        gradient = Z.T @ (p - y) + l2 * w
        hessian = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"weights": w, "mean": mean, "scale": scale}


def predict(model, X):
    Z = np.column_stack([np.ones(len(X)), (X - model["mean"]) / model["scale"]])
    return 1.0 / (1.0 + np.exp(-Z @ model["weights"]))


def sample_game(number):
    """A degraded sample game with its ball estimate and features, ready to train or test on."""
    game = sample_games.degrade(sample_games.load(number), seed=DEGRADE_SEED + number)
    ball_est = ball.estimate_ball(game, [tuple(b) for b in game["blocks"]])
    game["features"] = features(game, ball_est, sample_games.PITCH)
    return game


def train(games):
    """Fit on the graded frames of the given prepared sample games."""
    X = np.vstack([g["features"][g["labels"] != sample_games.X] for g in games])
    y = np.concatenate([g["labels"][g["labels"] != sample_games.X] == sample_games.D
                        for g in games]).astype(float)
    return fit(X, y)


def estimate_p_dead(tracking, ball_est, model, pitch=GAMES[GAME]["pitch"]):
    return predict(model, features(tracking, ball_est, pitch)).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Cache, as in ball.py: rebuilt when the tracking, VERSION or Params change.
# ---------------------------------------------------------------------------------------------

def cache_signature():
    text = json.dumps(PARAMS, sort_keys=True)
    return {"tracking": load_data.source_signature(), "version": VERSION,
            "params": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]}


def _normalise(signature):
    return json.loads(json.dumps(signature, sort_keys=True))


def build(refresh=False, cache_dir=CACHE_DIR):
    """p_dead for our match, from cache if current. Training reads the sample games."""
    signature = _normalise(cache_signature())
    path, sig_path = os.path.join(cache_dir, CACHE_FILE), os.path.join(cache_dir, SIGNATURE_FILE)
    if not refresh and os.path.isfile(path) and os.path.isfile(sig_path):
        with open(sig_path) as handle:
            if json.load(handle) == signature:
                return np.load(path)

    model = train([sample_game(n) for n in TRAIN_GAMES])
    p_dead = estimate_p_dead(load_data.load_tracking(), ball.load(), model)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, p_dead)
    with open(sig_path, "w") as handle:   # written last: an interrupted build never looks done
        json.dump(signature, handle, sort_keys=True)
    return p_dead


def load(refresh=False):
    """The one line for everyone: p_dead, shape (199319,), float32 in [0, 1]."""
    return build(refresh=refresh)


# ---------------------------------------------------------------------------------------------
# Report. Aggregates only, so it is safe to share.
# ---------------------------------------------------------------------------------------------

def runs(mask):
    """Half-open (start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def oracle_codes(labels, dead):
    """The true team on live frames, our dead/live call everywhere. Isolates Section 3's effect:
    a frame we call live inside a real stoppage keeps the team that had the ball before it."""
    live_team = np.where(labels < sample_games.D, labels, np.nan)
    team = pd.Series(live_team).ffill().bfill().to_numpy()
    codes = np.where(dead, sample_games.D, team).astype(np.int8)
    codes[labels == sample_games.X] = sample_games.X
    return codes


def compare(labels, dead, restarts):
    """Scores of one dead/live call against the event labels of a sample game."""
    import possession_metrics as pm
    graded = labels != sample_games.X
    truth = labels == sample_games.D
    tp = (truth & dead & graded).sum()
    true_runs = [r for r in runs(truth & graded)]
    pred_runs = [r for r in runs(dead & graded)]
    found, start_err, end_err = [], [], []
    for lo, hi in true_runs:
        overlaps = [(min(hi, b) - max(lo, a), a, b) for a, b in pred_runs if a < hi and b > lo]
        best = max(overlaps) if overlaps else None
        hit = best is not None and best[0] >= 0.5 * (hi - lo)
        found.append(hit)
        if hit:
            start_err.append((best[1] - lo) / FPS)
            end_err.append((best[2] - hi) / FPS)
    false_runs = sum(truth[a:b].mean() < 0.5 for a, b in pred_runs)
    by_kind = {}
    for kind, group in restarts.groupby("kind"):
        hits = [dead[lo:hi].mean() >= 0.5 for lo, hi in zip(group["dead_from"], group["frame"])
                if hi > lo]
        by_kind[kind] = (int(np.sum(hits)), len(hits))
    spells = pm.spell_scores(labels, oracle_codes(labels, dead), pm.valid_runs(labels))
    return {"spell_f1": spells["mean_f1"], "n_spells": spells["n_pred"],
            "n_true_spells": spells["n_gt"],
            "dead_f1": 2 * tp / max((truth & graded).sum() + (dead & graded).sum(), 1),
            "found": int(np.sum(found)), "n_stoppages": len(true_runs),
            "false_stoppages": int(false_runs), "n_pred_stoppages": len(pred_runs),
            "start_s": np.median(start_err), "end_s": np.median(end_err), "by_kind": by_kind}


def baseline_dead(ball_raw, half_window=50):
    """The baseline's dead call: ball missing, majority over +/- half_window frames."""
    missing = np.isnan(ball_raw[:, 0]).astype(float)
    return _rolling_mean(missing, 2 * half_window + 1) > 0.5


def sample_report():
    """Leave-one-game-out on the degraded sample games: train on two, test on the third."""
    games = {n: sample_game(n) for n in TRAIN_GAMES}
    lines = ["Held-out sample games (trained on the other two; ball degraded like ours).",
             "spellF1 uses the true team on live frames, so it measures the dead/live call only.",
             "", "game  method     spellF1  possessions  deadF1  stoppages found  false  "
             "start(s)  end(s)"]
    kinds = {}
    for test in games:
        model = train([g for n, g in games.items() if n != test])
        game = games[test]
        calls = {"baseline": baseline_dead(game["ball_raw"]),
                 "ours": predict(model, game["features"]) > THRESHOLD}
        for name, dead in calls.items():
            r = compare(game["labels"], dead, game["restarts"])
            lines.append("{:4d}  {:9s}  {:7.3f}  {:5d} / {:<5d} {:6.3f}  {:6d} / {:<5d} {:5d}  "
                         "{:+8.1f}  {:+6.1f}".format(
                             test, name, r["spell_f1"], r["n_spells"], r["n_true_spells"],
                             r["dead_f1"], r["found"], r["n_stoppages"], r["false_stoppages"],
                             r["start_s"], r["end_s"]))
            if name == "ours":
                for kind, (hit, n) in r["by_kind"].items():
                    kinds[kind] = (kinds.get(kind, (0, 0))[0] + hit, kinds.get(kind, (0, 0))[1] + n)
    lines += ["", "start/end: median signed error of found stoppages, + means late.",
              "", "Stoppages found by restart type (ours, all three games):"]
    lines += ["  {:12s} {:3d} / {:<3d} {:4.0%}".format(kind, hit, n, hit / n)
              for kind, (hit, n) in sorted(kinds.items(), key=lambda kv: -kv[1][1])]
    return lines


def our_match_report(p_dead):
    """Label-free checks on our match: shares, counts, lengths, agreement with Smart Tagging."""
    import tagging
    blocks = ball.scored_blocks()
    graded = np.zeros(len(p_dead), dtype=bool)
    for lo, hi in blocks:
        graded[lo:hi] = True
    dead = (p_dead > THRESHOLD) & graded
    stoppages = runs(dead)
    lengths = np.array([hi - lo for lo, hi in stoppages]) / FPS
    restart = tagging.load()["restart"] > 0
    windows = runs(restart & graded)
    with_stop = np.mean([dead[lo:hi].any() for lo, hi in windows]) if windows else float("nan")
    first, second = blocks
    kickoff_2nd = GAMES[GAME]["kickoff_2nd"]
    kickoff_frame = (kickoff_2nd["minute"] * 60 + kickoff_2nd["second"]) * FPS + kickoff_2nd["frame"]
    pre = slice(second[0], min(kickoff_frame, second[1]))
    return ["Our match (no labels used):",
            "  dead on {:.1%} of graded frames (sample games: 31-42%)".format(dead[graded].mean()),
            "  {} stoppages, median {:.1f} s, middle half {:.1f}-{:.1f} s".format(
                len(stoppages), np.median(lengths), *np.percentile(lengths, [25, 75])),
            "  {:.0%} of the {} Smart Tagging set-piece windows contain a stoppage".format(
                with_stop, len(windows)),
            "  graded frames before the 2nd-half kick-off called dead: {:.0%}".format(
                dead[pre].mean()),
            "  baseline rule (ball missing) would call {:.1%} dead".format(
                (baseline_dead(load_data.load_tracking()["ball_raw"]) & graded)[graded].mean())]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="rebuild even if cached")
    parser.add_argument("--report", action="store_true", help="evaluate and print checks")
    args = parser.parse_args()
    p_dead = load(refresh=args.refresh)
    print("p_dead {} {}, range {:.3f}-{:.3f}, NaN {}".format(
        p_dead.shape, p_dead.dtype, p_dead.min(), p_dead.max(), int(np.isnan(p_dead).sum())))
    if args.report:
        print("\n".join([""] + sample_report() + [""] + our_match_report(p_dead)))


if __name__ == "__main__":
    main()
