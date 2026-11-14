"""Section 4, step 2: turn per-frame evidence into whole possessions.

    import decode
    codes = decode.decode(p_home, p_dead, blocks)     # int8: 0 H, 1 A, 2 D, 3 X outside the blocks
    labels = decode.to_letters(codes)

The decoder has three states, H, A and D, and every change of state costs something. It returns
the single cheapest path through each half (Viterbi), so it only switches when the evidence for the
new state is strong enough, for long enough, to pay for the switch. That is what a majority vote
cannot do: a vote hides flicker, the decoder prices it.

    cost of a path = sum over frames of  -log(score of the chosen state)
                   + SWITCH_HA   for each H <-> A hand-over
                   + SWITCH_DEAD for each move into or out of D

where, with q = sigmoid(TEAM_WEIGHT * logit(p_home)):

    score_home = (1 - p_dead) * q          score_away = (1 - p_dead) * (1 - q)
    score_dead = p_dead

So a short dip in p_home (a deflection), a loose ball (p_home near 0.5) or a brief false "dead"
cannot pay for a switch and the last state is kept, while a sustained change of control or a real
stoppage can. Inside a ball gap p_home comes from the team shapes (team.py), so the decoder can
still hand the ball over there instead of holding one team for the whole gap.

SWITCH_HA, SWITCH_DEAD and TEAM_WEIGHT were chosen on Metrica's three sample games (python
starter/decode.py --tune), never on our match or its public 5 minutes. They were re-checked after
team.py gained the shape evidence and the same values are still on the flat top of the grid, so
they did not move. X is not written: it repeats
our last label, which is free inside a possession and costly across a hand-over, and a decoder
already holds the last possessor through weak evidence. See docs/section4_handoff.md.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
VERSION = "decode-v2"
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
EPS = 1e-4                  # floor on a state's score, so one certain frame cannot cost infinity
SWITCH_HA = 10.0            # chosen on the sample games (mean spellF1 0.495), see --tune
SWITCH_DEAD = 20.0
TEAM_WEIGHT = 1.5
PARAMS = {"switch_ha": SWITCH_HA, "switch_dead": SWITCH_DEAD, "team_weight": TEAM_WEIGHT}


def state_costs(logit, p_dead, team_weight=TEAM_WEIGHT):
    """(n, 3) cost of being in H, A, D on each frame. Lower is a better fit."""
    q = 1.0 / (1.0 + np.exp(-team_weight * logit))
    alive = 1.0 - p_dead
    scores = np.column_stack([alive * q, alive * (1.0 - q), p_dead])
    return -np.log(np.maximum(scores, EPS))


def viterbi(costs, switch_ha, switch_dead):
    """Cheapest state path, int8 array. States 0 H, 1 A, 2 D; staying is free.

    Written out for three states on plain floats, which is several times faster than numpy on
    arrays of three and keeps a full match at about a second.
    """
    n = len(costs)
    back = np.zeros((n, 3), dtype=np.int8)
    h, a, d = (float(c) for c in costs[0])
    rows = costs.tolist()
    for t in range(1, n):
        ch, ca, cd = rows[t]
        # into H: stay, or come from A (hand-over) or D (restart)
        best, arg = h, 0
        if a + switch_ha < best:
            best, arg = a + switch_ha, 1
        if d + switch_dead < best:
            best, arg = d + switch_dead, 2
        new_h, bh = best + ch, arg
        best, arg = a, 1
        if h + switch_ha < best:
            best, arg = h + switch_ha, 0
        if d + switch_dead < best:
            best, arg = d + switch_dead, 2
        new_a, ba = best + ca, arg
        best, arg = d, 2
        if h + switch_dead < best:
            best, arg = h + switch_dead, 0
        if a + switch_dead < best:
            best, arg = a + switch_dead, 1
        new_d, bd = best + cd, arg
        h, a, d = new_h, new_a, new_d
        back[t, 0], back[t, 1], back[t, 2] = bh, ba, bd
    path = np.zeros(n, dtype=np.int8)
    state = int(np.argmin((h, a, d)))
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return path


def decode(p_home, p_dead, blocks, switch_ha=SWITCH_HA, switch_dead=SWITCH_DEAD,
           team_weight=TEAM_WEIGHT, logit=None):
    """Codes for every frame: the best H/A/D path inside each block, X outside them.

    Blocks are half-open (lo, hi) frame ranges, one per half, decoded separately so a possession
    cannot run across half-time. Pass `logit` instead of p_home to skip the conversion.
    """
    if logit is None:
        p = np.clip(np.asarray(p_home, dtype=np.float64), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
    codes = np.full(len(p_dead), X, dtype=np.int8)
    for lo, hi in blocks:
        lo, hi = max(int(lo), 0), min(int(hi), len(codes))
        if hi > lo:
            costs = state_costs(logit[lo:hi], np.asarray(p_dead[lo:hi], dtype=np.float64),
                                team_weight)
            codes[lo:hi] = viterbi(costs, switch_ha, switch_dead)
    return codes


def to_letters(codes):
    return LETTERS[codes]


# ---------------------------------------------------------------------------------------------
# Comparison on the labelled sample games
# ---------------------------------------------------------------------------------------------

def nearest_team_codes(prepared, window=50):
    """The baseline's team call, as the dead-ball handoff describes it: the team of the nearest
    player to the OBSERVED ball, carried through ball gaps, D where p_dead > 0.5, then a centred
    majority vote of +/- `window` frames. Returns codes."""
    import baseline
    import team
    ball_raw = prepared["ball_raw"]
    d_home, d_away = team.nearest_distances(prepared["players"], ball_raw,
                                            prepared["team_of_track"])
    codes = np.where(np.isnan(d_home), -1, np.where(d_home <= d_away, 0, 1)).astype(np.int8)
    last = -1
    for t in range(len(codes)):              # carry the last nearest team through ball gaps
        if codes[t] >= 0:
            last = codes[t]
        else:
            codes[t] = last
    codes[prepared["p_dead"] > 0.5] = -1
    out = np.full(len(codes), X, dtype=np.int8)
    for lo, hi in prepared["blocks"]:
        letters = baseline.smooth(codes[lo:hi], window)
        out[lo:hi] = (letters == "A") * A + (letters == "D") * D    # "H" is 0
    return out


def score(labels, codes):
    """spellF1 (mean over IoU), macroF1 and the possession count of a prediction."""
    runs = pm.valid_runs(labels)
    spells = pm.spell_scores(labels, codes, runs)
    cm3 = pm.scored_confusion(pm.confusion(labels, codes))
    return {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(cm3),
            "n_pred": spells["n_pred"], "n_gt": spells["n_gt"],
            "team_acc": pm.team_accuracy(cm3)[1]}


def prepare_samples():
    """The three held-out sample games: p_dead, ball estimate and p_home, none fitted on itself.

    Each game gets `logit` (the evidence the pipeline uses), and its two parts for the ablation
    and --tune-shape: `ball_logit` (ball evidence only, which was team-v1) and `shape_logit`.
    """
    import team
    prepared = {n: team.sample_game(n) for n in team.TRAIN_GAMES}
    for n, g in prepared.items():
        models = team.fit_models(prepared[m] for m in team.TRAIN_GAMES if m != n)
        g["ball_logit"] = team.scaled_logit(models["ball"], g["features"], g["ball_conf"])
        g["shape_logit"] = team.shape_logit(models["shape"], g["shape"], g["shape_trust"])
        g["logit"] = team.blend(g["ball_logit"], g["shape_logit"], g["ball_conf"])
    return prepared


def run_decoder(g, logit=None, **params):
    p = dict(switch_ha=SWITCH_HA, switch_dead=SWITCH_DEAD, team_weight=TEAM_WEIGHT)
    p.update(params)
    return decode(None, g["p_dead"], g["blocks"], logit=g["logit"] if logit is None else logit,
                  **p)


def mean_spell_f1(prepared, **params):
    return float(np.mean([score(g["labels"], run_decoder(g, **params))["spell_f1"]
                          for g in prepared.values()]))


def report(prepared):
    print("{:<5} {:<36} {:>8} {:>8} {:>12} {:>9}".format(
        "game", "method", "spellF1", "macroF1", "possessions", "team acc"))
    totals = {}
    for n, g in prepared.items():
        rows = [("baseline vote, +/-50 (with p_dead)", nearest_team_codes(g, 50)),
                ("baseline vote, +/-125 (with p_dead)", nearest_team_codes(g, 125)),
                ("decoder, ball evidence only (v1)", run_decoder(g, logit=g["ball_logit"])),
                ("decoder, ball + team shape (v2)", run_decoder(g))]
        for name, codes in rows:
            s = score(g["labels"], codes)
            totals.setdefault(name, []).append(s["spell_f1"])
            print("{:<5} {:<36} {:>8.3f} {:>8.3f} {:>5d} / {:<5d} {:>9.3f}".format(
                n, name, s["spell_f1"], s["macro_f1"], s["n_pred"], s["n_gt"], s["team_acc"]))
    print("\nmean spellF1: " + ", ".join("{} {:.3f}".format(k, np.mean(v))
                                         for k, v in totals.items()))


def tune(prepared):
    """Grid search of the three parameters on the sample games, by mean spellF1."""
    results = []
    for team_weight in (0.5, 1.0, 1.5, 2.0):
        for switch_ha in (5, 10, 15, 25, 40, 80):
            for switch_dead in (10, 20, 40):
                f1 = mean_spell_f1(prepared, switch_ha=switch_ha, switch_dead=switch_dead,
                                   team_weight=team_weight)
                results.append((f1, team_weight, switch_ha, switch_dead))
    results.sort(reverse=True)
    print("mean spellF1 over the three sample games (held-out p_dead and p_home)")
    print("{:>8} {:>12} {:>10} {:>12}".format("spellF1", "team_weight", "switch_ha", "switch_dead"))
    for f1, w, ha, dead in results[:12]:
        print("{:>8.3f} {:>12} {:>10} {:>12}".format(f1, w, ha, dead))
    print("   ...\n{:>8.3f} {:>12} {:>10} {:>12}  (worst)".format(*results[-1]))


def tune_shape(prepared):
    """How much the team-shape evidence should count, on the sample games.

    w_gap is its weight where the ball is unknown, w_obs where the ball is observed; the pipeline
    uses (team.SHAPE_WEIGHT, 0). Also prints how often each kind of evidence, taken alone and
    frame by frame, names the right team.
    """
    import team
    print("frame-level team accuracy on live frames, held-out (ball evidence / shape evidence)")
    for n, g in prepared.items():
        live = g["labels"] < D
        for name, mask in (("ball observed", live & (g["ball_conf"] > 0.99)),
                           ("ball gap", live & (g["ball_conf"] < 0.5))):
            hit = [float((((x[mask] < 0).astype(np.int8)) == g["labels"][mask]).mean())
                   for x in (g["ball_logit"], g["shape_logit"])]
            print("  game {} {:<14} {:.3f} / {:.3f}".format(n, name, *hit))
    print("\nmean spellF1 by shape weight")
    print("{:>6} {:>6} {:>8}   per game".format("w_gap", "w_obs", "spellF1"))
    for w_gap, w_obs in ((0, 0), (0.25, 0), (0.5, 0), (1, 0), (1.5, 0), (1, 0.25), (1, 0.5),
                         (1, 1)):
        f1 = []
        for g in prepared.values():
            conf = np.clip(g["ball_conf"], 0.0, 1.0) ** team.CONF_POWER
            logit = g["ball_logit"] + (w_gap * (1 - conf) + w_obs * conf) * g["shape_logit"]
            f1.append(score(g["labels"], run_decoder(g, logit=logit))["spell_f1"])
        print("{:>6} {:>6} {:>8.3f}   {}".format(w_gap, w_obs, np.mean(f1),
                                                 " ".join("{:.3f}".format(x) for x in f1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tune", action="store_true", help="grid search on the sample games")
    parser.add_argument("--tune-shape", action="store_true",
                        help="weight of the team-shape evidence on the sample games")
    args = parser.parse_args()
    samples = prepare_samples()
    if args.tune:
        tune(samples)
    elif args.tune_shape:
        tune_shape(samples)
    else:
        report(samples)
