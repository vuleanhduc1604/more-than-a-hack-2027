# Section 4 handoff (team-v2, decode-v2)

Run from the repo root, with the match data in `data/` and the Metrica sample games in `data/metrica-sample-data` (README step 4).

```bash
python starter/make_submission.py        # writes evaluation/results/section4_v2.csv, validated
python starter/decode.py                 # decoder vs majority vote on the three sample games
python starter/decode.py --tune          # the grid search behind the three constants
python starter/decode.py --tune-shape    # how much the team-shape evidence should count
python -m pytest tests -q
```

```python
import sys; sys.path.insert(0, "starter")
import team, decode, dead_ball, ball
p_home = team.load()                     # (199319,) float32 in [0, 1], no NaN
codes = decode.decode(p_home, dead_ball.load(), ball.scored_blocks())   # int8, 0 H 1 A 2 D 3 X
```

The first `make_submission.py` call builds and caches `p_home` (it needs the sample games, about a minute). Later calls take a few seconds. Two runs write byte-identical files.

## What it is

**`starter/team.py` makes `p_home`.** Per frame it measures the distance from the ball to the nearest home player and the nearest away player, and feeds a plain logistic regression with:

- the signed margin between them, capped at 20 m
- that margin averaged over 0.5, 2 and 5 s (a team that *stays* closest)
- the margin's trend over +/- 0.5 s (who is moving towards the ball)
- the share of the last 2 s that home was closest
- a "someone is on the ball" term: the nearest player's distance, faded at 3 m
- the nearest player's distance (a loose ball is anybody's)

The regression is fitted on the three Metrica sample games (ball degraded like ours, live frames with an observed ball only), never on our match. Then the logit is multiplied by `ball_conf`, so on estimated ball positions the ball evidence fades out. That is the "weigh evidence less when the ball is uncertain" rule.

**New in v2: team-shape evidence for the frames without a ball (plan step 7).** v1 stopped there, so inside a ball gap the decoder could only hold the last possessor. Our gaps average 17 s (150 gaps, 85% of gap frames sit in gaps longer than 10 s), so several hand-overs happen inside one gap, and on held-out sample games the ball evidence named the right team on 90% of observed-ball frames but only about 55% of gap frames. With a perfect team signal the same decoder would score about 0.84 spellF1, so this was the biggest loss in the pipeline.

The fix reads the players instead of the ball. The team in possession is stretched: its keeper stays home while the others push up and spread out. The team without the ball is compact. `team.shape_features` measures each team's spread (standard deviation of its players along and across the pitch) and the home-minus-away difference, averaged over 1 and 4 s and over 3 s just before and just after. It needs no ball and no playing direction. A second logistic regression, fitted on every live frame of the same sample games, turns that into a logit, and the two are blended by `ball_conf`:

```text
logit = ball_conf * ball logit + (1 - ball_conf) * shape logit
```

So the ball decides where it is seen, the shape decides where it is not, and a frame where either team has fewer than 7 tracked players (7% of our graded frames) gets no shape evidence at all and falls back to "last possessor keeps it".

**`starter/decode.py` makes the labels.** A three-state Viterbi decoder (H, A, D) per half. Per-frame cost is `-log(score)` with `score_home = (1-p_dead)*q`, `score_away = (1-p_dead)*(1-q)`, `score_dead = p_dead`, and `q = sigmoid(1.5 * logit(p_home))`. Changing state costs 10 for an H<->A hand-over and 20 for entering or leaving D. Frames outside the scored halves are `X`.

**`starter/make_submission.py`** chains Sections 1 to 4, writes the CSV and runs the scorer's own validation.

## Results

### Sample games (held-out: the p_dead and p_home for each game come from models trained on the other two; ball degraded to 41% missing)

| game | vote +/-50 (with p_dead) | vote +/-125 | decoder, ball only (v1) | decoder, ball + shape (v2) |
| --- | --- | --- | --- | --- |
| 1 | 0.338 | 0.229 | 0.406 | **0.511** |
| 2 | 0.300 | 0.213 | 0.357 | **0.481** |
| 3 | 0.334 | 0.229 | 0.425 | **0.494** |
| mean spellF1 | 0.324 | 0.224 | 0.396 | **0.495** |

The v1 decoder beat the vote on every game by 0.06 to 0.09 spellF1. The shape evidence adds another 0.07 to 0.12 on every game, both well above the scorer's 0.04 noise band for a full match. With it, macroF1 goes from 0.755, 0.731, 0.790 to 0.822, 0.810, 0.831, team accuracy from 0.77, 0.75, 0.82 to 0.90, 0.89, 0.90, and the possession count moves closer to the truth (309/386, 288/365, 332/407, from 264, 254, 293). Absolute numbers are modest because the sample labels are event-based: the team is that of the last on-ball event, so they are noisier than the corrected-tracking ground truth.

Frame by frame, on held-out games, the share of live frames where each kind of evidence alone names the right team (`--tune-shape`):

| | ball evidence | shape evidence |
| --- | --- | --- |
| ball observed | 0.90, 0.91, 0.90 | 0.87, 0.88, 0.90 |
| ball gap | 0.53, 0.52, 0.63 | 0.83, 0.84, 0.84 |

The shape weight is not a sensitive setting: any weight from 0.5 to 1.5 on gap frames scores 0.492 to 0.495, and adding shape on observed-ball frames too changes nothing beyond noise (0.489 to 0.497), so the blend keeps the simplest form (weight 1 where the ball is unknown, 0 where it is seen).

The decoder constants were re-checked with the new evidence and did not move: `switch_ha` 10, `switch_dead` 20, `team_weight` 1.5 is still on the flat top of the grid (0.495; most settings with `team_weight` 1.0 to 2.0, `switch_ha` 10 to 25 and `switch_dead` 10 to 40 score 0.49 to 0.495; the worst corner, `switch_ha` 80 with `team_weight` 0.5, drops to 0.37).

**Does the shape signal carry over to our match?** There are no labels to check it with, so the check is label-free: on our frames where the ball is observed, play is live and the ball evidence is confident, the shape logit picks the same team 85.7% of the time. On the sample games that figure is 92 to 95%. So it carries over, somewhat weaker, which fits our noisier tracks (the spread statistics vary more in our match than in the sample games).

### Our match (label-free)

- 412 possessions over the two halves, mean 8.6 s (v1: 378, mean 9.2 s). For scale, the public 5 minutes hold 17 true possessions with a mean of 14.2 s.
- Labels: H 26.0%, A 18.7%, D 35.0%, X 20.3% (X is only the frames outside the halves).

### Public 5 minutes (correctness check, run once, nothing tuned on it)

| | spellF1 | macroF1 | D F1 | possessions (truth 17) |
| --- | --- | --- | --- | --- |
| baseline_smooth50 | 0.326 | 0.621 | 0.323 | 26 |
| baseline_smooth125 | 0.424 | 0.581 | 0.146 | 16 |
| section4_v1 | 0.410 | 0.815 | 0.797 | 22 |
| **section4_v2** | **0.432** | **0.798** | **0.749** | 20 |

These are 17 possessions, so spellF1 gaps of this size are noise: `baseline_smooth125` scores almost as high on this window, but the baseline docs show it is *worse* than `smooth50` on the full match, which is why the public window may not be used to choose a setting. v2 was chosen on the sample games alone; here it only has to be valid and not broken, and it is (v1 and v2 differ by one or two possessions on this window, far inside the noise). The large v2 gain on the sample games cannot show up in 17 possessions, so the full-match score is the first real test of it. One thing to watch: the D class is a little lower than v1 on this window (recall 0.69 vs 0.75), while on the sample games v2's macroF1 is higher on every game.

## Possession rules for the technical document

**Approach.** Each frame gets a probability that the ball is dead (Section 3) and a probability that the home team controls it (`p_home`). A Viterbi decoder picks the single best sequence of H, A and D for each half, charging a fixed cost per change of state, and the costs are set on labelled sample matches.

**Rules, as the code implements them:**

- **Touches and deflections** are too short to repay two switches (a dip has to be strong for roughly 10 frames or more to flip the label), so possession stays put.
- **Loose balls.** A ball far from everyone gives a `p_home` near 0.5, so weak evidence never repays a switch and the last possessor keeps the ball.
- **Ball gaps.** Where the ball is not tracked, the ball evidence is faded out by `ball_conf` and the team shapes take its place: the side that is stretched out has the ball, the side that is compact is defending. A sustained change of shape hands the ball over inside the gap. Where too few players are tracked to read a shape, the last possessor keeps it.
- **Hand-overs** need sustained evidence: the team margin has to favour the other side for about a second or more.
- **Stoppages.** A high `p_dead` for long enough moves the path to D, which ends the possession; a one-frame false alarm does not.
- **Restarts.** After D, the next team is whichever the evidence favours, with no memory of who had the ball before.
- **Halves** are decoded independently, so nothing runs across half-time.
- **X** is never written. A decoder already holds the last possessor through weak evidence, which is what `X` would do for free inside a possession and at high cost across a hand-over; there was no sample-game evidence that it helps.

**Training data for this section.** Metrica's three public sample games (tracking plus event data, anonymised, labelled by their event streams; team on the ball = team of the last on-ball event). No data from the provided match was used to fit either `p_home` model, to choose the constants, or to pick a version.

**Limitations.**

- The shape signal is a tendency, not a rule: a team defending high or countering with few players can look like the other side for a few seconds. It is right on about 84% of gap frames on the sample games, and the label-free check suggests somewhat less on our match.
- The sample games' player tracks are clean; ours drop players (7% of graded frames have a team with fewer than 7). Those frames get no shape evidence, but frames with 8 to 10 tracked players are used as they are.
- On observed-ball frames `p_home` is mostly "who is nearest, and for how long". A defender standing next to the ball while the attacker passes it is still called as the nearer team for a moment, and a long pass to a teammate reads as a loose ball until a receiver is close.
- The sample-game team labels are approximate (last on-ball event), so the constants are tuned against a slightly different target than the real ground truth.
- The decoder is a first-order model: it has no notion that possessions usually last longer after certain restarts, or that a team usually kicks off after conceding.
- Smart Tagging (`st_team`, `restart`) is unused here: the sample games have no tags to validate a weight on.

**Possible next steps, in order of value.**

1. Control vs proximity: whether the ball moves *with* a player (carried) or simply passes them, from ball speed and direction changes.
2. Degrade the sample games' player tracks like ours (dropouts, broken tracks) before fitting the shape model, so its weights see the same noise.
3. Smart Tagging as a prior at restarts (who takes the throw-in, corner or goal kick), once a labelled source exists to calibrate it.
4. Restart-aware switch costs (a hand-over right after a stoppage is cheap).
5. Try `X` inside long low-confidence stretches and keep it only if it wins on the sample games.

## Done when

- [x] The pipeline produces a valid 199,319-row CSV (`make_submission.py`, validated by the scorer's own checks)
- [x] The decoder beats majority voting on labelled games, beyond the noise band (+0.17 mean spellF1 on three games; +0.10 of it from the v2 shape evidence)
- [ ] Predicted possession counts and lengths look realistic against the sample matches (412 predicted over our match; the sample games have 365 to 407 true possessions, so the count is close to that range, but the full-match truth is not available to check)
- [ ] Transition timing shows no systematic lag (public window: median -2 frames, mean -5.3 frames, inside the 25-frame tolerance; only 20 transitions, so indicative only)
- [x] Costs and limits were set without using the public window
