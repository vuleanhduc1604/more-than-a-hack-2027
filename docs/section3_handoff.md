# Section 3 handoff (dead-ball-v1)

Run from the repo root, with the match data in `data/` and the Metrica sample games in `data/metrica-sample-data` (README step 4).

```python
import sys; sys.path.insert(0, "starter")
import dead_ball
p_dead = dead_ball.load()   # (199319,) float32 in [0, 1], no NaN; frame i is row i
```

The first call trains the model and caches the result in `cache/`, which takes about two minutes. Later calls are instant, and a rebuild gives a byte-identical array. `python starter/dead_ball.py --report` prints the numbers below.

## What it is

`p_dead` is the probability that play is stopped on each frame. It comes from a logistic regression trained on Metrica's three public sample games, where event data gives the true stoppages. It reads two things:

- **Team speed.** It uses the median and the 90th-percentile speed of the tracked players, averaged over windows of 0.5 to 10 s, plus a 3 s window 2 s ahead and 2 s behind. When play stops, everyone walks. Our match and the sample games have almost the same speed distributions, which is why a model trained on the samples transfers.
- **Ball over a line.** The observed ball was beyond a touchline or goal line in the last 3 s. Only observed positions count, after Section 2's outlier removal.

A missing ball is never treated as evidence. Nothing refers to a frame of our match.

## Results

### Held-out sample games

Each game is scored with a model trained on the other two. Before training and scoring, the sample ball was hidden on 41% of live frames, using our match's gap lengths. spellF1 here uses the true team on live frames, so it measures only the dead/live call.

| game | baseline rule (ball missing) | dead-ball-v1 |
| --- | --- | --- |
| 1 | 0.705 | **0.766** |
| 2 | 0.632 | **0.768** |
| 3 | 0.653 | **0.808** |

- The sample data is biased in the baseline's favour. Its ball is removed on nearly every dead frame, so "ball missing" looks far better there than it is on our match, where the ball is still tracked on about two thirds of dead frames.
- Stoppage boundaries land within a second of the truth (median start +0.2 to +0.8 s late, end −0.4 to +0.2 s).
- Stoppages found by restart type: kick-off 94%, corner 88%, free kick 80%, throw-in 71%, goal kick 54%.

### Our match (label-free checks)

- 43% of graded frames are called dead, against 31–42% in the sample games.
- There are 154 stoppages, with a median of 8 s.
- 96% of the Smart Tagging set-piece windows contain a stoppage, although the tags are not an input.

### Public 5 minutes (correctness check only, nothing was tuned on it)

Swapping only the dead-ball rule into the baseline (team = nearest player, carried through ball gaps):

| | alive/dead F1 | macroF1 | spellF1 |
| --- | --- | --- | --- |
| baseline_smooth50 | 0.899 | 0.621 | 0.326 |
| baseline team + p_dead > 0.5, smooth 50 | 0.952 | 0.820 | 0.432 |
| baseline_smooth125 | 0.899 | 0.581 | 0.424 |
| baseline team + p_dead > 0.5, smooth 125 | 0.952 | 0.826 | 0.507 |

These are five minutes and 17 possessions, so treat the numbers as "it works on our match", not as an estimate of the gain.

## For Lam (Section 4)

- **Use `p_dead` as the D evidence in the decoder.** It is a calibrated probability, so the plan's `score_dead = p_dead` works as written. It is already smooth on a scale of seconds, but it still has short wobbles: a brief false "dead" in the middle of a possession splits it. A switch cost in the decoder is the right place to suppress them, and setting it on the sample games is the honest way.
- **`starter/sample_games.py` gives labelled H/A/D frames for three games**, in the same array format as `load_data`. Use `sample_games.load(n)` and `sample_games.degrade(game)`. It is the "compare" data the plan wanted from Section 1, and you can set your switch costs on it. Its team labels come from events (the team of the last on-ball event) and are approximate. `dead_ball.compare()` and `dead_ball.oracle_codes()` show how to score one call against them.
- **Weak spots to expect:** short stoppages where players keep jogging (goal kicks and quick throw-ins are found only about half to two thirds of the time), and the two minutes before the second-half kick-off (only 75% called dead). A missed stoppage between two possessions of the same team merges them. A missed one between different teams is just a hand-over to you.

## For the technical document: dead-ball rules and limitations

**Rule.** A frame is dead when the players' collective movement says play is stopped. Specifically, a logistic regression on multi-scale team speed and a recent observed ball crossing of a line gives p_dead > 0.5. The model is trained on Metrica's public sample games, never on our match, and our public window was used once as a final correctness check.

**Why not "ball missing".** In our tracking the ball is missing on 41% of graded frames, most of them in open play. Treating them as dead splits possessions everywhere.

**Limitations.**

- **Restarts with movement.** Stoppages where players do not slow down (quick throw-ins, goal kicks) are missed more often.
- **Event-based labels.** The sample-game labels come from events, while the challenge's ground truth comes from corrected tracking, so the exact start and end frames of a stoppage may be defined slightly differently.
- **Untestable ball clues.** The sample games carry no ball during stoppages, so ball-based dead-ball signals (a ball lying still, a ball being carried to the restart spot) cannot be validated there and are not used.
- **Smart Tagging unused.** The `restart` hint is not an input, for the same reason: the sample games have no Smart Tagging to validate it on.

**Possible next steps, in order of value.**

1. A kick-off formation rule for the pre-kick-off stretch.
2. A ball-speed restart detector (a sudden kick after a slow phase), validated some other way.
3. Using `restart` as a prior inside tagged windows.
