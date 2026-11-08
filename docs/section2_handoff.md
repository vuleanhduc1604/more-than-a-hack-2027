# Section 2 handoff (ball-v0, tagging-v0)

From the repo root, with the data in `data/` or `MTAH_DATA_DIR`. First check that `wc -l data/tactical/BAR-ATH_tactical_atd.txt` gives 199319, because a cut-off download makes the loader stop with "expected 3 fields".

```python
import sys; sys.path.insert(0, "starter")
import ball, tagging
b = ball.load()      # ball_xy (199319, 2) float32, ball_conf (199319,) float32, ball_method (199319,) uint8
t = tagging.load()   # restart (199319,) float32, st_team (199319,) int8
```

Row i is frame i of the tactical video. Both calls cache to `cache/` and rebuild when the data, `VERSION` or the parameters change. Check `ball.VERSION` and `tagging.VERSION` if you cache anything derived from these arrays.

- `ball_xy`: ball position in metres, with no NaN on scored frames. Gaps inside a half are straight lines, and the ends of a half hold the nearest observation. It is never carried across halftime, and frames outside the scored blocks stay NaN where the tracker had no ball. Use `ball.scored_blocks()` to get the scored blocks as half-open ranges.
- **Most of the missing ball sits in long gaps.** Gaps over 10 s hold about 85% of the missing frames, and gaps over 3 s about 99%. Inside those gaps the straight line is close to a guess, so `ball_conf` is at its floor. Read those frames as "position unknown". Some long gaps may be stoppages, which is your call in Section 3.
- `ball_xy` is **not clipped to the pitch**. Points up to 8 m outside the lines are kept, so off-pitch detection works. Spikes, and points further out than 8 m, are removed and then filled. Points further out are fairly common, so v1 may expose them as a separate signal.
- `ball_conf`: 1.0 where the ball was observed. In a gap it falls with the distance to the nearest observation and with the gap length, with a floor of 0.05. It is 0.0 outside the scored blocks where there is no ball. It is only roughly calibrated until v1, so read it as an ordering rather than as a probability.
- `ball_method`: 0 observed, 1 linear, 2 hold, 3 player-based (last resort only), 255 outside the scored blocks.
- `restart`: in v0, 1.0 over each set-piece tag window and 0 elsewhere. **The windows are long, about 29 to 35 s**, so `restart` is 1 on about 28% of scored frames. Read it as "a set piece happens somewhere in this half-minute", not as the restart moment. In v1 it becomes a soft bump that peaks near the likely restart.
- `st_team`: 0 home, 1 away, -1 unknown. It is the team named by the phase tags, and -1 where tags disagree. Every set piece is -1 in v0, including throw-ins and kick-offs, which are both tagged "n/a".
- **Caveat for Lam:** each defending code appears to mirror an attacking code on the same window with the other team named. That is why about 41% of scored frames are -1 from conflicting tags. tagging-v1 fixes the polarity. Until then, do not use `st_team` as "team in possession".
- All four arrays are evidence. Nothing here says who has the ball.
