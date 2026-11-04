# More than a Hack 2027: Work Sections Plan

Repo copy of the [Claude Docs version](https://claude.ai/artifact/DsYaFuHVbPpNSGaWi5BNaq), as of Nov 2, 2026.

## Overview

We need to beat a full-match spellF1 of 0.2433 by labelling every frame of Barcelona vs Athletic as `H`, `A`, `D` or `X`. The work splits into four sections, connected only by per-frame arrays, and they run one after another: each part starts when the previous one finishes. The final submission is due 17 November 2026.

The baseline uses only the tracking. It gives each frame to the team of the player closest to the ball, marks missing-ball frames as `D`, then smooths with a majority vote. Each section replaces part of that logic.

| Section | Owner | Status | Main question | Baseline weakness it fixes |
| --- | --- | --- | --- | --- |
| 1. Data & validation | Harry | Done | Can we load the data and prove a change helped? | Only the public 5 minutes can be scored, and we can't tune on them |
| 2. Ball & context | Khai | To do | Where is the ball, and what do the tags suggest? | The ball is missing on 41% of scored frames; Smart Tagging is unused |
| 3. Dead ball | Milu | To do | Is the ball in play? | "Ball missing" is treated as "ball dead" |
| 4. Possession & decoder | Lam | To do | Which team has it, and where do possessions start and end? | Closest isn't control, nothing remembers the last possessor, and voting only hides flicker |

```text
1. Data & validation: gives every section the shared arrays
        |                     |                     |
        v                     v                     v
2. Ball & context ---> 3. Dead ball ---> 4. Possession & decoder ---> Final CSV
        |                                        ^                       |
        +---------- ball estimate and hints -----+                       |
                                                                         |
1. Data & validation <--------- validated and compared ------------------+
```

Section 1 supplies the data, Sections 2 and 3 add evidence, and Section 4 turns it into labels.

Slides sit outside these four sections.

## Shared formats and rules

Every per-frame array has 199,319 entries, indexed by tactical video frame, with `NaN` for missing values. Agree these formats before coding, because they're the only link between sections.

| Array | Shape | Meaning | Made by | Used by |
| --- | --- | --- | --- | --- |
| `players` | (199319, 46, 2) | Each track's x, y in metres | Section 1 | 2, 3, 4 |
| `speeds` | (199319, 46) | Each track's speed in m/s; missing on about 9.8% of tracked positions | Section 1 | 2, 3, 4 |
| `ball_raw` | (199319, 2) | Observed ball x, y in metres | Section 1 | 2, 3 |
| `team_of_track` | (46,) | 0 = home, 1 = away | Section 1 | 2, 3, 4 |
| Tag table | One row per tag | Code, team, type, half, start frame, end frame | Section 1 | 2 |
| `ball_xy` | (199319, 2) | Best estimate of the ball, gaps filled | Section 2 | 3, 4 |
| `ball_conf` | (199319,) | Trust in `ball_xy`, 0 to 1 | Section 2 | 3, 4 |
| `restart` | (199319,) | Soft hint that a restart is near | Section 2 | 3, 4 |
| `st_team` | (199319,) | Team suggested by the tags, or unknown | Section 2 | 4 |
| `p_dead` | (199319,) | Probability the ball is dead | Section 3 | 4 |
| `p_home` | (199319,) | Probability Barcelona has it, given the ball is alive | Section 4 | 4 |

**Rules for every section:**

- **Labels come from an algorithm.** No hand labels or hand corrections on this match, even for training or validation. Hand labels on other matches are allowed.
- **The public 5 minutes check correctness only.** Never train on them or use them to choose settings or versions.
- **No outside data about this match.** That covers event data, corrected tracking and any other human-produced data.
- **No constants tied to specific moments of this match.** Rules must work on any match.
- **Never commit match data.** Keep `data/`, `cache/`, our CSVs and notebook outputs out of the public repo.
- **Never upload match data to a cloud LLM.** Share only code, shapes, counts and error messages.
- **The repo must regenerate the exact CSV.** Pin dependencies and seed everything.

## Section 1: Data & validation

**Owner:** Harry · **Status:** Done

This section gives everyone clean data and a trustworthy way to compare versions. It doesn't raise the score directly, but it unblocks every other section and produces the file we submit.

**Files:** `starter/load_data.py`, `starter/validate/`, `starter/make_submission.py`

### In plain words

Section 1 does three jobs: it prepares the data, tests whether changes help, and produces the final file. Think of a kitchen: it prepares the ingredients, tastes each dish, and sends out the final plate. Numbers below are made up, to show the idea.

**Input:** the raw files in `data/tactical/`, plus the sample matches.

**Output:** the shared arrays, the tag table, a score for any version, and the final CSV.

**Job 1: prepare the data.** The tracking file has one long line of text per frame. Section 1 splits each line and stores the values in a table with one row per frame:

```text
Raw line:  1000 : track 1; track 2; ... track 46 : ball

Becomes:   players[999]   positions of all 46 tracks, in metres
           speeds[999]    speeds of all 46 tracks
           ball_raw[999]  ball position, or NaN if missing
```

The row is 999, not 1000, because tracking counts from 1 and video frames count from 0. Positions are multiplied by the pitch size to give metres.

```python
frame = int(frame_id) - 1
players[frame, slot] = float(x) * 105.3, float(y) * 68.0
```

**Job 2: test whether a change helped.** We can't tune on the public 5 minutes, so we use other matches where the answers can be derived from event data. Their tracking may be cleaner than ours, so we first make it messier, for example by hiding the ball on about 40% of frames.

```text
Version A on sample matches:  spellF1 0.30
Version B on sample matches:  spellF1 0.38
Gap is above the noise, so keep version B.
```

**Job 3: produce the final file.** One command runs every section in order, writes the CSV and validates it. Running it twice must give exactly the same file.

### Data needed

| Data | Where to get it | Used for |
| --- | --- | --- |
| Tracking positions | `data/tactical/BAR-ATH_tactical_atd.txt` | `players`, `speeds`, `ball_raw` |
| Track metadata | `data/tactical/BAR-ATH_tactical_atd.xml` | `team_of_track` |
| Smart Tagging | `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | Tag table |
| Pitch size and frame counts | `games.py` | Metre conversion and array length |
| Other matches with events | [Metrica open sample data](https://github.com/metrica-sports/sample-data); contents still to be checked | Labelled comparison matches |
| Public ground truth | `evaluation/gt/BAR-ATH_possession_gt_public.csv` | Pipeline correctness only |

### Steps

1. Create `starter/load_data.py` and add a gitignored `cache/` folder.
2. Parse the tracking into the shared arrays. A tested prototype reads all frames in about 5 seconds, with no malformed lines.
3. Save the arrays to `cache/`, and rebuild them if the source file changes.
4. Wrap `starter/load_smart_tagging.py` into a clean tag table. Normalise text case, and keep `N/A` teams as they are.
5. Run the starter alignment checks. Add tests for shapes, the `frame_id - 1` offset and the metre conversion.
6. Write `make_submission.py`. First make it reproduce `baseline_smooth50` exactly, then swap in the real modules.
7. Check what the sample matches contain, then turn their events into per-frame H/A/D labels.
8. Write a degrade step for the sample tracking: hide the ball on about 40% of frames, mostly at contested moments, and add noise and broken tracks.
9. Build a compare command that prints `spellF1`, `macroF1` and bootstrap intervals for any version.
10. Add label-free checks for our match: possession count and length, plus agreement with Smart Tagging.
11. Before submitting, pin dependencies, compare CSV hashes from two runs, and test a fresh clone.

### Outputs

- Shared arrays and the tag table, loadable in one line
- A compare command that all sections use
- `make_submission.py`, which writes and validates the CSV

### Intended outcome

Any team member can load the data in seconds, test a change on labelled matches, and regenerate the final CSV exactly.

### Done when

- [ ] All arrays match the agreed shapes, and alignment tests pass
- [ ] `make_submission.py` reproduces `baseline_smooth50` byte for byte
- [ ] The compare command runs on at least one degraded sample match
- [ ] Two runs give identical CSV hashes
- [ ] A fresh clone plus the data regenerates the submission
- [ ] `validate_submission.py` passes on the exact file we upload

## Section 2: Ball & context

**Owner:** Khai · **Status:** To do

This section gives later steps a usable ball estimate on every frame, and turns Smart Tagging into per-frame hints. It tackles the biggest data problem: the ball is missing on 41% of scored frames, often at the moments that matter.

**Files:** `starter/ball.py`, `starter/tagging.py`

**Deadline:** starts Nov 2, 2026 · finished by Nov 8, 2026, with its tech-doc part due the same day

### In plain words

Section 2 turns messy ball data and raw tags into four clean arrays, using two separate paths. It gives evidence, not decisions: it never says who has the ball. Numbers below are made up, to show the idea.

**Input:** `ball_raw`, `players`, `speeds`, `team_of_track` and the tag table.

**Output:** `ball_xy`, `ball_conf`, `restart` and `st_team`.

**Path A: fill in the ball.** First remove impossible positions. Then fill short gaps with a straight line between the known points. Long gaps are too uncertain for a line, so we estimate from players, such as where they are converging.

```text
Frame:      100    103    105    108    110
ball_raw:     A    NaN    NaN    NaN      B
ball_xy:      A    filled in along the line   B
ball_conf:   1.0   0.7    0.5    0.7    1.0
```

`ball_conf` says how far to trust each position. It drops further from a known point.

**Path B: turn tags into per-frame values.** Tags cover time ranges, but later sections work frame by frame. Picture a row of 199,319 boxes, and fill in the boxes each tag covers.

`restart` starts at 0 everywhere. A corner tagged from frame 5,000 to 5,125 raises those boxes:

```text
Frame:     4,999   5,000   5,060   5,124   5,125
restart:     0       1       1       1       0
```

`st_team` starts as unknown everywhere. A Barcelona build-up from frame 7,000 to 7,250 fills those boxes with home:

```text
Frame:      6,999     7,000    7,100    7,249    7,250
st_team:   unknown    home     home     home    unknown
```

```python
restart = np.zeros(199319)
restart[5000:5125] = 1          # for each set-piece tag

st_team = np.full(199319, -1)   # -1 = unknown
st_team[7000:7250] = 0          # 0 = home, for a Barcelona tag
```

The real code loops through the tag table rather than using fixed frames, so nothing is tied to this match.

**Two tricky cases:**

- **Throw-ins:** the team is `N/A`, so estimate it from the tracking, such as which team's players are near the ball at the touchline.
- **Overlapping tags:** if two tags disagree on a frame, the simplest rule is to write unknown.

### Data needed

| Data | Where to get it | Used for |
| --- | --- | --- |
| `ball_raw` | Section 1 | Known ball positions and gaps |
| `players`, `speeds`, `team_of_track` | Section 1 | Estimating the ball during long gaps |
| Tag table | Section 1 | Restart and team hints |
| Tactical video and homography (stretch goal) | `data/tactical/` through `games.asset_paths()` | Our own ball detection |

### Steps

1. Measure the gaps: how many, how long, and where they fall.
2. Remove implausible ball positions, such as impossible jumps between frames.
3. Fill short gaps by joining known positions with straight lines.
4. Estimate longer gaps from player movement. Start simply, such as the point players are converging on.
5. Set `ball_conf`: 1.0 where observed, falling as a gap gets longer.
6. Turn set-piece tags into `restart`, a soft signal over the tag's window. Tags open early, so don't treat the start frame as exact.
7. Turn phase tags into `st_team`: home, away or unknown. Keep `N/A` as unknown at first.
8. Infer the team for throw-ins from player positions. Every throw-in is tagged `N/A`.
9. Test gap filling: hide known ball positions, estimate them, and report the error in metres by gap length.
10. Stretch goal: run a ball detector on the tactical video. Any model must be accessible to the organisers.

### Outputs

- `ball_xy` and `ball_conf`
- `restart` and `st_team`
- A short report of gap-filling error by gap length

### Intended outcome

Sections 3 and 4 have a ball estimate on every frame, know when to distrust it, and receive restart and team hints. Those hints stay evidence, not answers, because Smart Tagging has false positives and misses.

### Done when

- [ ] `ball_xy` has no `NaN` on scored frames
- [ ] `ball_conf` falls as measured error rises
- [ ] Tag frames line up with the starter's time-base check
- [ ] Throw-ins have a team estimate wherever the evidence allows
- [ ] No tag interval is copied directly into the final labels

## Section 3: Dead ball

**Owner:** Milu · **Status:** To do

This section replaces "ball missing means dead" with a real estimate of whether the ball is in play. It matters directly to the ranking: a false `D` splits a possession, and a missed `D` can merge two.

**File:** `starter/dead_ball.py`

**Deadline:** starts Nov 8, 2026 · finished by Nov 11, 2026, with its tech-doc part due the same day

### In plain words

Section 3 answers one question for every frame: is the ball in play? It combines a few clues into one number, `p_dead`, from 0 (almost certainly in play) to 1 (almost certainly dead). Numbers below are made up, to show the idea.

**Input:** `players`, `speeds`, `team_of_track`, `ball_raw`, `ball_xy`, `ball_conf` and `restart`.

**Output:** `p_dead`, used by Section 4.

**The clues.** Each clue is its own row of boxes, from 0 to 1:

- **Ball off the pitch:** check `ball_xy` against the pitch lines, with a small margin for tracking error.
- **Players slowing down:** a low average speed suggests a stoppage.
- **Ball sitting still:** a ball that doesn't move may be waiting for a restart.
- **Tag restart:** the `restart` hint from Section 2.

**Example.** The ball goes out for a throw-in around frame 8,000:

```text
Frame:           7,990   8,000   8,050   8,100   8,150
Ball off pitch:    0       1       1       0       0
Players slow:      0      0.5      1       1       0
Ball still:        0       0       0       1       0
Tag restart:       0       0       0       1       1
                  ---     ---     ---     ---     ---
p_dead:           0.05    0.6     0.9     0.9     0.2
```

Read left to right, it tells a story: play is on, the ball goes out, players slow down, the throw-in is set up, and play restarts.

```python
ball_out = outside_pitch(ball_xy)     # 0 or 1 per frame
slow = low_team_speed(speeds)         # 0 to 1
still = ball_not_moving(ball_xy)      # 0 to 1

p_dead = 0.5 * ball_out + 0.3 * slow + 0.2 * still
```

The function names and weights are placeholders. Real weights come from the sample matches, never the public 5 minutes.

**The difference from the baseline:**

| Situation | Baseline | Section 3 |
| --- | --- | --- |
| Ball missing, players still running | Dead, which is wrong | In play |
| Ball visible, lying still before a throw-in | In play, which is wrong | Dead |

Section 3 never names a team. It only says whether the ball is in play, and Section 4 decides who has it.

### Data needed

| Data | Where to get it | Used for |
| --- | --- | --- |
| `players`, `speeds`, `team_of_track` | Section 1 | Slowdowns and restart formations |
| `ball_raw` | Section 1 | Line crossings and a stationary ball |
| `ball_xy`, `ball_conf` | Section 2 | Ball position during gaps |
| `restart` | Section 2 | Evidence that a stoppage is ending |
| Half boundaries | `games.half_video_blocks("BAR-ATH")` | Scored frame ranges |
| Pitch size | `games.py` | Pitch lines in metres |
| Labelled sample matches | Section 1 compare command | Measuring and calibrating |
| Public ground truth | `evaluation/gt/` | Understanding what counts as `D`, not tuning |

### Steps

1. Study how stoppages look in the public window, only to understand the definition.
2. Build a signal for the ball crossing a pitch line, with a margin for tracking error.
3. Build a signal for players slowing down together.
4. Build a signal for a ball that stays still.
5. Combine them into a first `p_dead`, using simple rules. Stop treating a missing ball as evidence by itself.
6. Detect kickoff formations, so the pre-kickoff stretch is `D` without hard-coded frames. In this match, the 2nd half is scored about 2 minutes before kickoff.
7. Add `restart` as evidence that a stoppage is ending.
8. Add restart-shape signals: corners, goal kicks, kickoffs, and a player near the touchline for a throw-in.
9. Find the exact restart moment from tracking, such as the first sharp ball or player movement.
10. If rules plateau, train a small classifier on the sample matches.
11. Evaluate by stoppage type, and list which types are missed.

### Outputs

- `p_dead`, with a value between 0 and 1 on every frame
- A report of hits and misses by stoppage type

### Intended outcome

The system finds real stoppages and their end points, and keeps play alive when tracking merely loses the ball. Section 4 then starts and ends possessions in the right places.

### Done when

- [ ] `p_dead` covers every frame and has no `NaN`
- [ ] Ball-missing frames aren't automatically treated as dead
- [ ] Stoppage detection beats the baseline on the compare command
- [ ] Restart timing error is measured on sample matches
- [ ] No rule depends on a specific frame in this match

## Section 4: Possession & decoder

**Owner:** Lam · **Status:** To do

This section decides which team has the ball and turns per-frame evidence into whole possessions. It produces the final labels, and it most directly targets `spellF1`, which rewards whole possessions rather than single frames.

**Files:** `starter/team.py`, `starter/decode.py`

**Deadline:** starts Nov 11, 2026 · finished by Nov 15, 2026, with its tech-doc part due the same day

### In plain words

Section 4 makes the final decision for every frame, in two steps. First it scores which team looks in control. Then a decoder turns those scores into whole possessions, instead of deciding each frame alone. Numbers below are made up, to show the idea.

**Input:** `players`, `speeds`, `team_of_track`, `ball_xy`, `ball_conf`, `st_team`, `restart` and `p_dead`.

**Output:** `p_home`, then one final label per frame: `H`, `A`, `D` or `X`.

**Step 1: score the team with `p_home`.** This is a number from 0 to 1 per frame, assuming the ball is in play. Near 1 means Barcelona looks in control, near 0 means Athletic, and 0.5 means unclear.

The clues are stronger than "who is closest": how much closer one team is than the other, how long a team has stayed closest, who is moving toward the ball, and whether the ball moves with a player or just passes them. When `ball_conf` is low, those clues count for less.

**Step 2: turn scores into possessions.** The decoder has three states, `H`, `A` and `D`, and changing state has a cost. So it only switches when the evidence is strong for long enough to justify that cost. It picks the single best path through the whole half, using Viterbi decoding.

```text
Frame:       9,000  9,010  9,020  9,030  9,040  9,050
p_home:       0.9    0.9    0.3    0.8    0.2    0.1
p_dead:       0.0    0.0    0.1    0.0    0.1    0.1

Per-frame:     H      H      A      H      A      A
Decoder:       H      H      H      H      A      A
```

Frame 9,020 is a brief dip, such as a deflection, so the decoder keeps `H`. From frame 9,040 the evidence stays with Athletic, so the decoder switches. Per-frame picking gives four possessions here, and the decoder gives two.

```python
# How well each state fits each frame
score_home = (1 - p_dead) * p_home
score_away = (1 - p_dead) * (1 - p_home)
score_dead = p_dead

labels = viterbi(score_home, score_away, score_dead, SWITCH_COST)
```

`viterbi` is a placeholder name, and `SWITCH_COST` is set from the sample matches, never the public 5 minutes.

**This gives us the possession rules:**

- **Touches and deflections:** too brief to justify a switch, so possession stays put.
- **Loose balls and gaps:** weak evidence never pays for a switch, so the last possessor keeps it.
- **Real hand-overs:** sustained evidence pays for the switch.
- **Stoppages:** a high `p_dead` moves the path to `D`, which ends the possession.

**Where `X` fits.** The scorer's ranking metric treats `X` as a repeat of our last label. So `X` is safe inside a possession, but it's costly across a hand-over. We keep it only if it helps on the sample matches.

### Data needed

| Data | Where to get it | Used for |
| --- | --- | --- |
| `players`, `speeds`, `team_of_track` | Section 1 | Proximity, movement and team shape |
| `ball_xy`, `ball_conf` | Section 2 | Team evidence, weighted by trust |
| `st_team`, `restart` | Section 2 | Weak team hints |
| `p_dead` | Section 3 | The `D` state in the decoder |
| Labelled sample matches | Section 1 compare command | Setting costs and comparing versions |
| Public ground truth | `evaluation/gt/` | Understanding hand-overs, not tuning |

### Steps

1. Start from the `p_dead` and ball estimates delivered by Sections 2 and 3.
2. Build a three-state H/A/D decoder with Viterbi decoding, including a cost for each type of switch.
3. Add a minimum possession length. Set costs and limits from the sample matches only.
4. Connect it to `make_submission.py` and write a valid CSV.
5. Build a better `p_home`: distance margin over the nearest opponent, how long a team stays closest, and who moves toward the ball.
6. Separate control from proximity by checking whether the ball moves with a player or simply passes them.
7. Add team-shape evidence for frames where `ball_conf` is low.
8. Reduce the weight of evidence when `ball_conf` is low, so the decoder keeps the last possessor through loose balls and gaps.
9. Use `st_team` and `restart` as weak hints, such as who starts after a stoppage.
10. Try writing `X` inside low-confidence stretches of a possession, never across a likely hand-over. Keep this only if it helps on the sample matches.
11. Check transition timing with `evaluation/reports/`, since late hand-overs cost points.

### Outputs

- `p_home`
- Final H/A/D/X labels for every frame
- The written possession rules for the tech doc

### Intended outcome

The labels form realistic possessions: brief touches and deflections don't flip the team, sustained control does, and loose balls stay with the last possessor. Hand-overs land close to the true frame.

### Done when

- [ ] The pipeline produces a valid 199,319-row CSV
- [ ] The decoder beats majority voting on the compare command, beyond the bootstrap noise
- [ ] Predicted possession counts and lengths look realistic against the sample matches
- [ ] Transition timing shows no systematic lag
- [ ] Costs and limits were set without using the public window

## Timeline and integration

The parts run as a relay: Harry, then Khai, then Milu, then Lam. Each has 3 to 6 days, and the last finishes at the 15 November freeze.

| Work | Dates (2026) |
| --- | --- |
| Section 1 (Harry) | Done Nov 2 |
| Section 2 (Khai) | Nov 2 to Nov 8 |
| Section 3 (Milu) | Nov 8 to Nov 11 |
| Section 4 (Lam) | Nov 11 to Nov 15 |
| Reproducibility checks | Nov 15 to Nov 16 |
| **Submit** | **Nov 16** |
| Official deadline | Nov 17 |

Each section builds on the one before it, so a late handover delays everyone after it.

**Working routine:**

- Run every change through the compare command, and keep it only if the gain is clearly above the noise.
- Run the public window only as a final correctness check.
- Merge to the shared branch only when `make_submission.py` still writes a valid CSV.
- Write your tech-doc part by the day your section finishes.

**Tech-doc ownership:**

| Section | Tech-doc part |
| --- | --- |
| 1. Data & validation | Reproduction and training data |
| 2. Ball & context | Assets used and dropped, and the ball estimate |
| 3. Dead ball | Dead-ball rules and limitations |
| 4. Possession & decoder | Approach and possession rules |
