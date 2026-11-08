"""Synthetic tracking and tags with the same structure as load_data, and a known truth.

Tests in this folder must pass on a fresh clone with no match data, so everything they need is
generated here. The generator plants the features Section 2 has to handle, at known places, so a
test can check each one:

    tracking, truth = make_tracking()
    tags = make_tags(len(tracking["ball_raw"]))

tracking has the keys and dtypes of load_data.load_tracking(): players (n, 46, 2) float32 metres,
speeds (n, 46) float32, ball_raw (n, 2) float32 with NaN where missing, team_of_track (46,) int8.
truth holds the noiseless ball path and the frames of every planted gap, spike and off-pitch spell.

Nothing here resembles the real match. Frames, positions and tags are made up.
"""

import numpy as np
import pandas as pd

# Params
SEED = 20261108
N_FRAMES = 6000
FPS = 25
PITCH = (105.3, 68.0)
N_TRACKS = 46
TRACKS_PER_TEAM = 23
ROLES = 11
# Scored blocks, half-open. The hole between them plays the part of halftime.
BLOCKS = ((500, 2700), (3300, 5600))

BALL_NOISE_M = 0.05
DRIBBLE_SPEED_MS = (2.0, 6.0)
PASS_SPEED_MS = (12.0, 30.0)
REST_FRAMES = (25, 100)

PLAYER_NOISE_M = 1.0
CHASER_SPREAD_M = (2.5, 4.0)  # home, away: how tightly the two nearest players crowd the ball
PLAYER_DROPOUT = 0.04         # gives 20 to 22 tracked players on most frames
SPEED_DROPOUT = 0.10          # speeds go missing more often than positions, as in the real data
SWAP_EVERY_FRAMES = 1500      # tracklets: each role moves to a different slot every so often

# Planted ball gaps (start, length), chosen to cover every report bucket.
GAPS = (
    (600, 1), (650, 3), (700, 5), (760, 8), (820, 18), (900, 40), (1000, 100), (1200, 300),
    (3400, 2), (3500, 12), (3600, 25), (3700, 60), (3900, 125), (4200, 260),
)
# Gaps that touch a block edge, so they must be held rather than interpolated.
EDGE_GAPS = ((2670, 30), (3300, 20))
SPIKE_FRAMES = (1700, 1900)   # one-frame impossible jumps
SPIKE_JUMP_M = 20.0
FAR_FRAME = 2000              # a point well beyond any off-pitch margin
FAR_X_M = PITCH[0] + 30.0
OFFPITCH_AFTER = 4700         # the ball goes just over a touchline at the first chance after this
OFFPITCH_POINT = (50.0, -3.0)
OFFPITCH_REST = 60


def _ball_path(n_frames, rng):
    """Noiseless ball path: rests, dribbles and fast passes between random points."""
    length, width = PITCH
    out = np.empty((n_frames, 2), dtype=np.float64)
    pos = np.array([length / 2, width / 2])
    frame, offpitch_done = 0, False
    while frame < n_frames:
        if not offpitch_done and frame >= OFFPITCH_AFTER:
            target, speed, rest = np.array(OFFPITCH_POINT), DRIBBLE_SPEED_MS[1], OFFPITCH_REST
            offpitch_done = True
        else:
            kind = rng.choice(3, p=[0.25, 0.45, 0.30])
            if kind == 0:
                rest_len = int(rng.integers(*REST_FRAMES))
                out[frame:frame + rest_len] = pos
                frame += rest_len
                continue
            target = rng.uniform([5.0, 5.0], [length - 5.0, width - 5.0])
            low, high = DRIBBLE_SPEED_MS if kind == 1 else PASS_SPEED_MS
            speed, rest = rng.uniform(low, high), 0
        distance = float(np.hypot(*(target - pos)))
        steps = max(2, int(np.ceil(distance / speed * FPS)))
        path = pos + (np.arange(1, steps + 1) / steps)[:, None] * (target - pos)
        path = np.vstack([path, np.repeat(target[None], rest, axis=0)])
        take = min(len(path), n_frames - frame)
        out[frame:frame + take] = path[:take]
        frame += take
        pos = target
    return out


def _smooth_noise(rng, shape, sd, window=50):
    """Slowly varying noise, so players drift instead of teleporting."""
    raw = rng.normal(0.0, sd, size=shape)
    kernel = np.ones(window) / np.sqrt(window)
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, raw)


def _players(ball, rng):
    """Two teams of 11 roles in a loose formation that follows the ball, written into 46 slots."""
    n_frames = len(ball)
    length, width = PITCH
    centre = np.array([length / 2, width / 2])
    players = np.full((n_frames, N_TRACKS, 2), np.nan)

    for team in (0, 1):
        sign = 1.0 if team == 0 else -1.0
        lines = np.array([[-45, 0]] + [[-30, y] for y in (-24, -8, 8, 24)]
                         + [[-12, y] for y in (-20, 0, 20)] + [[5, y] for y in (-15, 0, 15)],
                         dtype=np.float64)
        formation = centre + lines * np.array([sign * 0.9, 1.0])
        shift = (ball - centre) * np.array([0.35, 0.5])
        pos = formation[None] + shift[:, None] + _smooth_noise(
            rng, (n_frames, ROLES, 2), PLAYER_NOISE_M)

        # The two outfield roles nearest the ball close in on it.
        dist = np.hypot(*(pos - ball[:, None]).transpose(2, 0, 1))
        dist[:, 0] = np.inf  # the keeper stays home
        nearest = np.argsort(dist, axis=1)[:, :2]
        chase = ball[:, None] + rng.normal(0.0, CHASER_SPREAD_M[team], size=(n_frames, 2, 2))
        rows = np.arange(n_frames)[:, None]
        pos[rows, nearest] = chase

        # Tracklets: every SWAP_EVERY_FRAMES the roles move to the other bank of slots.
        bank = (np.arange(n_frames) // SWAP_EVERY_FRAMES) % 2
        for role in range(ROLES):
            slot = team * TRACKS_PER_TEAM + role + bank * ROLES
            players[np.arange(n_frames), slot] = pos[:, role]

    drop = rng.random((n_frames, N_TRACKS)) < PLAYER_DROPOUT
    players[drop] = np.nan
    return players


def _speeds(players, rng):
    step = np.diff(players, axis=0, prepend=players[:1])
    speeds = np.hypot(step[..., 0], step[..., 1]) * FPS
    speeds[rng.random(speeds.shape) < SPEED_DROPOUT] = np.nan
    return speeds


def make_tracking(n_frames=N_FRAMES, blocks=BLOCKS, seed=SEED):
    """(tracking, truth). See the module docstring. Same seed, same bytes."""
    rng = np.random.default_rng(seed)
    truth_ball = _ball_path(n_frames, rng)
    players = _players(truth_ball, rng)
    speeds = _speeds(players, rng)

    ball_raw = truth_ball + rng.normal(0.0, BALL_NOISE_M, size=truth_ball.shape)
    gaps = [(start, start + length) for start, length in GAPS + EDGE_GAPS
            if start + length <= n_frames]
    for start, end in gaps:
        ball_raw[start:end] = np.nan
    # Halftime hole: no ball at all between the blocks.
    halftime = (blocks[0][1], blocks[1][0]) if len(blocks) > 1 else None
    if halftime:
        ball_raw[halftime[0]:halftime[1]] = np.nan
    spikes = [f for f in SPIKE_FRAMES if f < n_frames]
    for frame in spikes:
        ball_raw[frame, 0] += SPIKE_JUMP_M
    far = [FAR_FRAME] if FAR_FRAME < n_frames else []
    for frame in far:
        ball_raw[frame, 0] = FAR_X_M

    length, width = PITCH
    outside = ((truth_ball[:, 0] < 0) | (truth_ball[:, 0] > length)
               | (truth_ball[:, 1] < 0) | (truth_ball[:, 1] > width))

    team_of_track = np.array([0] * TRACKS_PER_TEAM + [1] * TRACKS_PER_TEAM, dtype=np.int8)
    tracking = {
        "players": players.astype(np.float32),
        "speeds": speeds.astype(np.float32),
        "ball_raw": ball_raw.astype(np.float32),
        "team_of_track": team_of_track,
    }
    truth = {
        "ball_true": truth_ball.astype(np.float32),
        "blocks": [tuple(b) for b in blocks],
        "gaps": [g for g in gaps if not (halftime and halftime[0] <= g[0] < halftime[1])],
        "edge_gaps": [(s, s + n) for s, n in EDGE_GAPS],
        "halftime": halftime,
        "spikes": spikes,
        "far": far,
        "offpitch": np.flatnonzero(outside),
    }
    return tracking, truth


TAG_COLUMNS = ("code", "team", "half", "type", "side", "direction", "max_players_box",
               "start_s", "end_s", "start_frame", "end_frame")


def make_tags(n_frames=N_FRAMES, fps=FPS):
    """A small tag table with the columns and lower-case text of load_data.load_tags().

    Planted cases, by start frame (half-open windows):
      480-605  kick off (start), barcelona   overlaps the throw-in below
      600-725  throw-in, n/a                 no team, must stay unknown
      1000-1125 corner kick, barcelona
      1200-1450 build up, barcelona
      1250-1350 ball in final third, barcelona   agrees with build up, no conflict
      1400-1600 progression, atletic club        conflicts with build up on 1400-1450
      2000-2200 defending in middle third, atletic club
      2100-2150 goals, barcelona             not a phase tag
      3000-3100 players in the box, n/a      phase tag without a team
      n-50..n+100 long ball, atletic club    runs past the end of the array
      n-30..n+40 goal kick, atletic club     runs past the end of the array
    """
    plan = [
        ("set pieces", "barcelona", "kick off (start)", 480, 605),
        ("set pieces", "n/a", "throw-in", 600, 725),
        ("set pieces", "barcelona", "corner kick", 1000, 1125),
        ("build up", "barcelona", None, 1200, 1450),
        ("ball in final third", "barcelona", None, 1250, 1350),
        ("progression", "atletic club", None, 1400, 1600),
        ("defending in middle third", "atletic club", None, 2000, 2200),
        ("goals", "barcelona", None, 2100, 2150),
        ("players in the box", "n/a", None, 3000, 3100),
        ("long ball", "atletic club", None, n_frames - 50, n_frames + 100),
        ("set pieces", "atletic club", "goal kick", n_frames - 30, n_frames + 40),
    ]
    half_split = BLOCKS[1][0] if len(BLOCKS) > 1 else n_frames
    rows = []
    for code, team, kind, start, end in plan:
        rows.append({
            "code": code, "team": team,
            "half": "1st half" if start < half_split else "2nd half",
            "type": kind, "side": None, "direction": None, "max_players_box": None,
            "start_s": start / fps, "end_s": end / fps,
            "start_frame": int(start), "end_frame": int(end),
        })
    return pd.DataFrame(rows, columns=list(TAG_COLUMNS))
