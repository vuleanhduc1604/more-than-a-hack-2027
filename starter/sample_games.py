"""Labelled practice matches: Metrica's three public sample games, in our array format.

We may not tune anything on our own match (RULES.md), so every setting has to be chosen on other
matches where the answer is known. Metrica publishes three anonymised games with tracking AND
event data (https://github.com/metrica-sports/sample-data). The events say when the ball went out,
when a foul was given and when play restarted, which is enough to label every frame H, A or D.

    import sample_games
    game = sample_games.load(1)      # same keys as load_data.load_tracking(), plus:
    game["labels"]                   # (n_frames,) int8, 0 H, 1 A, 2 D, 3 X (the scorer's codes)
    game["blocks"]                   # half-open graded blocks, one per period
    game["restarts"]                 # DataFrame: frame, kind, dead_from -- one row per restart

    degraded = sample_games.degrade(game, seed=0)   # ball hidden like in our match

Get the data once (public, about 230 MB, gitignored like the rest of data/):

    git clone --depth 1 https://github.com/metrica-sports/sample-data data/metrica-sample-data

How the labels are derived (approximate, and the same rule for all three games):

  * A restart is a SET PIECE event (kick-off, throw-in, free kick, corner, penalty) or a goal kick
    (a PASS or BALL LOST with a GOAL KICK subtype; Metrica does not file those as set pieces).
  * The ball is dead from the end of the last event before the restart (the ball crossing the
    line, the foul, the goal) up to the restart frame. CARD events are ignored, because they
    happen during the stoppage. Before a period's kick-off the ball is dead too.
  * Live frames take the team of the last on-ball event (pass, carry, recovery, ball lost, shot,
    ball out, set piece). Challenges and fouls name both teams, so they are skipped.
  * A period is graded from its first tracked frame to the end of its last event.

The challenge's ground truth comes from corrected tracking, not events, so its stoppages may start
and end a few frames differently. These labels are for comparing versions, not for absolute claims.

Arrays follow load_data: frame f is row f (event and tracking frames count from 1), positions in
metres, NaN where missing. The sample data has no speeds, so they are computed from positions.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import DATA_DIR  # noqa: E402
import load_data  # noqa: E402

# Params
SAMPLE_DIR = os.path.join(DATA_DIR, "metrica-sample-data", "data")
CACHE_DIR = load_data.CACHE_DIR
VERSION = "sample-games-v1"
GAME_NUMBERS = (1, 2, 3)
FPS = 25
PITCH = (105.0, 68.0)          # all three sample games, per the sample-data README
SPEED_HALF_WINDOW = 2          # speed from positions +/- 2 frames (0.16 s), damps tracker jitter
TARGET_BALL_MISSING = 0.41     # our match hides the ball on 41% of graded frames; used for live ones
ARRAYS = ("players", "speeds", "ball_raw", "team_of_track", "labels", "blocks")

H, A, D, X = 0, 1, 2, 3        # the scorer's label codes (evaluation/possession_metrics.py)
ON_BALL_TYPES = ("PASS", "CARRY", "RECOVERY", "BALL LOST", "SHOT", "BALL OUT", "SET PIECE")


# ---------------------------------------------------------------------------------------------
# Reading the raw files
# ---------------------------------------------------------------------------------------------

def _paths(number):
    folder = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(number))
    stem = os.path.join(folder, "Sample_Game_{}".format(number))
    if number == 3:
        paths = {"tracking": stem + "_tracking.txt", "meta": stem + "_metadata.xml",
                 "events": stem + "_events.json"}
    else:
        paths = {"home": stem + "_RawTrackingData_Home_Team.csv",
                 "away": stem + "_RawTrackingData_Away_Team.csv",
                 "events": stem + "_RawEventsData.csv"}
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Metrica sample game {} not found ({}). Get it with:\n  git clone --depth 1 "
            "https://github.com/metrica-sports/sample-data data/metrica-sample-data"
            .format(number, missing[0]))
    return paths


def _read_csv_tracking(home_path, away_path):
    """Games 1 and 2: one CSV per team, three header rows, then period, frame, time, x/y pairs."""
    teams = []
    for path in (home_path, away_path):
        table = pd.read_csv(path, skiprows=3, header=None)
        frames = table[1].to_numpy() - 1
        xy = table.iloc[:, 3:].to_numpy(dtype=float)
        teams.append((frames, xy[:, :-2].reshape(len(table), -1, 2), xy[:, -2:]))
    (frames, home, ball), (away_frames, away, _) = teams
    if not np.array_equal(frames, away_frames) or not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("home and away tracking are not frame-aligned")
    players = np.concatenate([home, away], axis=1)
    team_of_track = np.r_[np.zeros(home.shape[1]), np.ones(away.shape[1])].astype(np.int8)
    return players, ball, team_of_track


def _read_epts_tracking(raw_path, meta_path):
    """Game 3: the FIFA EPTS text format, '<frame>:<x,y;x,y;...>:<ball x,y>'.

    The metadata has one format specification per frame range: substitutions change which players
    a line lists, and in what order. Each player gets one slot for the whole match.
    """
    root = ET.parse(meta_path).getroot()
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    home_id = _first_team_id(meta_path)
    channel_player = {ch.get("id"): ch.get("playerId") for ch in root.iter("PlayerChannel")}
    specs, slot_of = [], {}
    for spec in root.iter("DataFormatSpecification"):
        refs = [ref.get("playerChannelId") for ref in spec.iter("PlayerChannelRef")]
        order = [channel_player[name] for name in refs[::2]]   # x and y channels come in pairs
        for player in order:
            slot_of.setdefault(player, len(slot_of))
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      [slot_of[player] for player in order]))
    team_of_track = np.array([0 if team_of_player[p] == home_id else 1 for p in slot_of],
                             dtype=np.int8)

    lines = [line for line in open(raw_path).read().split("\n") if line]
    players = np.full((len(lines), len(slot_of), 2), np.nan)
    ball = np.full((len(lines), 2), np.nan)
    for row, line in enumerate(lines):
        frame, people, ball_text = line.split(":")
        frame = int(frame)
        if frame - 1 != row:
            raise ValueError("game 3 tracking is not one row per frame")
        slots = next(s for lo, hi, s in specs if lo <= frame <= hi)
        players[row, slots] = np.array(people.replace(";", ",").split(","),
                                       dtype=float).reshape(-1, 2)
        ball[row] = np.array(ball_text.split(",")[:2], dtype=float)
    return players, ball, team_of_track


def _read_events(number, paths):
    """One tidy event table: team (0 home / 1 away), type, subtype, period, start, end (rows)."""
    if number == 3:
        raw = json.load(open(paths["events"]))["data"]
        home_id = _first_team_id(paths["meta"])
        rows = []
        for event in raw:
            subtypes = event.get("subtypes") or []
            if isinstance(subtypes, dict):
                subtypes = [subtypes]
            rows.append({"team": 0 if event["team"]["id"] == home_id else 1,
                         "type": event["type"]["name"].upper(),
                         "subtype": "-".join(s["name"] for s in subtypes).upper(),
                         "period": event["period"],
                         "start": event["start"]["frame"], "end": event["end"]["frame"]})
        events = pd.DataFrame(rows)
    else:
        table = pd.read_csv(paths["events"])
        events = pd.DataFrame({
            "team": (table["Team"].str.lower() == "away").astype(int),
            "type": table["Type"].str.upper(),
            "subtype": table["Subtype"].fillna("").str.upper(),
            "period": table["Period"],
            "start": table["Start Frame"], "end": table["End Frame"]})
    events["start"] = events["start"].astype(int) - 1
    # Some events (set pieces, cards) carry no end frame; they end where they start.
    events["end"] = np.maximum(events["end"].fillna(0).astype(int) - 1, events["start"])
    return events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)


def _first_team_id(meta_path):
    """The first team in the EPTS metadata, which we call home."""
    return next(ET.parse(meta_path).getroot().iter("Team")).get("id")


# ---------------------------------------------------------------------------------------------
# Labels from events
# ---------------------------------------------------------------------------------------------

def restart_kind(event_type, subtype):
    """The kind of restart an event is, or None. Goal kicks are passes in Metrica's events."""
    if event_type == "SET PIECE":
        for kind in ("KICK OFF", "THROW IN", "FREE KICK", "CORNER KICK", "PENALTY"):
            if kind in subtype:
                return kind.lower()
        return "set piece"
    if event_type in ("PASS", "BALL LOST") and subtype.startswith("GOAL KICK"):
        return "goal kick"
    return None


def derive_labels(events, n_frames):
    """(labels, blocks, restarts) from an event table. See the module docstring for the rules."""
    labels = np.full(n_frames, X, dtype=np.int8)
    blocks = []
    for period, group in events.groupby("period"):
        lo = 0 if not blocks else blocks[-1][1]
        # The period's first tracked frame: after the previous block, at most a minute before
        # kick-off (the sample games keep a little pre-kick-off tracking, and halftime between).
        lo = max(lo, int(group["start"].min()) - 60 * FPS)
        hi = min(n_frames, int(group["end"].max()) + 1)
        blocks.append((lo, hi))

    live = np.zeros(n_frames, dtype=bool)
    for lo, hi in blocks:
        live[lo:hi] = True

    # Team of the last on-ball event, carried forward.
    team = np.full(n_frames, -1, dtype=np.int8)
    on_ball = events[events["type"].isin(ON_BALL_TYPES)]
    team[on_ball["start"].to_numpy()] = on_ball["team"].to_numpy()
    team = pd.Series(team).replace(-1, np.nan).ffill().bfill().to_numpy()
    labels[live] = team[live].astype(np.int8)

    restarts = []
    ignorable = events["type"] == "CARD"
    for index, event in events.iterrows():
        kind = restart_kind(event["type"], event["subtype"])
        if kind is None:
            continue
        block = next(b for b in blocks if b[0] <= event["start"] < b[1])
        before = events[(events["start"] < event["start"]) & ~ignorable
                        & (events["period"] == event["period"])]
        dead_from = block[0] if before.empty else int(before["end"].max()) + 1
        dead_from = min(max(dead_from, block[0]), event["start"])
        labels[dead_from:event["start"]] = D
        restarts.append({"frame": int(event["start"]), "kind": kind, "dead_from": dead_from})
    restarts = pd.DataFrame(restarts).drop_duplicates("frame").reset_index(drop=True)
    return labels, blocks, restarts


# ---------------------------------------------------------------------------------------------
# Speeds, caching, degrading
# ---------------------------------------------------------------------------------------------

def speeds_from_positions(players, half_window=SPEED_HALF_WINDOW, fps=FPS):
    """(n_frames, n_tracks) m/s from a centred difference. NaN where either end is missing."""
    k = half_window
    speeds = np.full(players.shape[:2], np.nan)
    step = players[2 * k:] - players[:-2 * k]
    speeds[k:-k] = np.linalg.norm(step, axis=2) * fps / (2 * k)
    return speeds


def parse(number):
    """Read one sample game from the raw files. No cache."""
    paths = _paths(number)
    if number == 3:
        players, ball, team_of_track = _read_epts_tracking(paths["tracking"], paths["meta"])
    else:
        players, ball, team_of_track = _read_csv_tracking(paths["home"], paths["away"])
    scale = np.array(PITCH)
    players, ball = players * scale, ball * scale
    events = _read_events(number, paths)
    labels, blocks, restarts = derive_labels(events, len(players))
    return {"players": players.astype(np.float32),
            "speeds": speeds_from_positions(players).astype(np.float32),
            "ball_raw": ball.astype(np.float32),
            "team_of_track": team_of_track,
            "labels": labels,
            "blocks": np.array(blocks, dtype=np.int64),
            "restarts": restarts}


def load(number, refresh=False):
    """One sample game as a dict, from cache/ where possible. See the module docstring."""
    path = os.path.join(CACHE_DIR, "sample_game_{}.npz".format(number))
    if not refresh and os.path.isfile(path):
        cached = np.load(path, allow_pickle=False)
        if str(cached["version"]) == VERSION:
            game = {name: cached[name] for name in ARRAYS}
            game["restarts"] = pd.DataFrame({"frame": cached["restart_frame"],
                                             "kind": cached["restart_kind"],
                                             "dead_from": cached["restart_dead_from"]})
            return game
    game = parse(number)
    os.makedirs(CACHE_DIR, exist_ok=True)
    restarts = game["restarts"]
    np.savez_compressed(path, version=VERSION, restart_frame=restarts["frame"].to_numpy(),
                        restart_kind=restarts["kind"].to_numpy(dtype=str),
                        restart_dead_from=restarts["dead_from"].to_numpy(),
                        **{name: game[name] for name in ARRAYS})
    return game


def degrade(game, seed=0, target_missing=TARGET_BALL_MISSING, gap_lengths=None):
    """A copy of a sample game with the ball hidden as often as in our match.

    The sample ball is corrected data: it is present on nearly every live frame and removed on
    nearly every dead one. Our ball is missing on 41% of graded frames, live ones included. So
    extra gaps are cut at random places, with lengths drawn from gap_lengths (frames; by default
    our match's own ball gaps, which are mostly long), until target_missing of the LIVE frames
    have no ball. The labels only count frames: gaps are placed blind, so a missing ball carries
    no hint about the game state. Player tracks are left alone: the dead-ball signals are
    team-level and need no identities.

    What this cannot fix: in our match the ball is still tracked on about two thirds of dead
    frames (carried back to the restart spot), and in the sample games it is not. Any signal that
    reads "ball present" as "ball in play" would look better here than it is on our match.
    """
    rng = np.random.default_rng(seed)
    if gap_lengths is None:
        gap_lengths = our_ball_gap_lengths()
    ball = game["ball_raw"].copy()
    live = game["labels"] < D
    starts = np.flatnonzero(game["labels"] != X)
    while np.isnan(ball[live, 0]).mean() < target_missing:
        start = rng.choice(starts)
        ball[start:start + int(rng.choice(gap_lengths))] = np.nan
    out = dict(game)
    out["ball_raw"] = ball
    return out


def our_ball_gap_lengths():
    """Lengths, in frames, of every ball gap inside the graded blocks of our match."""
    import ball as ball_module
    ball_raw = load_data.load_tracking()["ball_raw"]
    return np.array([hi - lo for lo, hi in ball_module.find_gaps(ball_raw,
                                                                 ball_module.scored_blocks())])


def summary_lines(number, game):
    """Aggregate description of one sample game: safe to print or share."""
    labels, blocks = game["labels"], game["blocks"]
    graded = labels != X
    restarts = game["restarts"]
    lines = ["sample game {}: {} frames, {} graded, {} periods".format(
        number, len(labels), int(graded.sum()), len(blocks))]
    for code, name in ((H, "H"), (A, "A"), (D, "D")):
        lines.append("  {}  {:5.1%} of graded frames".format(name, (labels == code).sum()
                                                            / graded.sum()))
    lines.append("  ball missing on {:.1%} of graded frames".format(
        np.isnan(game["ball_raw"][graded, 0]).mean()))
    lines.append("  {} restarts: {}".format(len(restarts), ", ".join(
        "{} {}".format(n, kind) for kind, n in restarts["kind"].value_counts().items())))
    return lines


if __name__ == "__main__":
    for number in GAME_NUMBERS:
        print("\n".join(summary_lines(number, load(number, refresh="--refresh" in sys.argv))))
