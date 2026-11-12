"""Section 2, Path B: Smart Tagging turned into per-frame hints.

Tags are time windows; later sections work frame by frame. This module writes two arrays of one
value per tactical video frame. Both are hints, not labels: the tags are automatic, they have false
positives and misses, and their windows (about 5 s) open before the event they describe.

    import tagging
    hints = tagging.load()  # dict: restart (n,) float32, st_team (n,) int8, restart_team (n,) int8

v1:

  restart       1.0 over [start_frame, end_frame) of every "set pieces" tag (all six types), 0
                elsewhere. The window opens early, so do not read its first frame as the restart.
                Still the v0 box; a soft bump that peaks near the restart is planned.
  st_team       0 home, 1 away, -1 unknown: the team the phase tags say HAS the ball. See polarity.
  restart_team  0 home, 1 away, -1 unknown over each set-piece window: the team taking the restart.
                See throw-ins.

Polarity (plan step 7). A phase tag names a team, but not always the team with the ball: some
codes describe the side defending. We do not keep a list of which codes those are. For every code,
each tag window gets the share of its observed-ball frames on which the nearest player belongs to
the tagged team. Over the code's tags, the mean share is tested against 0.5 (a t statistic over
tags, which are roughly independent where frames are not). Well above 0.5 is polarity +1 (the tag
names the team in possession), well below is -1 (it names the other team, so we flip it), and in
between is 0 (the code says nothing reliable about possession and is dropped). Nearest player is a
crude proxy for possession, but over dozens of tags its bias cancels; only the sign is used.
Then every frame takes a majority vote of the tags covering it; a tie (including none) is -1.

Throw-ins (plan step 8). Every throw-in is tagged "n/a". The thrower stands just beyond the
touchline, so on each frame of the window a team gets a vote when one of its players is between
THROW_OUT_MIN_M and THROW_OUT_MAX_M beyond a touchline and no opponent is, and another when that
player is within THROW_BALL_NEAR_M of an observed ball. The min distance absorbs tracking jitter
on the line; the max keeps out substitutes and staff further back. A team needs TAKER_MIN_VOTES
votes and TAKER_MIN_SHARE of them, else the throw-in stays -1. Set pieces that carry a team keep
it: the report checks that the tagged team of a corner is the one with a player at the flag.
Kick-offs are "n/a" too and stay -1 (both teams stand at the centre).

Tag team strings are mapped to home and away by fuzzy matching against the club names in the ATD
metadata, so no club name is written in this file and "atletic club" still finds its club.

    python starter/tagging.py            # build or load the cache, print shapes
    python starter/tagging.py --report   # aggregate tag report and time-base check, safe to share
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_names  # noqa: E402
import ball  # noqa: E402
import load_data  # noqa: E402
import load_smart_tagging  # noqa: E402

# Params
VERSION = "tagging-v1"
PITCH = GAMES[GAME]["pitch"]

SET_PIECE_CODE = "set pieces"
NON_PHASE_CODES = (SET_PIECE_CODE, "goals")
THROW_IN_TYPE = "throw-in"
CORNER_TYPE = "corner kick"

POLARITY_MIN_FRAMES = 25   # observed-ball frames a tag needs to count in the polarity test (1 s)
POLARITY_MIN_TAGS = 5      # fewer usable tags than this and a code is dropped
POLARITY_T = 2.0           # |t| needed to trust the sign of a code

THROW_OUT_MIN_M = 0.3      # beyond the touchline by at least this (tracking jitter on the line)
THROW_OUT_MAX_M = 3.0      # and at most this (substitutes and staff stand further back)
THROW_BALL_NEAR_M = 5.0    # a thrower near an observed ball gets an extra vote
TAKER_MIN_VOTES = 10
TAKER_MIN_SHARE = 0.8
CORNER_FLAG_M = 3.0        # report only: a player this close to a corner flag is the taker

NO_TEAM = ("n/a", "na", "none", "")
# Words that say "this is a club" rather than which club, dropped before matching names.
GENERIC_TOKENS = ("fc", "cf", "club", "cd", "sd", "ud", "ac", "sc", "afc", "de", "the")
TEAM_MATCH_MIN = 0.60      # best similarity needed to accept a match
TEAM_MATCH_MARGIN = 0.15   # and by how much it must beat the other club

HOME, AWAY, UNKNOWN = 0, 1, -1

CACHE_DIR = load_data.CACHE_DIR
SIGNATURE_FILE = "tagging_signature.json"
ARRAYS = ("restart", "st_team", "restart_team")

PARAMS = {
    "SET_PIECE_CODE": SET_PIECE_CODE, "NON_PHASE_CODES": list(NON_PHASE_CODES),
    "NO_TEAM": list(NO_TEAM), "GENERIC_TOKENS": list(GENERIC_TOKENS),
    "TEAM_MATCH_MIN": TEAM_MATCH_MIN, "TEAM_MATCH_MARGIN": TEAM_MATCH_MARGIN,
    "THROW_IN_TYPE": THROW_IN_TYPE, "POLARITY_MIN_FRAMES": POLARITY_MIN_FRAMES,
    "POLARITY_MIN_TAGS": POLARITY_MIN_TAGS, "POLARITY_T": POLARITY_T,
    "THROW_OUT_MIN_M": THROW_OUT_MIN_M, "THROW_OUT_MAX_M": THROW_OUT_MAX_M,
    "THROW_BALL_NEAR_M": THROW_BALL_NEAR_M, "TAKER_MIN_VOTES": TAKER_MIN_VOTES,
    "TAKER_MIN_SHARE": TAKER_MIN_SHARE, "PITCH": list(PITCH),
}


# ---------------------------------------------------------------------------------------------
# Team mapping
# ---------------------------------------------------------------------------------------------

def _tokens(name):
    words = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower()).split()
    core = [w for w in words if w not in GENERIC_TOKENS]
    return core or words


def name_similarity(a, b):
    """0..1. The better of whole-name similarity and the mean best-token similarity.

    The token score lets "barcelona" match "fc barcelona" and "atletic" match "athletic bilbao"
    without either name being written down anywhere.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    whole = difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    per_token = np.mean([max(difflib.SequenceMatcher(None, x, y).ratio() for y in tb)
                         for x in ta])
    return float(max(whole, per_token))


def map_teams(tag_team_values, club_names):
    """{tag team string: (code, best score)} for (home, away) club names.

    "n/a", None and empty give -1. A string that matches neither club well, or both about
    equally, gives -1 too. Two different strings that land on the same side raise, because then
    the tags would have only one team and every hint would be wrong.
    """
    out = {}
    for value in sorted({v for v in tag_team_values if v is not None}):
        text = str(value).strip().lower()
        if text in NO_TEAM:
            out[value] = (UNKNOWN, 0.0)
            continue
        scores = [name_similarity(text, club) for club in club_names]
        best = int(np.argmax(scores))
        ordered = sorted(scores, reverse=True)
        ok = ordered[0] >= TEAM_MATCH_MIN and ordered[0] - ordered[1] >= TEAM_MATCH_MARGIN
        out[value] = (best if ok else UNKNOWN, round(ordered[0], 3))

    sides = {}
    for value, (code, _) in out.items():
        if code != UNKNOWN:
            sides.setdefault(code, []).append(value)
    clash = {code: names for code, names in sides.items() if len(names) > 1}
    if clash:
        raise ValueError("tag team strings {} all map to the same side ({}) against clubs {}. "
                         "Check the ATD metadata and the tag export.".format(
                             sorted(next(iter(clash.values()))),
                             "home" if next(iter(clash)) == HOME else "away", list(club_names)))
    for value in tag_team_values:
        if value is None:
            out[None] = (UNKNOWN, 0.0)
    return out


def team_codes(tag_team_values, meta_path):
    """{tag team string: 0 home, 1 away, -1 unknown}, matched against the ATD club names."""
    clubs = team_names(meta_path)
    if len(clubs) != 2:
        raise ValueError("{}: expected 2 team names, found {}".format(meta_path, len(clubs)))
    return {value: code for value, (code, _) in map_teams(tag_team_values, clubs).items()}


# ---------------------------------------------------------------------------------------------
# Per-frame arrays
# ---------------------------------------------------------------------------------------------

def _clipped(tags, n_frames):
    """(start, end) arrays clipped to [0, n_frames); rows that end up empty have start == end."""
    start = np.clip(tags["start_frame"].to_numpy(dtype=np.int64), 0, n_frames)
    end = np.clip(tags["end_frame"].to_numpy(dtype=np.int64), 0, n_frames)
    return start, np.maximum(start, end)


def _coverage(start, end, n_frames):
    """How many of the half-open windows cover each frame. Vectorised with a difference array."""
    diff = np.zeros(n_frames + 1, dtype=np.int64)
    np.add.at(diff, start, 1)
    np.add.at(diff, end, -1)
    return np.cumsum(diff[:-1])


def build_restart(tags, n_frames):
    """float32 (n_frames,): 1.0 inside any set-piece window, 0 elsewhere. v0 box."""
    sel = (tags["code"] == SET_PIECE_CODE).to_numpy()
    start, end = _clipped(tags[sel], n_frames)
    return (_coverage(start, end, n_frames) > 0).astype(np.float32)


def _tag_teams(tags, home_away):
    return np.array([home_away.get(v, UNKNOWN) for v in tags["team"]], dtype=np.int64)


def nearest_team(tracking, blocks):
    """int8 per frame: team of the player nearest the observed ball, -1 where there is no ball.

    Uses the ball after ball.py's outlier removal, never the filled ball: a filled position in a
    long gap is close to a guess and would blur the polarity test.
    """
    clean, _ = ball.remove_outliers(tracking["ball_raw"], blocks)
    players = tracking["players"]
    out = np.full(len(clean), UNKNOWN, dtype=np.int8)
    team = np.asarray(tracking["team_of_track"])
    chunk = 20000   # bounds memory: one (chunk, n_tracks) distance block at a time
    for lo in range(0, len(clean), chunk):
        hi = min(len(clean), lo + chunk)
        diff = players[lo:hi].astype(np.float64) - clean[lo:hi, None, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        dist = np.where(np.isnan(dist), np.inf, dist)
        best = dist.argmin(axis=1)
        found = np.isfinite(dist.min(axis=1))
        out[lo:hi][found] = team[best[found]]
    return out


def code_polarity(tags, home_away, near_team):
    """{code: dict(polarity, n_tags, mean_share, t)} for every phase code. See the docstring."""
    phase = tags[~tags["code"].isin(NON_PHASE_CODES).to_numpy()]
    team = _tag_teams(phase, home_away)
    start, end = _clipped(phase, len(near_team))
    shares = {}
    for code, side, lo, hi in zip(phase["code"], team, start, end):
        if side == UNKNOWN:
            continue
        near = near_team[lo:hi]
        seen = near >= 0
        if seen.sum() >= POLARITY_MIN_FRAMES:
            shares.setdefault(code, []).append(float((near[seen] == side).mean()))
    out = {}
    for code in sorted(set(phase["code"])):
        values = np.array(shares.get(code, []))
        n = len(values)
        mean = float(values.mean()) if n else float("nan")
        sd = float(values.std(ddof=1)) if n > 1 else float("nan")
        if n < 2:
            t = float("nan")
        elif sd > 0:
            t = (mean - 0.5) / (sd / np.sqrt(n))
        else:   # every tag agrees exactly: the sign is as clear as it gets
            t = float(np.sign(mean - 0.5) * np.inf) if mean != 0.5 else 0.0
        polarity = 0
        if n >= POLARITY_MIN_TAGS and not np.isnan(t) and abs(t) >= POLARITY_T:
            polarity = 1 if t > 0 else -1
        out[code] = {"polarity": polarity, "n_tags": n, "mean_share": mean, "t": float(t)}
    return out


def team_votes(tags, n_frames, home_away, polarity=None):
    """(home_count, away_count) per frame from phase tags, for the team IN POSSESSION.

    polarity is {code: +1, -1 or 0}; a code missing from it counts as +1, so polarity=None is the
    v0 rule (the tagged team at face value).
    """
    phase = tags[~tags["code"].isin(NON_PHASE_CODES).to_numpy()]
    team = _tag_teams(phase, home_away)
    sign = np.array([(polarity or {}).get(c, 1) for c in phase["code"]], dtype=np.int64)
    team = np.where((sign == -1) & (team != UNKNOWN), 1 - team, team)
    team = np.where(sign == 0, UNKNOWN, team)
    start, end = _clipped(phase, n_frames)
    home = _coverage(start[team == HOME], end[team == HOME], n_frames)
    away = _coverage(start[team == AWAY], end[team == AWAY], n_frames)
    return home, away


def build_st_team(tags, n_frames, home_away, polarity=None):
    """int8 (n_frames,): majority team in possession by the phase tags, -1 on a tie or none."""
    home, away = team_votes(tags, n_frames, home_away, polarity)
    st_team = np.full(n_frames, UNKNOWN, dtype=np.int8)
    st_team[home > away] = HOME
    st_team[away > home] = AWAY
    return st_team


def throw_in_votes(players, team_of_track, ball_xy, pitch=PITCH):
    """(home_votes, away_votes) over one throw-in window. See the docstring.

    players (n, tracks, 2) and ball_xy (n, 2) cover the window only; ball_xy is NaN where unseen.
    """
    length, width = pitch
    x, y = players[..., 0].astype(np.float64), players[..., 1].astype(np.float64)
    with np.errstate(invalid="ignore"):
        beyond = (((y < -THROW_OUT_MIN_M) & (y > -THROW_OUT_MAX_M))
                  | ((y > width + THROW_OUT_MIN_M) & (y < width + THROW_OUT_MAX_M)))
        beyond &= (x > 0) & (x < length)
        dist = np.hypot(x - ball_xy[:, None, 0], y - ball_xy[:, None, 1])
        with_ball = beyond & (dist < THROW_BALL_NEAR_M)
    team = np.asarray(team_of_track)
    votes = []
    for side in (HOME, AWAY):
        mine = beyond[:, team == side].any(axis=1)
        theirs = beyond[:, team != side].any(axis=1)
        votes.append(int((mine & ~theirs).sum()) + int(with_ball[:, team == side].any(axis=1).sum()))
    return votes[0], votes[1]


def decide_taker(home_votes, away_votes):
    total = home_votes + away_votes
    if total < TAKER_MIN_VOTES:
        return UNKNOWN
    if home_votes >= TAKER_MIN_SHARE * total:
        return HOME
    if away_votes >= TAKER_MIN_SHARE * total:
        return AWAY
    return UNKNOWN


def set_piece_teams(tags, home_away, tracking=None, ball_clean=None):
    """(row positions of set pieces, team per set piece, source per set piece).

    source is "tag" (the tag names a team), "inferred" (throw-in decided from tracking) or
    "unknown". Without tracking, throw-ins stay unknown.
    """
    rows = np.flatnonzero((tags["code"] == SET_PIECE_CODE).to_numpy())
    sp = tags.iloc[rows]
    team = _tag_teams(sp, home_away)
    source = np.where(team != UNKNOWN, "tag", "unknown").astype(object)
    if tracking is not None:
        n_frames = len(tracking["players"])
        start, end = _clipped(sp, n_frames)
        for i, kind in enumerate(sp["type"]):
            if team[i] != UNKNOWN or kind != THROW_IN_TYPE or end[i] <= start[i]:
                continue
            lo, hi = start[i], end[i]
            votes = throw_in_votes(tracking["players"][lo:hi], tracking["team_of_track"],
                                   ball_clean[lo:hi])
            team[i] = decide_taker(*votes)
            if team[i] != UNKNOWN:
                source[i] = "inferred"
    return rows, team, source


def build_restart_team(tags, n_frames, set_piece_team):
    """int8 (n_frames,): team taking the restart over each set-piece window, -1 elsewhere and
    where windows of both teams overlap. set_piece_team is the team column of set_piece_teams()."""
    sp = tags[(tags["code"] == SET_PIECE_CODE).to_numpy()]
    start, end = _clipped(sp, n_frames)
    team = np.asarray(set_piece_team)
    home = _coverage(start[team == HOME], end[team == HOME], n_frames)
    away = _coverage(start[team == AWAY], end[team == AWAY], n_frames)
    out = np.full(n_frames, UNKNOWN, dtype=np.int8)
    out[(home > 0) & (away == 0)] = HOME
    out[(away > 0) & (home == 0)] = AWAY
    return out


def compute(tags, n_frames, home_away, tracking=None, blocks=None):
    """All arrays plus the evidence behind them. Without tracking this is the v0 tag-only rule."""
    polarity, ball_clean = None, None
    stats = {}
    if tracking is not None:
        blocks = ball.scored_blocks() if blocks is None else blocks
        near = nearest_team(tracking, blocks)
        stats = code_polarity(tags, home_away, near)
        polarity = {code: s["polarity"] for code, s in stats.items()}
        ball_clean, _ = ball.remove_outliers(tracking["ball_raw"], blocks)
    rows, sp_team, sp_source = set_piece_teams(tags, home_away, tracking, ball_clean)
    out = {"restart": build_restart(tags, n_frames),
           "st_team": build_st_team(tags, n_frames, home_away, polarity),
           "restart_team": build_restart_team(tags, n_frames, sp_team)}
    evidence = {"polarity": stats, "set_piece_rows": rows, "set_piece_team": sp_team,
                "set_piece_source": sp_source, "ball_clean": ball_clean}
    return out, evidence


# ---------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------

def params_hash():
    text = json.dumps(PARAMS, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def source_signature(game=GAME, data_dir=None):
    """The tag file and the ATD metadata (club names come from it), by size and mtime."""
    paths = require(["smart_tagging", "atd_meta"], game, data_dir)
    return {"game": game,
            "files": [[os.path.basename(p), os.path.getsize(p), os.path.getmtime(p)]
                      for p in paths]}


def cache_signature(n_frames=None):
    return {"tags": source_signature(), "tracking": load_data.source_signature(),
            "version": VERSION, "ball_version": ball.VERSION, "ball_params": ball.params_hash(),
            "params": params_hash(), "n_frames": n_frames or source_frames("tactical")}


def _normalise(signature):
    return json.loads(json.dumps(signature, sort_keys=True))


def _cache_is_current(signature, cache_dir):
    path = os.path.join(cache_dir, SIGNATURE_FILE)
    if not os.path.isfile(path):
        return False
    if not all(os.path.isfile(os.path.join(cache_dir, name + ".npy")) for name in ARRAYS):
        return False
    with open(path) as handle:
        return json.load(handle) == _normalise(signature)


def build(refresh=False, cache_dir=CACHE_DIR, tags=None, home_away=None, n_frames=None,
          signature=None, tracking=None, blocks=None):
    """Compute the arrays and cache them, or return the cache if it is current.

    With no arguments this reads the match, tags and tracking. Tests pass tags, home_away,
    n_frames, cache_dir and signature (and tracking and blocks if they want the tracking evidence),
    so they never touch the match data. Tags passed without tracking give the tag-only rule.
    """
    n_frames = n_frames or source_frames("tactical")
    if signature is None:
        signature = cache_signature(n_frames)
    if not refresh and _cache_is_current(signature, cache_dir):
        return {name: np.load(os.path.join(cache_dir, name + ".npy")) for name in ARRAYS}

    if tags is None:
        tags = load_data.load_tags()
        if tracking is None:
            tracking = load_data.load_tracking()
    if home_away is None:
        home_away = team_codes(tags["team"].tolist(), require(["atd_meta"])[0])
    out, _ = compute(tags, n_frames, home_away, tracking, blocks)
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    for name in ARRAYS:
        np.save(os.path.join(cache_dir, name + ".npy"), out[name])
    # Written last, so an interrupted build never looks complete.
    with open(os.path.join(cache_dir, SIGNATURE_FILE), "w") as handle:
        json.dump(_normalise(signature), handle, sort_keys=True)
    return out


def load(refresh=False):
    """The one line for everyone: dict(restart, st_team, restart_team), rebuilt if stale."""
    return build(refresh=refresh)


# ---------------------------------------------------------------------------------------------
# Report. Aggregates only: counts, percentiles, shares. Nothing per frame.
# ---------------------------------------------------------------------------------------------

def _corner_check(tags, home_away, tracking, pitch=PITCH):
    """(n corners with a clear taker at a flag, n of them agreeing with the tagged team)."""
    length, width = pitch
    sp = tags[((tags["code"] == SET_PIECE_CODE) & (tags["type"] == CORNER_TYPE)).to_numpy()]
    start, end = _clipped(sp, len(tracking["players"]))
    team = _tag_teams(sp, home_away)
    side_of = np.asarray(tracking["team_of_track"])
    decided = agree = 0
    for lo, hi, tagged in zip(start, end, team):
        p = tracking["players"][lo:hi].astype(np.float64)
        flag = np.min([np.hypot(p[..., 0] - cx, p[..., 1] - cy)
                       for cx in (0.0, length) for cy in (0.0, width)], axis=0)
        near = np.nan_to_num(flag, nan=np.inf) < CORNER_FLAG_M
        taker = decide_taker(int(near[:, side_of == HOME].any(axis=1).sum()),
                             int(near[:, side_of == AWAY].any(axis=1).sum()))
        if taker != UNKNOWN and tagged != UNKNOWN:
            decided += 1
            agree += int(taker == tagged)
    return decided, agree


def report_lines(tags, out, mapping, blocks, time_base=None, evidence=None, tracking=None):
    """The --report block as a list of lines. mapping is map_teams() output."""
    n_frames = len(out["restart"])
    scored = np.zeros(n_frames, dtype=bool)
    for lo, hi in blocks:
        scored[lo:hi] = True
    n_scored = int(scored.sum())
    pct = lambda part, whole: 100.0 * part / whole if whole else float("nan")  # noqa: E731
    label = lambda v: "None" if v is None else v  # noqa: E731

    lines = ["=== tagging.py --report ({}) ===".format(VERSION)]
    for name in ARRAYS:
        lines.append("{:<8} shape {} dtype {}".format(name, out[name].shape, out[name].dtype))
    lines.append("tags {}".format(len(tags)))
    lines.append("team mapping (tag string -> side, similarity):")
    for value, (code, score) in sorted(mapping.items(), key=lambda kv: str(kv[0])):
        side = {HOME: "home", AWAY: "away"}.get(code, "unknown")
        lines.append("  {:<16} -> {:<7} {:.2f}".format(label(value), side, score))

    lines.append("by code:")
    for code, n in tags["code"].value_counts().sort_index().items():
        lines.append("  {:<32} {:>4}".format(code, n))
    lines.append("by team:")
    for team, n in tags["team"].fillna("None").value_counts().sort_index().items():
        lines.append("  {:<16} {:>4}".format(team, n))
    sp = tags[tags["code"] == SET_PIECE_CODE]
    lines.append("set pieces by type (team n/a count):")
    for kind, group in sp.groupby(sp["type"].fillna("None")):
        lines.append("  {:<24} {:>3}  n/a {:>3}".format(
            kind, len(group), int(group["team"].isin(NO_TEAM).sum())))

    lines.append("window length (s) by code: n  p10  p50  p90")
    length = (tags["end_s"] - tags["start_s"]).to_numpy(dtype=np.float64)
    for code in sorted(tags["code"].unique()):
        values = length[(tags["code"] == code).to_numpy()]
        q = np.percentile(values, [10, 50, 90])
        lines.append("  {:<32} {:>4} {:>5.1f} {:>5.1f} {:>5.1f}".format(code, len(values), *q))

    home_away = {v: c for v, (c, _) in mapping.items()}
    if evidence and evidence["polarity"]:
        lines.append("polarity by code (share of tag frames where the tagged team is nearest):")
        lines.append("  {:<32} {:>4} {:>6} {:>6} {:>4}".format("code", "tags", "share", "t", "pol"))
        for code, st in sorted(evidence["polarity"].items()):
            lines.append("  {:<32} {:>4} {:>6.2f} {:>6.1f} {:>+4d}".format(
                code, st["n_tags"], st["mean_share"], st["t"], st["polarity"]))
        polarity = {code: st["polarity"] for code, st in evidence["polarity"].items()}
    else:
        polarity = None
    st = out["st_team"][scored]
    lines.append("st_team on scored frames: home {:.1f}%  away {:.1f}%  unknown {:.1f}%".format(
        pct((st == HOME).sum(), n_scored), pct((st == AWAY).sum(), n_scored),
        pct((st == UNKNOWN).sum(), n_scored)))
    for name, pol in (("v0, face value", None), ("with polarity", polarity)):
        if name != "v0, face value" and pol is None:
            continue
        home, away = team_votes(tags, n_frames, home_away, pol)
        clash = (home > 0) & (away > 0)
        tie = clash & (home == away)
        lines.append("both teams voted ({}): {:.1f}% of scored frames, tied {:.1f}%".format(
            name, pct((clash & scored).sum(), n_scored), pct((tie & scored).sum(), n_scored)))
    if evidence is not None:
        sp_type = tags.iloc[evidence["set_piece_rows"]]["type"].fillna("None").to_numpy()
        src = evidence["set_piece_source"]
        lines.append("restart_team per set piece (tag / inferred / unknown):")
        for kind in sorted(set(sp_type)):
            sel = sp_type == kind
            lines.append("  {:<24} {:>3} {:>3} {:>3}".format(
                kind, int((src[sel] == "tag").sum()), int((src[sel] == "inferred").sum()),
                int((src[sel] == "unknown").sum())))
        inferred = src == "inferred"
        team = evidence["set_piece_team"]
        lines.append("  inferred throw-ins: home {}  away {}".format(
            int((team[inferred] == HOME).sum()), int((team[inferred] == AWAY).sum())))
    if tracking is not None:
        decided, agree = _corner_check(tags, home_away, tracking)
        lines.append("corner check (player at the flag vs tagged team): {}/{} agree".format(
            agree, decided))
    rt = out["restart_team"][scored]
    lines.append("restart_team on scored frames: home {:.1f}%  away {:.1f}%".format(
        pct((rt == HOME).sum(), n_scored), pct((rt == AWAY).sum(), n_scored)))
    lines.append("restart > 0 on scored frames: {:.2f}%".format(
        pct((out["restart"][scored] > 0).sum(), n_scored)))
    if time_base is not None:
        lines.append("time base check: {} kick off (start) tags, {}".format(
            time_base["n_kickoff_tags"], "OK" if time_base["agrees"] else "MISMATCH"))
    lines.append("=== end ===")
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="print the aggregate tag report")
    parser.add_argument("--refresh", action="store_true", help="rebuild the cache")
    args = parser.parse_args()

    out = build(refresh=args.refresh)
    if args.report:
        tags = load_data.load_tags()
        tracking = load_data.load_tracking()
        mapping = map_teams(tags["team"].tolist(), team_names(require(["atd_meta"])[0]))
        home_away = {v: c for v, (c, _) in mapping.items()}
        _, evidence = compute(tags, len(out["restart"]), home_away, tracking)
        time_base = load_smart_tagging.check_time_base(load_smart_tagging.load())
        print("\n".join(report_lines(tags, out, mapping, ball.scored_blocks(), time_base,
                                      evidence, tracking)))
    else:
        for name in ARRAYS:
            print("{:<8} {} {}".format(name, out[name].shape, out[name].dtype))
